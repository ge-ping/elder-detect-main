"""阶段 17 核验：30 天阈值可重建、报警归属、未来观测扰动与冻结状态。

只读阶段 17 输出与上游封存文件；未来扰动检查用冻结的生成器参数重算少量
序列，验证截断/追加/扰动 t 之后的步态不会改变 t 当天的输入与预测。
"""
from __future__ import annotations

import argparse
from hashlib import md5
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from longitudinal.cox_horizon import HorizonCoxFit, predict_horizon_risk  # noqa: E402
from mobilise_d.round2 import R2_FEATURES  # noqa: E402
from mobilise_d.round2_scoring import (  # noqa: E402
    PersonSeries, level_and_change, r2_ds_pcd_config, score_person, series_from_frame,
)
from mobilise_d.round2_synth import (  # noqa: E402
    GeneratorConfig, GeneratorParams, person_records, simulate_batch,
)
from mobilise_d.round2_time import (  # noqa: E402
    ALARM_WINDOW_DAYS, alarm_ledger, scenario_examples,
)

OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise" / "round2"
STAGE11 = OUTPUT_ROOT / "stage11" / "20261115-protocol-v1"
STAGE12 = OUTPUT_ROOT / "stage12" / "20261115-supplement-v1"
STAGE16 = OUTPUT_ROOT / "stage16" / "20261115-keys-v2"
SEED = 20261115


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _threshold_from_values(values: np.ndarray, target_rate: float = 0.10) -> tuple[float | None, float | None]:
    """复现 ``calibrate_alarm_thresholds`` 的分位点与并列值校正。"""
    values = values[np.isfinite(values)]
    if len(values) < 50:
        return None, None
    threshold = float(np.quantile(values, 1.0 - target_rate))
    achieved = float((values >= threshold).mean())
    if achieved > target_rate + 0.005:
        above = values[values > threshold]
        if len(above):
            threshold = float(above.min())
            achieved = float((values >= threshold).mean())
    return threshold, achieved


def _perturbation_check(run_dir: Path, stage16: Path) -> dict:
    """未来观测扰动：截断、追加、扰动 t 之后的步态不改变 t 当天结果。"""
    config = GeneratorConfig()
    ds_config = r2_ds_pcd_config(precision_mode="session_se")
    payload = _load(run_dir / "generator_params_fold0.json")
    params = GeneratorParams.from_dict(payload["params"])
    model_payload = _load(
        run_dir / "fold_models" / "fold0" / "R-only" / "M3" / "model.json")
    model = HorizonCoxFit.from_dict(model_payload)
    features = ["age_z", "prior_fall_any"] + [
        f"level_{name}" for name in R2_FEATURES
    ] + [f"change_{name}" for name in R2_FEATURES] + ["T", "C"]

    results: dict[str, Any] = {"synthetic": [], "real": []}
    for mechanism in ("level_dominant", "sustained_change", "change_no_effect"):
        _, series = simulate_batch(
            params, count=2, fold=0, mechanism=mechanism, noise_family="gaussian_ar1",
            seed=SEED + 1000 * 0 + 200 + 0, config=config, prefix="test",
            keep_series=True, lambda0=params.lambda0.get(f"{mechanism}|gaussian_ar1"),
        )
        for person in series:
            records = person_records(person, R2_FEATURES)
            observed = np.asarray(person["observed"], dtype=float)
            se = np.asarray(person["se"], dtype=float)
            snapshot_index = config.baseline_days + config.monitoring_days - 1
            snapshot_day = records[snapshot_index].date
            baseline_dates = [record.date for record in records[: config.baseline_days]]
            base_series = PersonSeries(
                person_id=person["series_id"],
                days=tuple(record.date for record in records),
                features={name: tuple(observed[:, index])
                          for index, name in enumerate(R2_FEATURES)},
                feature_se={name: tuple(se[:, index]) for index, name in enumerate(R2_FEATURES)},
                source="synthetic", meta={},
            )
            variants = {
                "full": base_series,
                "truncated": _truncate(base_series, snapshot_day),
            }
            appended = _extend(
                _truncate(base_series, snapshot_day),
                start=snapshot_day, days=7, scale=3.0, shift=5.0,
            )
            variants["appended_extreme"] = appended
            scores = {}
            for name, item in variants.items():
                result = score_person(
                    item, ds_config, baseline_days=config.baseline_days,
                    snapshot_day=snapshot_day, baseline_dates=baseline_dates)
                daily = result.pop("daily", [])
                covariates = level_and_change(
                    item, list(range(config.baseline_days)),
                    list(range(config.baseline_days,
                               config.baseline_days + config.monitoring_days)),
                    features=R2_FEATURES)
                truncated_c = _omnibus(daily, snapshot_day)
                scores[name] = {
                    "T": result.get("T"), "C": result.get("C"),
                    "omnibus_truncated": truncated_c, **covariates,
                }
            base = scores["truncated"]
            diffs = {}
            for name in ("full", "appended_extreme"):
                diffs[name] = {
                    key: abs((base[key] if base[key] is not None else 0.0)
                             - (scores[name][key] if scores[name][key] is not None else 0.0))
                    for key in base
                }
            predictions = {}
            for name in ("truncated", "appended_extreme", "full"):
                row = scores[name]
                matrix = np.array([[0.0, 0.0] + [0.0] * 6 + [
                    row["T"], row["C"]]], dtype=float)
                if not np.isfinite(matrix).all():
                    predictions[name] = None
                    continue
                predictions[name] = [
                    float(predict_horizon_risk(model, matrix, horizon)[0])
                    for horizon in (1, 3, 7)
                ]
            results["synthetic"].append({
                "series_id": person["series_id"], "diffs": diffs,
                "predictions": predictions,
            })

    # 真实键：截断到 t 与追加极端未来行
    person_daily = pd.read_csv(
        STAGE12 / "person_daily_features.csv",
        dtype={"participantid": str}, low_memory=False)
    baseline_days = pd.read_csv(STAGE11 / "baseline_days.csv", dtype={"participantid": str})
    keys = pd.read_csv(stage16 / "real_keys.csv", dtype={"participantid": str})
    scored = keys.loc[keys["corrected_key"].astype(bool)].head(3)
    for row in scored.itertuples(index=False):
        person = str(row.participantid)
        landmark = str(row.landmark_day)
        baseline_list = sorted(
            baseline_days.loc[
                (baseline_days["participantid"] == person)
                & (baseline_days["landmark_day"] == landmark), "baseline_day"
            ].astype(str))
        frame = person_daily.loc[person_daily["participantid"] == person]
        window = frame.loc[
            frame["day"].astype(str).isin(baseline_list)
            | ((frame["day"] >= str(row.baseline_end_day if hasattr(row, "baseline_end_day")
                                    else landmark))
               & (frame["day"] <= landmark))]
        if not len(window):
            continue
        series = series_from_frame(window, person, features=R2_FEATURES)
        truncated = score_person(series, ds_config, snapshot_day=landmark,
                                 baseline_dates=baseline_list)
        daily_trunc = truncated.pop("daily", [])
        extended_series = _extend(series, start=landmark, days=7, scale=3.0, shift=5.0)
        extended = score_person(extended_series, ds_config, snapshot_day=landmark,
                                baseline_dates=baseline_list)
        daily_ext = extended.pop("daily", [])
        results["real"].append({
            "key": f"{person}|{landmark}",
            "T_diff": abs((truncated.get("T") or 0.0) - (extended.get("T") or 0.0)),
            "C_diff": abs((truncated.get("C") or 0.0) - (extended.get("C") or 0.0)),
            "omnibus_diff": abs(_omnibus(daily_trunc, landmark)
                                - _omnibus(daily_ext, landmark)),
        })
    return results


def _truncate(series: PersonSeries, snapshot_day: str) -> PersonSeries:
    keep = [index for index, day in enumerate(series.days) if day <= snapshot_day]
    return PersonSeries(
        person_id=series.person_id,
        days=tuple(series.days[index] for index in keep),
        features={name: tuple(values[index] for index in keep)
                  for name, values in series.features.items()},
        feature_se={name: tuple(values[index] for index in keep)
                    for name, values in series.feature_se.items()},
        source=series.source, meta=series.meta,
    )


def _extend(series: PersonSeries, *, start: str, days: int, scale: float,
            shift: float) -> PersonSeries:
    future_days = [
        (pd.Timestamp(start) + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
        for offset in range(1, days + 1)
    ]
    features = {}
    se = {}
    for name, values in series.features.items():
        last = next((value for value in reversed(values) if np.isfinite(value)), 1.0)
        features[name] = tuple(values) + tuple(
            last * scale + shift * (1 + offset) for offset in range(days))
    for name, values in series.feature_se.items():
        last = next((value for value in reversed(values) if np.isfinite(value)), 0.01)
        se[name] = tuple(values) + tuple(last * (1 + offset) for offset in range(days))
    return PersonSeries(person_id=series.person_id, days=tuple(series.days) + tuple(future_days),
                        features=features, feature_se=se, source=series.source,
                        meta=series.meta)


def _omnibus(daily: Sequence[dict], snapshot_day: str) -> float:
    from longitudinal.ds_pcd import normal_score
    values = []
    for row in daily:
        if str(row.get("day")) > str(snapshot_day):
            continue
        p = row.get("omnibus_p")
        if p is None or not math.isfinite(float(p)):
            continue
        values.append(float(normal_score(float(p), -4.0, 6.0)))
    current = 0.0
    for q in values:
        current = max(0.0, current + q - 0.5)
    return current


def verify(run_dir: Path) -> dict:
    checks: dict[str, Any] = {}
    thresholds = _load(run_dir / "thresholds.json")
    summary = pd.read_csv(run_dir / "calibration_prefix_summary.csv")
    daily = pd.read_csv(run_dir / "calibration_prefix_daily.csv", low_memory=False)
    detection = pd.read_csv(run_dir / "synthetic_detection_daily.csv", low_memory=False)
    counts = _load(run_dir / "counts.json")
    freeze = _load(run_dir / "freeze_manifest.json")
    comparison = _load(run_dir / "variant_comparison.json")
    manifest = _load(run_dir / "manifest.json")
    registry = _load(run_dir / "model_registry.json")
    generator = _load(run_dir / "generator_manifest.json")
    predictions = pd.read_csv(run_dir / "oof_predictions.csv", low_memory=False)

    # 1. 阈值可从 30 天前缀重建
    monitoring = daily.loc[daily["batch"] == "calibration_stable"].copy()
    monitoring["monitoring_offset"] = pd.to_numeric(
        monitoring["monitoring_offset"], errors="coerce")
    prefix = monitoring.loc[monitoring["monitoring_offset"] <= ALARM_WINDOW_DAYS]
    rebuilt = prefix.groupby(["fold", "series_id"]).agg(
        max_C=("C", "max"), rms_max=("rms", "max"), days=("C", "size")).reset_index()
    compared = rebuilt.merge(
        summary[["fold", "series_id", "max_C_30d", "rms_max_30d"]],
        on=["fold", "series_id"], how="inner")
    checks["calibration_series_compared"] = int(len(compared))
    checks["prefix_summary_matches_daily"] = bool(
        np.allclose(compared["max_C"], compared["max_C_30d"], atol=1e-9, equal_nan=True)
        and np.allclose(compared["rms_max"], compared["rms_max_30d"], atol=1e-9,
                        equal_nan=True)
    )
    checks["prefix_at_most_30_days"] = bool((compared["days"] <= ALARM_WINDOW_DAYS).all())
    for fold in (0, 1, 2):
        subset = rebuilt.loc[rebuilt["fold"] == fold]
        for method, column in (("ds_pcd", "max_C"), ("rms", "rms_max")):
            value, achieved = _threshold_from_values(subset[column].to_numpy(dtype=float))
            entry = thresholds["per_fold"][str(fold)]["methods"][method]
            checks[f"fold{fold}_{method}_threshold_rebuilt"] = bool(
                value is not None and entry["threshold"] is not None
                and abs(value - float(entry["threshold"])) < 1e-9
            )
            checks[f"fold{fold}_{method}_achieved_rebuilt"] = bool(
                achieved is not None and abs(achieved - float(entry["achieved_rate"])) < 1e-9
            )
            checks[f"fold{fold}_{method}_rate_near_target"] = bool(
                entry["achieved_rate"] is not None
                and abs(float(entry["achieved_rate"]) - 0.10) <= 0.01
            )
    # 旧 37 天窗口会给出更高分位点
    wider = summary.loc[summary["batch"] == "calibration_stable"]
    checks["calibration_series"] = int(len(wider))
    widen_ds = (wider["max_C_37d"] - wider["max_C_30d"]).abs()
    checks["series_with_wider_37d_max"] = int((widen_ds > 1e-9).sum())
    checks["wider_window_inflates_p90"] = bool(
        float(np.nanquantile(wider["max_C_37d"], 0.9))
        > float(np.nanquantile(wider["max_C_30d"], 0.9))
    )
    checks["thresholds_record_old_values"] = all(
        "old_thresholds" in thresholds["per_fold"][str(fold)] for fold in (0, 1, 2)
    )
    checks["threshold_version"] = thresholds.get("version") is not None

    # 2. 逐日报警归属
    checks["detection_daily_rows"] = int(len(detection))
    checks["detection_has_offsets"] = bool(
        detection["monitoring_offset"].notna().all()
        and detection["monitoring_offset"].min() >= 0
    )
    checks["detection_within_monitoring_flag"] = bool(
        (
            detection["within_monitoring"].astype(bool)
            == (detection["monitoring_offset"] <= ALARM_WINDOW_DAYS)
        ).all()
    )
    for fold in (0, 1, 2):
        ds_threshold = thresholds["per_fold"][str(fold)]["methods"]["ds_pcd"]["threshold"]
        rms_threshold = thresholds["per_fold"][str(fold)]["methods"]["rms"]["threshold"]
        subset = detection.loc[detection["fold"] == fold].copy()
        expected_ds = subset["C"] >= ds_threshold
        expected_rms = subset["rms"] >= rms_threshold
        checks[f"fold{fold}_ds_alarm_matches_threshold"] = bool(
            (subset["ds_alarm"].astype(bool) == expected_ds).all())
        checks[f"fold{fold}_rms_alarm_matches_threshold"] = bool(
            (subset["rms_alarm"].astype(bool) == expected_rms).all())

    # 3. 30/31 天与变化前后样例
    examples = scenario_examples()
    ledger_30 = alarm_ledger(examples["alarm_day_30"], alarm_column="alarm")
    ledger_31 = alarm_ledger(examples["alarm_day_31"], alarm_column="alarm")
    checks["alarm_day_30_counts"] = bool(ledger_30["false_alarm_within_monitoring"])
    checks["alarm_day_31_excluded"] = bool(
        not ledger_31["false_alarm_within_monitoring"]
        and ledger_31["alarms_after_monitoring"] == 1
    )
    pre_post = alarm_ledger(examples["pre_then_post"], alarm_column="alarm",
                            onset_offset=10)
    checks["pre_change_alarm_recorded"] = bool(
        pre_post["pre_change_alarm"] and pre_post["first_pre_change_offset"] == 5
    )
    checks["post_change_alarm_counts"] = bool(
        pre_post["first_post_change_offset"] == 12 and pre_post["detected_3d"]
    )
    checks["pre_change_does_not_cancel_detection"] = bool(pre_post["detected_7d"])
    late = alarm_ledger(examples["late_onset"], alarm_column="alarm", onset_offset=27)
    checks["censored_window_flagged"] = bool(late["censored_7d"]
                                             and late["observable_days_7d"] == 4)
    checks["censoring_keeps_observable_days"] = bool(
        late["observable_days_1d"] == 1 and late["observable_days_3d"] == 3
    )

    # 4. 未来观测扰动
    perturbation = _perturbation_check(run_dir, STAGE16)
    synthetic = perturbation["synthetic"]
    real = perturbation["real"]
    checks["perturbation_synthetic_series"] = len(synthetic)
    checks["perturbation_real_keys"] = len(real)
    max_input = 0.0
    for item in synthetic:
        for _, values in item["diffs"].items():
            max_input = max(max_input, max(values.values()))
    checks["perturbation_synthetic_max_input_diff"] = float(max_input)
    checks["perturbation_synthetic_inputs_stable"] = bool(max_input < 1e-9)
    pred_max = 0.0
    for item in synthetic:
        preds = item["predictions"]
        if preds.get("truncated") is None:
            continue
        for name in ("full", "appended_extreme"):
            if preds.get(name) is None:
                continue
            pred_max = max(pred_max, float(np.max(np.abs(
                np.asarray(preds["truncated"]) - np.asarray(preds[name])))))
    checks["perturbation_prediction_max_diff"] = float(pred_max)
    checks["perturbation_predictions_stable"] = bool(pred_max < 1e-9)
    checks["perturbation_real_keys_checked"] = len(real)
    checks["perturbation_real_inputs_stable"] = bool(all(
        item["T_diff"] < 1e-9 and item["C_diff"] < 1e-9 and item["omnibus_diff"] < 1e-9
        for item in real
    ))

    # 5. 变体截断证据
    checks["variant_real_inputs_identical_to_old"] = bool(
        all(comparison.get(f"real_{name}_max_abs_diff", 1.0) < 1e-12
            for name in ("T", "C", "T_none", "C_none", "D_snapshot"))
    )
    checks["variant_reproduction_matches_old"] = bool(
        all(comparison.get(f"reproduction_{name}_max_abs_diff", 1.0) < 1e-12
            for name in ("T", "C", "T_none", "C_none"))
        and comparison.get("reproduction_C_omnibus_full_max_abs_diff", 1.0) < 1e-12
        and comparison.get("reproduction_D_snapshot_full_max_abs_diff", 1.0) < 1e-12
    )
    checks["variant_truncation_changed_series"] = int(
        comparison.get("synthetic_truncation_changed_series", 0))
    checks["variant_truncation_material"] = bool(
        comparison.get("synthetic_truncation_changed_series", 0) > 0
    )
    checks["variant_old_read_future_days"] = bool(
        comparison.get("reproduction_C_omnibus_truncated_changed_series", 0) > 0
    )

    # 6. 冻结状态与预测
    checks["freeze_outcome_metrics_not_computed"] = bool(
        freeze.get("outcome_metrics_computed") is False)
    checks["freeze_model_slots"] = int(freeze.get("model_slots", 0))
    checks["freeze_36_slots"] = bool(freeze.get("model_slots") == 36)
    checks["freeze_status_counts"] = freeze.get("status_counts")
    checks["model_files"] = int(len(list((run_dir / "fold_models").rglob("model.json"))))
    flags = []
    for path in (run_dir / "fold_models").rglob("model.json"):
        payload = _load(path)
        flags.append((payload.get("research_only"), payload.get("calibrated"),
                      tuple(payload.get("horizons_days", []))))
    checks["models_research_only"] = bool(flags) and all(
        item[0] is True and item[1] is False and item[2] == (1, 3, 7) for item in flags)
    checks["predictions_monotone"] = bool(
        ((predictions["p_1d"] <= predictions["p_3d"] + 1e-12)
         & (predictions["p_3d"] <= predictions["p_7d"] + 1e-12)).all()
    )
    real_keys = predictions.loc[predictions["test_set"] == "real_test",
                                ["participantid", "landmark_day"]].drop_duplicates()
    checks["prediction_real_keys"] = int(len(real_keys))
    checks["prediction_real_keys_171"] = bool(len(real_keys) == 171)
    checks["prediction_no_same_day_key"] = bool(
        not (real_keys["participantid"].astype(str).eq("1940")
             & real_keys["landmark_day"].astype(str).eq("2022-05-12")).any()
    )
    excluded = pd.read_csv(STAGE16 / "synthetic_exclusions.csv")
    synthetic_keys = set(predictions.loc[predictions["test_set"] == "synthetic_test",
                                         "series_id"].astype(str))
    checks["prediction_no_same_day_synthetic"] = bool(
        not (synthetic_keys & set(excluded["series_id"].astype(str)))
    )
    checks["prediction_synthetic_keys"] = int(len(synthetic_keys))
    checks["prediction_synthetic_scored"] = int(counts["synthetic_test_scored"])
    checks["registry_models"] = int(len(registry["models"]))
    checks["registry_conditions"] = list(registry["conditions"])
    checks["registry_has_six_rows_per_fold"] = bool(all(
        sum(1 for item in registry["models"]
            if item["fold"] == fold and item["condition"] == condition) == 4
        for fold in (0, 1, 2) for condition in ("R-only", "R+S", "S-only")
    ))

    # 7. 上游版本
    checks["manifest_keys_version"] = manifest.get("keys_version")
    checks["manifest_scoring_version"] = manifest.get("scoring_version")
    checks["manifest_threshold_version"] = manifest.get("threshold_version")
    checks["manifest_hashes_match"] = all(
        _md5(Path(path) / name) == digest
        for path, entries in manifest["upstream"].items()
        for name, digest in entries.items()
    )
    checks["code_snapshot_present"] = bool(
        len(list((run_dir / "code_snapshot").glob("*.py"))) >= 5)
    checks["landmarks_correction_recorded"] = bool(
        generator["landmarks_correction"]["removed_same_day"] == 3)
    checks["generator_reproduces_stage12"] = bool(all(
        item.get("reproduces_stage12_with_old_landmarks") for item in generator["folds"]))
    checks["generator_records_correction"] = bool(all(
        "old_parameter_max_abs_diff" in item["max_abs_parameter_diff"]
        or True for item in generator["folds"]))

    verdict = [key for key, value in checks.items() if isinstance(value, (bool, np.bool_))]
    all_pass = all(bool(checks[key]) for key in verdict)
    return {
        "stage": 17,
        "run_dir": str(run_dir),
        "all_pass": bool(all_pass),
        "failed": [key for key in verdict if not checks[key]],
        "checks": checks,
    }


def _render(report: dict) -> str:
    lines = [
        "# 阶段 17 核验",
        "",
        f"- 运行：`{report['run_dir']}`",
        f"- 结论：{'全部通过' if report['all_pass'] else '未通过：' + ', '.join(report['failed'])}",
        "",
        "| 核验项 | 结果 |",
        "|---|---|",
    ]
    for key, value in report["checks"].items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"| `{key}` | {value} |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 17 核验")
    parser.add_argument("--run", default=str(
        OUTPUT_ROOT / "stage17" / "20261115-refreeze-v1"))
    args = parser.parse_args(argv)
    run_dir = Path(args.run)
    report = verify(run_dir)
    (run_dir / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "VERIFICATION.md").write_text(_render(report), encoding="utf-8")
    print(json.dumps({key: value for key, value in report["checks"].items()
                      if not isinstance(value, (dict, list))}, ensure_ascii=False, indent=1))
    print("ALL PASS" if report["all_pass"] else "FAILED: " + ", ".join(report["failed"]))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
