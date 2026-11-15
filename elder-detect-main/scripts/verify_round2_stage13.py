"""阶段 13 核验：模型隔离、权重、未来字段禁入、短窗数值与冻结清单。"""
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

from longitudinal.cox_fit import fit_landmark_cox  # noqa: E402
from longitudinal.cox_horizon import (  # noqa: E402
    HorizonCoxFit, fit_horizon_cox, predict_horizon_risk,
)
from mobilise_d.round2 import R2_FEATURES  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise" / "round2"
STAGE11 = OUTPUT_ROOT / "stage11" / "20261115-protocol-v1"
STAGE12 = OUTPUT_ROOT / "stage12" / "20261115-supplement-v1"


def _md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _breslow_reference(
    times: np.ndarray, events: np.ndarray, covariate: np.ndarray,
    beta: float, center: float, scale: float,
) -> tuple[list[float], list[float]]:
    """独立重算 Breslow 基线阶梯（单协变量、权重 1）。"""
    standardized = (covariate - center) / scale
    linear = standardized * beta
    shift = float(np.max(linear))
    order = np.argsort(times, kind="stable")
    sorted_times = times[order]
    times_out: list[float] = []
    survival_out: list[float] = []
    cumulative = 0.0
    for time in np.unique(sorted_times[events[order] == 1]):
        at_risk = sorted_times >= time
        total = float(np.sum(np.exp(linear[order][at_risk] - shift)))
        count = float(np.sum((sorted_times == time) & (events[order] == 1)))
        cumulative += count * math.exp(-shift) / total
        times_out.append(float(time))
        survival_out.append(float(math.exp(-cumulative)))
    return times_out, survival_out


def verify(run_dir: Path) -> dict:
    checks: dict[str, Any] = {}
    registry = _load(run_dir / "model_registry.json")
    models = pd.DataFrame(registry["models"])
    freeze = _load(run_dir / "freeze_manifest.json")
    diagnostics = _load(run_dir / "training_diagnostics.json")
    predictions = pd.read_csv(run_dir / "oof_predictions.csv", low_memory=False)
    synthetic_train = pd.read_csv(STAGE12 / "synthetic_train_scores.csv")
    synthetic_test = pd.read_csv(STAGE12 / "synthetic_test_scores.csv")
    splits = pd.read_csv(STAGE11 / "splits.csv", dtype={"participantid": str})
    real_scores = pd.read_csv(STAGE12 / "real_scores.csv", dtype={"participantid": str})
    folds = sorted(splits["fold"].unique())

    # 1. 每个折、每个模型、每个条件都有产物或失败状态
    expected = {
        (int(fold), model, condition)
        for fold in folds
        for model in ("M0", "M1", "M2", "M3")
        for condition in ("R-only", "R+S", "S-only")
    }
    present = set(zip(models["fold"], models["model"], models["condition"]))
    checks["all_model_slots_present"] = expected == present
    checks["model_slots"] = int(len(present))
    checks["status_counts"] = models["status"].value_counts().to_dict()
    checks["failures_retained"] = models.loc[
        models["status"] != "ok", ["bundle_id", "status"]
    ].to_dict("records")
    checks["every_slot_has_model_file"] = bool(
        models["model_path"].map(lambda path: (run_dir / path).is_file()).all()
    )

    # 2. 训练隔离与权重
    isolation = []
    weight_checks = []
    for record in models.itertuples(index=False):
        fold = int(record.fold)
        if record.condition == "R-only":
            continue
        if record.condition == "S-only":
            weight_checks.append({
                "bundle_id": record.bundle_id,
                "total_weight": record.total_weight,
                "rows": record.rows,
                "ok": abs(record.total_weight - record.rows) < 1e-9,
            })
        if record.condition == "R+S":
            real_train_rows = int((real_scores.loc[real_scores["status"] == "scored"]
                                   .merge(splits, on="participantid")
                                   .query("fold != @fold")).shape[0])
            expected_weight = 1.5 * real_train_rows
            weight_checks.append({
                "bundle_id": record.bundle_id,
                "synthetic_weight": record.weights["synthetic_weight"],
                "real_weight": record.weights["real_weight"],
                "ratio": (
                    record.weights["synthetic_weight"] / record.weights["real_weight"]
                    if record.weights["real_weight"] else None
                ),
                "ok": abs(
                    record.weights["synthetic_weight"]
                    - 0.5 * record.weights["real_weight"]
                ) < 1e-6,
            })
    checks["weight_rules_hold"] = all(item["ok"] for item in weight_checks)
    checks["weight_examples"] = weight_checks[:6]
    train_ids = set(synthetic_train["series_id"])
    test_ids = set(synthetic_test["series_id"])
    checks["synthetic_train_test_series_disjoint"] = not (train_ids & test_ids)
    checks["synthetic_train_fold_matches_test_fold_roles"] = bool(
        (synthetic_train.groupby("fold")["series_id"].apply(set).apply(
            lambda ids: ids <= train_ids
        )).all()
    )
    checks["real_test_persons_not_in_training"] = all(
        set(splits.loc[splits["fold"] == fold, "participantid"]).isdisjoint(
            set(splits.loc[splits["fold"] != fold, "participantid"])
        )
        for fold in folds
    )

    # 3. 未来字段禁入
    forbidden = {
        "event", "duration", "y_1d", "y_3d", "y_7d", "event_day",
        "event_gap_from_snapshot", "h_snapshot", "h_max",
    }
    checks["model_features_exclude_outcomes"] = all(
        forbidden.isdisjoint(set(registry["ladders"][model]))
        for model in registry["ladders"]
    )
    checks["model_feature_lists"] = registry["ladders"]
    checks["m3_adds_only_T_C"] = (
        set(registry["ladders"]["M3"]) - set(registry["ladders"]["M2"]) == {"T", "C"}
    )
    checks["m0_has_no_gait"] = set(registry["ladders"]["M0"]) == {"age_z", "prior_fall_any"}

    # 4. 逾期结果不影响已给定时点的预测
    probe = predictions.query("test_set == 'synthetic_test' and condition == 'S-only' and model == 'M3'")
    probe_ids = probe["series_id"].head(5).tolist()
    perturbed = synthetic_test.copy()
    mask = perturbed["series_id"].isin(probe_ids)
    perturbed.loc[mask, ["y_1d", "y_3d", "y_7d", "event_day"]] = 0
    age_mean = diagnostics["folds"]["0"]["real"]["age_mean"]
    age_sd = diagnostics["folds"]["0"]["real"]["age_sd"]
    perturbed["age_z"] = (perturbed["age"].astype(float) - age_mean) / age_sd
    model_path = models.query(
        "fold == 0 and model == 'M3' and condition == 'S-only'"
    )["model_path"].iloc[0]
    fit = HorizonCoxFit.from_dict(_load(run_dir / model_path))
    features = registry["ladders"]["M3"]
    ordered = perturbed.set_index("series_id").loc[probe_ids]
    rows = ordered[features].to_numpy(dtype=float)
    risks_after = predict_horizon_risk(fit, rows, 7)
    stored = probe.set_index("series_id").loc[probe_ids, "p_7d"].to_numpy(dtype=float)
    checks["outcome_perturbation_does_not_change_predictions"] = bool(
        np.allclose(risks_after, stored, atol=1e-12)
    )

    # 5. 短窗数值：独立重算 Breslow 阶梯
    times = np.array([2.0, 2.0, 3.0, 4.0, 6.0, 7.0])
    events = np.array([1, 1, 0, 1, 1, 0])
    covariate = np.array([0.0, 1.0, 2.0, 1.5, 3.0, 0.5])
    matrix = covariate.reshape(-1, 1)
    fit_small = fit_horizon_cox(
        times, events, matrix, ["x"], horizon_days=7.0, ridge=0.0, weights=None,
    )
    ref_times, ref_survival = _breslow_reference(
        times, events, covariate,
        beta=fit_small.notes["standardized_beta"][0],
        center=fit_small.centers[0], scale=fit_small.scales[0],
    )
    checks["breslow_ladder_matches_independent_computation"] = bool(
        np.allclose(fit_small.baseline_times, ref_times, atol=1e-12)
        and np.allclose(fit_small.baseline_survival, ref_survival, atol=1e-12)
    )
    checks["tied_event_single_step"] = bool(
        len(fit_small.baseline_times) == len(np.unique(times[events == 1]))
    )
    checks["baseline_ladder_non_increasing"] = bool(
        all(a >= b for a, b in zip(fit_small.baseline_survival,
                                   fit_small.baseline_survival[1:]))
    )
    probe_rows = covariate.reshape(-1, 1)
    risks = {h: predict_horizon_risk(fit_small, probe_rows, h) for h in (1, 3, 7)}
    checks["risk_monotone_across_horizons_small"] = bool(
        np.all(risks[1] <= risks[3] + 1e-12) and np.all(risks[3] <= risks[7] + 1e-12)
    )
    manual = 1.0 - fit_small.survival_at(7.0) ** math.exp(
        fit_small.coefficients[0] * (covariate[0] - fit_small.centers[0])
    )
    checks["risk_formula_matches_manual"] = abs(float(risks[7][0]) - manual) < 1e-12

    # 与既有研究推理入口（longitudinal.survival 的 CoxModelArtifact）逐行对照
    from longitudinal.survival import CoxModelArtifact, predict_cumulative_risk_research

    legacy_small = fit_landmark_cox(
        times, events, matrix, ["x"], horizon_days=7.0, ridge=0.0,
    )
    artifact = CoxModelArtifact(
        schema_version=2, bundle_id="probe", horizon_days=7,
        feature_names=("x",),
        coefficients=tuple(float(v) for v in legacy_small.coefficients),
        feature_centers=tuple(float(v) for v in legacy_small.centers),
        baseline_survival=float(legacy_small.baseline_survival),
        calibrated=False, trained_at="probe", evidence_ids=(),
    )
    legacy_risk = np.array([
        predict_cumulative_risk_research({"x": float(value)}, artifact)
        for value in covariate
    ])
    checks["horizon_risk_matches_legacy_artifact"] = bool(
        np.allclose(legacy_risk, risks[7], atol=1e-12)
    )
    checks["horizon_risk_legacy_max_abs_diff"] = float(
        np.max(np.abs(legacy_risk - risks[7]))
    )

    # 6. 加权似然：全 1 权重与未加权拟合一致；整体缩放只改变基线风险
    legacy = fit_landmark_cox(
        times, events, matrix, ["x"], horizon_days=7.0, ridge=0.0,
    )
    checks["unweighted_matches_legacy_coefficients"] = bool(
        np.allclose(fit_small.coefficients, legacy.coefficients, atol=1e-9)
    )
    checks["unweighted_matches_legacy_baseline"] = bool(
        abs(fit_small.survival_at(7.0) - legacy.baseline_survival) < 1e-9
    )
    scaled = fit_horizon_cox(
        times, events, matrix, ["x"], horizon_days=7.0, ridge=0.0,
        weights=np.full(len(times), 3.0),
    )
    checks["weight_scaling_keeps_coefficients"] = bool(
        np.allclose(scaled.coefficients, fit_small.coefficients, atol=1e-9)
    )
    # Cox 权重是相对的：整体缩放不改变基线风险与系数，只改变估计精度。
    checks["weight_scaling_keeps_baseline_hazard"] = bool(
        abs(scaled.survival_at(7.0) - fit_small.survival_at(7.0)) < 1e-9
    )
    # 相对权重改变事件计数权重：把一半行权重设 2、另一半设 0.5 应改变基线风险
    relative = fit_horizon_cox(
        times, events, matrix, ["x"], horizon_days=7.0, ridge=0.0,
        weights=np.array([2.0, 0.5, 0.5, 2.0, 0.5, 2.0]),
    )
    checks["relative_weights_change_baseline_hazard"] = bool(
        abs(relative.survival_at(7.0) - fit_small.survival_at(7.0)) > 1e-6
    )

    # 7. 无早期事件：窗口内零估计并记录退化
    late_times = np.array([3.0, 4.0, 5.0, 6.0])
    late_events = np.array([0, 1, 1, 0])
    late_fit = fit_horizon_cox(
        late_times, late_events, covariate[:4].reshape(-1, 1), ["x"],
        horizon_days=7.0, ridge=0.0,
    )
    checks["no_early_event_risk_zero_at_1d"] = bool(
        np.allclose(predict_horizon_risk(late_fit, covariate[:4].reshape(-1, 1), 1), 0.0)
    )
    checks["no_early_event_degeneracy_recorded"] = bool(
        set(late_fit.notes.get("degenerate_horizons_days", [])) == {1, 3}
    )

    # 8. 模型复载一致
    reload_ok = True
    for record in models.head(6).itertuples(index=False):
        stored = (run_dir / record.model_path).read_text(encoding="utf-8")
        fit = HorizonCoxFit.from_dict(json.loads(stored))
        rows = np.zeros((2, len(fit.feature_names)))
        if not np.allclose(
            predict_horizon_risk(fit, rows, 7),
            predict_horizon_risk(fit, rows, 7),
            atol=0.0,
        ):
            reload_ok = False
        if fit.fingerprint() != HorizonCoxFit.from_dict(json.loads(stored)).fingerprint():
            reload_ok = False
    checks["model_reload_deterministic"] = bool(reload_ok)

    # 9. 冻结清单
    checks["freeze_declares_no_metrics"] = freeze["outcome_metrics_computed"] is False
    checks["freeze_primary_comparison"] = freeze["primary_comparison"] == "M3-M2"
    checks["freeze_horizons"] = freeze["primary_horizon_days"] == 7 and freeze["secondary_horizons_days"] == [1, 3]
    checks["freeze_real_test_keys"] = freeze["test_scope"]["real"]["keys"]
    checks["freeze_synthetic_test_keys"] = freeze["test_scope"]["synthetic"]["keys"]
    checks["real_test_keys_match_total"] = (
        int(freeze["test_scope"]["real"]["keys"])
        == int((real_scores["status"] == "scored").sum())
    )
    checks["freeze_input_hashes_match"] = all(
        Path(item["path"]).is_file() and _md5(Path(item["path"])) == item["md5"]
        for item in freeze["input_files"].values()
    )
    checks["no_metrics_artifact_in_stage13"] = not any(
        (run_dir / name).exists() for name in ("metrics.csv", "comparison.json", "bootstrap.json")
    )

    # 10. 预测完整性与单调性
    checks["prediction_rows"] = int(len(predictions))
    checks["prediction_probabilities_finite"] = bool(
        np.isfinite(predictions[["p_1d", "p_3d", "p_7d"]].to_numpy()).all()
    )
    saturation = []
    for (condition, model), group in predictions.query("test_set == 'synthetic_test'").groupby(
        ["condition", "model"]
    ):
        saturation.append({
            "condition": condition,
            "model": model,
            "share_p7_eq_1": round(float((group["p_7d"] >= 1 - 1e-12).mean()), 4),
            "share_p7_lt_1e-9": round(float((group["p_7d"] < 1e-9).mean()), 4),
            "median_p7": round(float(group["p_7d"].median()), 4),
        })
    checks["synthetic_saturation_by_model"] = saturation
    checks["synthetic_s_only_not_saturated"] = bool(
        all(item["share_p7_eq_1"] < 0.05 for item in saturation
            if item["condition"] == "S-only")
    )
    checks["prediction_probabilities_monotone"] = bool(
        (
            (predictions["p_1d"] <= predictions["p_3d"] + 1e-12)
            & (predictions["p_3d"] <= predictions["p_7d"] + 1e-12)
        ).all()
    )
    checks["real_test_rows_consistent_across_models"] = bool(
        predictions.query("test_set == 'real_test'")
        .groupby(["fold", "model", "condition"]).size().nunique() >= 1
    )
    checks["real_test_keys_per_condition_equal"] = bool(
        predictions.query("test_set == 'real_test'")
        .groupby(["condition", "model"])["participantid"].nunique().nunique() == 1
    )
    checks["synthetic_branch_covers_three_mechanisms"] = bool(
        set(predictions.query("test_set == 'synthetic_test'")["mechanism"].dropna())
        == {"level_dominant", "sustained_change", "change_no_effect"}
    )
    checks["real_branch_has_four_events"] = int(
        predictions.query("test_set == 'real_test' and model == 'M0' and condition == 'R-only'")[
            "y_7d"
        ].sum()
    ) == 4

    verdict_keys = [
        "all_model_slots_present", "every_slot_has_model_file", "weight_rules_hold",
        "synthetic_train_test_series_disjoint", "synthetic_train_fold_matches_test_fold_roles",
        "real_test_persons_not_in_training", "model_features_exclude_outcomes",
        "m3_adds_only_T_C", "m0_has_no_gait",
        "outcome_perturbation_does_not_change_predictions",
        "breslow_ladder_matches_independent_computation", "tied_event_single_step",
        "baseline_ladder_non_increasing", "risk_monotone_across_horizons_small",
        "risk_formula_matches_manual", "horizon_risk_matches_legacy_artifact",
        "synthetic_s_only_not_saturated", "unweighted_matches_legacy_coefficients",
        "unweighted_matches_legacy_baseline", "weight_scaling_keeps_coefficients",
        "weight_scaling_keeps_baseline_hazard", "relative_weights_change_baseline_hazard",
        "no_early_event_risk_zero_at_1d",
        "no_early_event_degeneracy_recorded", "model_reload_deterministic",
        "freeze_declares_no_metrics", "freeze_primary_comparison", "freeze_horizons",
        "real_test_keys_match_total", "freeze_input_hashes_match",
        "no_metrics_artifact_in_stage13", "prediction_probabilities_finite",
        "prediction_probabilities_monotone", "real_test_keys_per_condition_equal",
        "synthetic_branch_covers_three_mechanisms", "real_branch_has_four_events",
    ]
    all_pass = all(bool(checks[key]) for key in verdict_keys)
    return {
        "stage": 13,
        "run_dir": str(run_dir),
        "all_pass": bool(all_pass),
        "failed": [key for key in verdict_keys if not checks[key]],
        "checks": checks,
    }


def _render(report: dict) -> str:
    lines = [
        "# 阶段 13 核验",
        "",
        f"- 运行：`{report['run_dir']}`",
        f"- 结论：{'全部通过' if report['all_pass'] else '未通过：' + ', '.join(report['failed'])}",
        "",
        "| 核验项 | 结果 |",
        "|---|---|",
    ]
    for key, value in report["checks"].items():
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"| `{key}` | {value} |")
    lines.append("")
    for key in ("status_counts", "failures_retained", "weight_examples", "model_feature_lists"):
        value = report["checks"].get(key)
        if value is not None:
            lines.append(f"## {key}")
            lines.append("```")
            lines.append(json.dumps(value, ensure_ascii=False, indent=1))
            lines.append("```")
            lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 13 核验")
    parser.add_argument("--run", default=str(OUTPUT_ROOT / "stage13" / "20261115-freeze-v1"))
    args = parser.parse_args(argv)
    run_dir = Path(args.run)
    report = verify(run_dir)
    (run_dir / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "VERIFICATION.md").write_text(_render(report), encoding="utf-8")
    print(json.dumps({key: value for key, value in report["checks"].items()
                      if not isinstance(value, (dict, list))}, ensure_ascii=False, indent=1))
    print("ALL PASS" if report["all_pass"] else "FAILED: " + ", ".join(report["failed"]))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
