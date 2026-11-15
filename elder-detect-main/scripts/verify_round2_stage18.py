"""阶段 18 核验：从逐条预测/报警/事件独立重算关键数字。

* 分母、7 天 ΔBrier、可估窗口的 ΔAUROC 全部从 ``paired_predictions.csv``
  重算（自己实现 Mann–Whitney 与 Brier），不调用评价脚本的函数；
* 误报与变化后检出从 ``synthetic_detection_daily.csv`` + 冻结阈值重算；
* 检查三项更正是否真的反映在表里（171 键、同日剔除、截断消融、30 天窗）；
* 与旧运行的差异表完整性；
* 复现目录（若存在）的键、点估计与容差。
"""
from __future__ import annotations

import argparse
from hashlib import md5
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise" / "round2"
STAGE16 = OUTPUT_ROOT / "stage16" / "20261115-keys-v2"
STAGE17 = OUTPUT_ROOT / "stage17" / "20261115-refreeze-v1"
STAGE18 = OUTPUT_ROOT / "stage18" / "20261115-corrected-eval-v1"
OLD_STAGE14 = OUTPUT_ROOT / "stage14" / "20261115-eval-v1"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Mann–Whitney U 形式的 AUROC（并列取平均秩）。"""
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(labels) & np.isfinite(scores)
    labels, scores = labels[mask], scores[mask]
    positives = labels.sum()
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0 or positives < 5:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    index = 0
    while index < len(sorted_scores):
        end = index
        while end + 1 < len(sorted_scores) and sorted_scores[end + 1] == sorted_scores[index]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        ranks[order[index:end + 1]] = average
        index = end + 1
    positive_ranks = ranks[labels == 1].sum()
    return float((positive_ranks - positives * (positives + 1) / 2.0)
                 / (positives * negatives))


def _brier_delta(frame: pd.DataFrame, base: str, other: str) -> dict:
    wide = frame.pivot_table(index=["key", "label"], columns="model",
                             values="prediction", aggfunc="first").reset_index()
    wide = wide.dropna(subset=[base, other, "label"])
    labels = wide["label"].to_numpy(dtype=float)
    a = wide[base].to_numpy(dtype=float)
    b = wide[other].to_numpy(dtype=float)
    return {
        "n": int(len(wide)),
        "positives": int(labels.sum()),
        "brier_a": float(np.mean((a - labels) ** 2)),
        "brier_b": float(np.mean((b - labels) ** 2)),
        "delta_brier": float(np.mean((b - labels) ** 2) - np.mean((a - labels) ** 2)),
        "auroc_a": _auroc(labels, a),
        "auroc_b": _auroc(labels, b),
        "delta_auroc": (
            _auroc(labels, b) - _auroc(labels, a)
            if _auroc(labels, b) is not None and _auroc(labels, a) is not None else None
        ),
    }


def _replay_check(run_dir: Path, replay_dir: Path | None) -> dict:
    if replay_dir is None or not (replay_dir / "comparison.csv").is_file():
        return {"available": False, "reason": "未运行复现目录"}
    new_pairs = pd.read_csv(run_dir / "paired_predictions.csv", low_memory=False)
    replay_pairs = pd.read_csv(replay_dir / "paired_predictions.csv", low_memory=False)
    key_columns = ["branch", "test_set", "key", "fold", "model", "condition", "ablation",
                   "horizon_days"]
    merged = new_pairs.merge(replay_pairs, on=key_columns, suffixes=("_new", "_replay"))
    delta = (merged["prediction_new"] - merged["prediction_replay"]).abs()
    comparison_a = pd.read_csv(run_dir / "comparison.csv")
    comparison_b = pd.read_csv(replay_dir / "comparison.csv")
    joined = comparison_a.merge(
        comparison_b, on=["branch", "condition", "horizon_days", "scenario"],
        suffixes=("_new", "_replay"))
    return {
        "available": True,
        "prediction_rows_new": int(len(new_pairs)),
        "prediction_rows_replay": int(len(replay_pairs)),
        "prediction_rows_joined": int(len(merged)),
        "max_abs_prediction_diff": float(delta.max()) if len(delta) else None,
        "predictions_identical": bool(len(merged) == len(new_pairs) == len(replay_pairs)
                                     and float(delta.max()) < 1e-12),
        "comparisons_new": int(len(comparison_a)),
        "comparisons_replay": int(len(comparison_b)),
        "comparisons_joined": int(len(joined)),
        "max_abs_delta_brier_diff": float(
            (joined["delta_brier_new"] - joined["delta_brier_replay"]).abs().max()),
        "max_abs_delta_auroc_diff": (
            float((pd.to_numeric(joined["delta_auroc_new"], errors="coerce")
                   - pd.to_numeric(joined["delta_auroc_replay"], errors="coerce")).abs().max())
            if len(joined) else None
        ),
        "n_equal": bool((joined["n_new"] == joined["n_replay"]).all()),
        "positives_equal": bool((joined["positives_new"] == joined["positives_replay"]).all()),
        "bootstrap_note": (
            "复现运行使用较小的 bootstrap 次数；点估计必须完全一致，"
            "区间宽度允许不同"
        ),
    }


def verify(run_dir: Path, replay_dir: Path | None = None) -> dict:
    checks: dict[str, Any] = {}
    paired = pd.read_csv(run_dir / "paired_predictions.csv", low_memory=False)
    comparison = pd.read_csv(run_dir / "comparison.csv")
    composite = pd.read_csv(run_dir / "composite_equal_weight.csv")
    metrics = pd.read_csv(run_dir / "metrics_by_cell.csv")
    ablation = pd.read_csv(run_dir / "ablation.csv")
    augmentation = pd.read_csv(run_dir / "training_augmentation.csv")
    detection = pd.read_csv(run_dir / "detection_table.csv")
    diagnostic = pd.read_csv(run_dir / "alarm_diagnostic.csv")
    difference = pd.read_csv(run_dir / "difference_table.csv")
    samples = pd.read_csv(run_dir / "sample_counts.csv")
    counts = _load(run_dir / "counts.json")
    corrections = _load(run_dir / "corrections_summary.json")
    thresholds = _load(STAGE17 / "thresholds.json")
    keys = pd.read_csv(STAGE16 / "real_keys.csv", dtype={"participantid": str})
    excluded = pd.read_csv(STAGE16 / "synthetic_exclusions.csv")
    manifest = _load(run_dir / "manifest.json")

    real = paired.loc[(paired["branch"] == "R") & (paired["ablation"] == "full")]
    synthetic = paired.loc[(paired["branch"] == "S") & (paired["ablation"] == "full")]

    # 1. 分母与阳性
    real_keys = real[["key"]].drop_duplicates()
    checks["real_test_keys"] = int(len(real_keys))
    checks["real_test_keys_171"] = bool(len(real_keys) == 171)
    checks["real_test_people"] = int(
        real["key"].astype(str).str.split("|").str[0].nunique())
    checks["real_test_people_133"] = bool(checks["real_test_people"] == 133)
    checks["real_no_same_day_key"] = bool(
        not real["key"].astype(str).eq("1940|2022-05-12").any())
    checks["synthetic_test_keys"] = int(
        synthetic.loc[synthetic["test_set"] == "synthetic_test", "key"].nunique())
    checks["synthetic_no_same_day"] = bool(
        not (set(synthetic["key"].astype(str)) & set(excluded["series_id"].astype(str))))
    real_counts = {}
    for horizon in (1, 3, 7):
        subset = real.loc[real["horizon_days"] == horizon]
        labels = subset.drop_duplicates("key")["label"].astype(float)
        real_counts[f"{horizon}d"] = int(labels.sum())
    checks["real_positives_1_3_7"] = [real_counts["1d"], real_counts["3d"], real_counts["7d"]]
    checks["real_positives_match_stage16"] = bool(
        checks["real_positives_1_3_7"] == [0, 1, 3])
    sample_real = samples.loc[samples["branch"] == "R"]
    checks["sample_counts_match"] = bool(all(
        int(sample_real.loc[sample_real["horizon_days"] == horizon, "positives"].iloc[0])
        == real_counts[f"{horizon}d"] for horizon in (1, 3, 7)))

    # 2. 7 天 ΔBrier（S-only 综合，独立重算）
    composite_rows = []
    for horizon in (1, 3, 7):
        subset = paired.loc[
            (paired["branch"] == "S") & (paired["ablation"] == "full")
            & (paired["condition"] == "S-only") & (paired["horizon_days"] == horizon)
            & (paired["test_set"] == "synthetic_test")
        ]
        per_mechanism = []
        for scenario, group in subset.groupby("scenario"):
            result = _brier_delta(group, "M2", "M3")
            per_mechanism.append(result)
        composite_rows.append({
            "horizon_days": horizon,
            "delta_brier_equal_weight": float(np.mean([r["delta_brier"] for r in per_mechanism])),
            "delta_auroc_equal_weight": float(np.mean(
                [r["delta_auroc"] for r in per_mechanism
                 if r["delta_auroc"] is not None])) if any(
                r["delta_auroc"] is not None for r in per_mechanism) else None,
            "positives_mean": float(np.mean([r["positives"] for r in per_mechanism])),
        })
    for row in composite_rows:
        stored = composite.loc[
            (composite["condition"] == "S-only")
            & (composite["horizon_days"] == row["horizon_days"])]
        checks[f"composite_{row['horizon_days']}d_matches"] = bool(
            len(stored) == 1
            and abs(float(stored["delta_brier_equal_weight"].iloc[0])
                    - row["delta_brier_equal_weight"]) < 1e-9
        )
        checks[f"composite_{row['horizon_days']}d_delta_brier"] = float(
            row["delta_brier_equal_weight"])
    checks["composite_7d_delta_brier"] = composite_rows[2]["delta_brier_equal_weight"]
    checks["composite_7d_delta_auroc"] = composite_rows[2]["delta_auroc_equal_weight"]

    # 3. 全部比较的 ΔBrier/ΔAUROC 点估计可重算
    worst_brier = 0.0
    worst_auroc = 0.0
    matched = 0
    for row in comparison.itertuples(index=False):
        subset = paired.loc[
            (paired["branch"] == row.branch)
            & (paired["condition"] == row.condition)
            & (paired["horizon_days"] == row.horizon_days)
            & (paired["scenario"] == row.scenario)
            & (paired["ablation"] == "full")
        ]
        if not len(subset):
            continue
        result = _brier_delta(subset, "M2", "M3")
        matched += 1
        worst_brier = max(worst_brier, abs(result["delta_brier"] - float(row.delta_brier)))
        if row.delta_auroc is not None and not pd.isna(row.delta_auroc) \
                and result["delta_auroc"] is not None:
            worst_auroc = max(worst_auroc,
                              abs(result["delta_auroc"] - float(row.delta_auroc)))
        if int(result["positives"]) != int(row.positives):
            worst_brier = 1.0
    checks["comparisons_recomputed"] = matched
    checks["comparisons_total"] = int(len(comparison))
    checks["delta_brier_max_abs_diff"] = float(worst_brier)
    checks["delta_auroc_max_abs_diff"] = float(worst_auroc)
    checks["delta_brier_recomputable"] = bool(worst_brier < 1e-12)
    checks["delta_auroc_recomputable"] = bool(worst_auroc < 1e-12)
    checks["real_auroc_not_estimable"] = bool(
        comparison.loc[comparison["branch"] == "R", "auroc_not_in_row"
                       if "auroc_not_in_row" in comparison.columns else "state"]
        .isin(["not_estimable"]).all()
    )

    # 4. 报警表独立重算
    raw = pd.read_csv(STAGE17 / "synthetic_detection_daily.csv", low_memory=False)
    raw["fold"] = raw["fold"].astype(int)
    raw["threshold_C"] = raw["fold"].map(
        lambda f: thresholds["per_fold"][str(int(f))]["methods"]["ds_pcd"]["threshold"])
    raw["threshold_rms"] = raw["fold"].map(
        lambda f: thresholds["per_fold"][str(int(f))]["methods"]["rms"]["threshold"])
    raw["alarm_C"] = raw["C"] >= raw["threshold_C"]
    raw["alarm_rms"] = raw["rms"] >= raw["threshold_rms"]
    raw["offset"] = pd.to_numeric(raw["monitoring_offset"], errors="coerce")
    raw = raw.loc[~raw["series_id"].astype(str).isin(set(excluded["series_id"].astype(str)))]
    stable = raw.loc[raw["trajectory"] == "stable"]
    recomputed = {}
    for method, column in (("ds_pcd", "alarm_C"), ("rms", "alarm_rms")):
        within = stable.loc[stable["offset"] <= 30]
        per_series = within.groupby("series_id")[column].any()
        recomputed[method] = float(per_series.mean())
        stored = detection.loc[
            (detection["table"] == "stable_null") & (detection["method"] == method),
            "false_alarm_30d"]
        checks[f"stable_false_alarm_{method}_recomputed"] = recomputed[method]
        checks[f"stable_false_alarm_{method}_matches"] = bool(
            len(stored) == 1 and abs(float(stored.iloc[0]) - recomputed[method]) < 1e-12)
    changed = raw.loc[
        raw["trajectory"].isin(("slow_linear", "rapid_step"))
        & raw["onset_offset"].notna()
    ]
    for method, column in (("ds_pcd", "alarm_C"), ("rms", "alarm_rms")):
        subset = changed.copy()
        onset = pd.to_numeric(subset["onset_offset"]).astype(int)
        for window in (1, 3, 7):
            alarmed = subset.loc[subset[column] & (subset["offset"] >= onset)
                                 & (subset["offset"] <= onset + window - 1)
                                 & (subset["offset"] <= 30)]
            detected = alarmed.groupby("series_id").size()
            rate = float(len(detected) / max(subset["series_id"].nunique(), 1))
            stored = detection.loc[
                (detection["table"] == "known_change")
                & (detection["method"] == method)
                & (detection["window_days"] == window),
                "detected_within_window"]
            checks[f"change_detect_{method}_{window}d_recomputed"] = rate
            checks[f"change_detect_{method}_{window}d_matches"] = bool(
                len(stored) == 1 and abs(float(stored.iloc[0]) - rate) < 1e-12)
        pre = subset.loc[subset[column] & (subset["offset"] < onset)]
        stored_pre = detection.loc[
            (detection["table"] == "known_change") & (detection["method"] == method),
            "pre_change_alarm_share"]
        if len(stored_pre):
            pre_series = pre.groupby("series_id").size()
            expected = len(pre_series) / max(subset["series_id"].nunique(), 1)
            checks[f"pre_change_share_{method}_matches"] = bool(
                abs(float(stored_pre.iloc[0]) - expected) < 1e-12)
    checks["stable_series_count"] = int(stable["series_id"].nunique())
    checks["change_series_count"] = int(changed["series_id"].nunique())
    checks["alarm_diagnostic_has_old_30d"] = bool(
        (diagnostic["definition"] == "old_threshold_30d").any())
    checks["old_threshold_30d_ds_pcd"] = float(diagnostic.loc[
        (diagnostic["method"] == "ds_pcd")
        & (diagnostic["definition"] == "old_threshold_30d"), "false_alarm_rate"].iloc[0])
    checks["old_threshold_30d_rms"] = float(diagnostic.loc[
        (diagnostic["method"] == "rms")
        & (diagnostic["definition"] == "old_threshold_30d"), "false_alarm_rate"].iloc[0])

    # 5. 三项更正的落地
    checks["corrections_same_day_new_rows"] = int(
        corrections["same_day"]["new_real_test_rows"])
    checks["corrections_same_day_new_positives"] = corrections["same_day"][
        "new_positives_1_3_7"]
    checks["corrections_thresholds_lower_than_old"] = bool(all(
        corrections["alarm_window"]["new_thresholds"][str(fold)][method]
        <= corrections["alarm_window"]["old_thresholds"][str(fold)][method]
        for fold in (0, 1, 2) for method in ("ds_pcd", "rms")
    ))
    variant_models = ablation["variant"].unique().tolist()
    checks["ablation_variants"] = sorted(variant_models)
    checks["ablation_has_four_variants"] = bool(set(variant_models) == {
        "full", "no_precision", "direction_only", "no_direction_cumulative"})
    fit_status = _load(run_dir / "ablation_fit_status.json")["models"]
    checks["ablation_uses_truncated_c"] = bool(any(
        "C_omnibus_truncated" in item["features"] for item in fit_status))
    checks["ablation_no_full_omnibus"] = bool(not any(
        "C_omnibus" in item["features"] for item in fit_status))

    # 6. 旧/新差异表
    checks["difference_rows"] = int(len(difference))
    checks["difference_all_matched"] = bool(difference["matched"].all())
    checks["difference_matches_comparisons"] = bool(len(difference) == len(comparison))
    real_7d = difference.loc[
        (difference["branch"] == "R") & (difference["horizon_days"] == 7)
        & (difference["condition"] == "R-only")]
    checks["difference_real_7d_present"] = bool(len(real_7d) == 1)
    if len(real_7d):
        checks["difference_real_7d_old_brier"] = float(real_7d["delta_brier_old"].iloc[0])
        checks["difference_real_7d_new_brier"] = float(real_7d["delta_brier_new"].iloc[0])
    synthetic_7d = difference.loc[
        (difference["branch"] == "S") & (difference["horizon_days"] == 7)
        & (difference["condition"] == "S-only")]
    checks["difference_synthetic_7d_cells"] = int(len(synthetic_7d))

    # 7. 研究用途边界
    checks["clinical_outlet_untouched"] = bool(
        "round2" not in (PROJECT_ROOT / "longitudinal" / "service.py")
        .read_text(encoding="utf-8"))
    checks["all_metrics_scoring_v2"] = bool(
        (metrics["scoring_version"] == "round2-scoring-v2-truncated").all())

    # 8. 复现
    replay = _replay_check(run_dir, replay_dir)
    checks["replay"] = replay
    checks["replay_predictions_identical"] = bool(replay.get("predictions_identical"))
    checks["replay_point_estimates_identical"] = bool(
        replay.get("available")
        and replay.get("max_abs_delta_brier_diff", 1) < 1e-12
        and (replay.get("max_abs_delta_auroc_diff") in (None, 0.0)
             or replay.get("max_abs_delta_auroc_diff", 1) < 1e-12)
        and replay.get("n_equal") and replay.get("positives_equal")
    )

    # 9. 上游哈希
    checks["manifest_hashes_match"] = all(
        _md5(Path(path) / name) == digest
        for path, entries in manifest["upstream"].items()
        for name, digest in entries.items()
    )

    verdict = [key for key, value in checks.items()
               if isinstance(value, (bool, np.bool_)) and key != "replay"]
    all_pass = all(bool(checks[key]) for key in verdict)
    return {
        "stage": 18,
        "run_dir": str(run_dir),
        "all_pass": bool(all_pass),
        "failed": [key for key in verdict if not checks[key]],
        "checks": checks,
    }


def _render(report: dict) -> str:
    lines = [
        "# 阶段 18 核验",
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
    parser = argparse.ArgumentParser(description="阶段 18 核验")
    parser.add_argument("--run", default=str(STAGE18))
    parser.add_argument("--replay", default=str(
        OUTPUT_ROOT / "stage18" / "20261115-corrected-eval-replay"))
    args = parser.parse_args(argv)
    run_dir = Path(args.run)
    replay_dir = Path(args.replay) if args.replay and Path(args.replay).is_dir() else None
    report = verify(run_dir, replay_dir)
    (run_dir / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "VERIFICATION.md").write_text(_render(report), encoding="utf-8")
    print(json.dumps({key: value for key, value in report["checks"].items()
                      if not isinstance(value, (dict, list))}, ensure_ascii=False, indent=1))
    print("ALL PASS" if report["all_pass"] else "FAILED: " + ", ".join(report["failed"]))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
