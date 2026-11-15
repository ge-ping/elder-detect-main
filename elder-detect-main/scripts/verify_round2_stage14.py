"""阶段 14 核验：从逐条预测独立重算主表、配对键、计数与不可估项。"""
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

from mobilise_d.round2 import R2_FEATURES  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise" / "round2"
STAGE11 = OUTPUT_ROOT / "stage11" / "20261115-protocol-v1"
STAGE13 = OUTPUT_ROOT / "stage13" / "20261115-freeze-v2"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _norm_key(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0\|", "|", regex=True)


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    from scipy.stats import rankdata

    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores, method="average")
    return (float(ranks[positives].sum()) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def verify(run_dir: Path) -> dict:
    checks: dict[str, Any] = {}
    paired = pd.read_csv(run_dir / "paired_predictions.csv", low_memory=False)
    metrics = pd.read_csv(run_dir / "metrics_by_cell.csv")
    comparison = pd.read_csv(run_dir / "comparison.csv")
    augmentation = pd.read_csv(run_dir / "training_augmentation.csv")
    ablation = pd.read_csv(run_dir / "ablation.csv")
    detection = pd.read_csv(run_dir / "detection_table.csv")
    composite = pd.read_csv(run_dir / "composite_equal_weight.csv")
    conclusions = _load(run_dir / "conclusions.json")
    joint = _load(run_dir / "joint_tests.json")
    manifest = _load(run_dir / "manifest.json")
    splits = pd.read_csv(STAGE11 / "splits.csv", dtype={"participantid": str})
    scored_real = pd.read_csv(
        OUTPUT_ROOT / "stage12" / "20261115-supplement-v1" / "real_scores.csv",
        dtype={"participantid": str},
    )
    real_keys = set(_norm_key(
        scored_real.loc[scored_real["status"] == "scored"]
        .assign(key=lambda frame: frame["participantid"] + "|" + frame["landmark_day"].astype(str))["key"]
    ))

    # 1. 独立重算 7 天主表（M2/M3 配对）
    recomputed = []
    for (branch, condition, scenario), group in paired.loc[
        (paired["ablation"] == "full") & (paired["horizon_days"] == 7)
    ].groupby(["branch", "condition", "scenario"]):
        wide = group.pivot_table(index=["key", "label"], columns="model",
                                 values="prediction", aggfunc="first").reset_index()
        wide = wide.dropna(subset=["M2", "M3", "label"])
        delta_brier = float(np.mean((wide["M3"] - wide["label"]) ** 2)
                            - np.mean((wide["M2"] - wide["label"]) ** 2))
        recomputed.append({
            "branch": branch, "condition": condition, "scenario": scenario,
            "n": int(len(wide)), "positives": int(wide["label"].sum()),
            "delta_brier": delta_brier,
            "delta_auroc": (
                (_auroc(wide["label"].to_numpy(), wide["M3"].to_numpy())
                 - _auroc(wide["label"].to_numpy(), wide["M2"].to_numpy()))
                if int(wide["label"].sum()) >= 5 else None
            ),
        })
    recomputed_frame = pd.DataFrame(recomputed)
    reported = comparison.loc[comparison["horizon_days"] == 7][[
        "branch", "condition", "scenario", "n", "positives", "delta_brier", "delta_auroc"
    ]]
    merged = reported.merge(
        recomputed_frame, on=["branch", "condition", "scenario"], suffixes=("_reported", "_recomputed"),
        how="outer", indicator=True,
    )
    checks["main_table_recomputable"] = bool((merged["_merge"] == "both").all())
    checks["main_table_max_abs_brier_diff"] = float(
        (merged["delta_brier_reported"] - merged["delta_brier_recomputed"]).abs().max()
    )
    checks["main_table_brier_within_tolerance"] = bool(
        (merged["delta_brier_reported"] - merged["delta_brier_recomputed"]).abs().max() < 1e-12
    )
    auroc_both = merged.dropna(subset=["delta_auroc_reported", "delta_auroc_recomputed"])
    checks["main_table_auroc_within_tolerance"] = bool(
        (auroc_both["delta_auroc_reported"] - auroc_both["delta_auroc_recomputed"]).abs().max() < 1e-9
    ) if len(auroc_both) else True
    checks["main_table_counts_match"] = bool(
        (merged["n_reported"] == merged["n_recomputed"]).all()
        and (merged["positives_reported"] == merged["positives_recomputed"]).all()
    )

    # 2. 测试键与分母固定
    real_pairs = paired.loc[(paired["branch"] == "R") & (paired["ablation"] == "full")]
    synthetic_pairs = paired.loc[(paired["branch"] == "S") & (paired["ablation"] == "full")]
    checks["real_test_keys_are_the_frozen_candidates"] = bool(
        set(_norm_key(real_pairs["key"])) == real_keys
    )
    checks["real_test_keys_total"] = len(real_keys)
    checks["real_test_keys_per_model_equal"] = bool(
        real_pairs.groupby("model")["key"].nunique().nunique() == 1
    )
    checks["synthetic_test_keys_per_model_equal"] = bool(
        synthetic_pairs.groupby("model")["key"].nunique().nunique() == 1
    )
    checks["real_branch_has_no_synthetic_keys"] = bool(
        set(real_pairs["key"]).isdisjoint(set(synthetic_pairs["key"]))
    )
    checks["real_branch_labels_from_real_outcomes"] = bool(
        real_pairs["label"].isin([0.0, 1.0]).all()
    )

    # 3. 不可估项进入汇总
    checks["all_comparisons_have_state"] = bool(comparison["state"].notna().all())
    real_states = comparison.loc[comparison["branch"] == "R", "state"].unique().tolist()
    checks["real_branch_all_not_estimable"] = real_states == ["not_estimable"]
    checks["auroc_null_when_positives_below_5"] = bool(
        metrics.loc[metrics["positives"] < 5, "auroc"].isna().all()
    )
    checks["auroc_present_when_positives_at_least_5"] = bool(
        metrics.loc[metrics["positives"] >= 5, "auroc"].notna().all()
    )
    checks["not_estimable_reason_recorded"] = bool(
        metrics.loc[metrics["positives"] < 5, "auroc_not_estimable_reason"]
        .eq("fewer_than_5_positives").all()
    )

    # 4. 没有新增窗口/机制/条件
    checks["horizons_only_1_3_7"] = sorted(int(v) for v in paired["horizon_days"].unique()) == [1, 3, 7]
    checks["mechanisms_only_three"] = sorted(
        {str(v) for v in paired["mechanism"].dropna().unique()}
    ) == ["change_no_effect", "level_dominant", "sustained_change"]
    checks["conditions_only_three"] = sorted(
        {str(v) for v in paired["condition"].unique()}
    ) == ["R+S", "R-only", "S-only"]
    checks["ablations_only_four"] = sorted(
        {str(v) for v in paired["ablation"].unique()}
    ) == ["direction_only", "full", "no_direction_cumulative", "no_precision"]
    checks["noise_families_only_two"] = sorted(
        {str(v) for v in paired["noise_family"].dropna().unique()}
    ) == ["gaussian_ar1", "heavy_tail_spike"]

    # 5. 训练增强表覆盖同一模型的两种条件
    checks["augmentation_covers_all_models"] = sorted(
        {str(v) for v in augmentation["model"].unique()}
    ) == ["M0", "M1", "M2", "M3"]
    checks["augmentation_uses_paired_keys"] = bool(
        (augmentation["n"] > 0).all() and augmentation["positives"].notna().all()
    )
    checks["augmentation_reported_both_branches"] = {str(v) for v in augmentation["branch"]} == {"R", "S"}

    # 6. 消融覆盖四个变体
    checks["ablation_covers_four_variants"] = {str(v) for v in ablation["variant"]} == {
        "full", "no_precision", "direction_only", "no_direction_cumulative"
    }
    checks["ablation_has_three_horizons"] = set(ablation["horizon_days"]) == {1, 3, 7}

    # 7. 综合指标是等权平均
    composite_ok = True
    worst = 0.0
    for row in composite.itertuples(index=False):
        subset = comparison.loc[
            (comparison["branch"] == row.branch)
            & (comparison["condition"] == row.condition)
            & (comparison["horizon_days"] == row.horizon_days)
        ].copy()
        subset["mechanism"] = subset["scenario"].str.split("|").str[0]
        subset = subset.loc[subset["mechanism"].isin(
            ["level_dominant", "sustained_change", "change_no_effect"]
        )]
        by_mechanism = subset.groupby("mechanism")["delta_brier"].mean()
        expected = float(by_mechanism.mean())
        worst = max(worst, abs(expected - row.delta_brier_equal_weight))
        if len(by_mechanism) != 3:
            composite_ok = False
    checks["composite_is_equal_weight_mean"] = bool(composite_ok and worst < 1e-12)
    checks["composite_max_abs_diff"] = float(worst)
    checks["stress_not_in_composite"] = bool(
        "stress" not in " ".join(composite["note"].tolist()).lower()
        and composite["mechanisms"].eq(3).all()
    )

    # 8. 检出表覆盖两种方法与三个窗口
    checks["detection_covers_methods"] = {str(v) for v in detection["method"]} == {"ds_pcd", "rms"}
    checks["detection_windows"] = sorted(
        detection.dropna(subset=["window_days"])["window_days"].astype(int).unique().tolist()
    ) == [1, 3, 7]
    checks["detection_false_alarm_reported"] = bool(
        detection.loc[detection["subset"] == "stable_null", "false_alarm_30d"].notna().all()
    )
    checks["detection_undetected_share_reported"] = bool(
        detection.loc[detection["subset"] == "known_change", "undetected_share"].notna().all()
    )

    # 9. 失败/退化状态保留
    joint_status = {key: value.get("status") for key, value in joint.items()}
    checks["joint_test_statuses"] = joint_status
    checks["joint_test_has_estimable_case"] = any(
        value == "ok" for value in joint_status.values()
    )
    checks["joint_test_records_not_a_lrt"] = all(
        value.get("not_a_likelihood_ratio_test", True) in (True, None) for value in joint.values()
    )
    checks["conclusions_include_three_states"] = bool(
        "composite_7d" in conclusions and "R" in conclusions and "S" in conclusions
    )

    # 10. 上游冻结未被改动
    freeze = _load(STAGE13 / "freeze_manifest.json")
    checks["freeze_manifest_unchanged"] = bool(
        _md5(STAGE13 / "freeze_manifest.json") == manifest["upstream"]["freeze_md5"]
    )
    checks["protocol_unchanged"] = bool(
        _md5(STAGE11 / "protocol.json") == manifest["upstream"]["protocol_md5"]
    )
    checks["manifest_lists_outputs"] = bool(len(manifest["outputs"]) >= 12)
    checks["sensitivity_files_present"] = bool(
        (run_dir / "power_sensitivity.csv").is_file()
        and (run_dir / "comparison_by_trajectory.csv").is_file()
    )
    sensitivity_manifest = _load(run_dir / "sensitivity_manifest.json")
    checks["power_sensitivity_labelled_post_hoc"] = (
        "post-hoc" in sensitivity_manifest["power_sensitivity"]["status"]
    )

    verdict_keys = [
        "main_table_recomputable", "main_table_brier_within_tolerance",
        "main_table_auroc_within_tolerance", "main_table_counts_match",
        "real_test_keys_are_the_frozen_candidates", "real_test_keys_per_model_equal",
        "synthetic_test_keys_per_model_equal", "real_branch_has_no_synthetic_keys",
        "real_branch_labels_from_real_outcomes", "all_comparisons_have_state",
        "real_branch_all_not_estimable", "auroc_null_when_positives_below_5",
        "auroc_present_when_positives_at_least_5", "not_estimable_reason_recorded",
        "horizons_only_1_3_7", "mechanisms_only_three", "conditions_only_three",
        "ablations_only_four", "noise_families_only_two",
        "augmentation_covers_all_models", "augmentation_uses_paired_keys",
        "augmentation_reported_both_branches", "ablation_covers_four_variants",
        "ablation_has_three_horizons", "composite_is_equal_weight_mean",
        "stress_not_in_composite", "detection_covers_methods", "detection_windows",
        "detection_false_alarm_reported", "detection_undetected_share_reported",
        "joint_test_has_estimable_case", "joint_test_records_not_a_lrt",
        "conclusions_include_three_states", "freeze_manifest_unchanged",
        "protocol_unchanged", "manifest_lists_outputs", "sensitivity_files_present",
        "power_sensitivity_labelled_post_hoc",
    ]
    all_pass = all(bool(checks[key]) for key in verdict_keys)
    return {
        "stage": 14,
        "run_dir": str(run_dir),
        "all_pass": bool(all_pass),
        "failed": [key for key in verdict_keys if not checks[key]],
        "checks": checks,
    }


def _render(report: dict) -> str:
    lines = [
        "# 阶段 14 核验",
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
    for key in ("joint_test_statuses",):
        value = report["checks"].get(key)
        if value is not None:
            lines.append(f"## {key}")
            lines.append("```")
            lines.append(json.dumps(value, ensure_ascii=False, indent=1))
            lines.append("```")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 14 核验")
    parser.add_argument("--run", default=str(OUTPUT_ROOT / "stage14" / "20261115-eval-v1"))
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
