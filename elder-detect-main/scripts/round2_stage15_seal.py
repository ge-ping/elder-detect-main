"""阶段 15：封存本轮证据、生成图表、交付索引并复现冻结主流程。

* ``round2_index.json``：阶段 11–14（含复现运行）的精确引用、版本、哈希、种子与命令；
* ``figures/``：主比较、训练增强、组件消融、误报/检出、真实分支五张图；
* ``replay_verification.json``：冻结主流程一次复现的键、计数与指标核对；
* 不新增数据、不改主协议、不启动视频采集或临床部署。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise" / "round2"
STAGE11 = OUTPUT_ROOT / "stage11" / "20261115-protocol-v1"
STAGE12 = OUTPUT_ROOT / "stage12" / "20261115-supplement-v1"
STAGE13 = OUTPUT_ROOT / "stage13" / "20261115-freeze-v2"
STAGE14 = OUTPUT_ROOT / "stage14" / "20261115-eval-v1"
REPLAY13 = OUTPUT_ROOT / "stage13" / "20261115-freeze-replay"
REPLAY14 = OUTPUT_ROOT / "stage14" / "20261115-eval-replay"

COLORS = {
    "full": "#1f4e79", "no_precision": "#2e75b6", "direction_only": "#548235",
    "no_direction_cumulative": "#bf9000",
}
LABELS = {
    "full": "完整 DS", "no_precision": "无精度权重",
    "direction_only": "仅方向 T", "no_direction_cumulative": "不分方向但累积",
}
SCENARIO_LABELS = {
    "level_dominant|gaussian_ar1": "水平主导·高斯",
    "level_dominant|heavy_tail_spike": "水平主导·重尾",
    "sustained_change|gaussian_ar1": "持续变化·高斯",
    "sustained_change|heavy_tail_spike": "持续变化·重尾",
    "change_no_effect|gaussian_ar1": "变化无关联·高斯",
    "change_no_effect|heavy_tail_spike": "变化无关联·重尾",
}


def _md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ci(value: Any) -> tuple[float, float] | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        import ast

        value = ast.literal_eval(value)
    return (float(value[0]), float(value[1]))


def figure_primary(run_dir: Path, out: Path) -> dict:
    comparison = pd.read_csv(run_dir / "comparison.csv")
    subset = comparison.loc[(comparison["branch"] == "S") & (comparison["condition"] == "S-only")]
    scenarios = list(SCENARIO_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for panel, (axis, horizon) in enumerate(zip(axes, (7, 3, 1))):
        block = subset.loc[subset["horizon_days"] == horizon]
        positions = []
        values = []
        errors = []
        labels = []
        for index, scenario in enumerate(scenarios):
            row = block.loc[block["scenario"] == scenario]
            if not len(row):
                continue
            row = row.iloc[0]
            ci = _ci(row["delta_brier_ci"])
            positions.append(index)
            values.append(float(row["delta_brier"]))
            errors.append(
                [float(row["delta_brier"]) - ci[0], ci[1] - float(row["delta_brier"])]
                if ci else [0, 0]
            )
            labels.append(SCENARIO_LABELS[scenario])
        errors = np.asarray(errors).T if errors else np.zeros((2, 0))
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.errorbar(values, positions, xerr=errors, fmt="o", color="#1f4e79",
                      capsize=3, markersize=5)
        axis.set_yticks(positions)
        axis.set_yticklabels(labels if panel == 0 else [""] * len(labels), fontsize=8)
        axis.set_xlabel("ΔBrier（M3−M2，负=改善）", fontsize=9)
        axis.set_title(f"{horizon} 天窗口（S-only，半合成测试）", fontsize=10)
        axis.grid(axis="x", alpha=0.3)
    fig.suptitle("DS-PCD 相对朴素变化的增量：半合成 S 分支（每格约 6,000 序列、130–160 事件）",
                 fontsize=11)
    fig.tight_layout()
    path = out / "fig_primary_forest.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"file": path.name, "caption": "7/3/1 天主比较 ΔBrier 与 95% 区间（S 分支，仅半合成，训练条件 S-only）"}


def figure_augmentation(run_dir: Path, out: Path) -> dict:
    augmentation = pd.read_csv(run_dir / "training_augmentation.csv")
    subset = augmentation.loc[augmentation["horizon_days"] == 7]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for axis, branch, title in (
        (axes[0], "S", "S 分支：半合成测试（同一批虚拟序列）"),
        (axes[1], "R", "R 分支：真实 PD 测试（172 个预测日、4 个阳性）"),
    ):
        block = subset.loc[subset["branch"] == branch]
        for index, model in enumerate(("M0", "M1", "M2", "M3")):
            rows = block.loc[block["model"] == model]
            if not len(rows):
                continue
            values = rows["delta_brier"].astype(float).to_numpy()
            axis.scatter(np.full(len(values), index), values, s=18,
                         color="#1f4e79" if branch == "S" else "#a61c00", alpha=0.75)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(range(4))
        axis.set_xticklabels(["M0", "M1", "M2", "M3"])
        axis.set_ylabel("ΔBrier（R+S − R-only，负=加入半合成训练后改善）", fontsize=9)
        axis.set_title(title, fontsize=10)
        axis.grid(axis="y", alpha=0.3)
    fig.suptitle("训练增强贡献（7 天；每点为一个机制×噪声族单元）", fontsize=11)
    fig.tight_layout()
    path = out / "fig_augmentation.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"file": path.name, "caption": "R+S 相对 R-only 的 ΔBrier（按模型；S 与 R 分支分别）"}


def figure_ablation(run_dir: Path, out: Path) -> dict:
    ablation = pd.read_csv(run_dir / "ablation.csv")
    subset = ablation.loc[
        (ablation["branch"] == "S") & (ablation["condition"] == "S-only")
        & (ablation["horizon_days"] == 7)
    ]
    scenarios = list(SCENARIO_LABELS)
    variants = ["full", "no_precision", "direction_only", "no_direction_cumulative"]
    fig, axis = plt.subplots(figsize=(9, 4.4))
    width = 0.2
    for position, variant in enumerate(variants):
        values = []
        errors = []
        for scenario in scenarios:
            row = subset.loc[(subset["scenario"] == scenario) & (subset["variant"] == variant)]
            if not len(row):
                values.append(np.nan)
                errors.append([np.nan, np.nan])
                continue
            row = row.iloc[0]
            ci = _ci(row["delta_brier_ci"])
            values.append(float(row["delta_brier"]))
            errors.append(
                [float(row["delta_brier"]) - ci[0], ci[1] - float(row["delta_brier"])]
                if ci else [0, 0]
            )
        xs = np.arange(len(scenarios)) + (position - 1.5) * width
        axis.errorbar(xs, values, yerr=np.asarray(errors).T, fmt="o", capsize=2,
                      markersize=4, label=LABELS[variant], color=COLORS[variant])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(range(len(scenarios)))
    axis.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], fontsize=8, rotation=15)
    axis.set_ylabel("ΔBrier（相对 M2，负=改善）", fontsize=9)
    axis.set_title("组件消融：7 天、S-only、半合成测试（每点约 6,000 序列）", fontsize=10)
    axis.legend(fontsize=8, ncol=4)
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / "fig_ablation.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"file": path.name, "caption": "四种组件规格相对 M2 的 ΔBrier（7 天，S-only，半合成）"}


def figure_detection(run_dir: Path, out: Path) -> dict:
    detection = pd.read_csv(run_dir / "detection_table.csv")
    stable = detection.loc[detection["subset"] == "stable_null"]
    changed = detection.loc[detection["subset"] == "known_change"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    methods = ["ds_pcd", "rms"]
    colors = {"ds_pcd": "#1f4e79", "rms": "#a61c00"}
    names = {"ds_pcd": "DS-PCD", "rms": "RMS"}
    for method in methods:
        row = stable.loc[stable["method"] == method]
        if len(row):
            rate = float(row["false_alarm_30d"].iloc[0])
            ci = _ci(row["false_alarm_ci"].iloc[0])
            axes[0].bar(names[method], rate, color=colors[method], alpha=0.8)
            if ci:
                axes[0].errorbar(names[method], rate,
                                 yerr=[[rate - ci[0]], [ci[1] - rate]], fmt="none",
                                 ecolor="black", capsize=4)
    axes[0].axhline(0.10, color="gray", linestyle="--", linewidth=0.9)
    axes[0].set_ylabel("30 天至少一次误报的比例", fontsize=9)
    axes[0].set_title("独立稳定测试序列的实际误报（虚线=标定目标 10%）", fontsize=10)
    for method in methods:
        block = changed.loc[changed["method"] == method]
        axes[1].plot(block["window_days"], block["detected_within_window"], marker="o",
                     color=colors[method], label=f"{names[method]} 检出")
        axes[1].plot(block["window_days"], block["undetected_share"], marker="s",
                     linestyle="--", color=colors[method], label=f"{names[method]} 未检出")
    axes[1].set_xlabel("变化发生后窗口（监测日）", fontsize=9)
    axes[1].set_ylabel("比例", fontsize=9)
    axes[1].set_xticks([1, 3, 7])
    axes[1].set_title("已知变化检出率与从未检出比例（17,967 条变化序列）", fontsize=10)
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    # 提前量：已检出者的中位延迟 + 检出偏晚比例（7 天窗口）
    window7 = changed.loc[changed["window_days"] == 7]
    labels = {"ds_pcd": "DS-PCD", "rms": "RMS"}
    latencies = [float(window7.loc[window7["method"] == m, "median_latency_days"].iloc[0])
                 for m in methods if len(window7.loc[window7["method"] == m])]
    late = [float(window7.loc[window7["method"] == m, "detected_late"].iloc[0]) * 100
            for m in methods if len(window7.loc[window7["method"] == m])]
    xs = np.arange(len(methods))
    axes[2].bar(xs - 0.18, latencies, width=0.36, color="#1f4e79", label="中位延迟（天）")
    axes[2].bar(xs + 0.18, late, width=0.36, color="#bf9000", label="7 天后才检出（%）")
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels([labels[m] for m in methods])
    axes[2].set_title("提前量：已检出者的延迟与偏晚比例", fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.3)
    fig.suptitle("变化检出层面：误报对齐下的 DS-PCD 与 RMS（半合成场景，非风险预测主表）", fontsize=11)
    fig.tight_layout()
    path = out / "fig_detection.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"file": path.name, "caption": "误报对齐下 DS-PCD 与 RMS 的误报率、1/3/7 天检出率与未检出比例"}


def figure_real(run_dir: Path, out: Path) -> dict:
    comparison = pd.read_csv(run_dir / "comparison.csv")
    subset = comparison.loc[comparison["branch"] == "R"]
    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    conditions = ["R-only", "R+S", "S-only"]
    colors = {"R-only": "#1f4e79", "R+S": "#548235", "S-only": "#bf9000"}
    for position, horizon in enumerate((7, 3, 1)):
        for offset, condition in enumerate(conditions):
            row = subset.loc[(subset["horizon_days"] == horizon) & (subset["condition"] == condition)]
            if not len(row):
                continue
            row = row.iloc[0]
            ci = _ci(row["delta_brier_ci"])
            value = float(row["delta_brier"])
            x = position + (offset - 1) * 0.2
            axis.errorbar(x, value, yerr=[[value - ci[0]], [ci[1] - value]] if ci else None,
                          fmt="o", color=colors[condition], capsize=3, markersize=5,
                          label=condition if position == 0 else None)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(range(3))
    axis.set_xticklabels(["7 天（主）", "3 天", "1 天"])
    axis.set_ylabel("ΔBrier（M3−M2）", fontsize=9)
    axis.set_title("R 分支：真实 PD（172 个预测日、阳性 4/2/1，全部不可估）", fontsize=10)
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / "fig_real_branch.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"file": path.name, "caption": "真实分支 ΔBrier 与区间（1/3/7 天；阳性 1/2/4，区分度不可估）"}


def figure_cohort_flow(out: Path) -> dict:
    """真实队列流程图（PD 从发布受试者到可评分预测日）。"""
    flow = pd.read_csv(STAGE11 / "cohort_flow.csv")
    pd_flow = flow.loc[flow["cohort"] == "pd"].copy()
    steps = [
        ("发布受试者", 2316),
        ("PD ≥65 岁", None),
        ("有有效步态", None),
        ("候选预测日", None),
        ("14 天基线就绪", None),
        ("可评分（172）", 172),
    ]
    values = []
    for _, row in pd_flow.iterrows():
        step = str(row["step"])
        if step == "older_pd":
            values.append(int(row["rows_out"]))
        elif step == "older_pd_with_valid_gait":
            values.append(int(row["rows_out"]))
        elif step == "candidate_landmarks":
            values.append(int(row["rows_out"]))
    baseline_ready = 186
    sequence = [2316, values[0] if values else None, values[1] if len(values) > 1 else None,
                values[2] if len(values) > 2 else None, baseline_ready, 172]
    labels = ["发布受试者(人)", "PD >=65 岁(人)", "有有效步态(人)",
              "候选预测日(人 x 日)", "14 天基线就绪(186)", "可评分短窗(172)"]
    fig, axis = plt.subplots(figsize=(11, 3.6))
    positions = np.arange(len(sequence))
    axis.barh(positions, [v if v else 0 for v in sequence], color="#1f4e79", alpha=0.85)
    for position, value in zip(positions, sequence):
        if value:
            axis.text(value * 1.01, position, f"{value:,}", va="center", fontsize=9)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=9)
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlabel("计数（对数刻度；单位已在标签中注明）", fontsize=9)
    axis.set_title("真实队列流程：PD ≥65 岁从发布到可评分短窗预测日（数据来源：阶段 11 cohort_flow.csv）", fontsize=10)
    axis.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = out / "fig_cohort_flow.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"file": path.name, "caption": "真实 PD 队列流程（发布 2,316 → PD≥65 341 → 有效步态 331 → 候选 463 → 基线就绪 186 → 可评分 172）"}


def replay_check() -> dict:
    """冻结主流程复现：键、计数与点指标逐项核对。"""
    result: dict[str, Any] = {"replay_stage13": str(REPLAY13), "replay_stage14": str(REPLAY14)}
    if not (REPLAY13 / "freeze_manifest.json").is_file():
        result["status"] = "replay_not_available"
        return result
    original_freeze = _load(STAGE13 / "freeze_manifest.json")
    replay_freeze = _load(REPLAY13 / "freeze_manifest.json")
    result["stage13"] = {
        "status_counts_equal": original_freeze["status_counts"] == replay_freeze["status_counts"],
        "model_slots_equal": len(original_freeze["model_fingerprints"]) == len(replay_freeze["model_fingerprints"]),
        "test_scope_equal": original_freeze["test_scope"] == replay_freeze["test_scope"],
    }
    original_pred = pd.read_csv(STAGE13 / "oof_predictions.csv", low_memory=False)
    replay_pred = pd.read_csv(REPLAY13 / "oof_predictions.csv", low_memory=False)
    joined = original_pred.merge(
        replay_pred,
        on=["bundle_id", "participantid", "landmark_day", "series_id"],
        suffixes=("_orig", "_replay"), how="inner",
    )
    result["stage13"]["prediction_rows_original"] = int(len(original_pred))
    result["stage13"]["prediction_rows_replay"] = int(len(replay_pred))
    result["stage13"]["joined_rows"] = int(len(joined))
    diffs = [
        float((joined[f"p_{h}d_orig"] - joined[f"p_{h}d_replay"]).abs().max())
        for h in (1, 3, 7)
    ]
    result["stage13"]["max_abs_prediction_diff"] = diffs
    result["stage13"]["predictions_identical_within_1e-12"] = all(value < 1e-12 for value in diffs)

    if not (REPLAY14 / "comparison.csv").is_file():
        result["status"] = "stage14_replay_pending"
        return result
    original_cmp = pd.read_csv(STAGE14 / "comparison.csv")
    replay_cmp = pd.read_csv(REPLAY14 / "comparison.csv")
    keys = ["branch", "condition", "horizon_days", "scenario"]
    merged = original_cmp.merge(replay_cmp, on=keys, suffixes=("_orig", "_replay"), how="outer",
                                indicator=True)
    result["stage14"] = {
        "rows_original": int(len(original_cmp)),
        "rows_replay": int(len(replay_cmp)),
        "joined": bool((merged["_merge"] == "both").all()),
        "n_equal": bool((merged["n_orig"] == merged["n_replay"]).all()),
        "positives_equal": bool((merged["positives_orig"] == merged["positives_replay"]).all()),
        "max_abs_delta_brier_diff": float(
            (merged["delta_brier_orig"] - merged["delta_brier_replay"]).abs().max()
        ),
        "max_abs_delta_auroc_diff": float(
            (merged["delta_auroc_orig"] - merged["delta_auroc_replay"]).abs().max()
        ),
        "bootstrap_note": "复现运行使用 200 次 bootstrap（原运行 2000 次），点估计应完全一致，区间宽度不同",
    }
    result["status"] = "ok"
    return result


def build_index() -> dict:
    freeze = _load(STAGE13 / "freeze_manifest.json")
    stage11_manifest = _load(STAGE11 / "manifest.json")
    stage12_manifest = _load(STAGE12 / "manifest.json")
    stage14_manifest = _load(STAGE14 / "manifest.json")
    outputs = {
        "stage11": {
            "run": str(STAGE11),
            "protocol": {"path": str(STAGE11 / "protocol.json"), "md5": _md5(STAGE11 / "protocol.json")},
            "real_candidates": str(STAGE11 / "real_candidates.csv"),
            "data_gap_matrix": str(STAGE11 / "data_gap_matrix.csv"),
            "splits": {"path": str(STAGE11 / "splits.csv"), "md5": _md5(STAGE11 / "splits.csv")},
            "clinical_reference": str(STAGE11 / "clinical_reference_corrected.json"),
            "report": str(STAGE11 / "REPORT.md"),
            "verification": str(STAGE11 / "verification.json"),
            "release": stage11_manifest["release"],
            "inputs": {name: item["md5"] for name, item in stage11_manifest["inputs"].items()},
            "environment": stage11_manifest["environment"],
            "seed": stage11_manifest["seed"],
        },
        "stage12": {
            "run": str(STAGE12),
            "report": str(STAGE12 / "REPORT.md"),
            "verification": str(STAGE12 / "verification.json"),
            "generator_manifest": str(STAGE12 / "generator_manifest.json"),
            "feature_provenance": str(STAGE12 / "feature_provenance.json"),
            "threshold_calibration": str(STAGE12 / "threshold_calibration.json"),
            "lineage": str(STAGE12 / "lineage.csv"),
            "inputs": {name: item["md5"] for name, item in stage12_manifest["inputs"].items()},
            "scoring": stage12_manifest["scoring"],
            "seed": stage12_manifest["seed"],
        },
        "stage13": {
            "run": str(STAGE13),
            "report": str(STAGE13 / "REPORT.md"),
            "verification": str(STAGE13 / "verification.json"),
            "freeze_manifest": {"path": str(STAGE13 / "freeze_manifest.json"),
                                "md5": _md5(STAGE13 / "freeze_manifest.json")},
            "model_registry": str(STAGE13 / "model_registry.json"),
            "predictions": str(STAGE13 / "oof_predictions.csv"),
            "outcome_metrics_computed": freeze["outcome_metrics_computed"],
            "test_scope": freeze["test_scope"],
            "prefix_run": str(OUTPUT_ROOT / "stage13" / "20261115-freeze-v1-prefix-relative-risk-bug"),
        },
        "stage14": {
            "run": str(STAGE14),
            "report": str(STAGE14 / "REPORT.md"),
            "verification": str(STAGE14 / "verification.json"),
            "paired_predictions": str(STAGE14 / "paired_predictions.csv"),
            "primary_7d": str(STAGE14 / "primary_7d.csv"),
            "secondary_1_3d": str(STAGE14 / "secondary_1_3d.csv"),
            "composite": str(STAGE14 / "composite_equal_weight.csv"),
            "augmentation": str(STAGE14 / "training_augmentation.csv"),
            "ablation": str(STAGE14 / "ablation.csv"),
            "detection": str(STAGE14 / "detection_table.csv"),
            "trajectory": str(STAGE14 / "comparison_by_trajectory.csv"),
            "power_sensitivity": str(STAGE14 / "power_sensitivity.csv"),
            "joint_tests": str(STAGE14 / "joint_tests.json"),
            "conclusions": str(STAGE14 / "conclusions.json"),
            "inputs": {
                name: item["md5"] for name, item in stage14_manifest["upstream"].items()
                if isinstance(item, dict) and "md5" in item
            },
            "seed": stage14_manifest["seed"],
            "bootstrap": stage14_manifest["bootstrap"],
        },
    }
    return {
        "round": "ds_pcd_mobilise_round2",
        "title": "DS-PCD 未来 1/3/7 天跌倒风险预警再实验",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plan": "planning-files/ds-pcd-mobilise-implementation-plan.md",
        "goal_prompt": "planning-files/ds-pcd-mobilise-goal-prompt.md",
        "environment": {
            "python": "D:/AnacondaData/envs_dirs/py3.13/python.exe",
            "python_version": stage11_manifest["environment"]["python"],
            "numpy": stage11_manifest["environment"]["numpy"],
            "pandas": stage11_manifest["environment"]["pandas"],
        },
        "stages": outputs,
        "preserved_history": {
            "stage10": str(PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise"
                           / "stage10" / "20261114-attribution-v1"),
            "fixed_run": str(PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise" / "20261114-fixed"),
            "old_run": str(PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise" / "20261114-main"),
            "not_overwritten": True,
        },
        "corrections": str(OUTPUT_ROOT / "CORRECTIONS.md"),
        "claims_index": {
            "real_labels_not_fabricated": "仅 PD 有逐次跌倒日期；MS/PFF/COPD 无日期，不进入短窗监督测试",
            "no_half_year_endpoint": "本轮窗口固定 1/3/7 天，未恢复半年或 30 天终点",
            "research_only": "所有模型 research_only=true、calibrated=false；临床概率出口未改动",
            "synthetic_not_real": "S 分支为半合成评价，不称为新增真实病例或临床验证",
        },
        "reproduction": [
            "python scripts/round2_stage11_protocol.py --run-id 20261115-protocol-v1",
            "python scripts/round2_stage12_supplement.py --run-id 20261115-supplement-v1",
            "python scripts/round2_stage13_train_freeze.py --run-id 20261115-freeze-v2",
            "python scripts/round2_stage14_ablation_scores.py --run-id 20261115-eval-v1",
            "python scripts/round2_stage14_evaluate.py --run-id 20261115-eval-v1",
            "python scripts/round2_stage14_sensitivity.py",
            "python scripts/round2_stage15_seal.py",
        ],
    }


def run(args: argparse.Namespace) -> int:
    run_dir = OUTPUT_ROOT / "stage15" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    figures = []
    builders = (figure_primary, figure_augmentation, figure_ablation,
                figure_detection, figure_real)
    for builder in builders:
        try:
            figures.append(builder(STAGE14, figures_dir))
        except Exception as error:  # noqa: BLE001
            figures.append({"file": None, "error": f"{type(error).__name__}: {error}"})
    try:
        figures.append(figure_cohort_flow(figures_dir))
    except Exception as error:  # noqa: BLE001
        figures.append({"file": None, "error": f"{type(error).__name__}: {error}"})
    index = build_index()
    (run_dir / "round2_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    replay = replay_check()
    (run_dir / "replay_verification.json").write_text(
        json.dumps(replay, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "figures_manifest.json").write_text(
        json.dumps(figures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"figures": figures, "replay": replay.get("status")},
                     ensure_ascii=False, indent=1))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 15：封存与交付")
    parser.add_argument("--run-id", default="20261115-seal-v1")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
