"""阶段 18 封存：图表、交付索引与复现核对。

读取阶段 16–18 的封存产物，生成：
* ``figures/``：主比较、训练增强、组件消融、报警/检出、真实分支、旧新差异六张图；
* ``round2_delivery_index.json``：从原包到更正版结果的可追溯引用（绝对路径 + md5）；
* ``report_data.json``：最终报告引用的数字（从表直接汇总，避免手抄）；
* ``replay_verification.json``：主流程与新目录复现的键、预测与点估计核对。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import md5
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

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
STAGE16 = OUTPUT_ROOT / "stage16" / "20261115-keys-v2"
STAGE17 = OUTPUT_ROOT / "stage17" / "20261115-refreeze-v1"
STAGE18 = OUTPUT_ROOT / "stage18" / "20261115-corrected-eval-v1"
REPLAY17 = OUTPUT_ROOT / "stage17" / "20261115-refreeze-replay"
REPLAY18 = OUTPUT_ROOT / "stage18" / "20261115-corrected-eval-replay"
HISTORY = PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise"

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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")


def _ci(value: Any) -> tuple[float, float] | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        import ast

        value = ast.literal_eval(value)
    return (float(value[0]), float(value[1]))


def figure_primary(run_dir: Path, out: Path) -> dict:
    comparison = pd.read_csv(run_dir / "comparison.csv")
    subset = comparison.loc[(comparison["branch"] == "S")
                            & (comparison["condition"] == "S-only")]
    scenarios = list(SCENARIO_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for panel, (axis, horizon) in enumerate(zip(axes, (7, 3, 1))):
        block = subset.loc[subset["horizon_days"] == horizon]
        positions, values, errors, labels = [], [], [], []
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
                if ci else [0, 0])
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
    fig.suptitle("更正版：DS-PCD 相对朴素变化的增量（同日剔除、30 天报警窗、截断消融）",
                 fontsize=11)
    fig.tight_layout()
    path = out / "fig_primary_forest.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"file": path.name,
            "caption": "7/3/1 天主比较 ΔBrier 与 95% 区间（S 分支半合成，训练条件 S-only）"}


def figure_augmentation(run_dir: Path, out: Path) -> dict:
    augmentation = pd.read_csv(run_dir / "training_augmentation.csv")
    subset = augmentation.loc[augmentation["horizon_days"] == 7]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, branch, title in (
        (axes[0], "S", "S 分支：半合成测试（同一批虚拟序列）"),
        (axes[1], "R", "R 分支：真实 PD 测试（171 个预测日、3 个 7 天阳性）"),
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
    return {"file": path.name,
            "caption": "R+S 相对 R-only 的 ΔBrier（按模型；S 与 R 分支分别）"}


def figure_ablation(run_dir: Path, out: Path) -> dict:
    ablation = pd.read_csv(run_dir / "ablation.csv")
    subset = ablation.loc[
        (ablation["branch"] == "S") & (ablation["condition"] == "S-only")
        & (ablation["horizon_days"] == 7)]
    scenarios = list(SCENARIO_LABELS)
    variants = ["full", "no_precision", "direction_only", "no_direction_cumulative"]
    fig, axis = plt.subplots(figsize=(9.5, 4.6))
    width = 0.2
    for position, variant in enumerate(variants):
        values, errors = [], []
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
                if ci else [0, 0])
        xs = np.arange(len(scenarios)) + (position - 1.5) * width
        axis.errorbar(xs, values, yerr=np.asarray(errors).T, fmt="o", capsize=2,
                      markersize=4, label=LABELS[variant], color=COLORS[variant])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(range(len(scenarios)))
    axis.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], fontsize=8, rotation=15)
    axis.set_ylabel("ΔBrier（相对 M2，负=改善）", fontsize=9)
    axis.set_title("组件消融：7 天、S-only、半合成测试（输入截断到 t）", fontsize=10)
    axis.legend(fontsize=8, ncol=4)
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / "fig_ablation.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"file": path.name,
            "caption": "四种组件规格相对 M2 的 ΔBrier（7 天，S-only，半合成，C 截断到 t）"}


def figure_detection(run_dir: Path, out: Path) -> dict:
    detection = pd.read_csv(run_dir / "detection_table.csv")
    diagnostic = pd.read_csv(run_dir / "alarm_diagnostic.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    methods = ["ds_pcd", "rms"]
    labels = {"ds_pcd": "DS-PCD", "rms": "RMS"}
    colors = {"ds_pcd": "#1f4e79", "rms": "#bf9000"}
    # 面板 1：独立稳定测试的实际误报（新阈值 30 天 vs 旧阈值 30/37 天诊断）
    defs = ["new_threshold_30d", "old_threshold_30d", "old_threshold_37d"]
    def_labels = ["新阈值·30 天", "旧阈值·30 天", "旧阈值·37 天"]
    width = 0.26
    for index, method in enumerate(methods):
        values = [
            float(diagnostic.loc[(diagnostic["method"] == method)
                                 & (diagnostic["definition"] == definition),
                                 "false_alarm_rate"].iloc[0])
            for definition in defs
        ]
        axes[0].bar(np.arange(len(defs)) + (index - 0.5) * width, values, width,
                    label=labels[method], color=colors[method])
    axes[0].set_xticks(range(len(defs)))
    axes[0].set_xticklabels(def_labels, fontsize=8)
    axes[0].set_ylabel("至少一次误报的比例", fontsize=9)
    stable_n = int(detection.loc[detection["table"] == "stable_null", "n_series"].iloc[0])
    axes[0].set_title(f"独立稳定测试的误报（{stable_n:,} 序列）", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    # 面板 2：已知变化的 1/3/7 天检出率
    change = detection.loc[detection["table"] == "known_change"]
    for index, method in enumerate(methods):
        block = change.loc[change["method"] == method].sort_values("window_days")
        axes[1].plot(block["window_days"], block["detected_within_window"],
                     marker="o", label=labels[method], color=colors[method])
        axes[1].plot(block["window_days"], block["detected_extended_share"],
                     marker="x", linestyle="--", alpha=0.6,
                     color=colors[method])
    axes[1].set_xticks([1, 3, 7])
    axes[1].set_xlabel("变化后窗口（天）", fontsize=9)
    axes[1].set_ylabel("检出率", fontsize=9)
    axes[1].set_title("已知变化的检出（实线=监测 30 天内；虚线=到第 37 天）", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    # 面板 3：变化前报警与截尾
    for index, method in enumerate(methods):
        block = change.loc[change["method"] == method].sort_values("window_days")
        axes[2].plot(block["window_days"], block["pre_change_alarm_share"],
                     marker="s", label=f"{labels[method]}·变化前报警",
                     color=colors[method])
        axes[2].plot(block["window_days"], block["censored_share"], marker="^",
                     linestyle=":", color=colors[method],
                     label=f"{labels[method]}·截尾")
    axes[2].set_xticks([1, 3, 7])
    axes[2].set_xlabel("变化后窗口（天）", fontsize=9)
    axes[2].set_ylabel("比例", fontsize=9)
    axes[2].set_title("变化前已报警与监测结束截尾", fontsize=10)
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)
    fig.suptitle("更正版报警表：误报只在监测第 1–30 天归属，变化前报警单记", fontsize=11)
    fig.tight_layout()
    path = out / "fig_detection.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"file": path.name,
            "caption": "误报（新旧阈值）、已知变化 1/3/7 天检出、变化前报警与截尾"}


def figure_real(run_dir: Path, out: Path) -> dict:
    comparison = pd.read_csv(run_dir / "comparison.csv")
    samples = pd.read_csv(run_dir / "sample_counts.csv")
    subset = comparison.loc[comparison["branch"] == "R"]
    conditions = ["R-only", "R+S", "S-only"]
    colors = {"R-only": "#1f4e79", "R+S": "#548235", "S-only": "#bf9000"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for condition in conditions:
        block = subset.loc[subset["condition"] == condition].sort_values("horizon_days")
        values = block["delta_brier"].astype(float).to_numpy()
        errors = []
        for value in block["delta_brier_ci"]:
            ci = _ci(value)
            errors.append([np.nan, np.nan] if ci is None else None)
        lows, highs = [], []
        for value, ci in zip(values, block["delta_brier_ci"]):
            parsed = _ci(ci)
            lows.append(np.nan if parsed is None else value - parsed[0])
            highs.append(np.nan if parsed is None else parsed[1] - value)
        axes[0].errorbar(block["horizon_days"], values, yerr=[lows, highs],
                         marker="o", capsize=3, label=condition, color=colors[condition])
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks([1, 3, 7])
    axes[0].set_xlabel("预测窗口（天）", fontsize=9)
    axes[0].set_ylabel("ΔBrier（M3−M2）", fontsize=9)
    axes[0].set_title("真实 PD 分支：171 个预测日 / 133 人", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    real_samples = samples.loc[samples["branch"] == "R"].sort_values("horizon_days")
    axes[1].bar(real_samples["horizon_days"].astype(str), real_samples["positives"],
                color="#a61c00")
    for index, row in enumerate(real_samples.itertuples(index=False)):
        axes[1].text(index, row.positives + 0.05, f"{int(row.positives)} 阳性",
                     ha="center", fontsize=9)
    axes[1].axhline(5, color="black", linestyle="--", linewidth=0.8)
    axes[1].text(1.6, 5.15, "AUROC 可估门槛（≥5 阳性）", fontsize=8)
    axes[1].set_ylim(0, 6)
    axes[1].set_xlabel("预测窗口（天）", fontsize=9)
    axes[1].set_ylabel("阳性预测日数", fontsize=9)
    axes[1].set_title("真实阳性数：1/3/7 天 = 0/1/3（AUROC 不可估）", fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("真实证据边界：区间极宽、区分度不可估（同日跌倒整日剔除后）", fontsize=11)
    fig.tight_layout()
    path = out / "fig_real_branch.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"file": path.name,
            "caption": "真实分支 ΔBrier 与区间（1/3/7 天；阳性 0/1/3，AUROC 不可估）"}


def figure_difference(run_dir: Path, out: Path) -> dict:
    difference = pd.read_csv(run_dir / "difference_table.csv")
    subset = difference.dropna(subset=["delta_brier_old", "delta_brier_new"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    styles = {"R": ("#a61c00", "o", "真实 R"), "S": ("#1f4e79", "s", "半合成 S")}
    for branch, (color, marker, label) in styles.items():
        block = subset.loc[subset["branch"] == branch]
        axes[0].scatter(block["delta_brier_old"], block["delta_brier_new"], s=22,
                        color=color, marker=marker, alpha=0.7, label=label)
    limits = [min(subset["delta_brier_old"].min(), subset["delta_brier_new"].min()),
              max(subset["delta_brier_old"].max(), subset["delta_brier_new"].max())]
    axes[0].plot(limits, limits, color="black", linewidth=0.8, linestyle="--")
    axes[0].axhline(0.0, color="gray", linewidth=0.6)
    axes[0].axvline(0.0, color="gray", linewidth=0.6)
    axes[0].set_xlabel("旧版 ΔBrier（M3−M2）", fontsize=9)
    axes[0].set_ylabel("更正版 ΔBrier（M3−M2）", fontsize=9)
    axes[0].set_title("逐格 ΔBrier：旧 vs 更正（每点一个条件×单元×窗口）", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    changes = (subset["delta_brier_new"] - subset["delta_brier_old"]).abs()
    axes[1].hist(changes, bins=20, color="#548235", alpha=0.85)
    axes[1].set_xlabel("|ΔBrier 变化|", fontsize=9)
    axes[1].set_ylabel("单元数", fontsize=9)
    axes[1].set_title(
        f"三项更正造成的偏移：中位 {changes.median():.2e}，最大 {changes.max():.2e}",
        fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("旧/新逐格差异（同日剔除、30 天阈值、截断消融）", fontsize=11)
    fig.tight_layout()
    path = out / "fig_difference.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"file": path.name,
            "caption": "旧版与更正版 ΔBrier 的逐格对照及变化幅度分布"}


def _replay_payload() -> dict:
    payload: dict[str, Any] = {"stage17": {}, "stage18": {}}
    if REPLAY17.is_dir() and (REPLAY17 / "oof_predictions.csv").is_file():
        new = pd.read_csv(STAGE17 / "oof_predictions.csv", low_memory=False)
        replay = pd.read_csv(REPLAY17 / "oof_predictions.csv", low_memory=False)
        keys = ["bundle_id", "fold", "model", "condition", "test_set"]
        merged = new.merge(replay, on=keys + ["participantid", "landmark_day", "series_id"],
                           suffixes=("_new", "_replay")) if "series_id" in new.columns else None
        # 统一以 (test_set, key) 对齐
        for frame in (new, replay):
            frame["key"] = np.where(
                frame["test_set"] == "real_test",
                frame["participantid"].astype(str) + "|" + frame["landmark_day"].astype(str),
                frame["series_id"].astype(str))
        merged = new.merge(
            replay, on=keys + ["key"], suffixes=("_new", "_replay"),
            validate="one_to_one")
        diffs = {}
        for horizon in (1, 3, 7):
            diffs[f"p_{horizon}d"] = float(
                (merged[f"p_{horizon}d_new"] - merged[f"p_{horizon}d_replay"]).abs().max())
        payload["stage17"] = {
            "rows_new": int(len(new)),
            "rows_replay": int(len(replay)),
            "rows_joined": int(len(merged)),
            "max_abs_prediction_diff": max(diffs.values()),
            "per_horizon_max_abs_diff": diffs,
            "predictions_identical_within_1e-12": bool(max(diffs.values()) < 1e-12),
            "real_test_keys_new": int(new.loc[new["test_set"] == "real_test", "key"].nunique()),
            "real_test_keys_replay": int(
                replay.loc[replay["test_set"] == "real_test", "key"].nunique()),
        }
    if REPLAY18.is_dir() and (REPLAY18 / "comparison.csv").is_file():
        new = pd.read_csv(STAGE18 / "comparison.csv")
        replay = pd.read_csv(REPLAY18 / "comparison.csv")
        keys = ["branch", "condition", "horizon_days", "scenario"]
        joined = new.merge(replay, on=keys, suffixes=("_new", "_replay"))
        payload["stage18"] = {
            "comparisons_new": int(len(new)),
            "comparisons_replay": int(len(replay)),
            "joined": bool(len(joined) == len(new) == len(replay)),
            "n_equal": bool((joined["n_new"] == joined["n_replay"]).all()),
            "positives_equal": bool(
                (joined["positives_new"] == joined["positives_replay"]).all()),
            "max_abs_delta_brier_diff": float(
                (joined["delta_brier_new"] - joined["delta_brier_replay"]).abs().max()),
            "max_abs_delta_auroc_diff": float(
                (pd.to_numeric(joined["delta_auroc_new"], errors="coerce")
                 - pd.to_numeric(joined["delta_auroc_replay"], errors="coerce")).abs().max()),
            "bootstrap_note": (
                "复现运行使用较少的 bootstrap 次数；点估计一致，区间宽度允许不同"
            ),
        }
    return payload


def _report_data(run_dir: Path) -> dict:
    comparison = pd.read_csv(run_dir / "comparison.csv")
    composite = pd.read_csv(run_dir / "composite_equal_weight.csv")
    augmentation = pd.read_csv(run_dir / "training_augmentation.csv")
    ablation = pd.read_csv(run_dir / "ablation.csv")
    detection = pd.read_csv(run_dir / "detection_table.csv")
    diagnostic = pd.read_csv(run_dir / "alarm_diagnostic.csv")
    difference = pd.read_csv(run_dir / "difference_table.csv")
    samples = pd.read_csv(run_dir / "sample_counts.csv")
    metrics = pd.read_csv(run_dir / "metrics_by_cell.csv")
    keys = pd.read_csv(STAGE16 / "real_keys.csv", dtype={"participantid": str})
    counts = _load(STAGE16 / "counts.json")
    stage17_counts = _load(STAGE17 / "counts.json")
    thresholds = _load(STAGE17 / "thresholds.json")
    generators = _load(STAGE17 / "generator_manifest.json")
    variant_comparison = _load(STAGE17 / "variant_comparison.json")
    freeze = _load(STAGE17 / "freeze_manifest.json")
    corrections = _load(run_dir / "corrections_summary.json")
    deep = lambda value: json.loads(json.dumps(value, default=str))  # noqa: E731

    data: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "samples": {
            "stage16_counts": deep(counts),
            "stage17_counts": deep(stage17_counts),
            "stage18_sample_counts": deep(samples.to_dict("records")),
        },
        "thresholds": deep(thresholds),
        "generators": [
            {
                "fold": item["fold"],
                "donor_count": item["donor_count"],
                "donor_overlap_same_day_persons": item.get("donor_overlap_same_day_persons"),
                "event_rate_anchor_7d": item["event_rate_anchor_7d"],
                "old_event_rate_anchor_7d": item.get("old_event_rate_anchor_7d"),
                "lambda0_default": item["lambda0"].get("default"),
                "old_lambda0_default": (item.get("old_lambda0") or {}).get("default"),
                "reproduces_stage12_with_old_landmarks": item.get(
                    "reproduces_stage12_with_old_landmarks"),
            }
            for item in generators["folds"]
        ],
        "variant_comparison": deep(variant_comparison),
        "freeze": deep(freeze),
        "corrections": deep(corrections),
        "primary": {
            "s_only_7d": deep(comparison.loc[
                (comparison["branch"] == "S") & (comparison["condition"] == "S-only")
                & (comparison["horizon_days"] == 7)][[
                    "scenario", "n", "positives", "delta_brier", "delta_brier_ci",
                    "delta_auroc", "delta_auroc_ci", "state"]].to_dict("records")),
            "composite": deep(composite.to_dict("records")),
            "real": deep(comparison.loc[comparison["branch"] == "R"][[
                "condition", "horizon_days", "n", "positives", "delta_brier",
                "delta_brier_ci", "delta_auroc", "delta_auroc_ci", "state"]]
                .to_dict("records")),
            "s_only_other_conditions_7d": deep(comparison.loc[
                (comparison["branch"] == "S") & (comparison["horizon_days"] == 7)
                & comparison["condition"].isin(["R+S", "R-only"])][[
                    "condition", "scenario", "n", "positives", "delta_brier",
                    "delta_brier_ci", "delta_auroc", "delta_auroc_ci", "state"]]
                .to_dict("records")),
        },
        "training_augmentation": deep(augmentation.loc[augmentation["horizon_days"] == 7][[
            "branch", "model", "scenario", "n", "positives", "delta_brier",
            "delta_brier_ci", "delta_auroc", "delta_auroc_ci"]].to_dict("records")),
        "ablation": deep(ablation.loc[
            (ablation["horizon_days"] == 7) & (ablation["condition"] == "S-only")][[
                column for column in (
                    "variant", "scenario", "n", "positives", "delta_brier",
                    "delta_brier_ci", "delta_auroc", "delta_auroc_ci", "state")
                if column in ablation.columns]].to_dict("records")),
        "ablation_all_cells_summary": {
            "cells": int(len(ablation)),
            "ci_excludes_zero": int(ablation["delta_brier_ci"].notna().sum()),
        },
        "alarm": {
            "detection": deep(detection.to_dict("records")),
            "diagnostic": deep(diagnostic.to_dict("records")),
        },
        "difference": {
            "rows": int(len(difference)),
            "median_abs_change": float(
                (difference["delta_brier_new"] - difference["delta_brier_old"]).abs().median()),
            "max_abs_change": float(
                (difference["delta_brier_new"] - difference["delta_brier_old"]).abs().max()),
            "real_rows": deep(difference.loc[difference["branch"] == "R"][[
                "condition", "horizon_days", "n_old", "n_new", "positives_old",
                "positives_new", "delta_brier_old", "delta_brier_new",
                "delta_brier_change", "state_old", "state_new"]].to_dict("records")),
            "synthetic_7d": deep(difference.loc[
                (difference["branch"] == "S") & (difference["horizon_days"] == 7)][[
                    "condition", "scenario", "n_old", "n_new", "positives_old",
                    "positives_new", "delta_brier_old", "delta_brier_new",
                    "delta_brier_change", "state_old", "state_new"]].to_dict("records")),
        },
        "metric_cells": {
            "rows": int(len(metrics)),
            "auroc_estimable": int(metrics["auroc"].notna().sum()),
            "auroc_not_estimable": int(metrics["auroc"].isna().sum()),
        },
        "keys": {
            "corrected_keys": int(counts["real"]["corrected_keys"]),
            "corrected_people": int(counts["real"]["corrected_people"]),
            "positives": counts["real"]["positives_corrected"],
        },
    }
    return data


def build_index(run_dir: Path, seal_dir: Path | None = None) -> dict:
    def entry(path: Path, note: str = "") -> dict:
        return {
            "path": str(path),
            "exists": path.exists(),
            "md5": _md5(path) if path.is_file() else None,
            "note": note,
        }

    release_manifest = _load(STAGE11 / "manifest.json")
    protocol = _load(STAGE11 / "protocol.json")
    windows = _load(STAGE16 / "window_definitions.json")
    thresholds = _load(STAGE17 / "thresholds.json")
    freeze = _load(STAGE17 / "freeze_manifest.json")
    index = {
        "package": "DS-PCD 短窗预测更正（阶段 16–18）",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "release": {
            "doi": "10.5281/zenodo.21060991",
            "version": "V1.0.0",
            "license": "CC BY-NC-ND 4.0",
            "files": release_manifest.get("release", {}).get("files", []),
        },
        "protocol": {
            **entry(STAGE11 / "protocol.json",
                    "阶段 11 冻结的研究协议（1/3/7 天、M0–M3、M3−M2）"),
            "horizons_days": protocol.get("horizons_days"),
            "primary_horizon_days": protocol.get("primary_horizon_days"),
        },
        "corrected_windows": entry(STAGE16 / "window_definitions.json",
                                   "阶段 16 冻结的三个时段端点"),
        "keys": {
            "real_keys": entry(STAGE16 / "real_keys.csv"),
            "exclusions": entry(STAGE16 / "exclusions.csv"),
            "same_day_records": entry(STAGE16 / "same_day_records.csv"),
            "cohort_flow": entry(STAGE16 / "cohort_flow_corrected.csv"),
            "synthetic_exclusions": entry(STAGE16 / "synthetic_exclusions.csv"),
            "reference_comparison": entry(STAGE16 / "reference_comparison.json"),
            "boundary_samples": entry(STAGE16 / "boundary_samples.csv"),
        },
        "thresholds": {
            "thresholds": entry(STAGE17 / "thresholds.json"),
            "calibration_prefix_summary": entry(STAGE17 / "calibration_prefix_summary.csv"),
            "calibration_prefix_daily": entry(STAGE17 / "calibration_prefix_daily.csv"),
            "window_days": thresholds.get("window_days"),
        },
        "generators": {
            "manifest": entry(STAGE17 / "generator_manifest.json"),
            **{f"fold{fold}": entry(STAGE17 / f"generator_params_fold{fold}.json")
               for fold in (0, 1, 2)},
        },
        "scores": {
            "real_scores": entry(STAGE17 / "real_scores.csv"),
            "synthetic_train": entry(STAGE17 / "synthetic_train_scores.csv"),
            "synthetic_test": entry(STAGE17 / "synthetic_test_scores.csv"),
            "synthetic_detection_daily": entry(STAGE17 / "synthetic_detection_daily.csv"),
            "variant_scores": entry(STAGE17 / "variant_scores.csv"),
            "variant_comparison": entry(STAGE17 / "variant_comparison.json"),
        },
        "models": {
            "freeze_manifest": entry(STAGE17 / "freeze_manifest.json"),
            "model_registry": entry(STAGE17 / "model_registry.json"),
            "fold_models": {
                "path": str(STAGE17 / "fold_models"),
                "exists": (STAGE17 / "fold_models").is_dir(),
                "count": len(list((STAGE17 / "fold_models").rglob("model.json"))),
                "slots": freeze.get("model_slots"),
            },
        },
        "experiments": {
            name: entry(run_dir / name) for name in (
                "paired_predictions.csv", "metrics_by_cell.csv", "comparison.csv",
                "primary_7d.csv", "secondary_1_3d.csv", "composite_equal_weight.csv",
                "training_augmentation.csv", "ablation.csv", "detection_table.csv",
                "alarm_ledger.csv", "alarm_diagnostic.csv", "difference_table.csv",
                "sample_counts.csv", "joint_tests.json", "conclusions.json",
                "corrections_summary.json", "counts.json", "verification.json",
            )
        },
        "deliverables": {
            "report": entry((seal_dir or run_dir) / "REPORT.md", "更正版中文报告"),
            "corrections": entry((seal_dir or run_dir) / "CORRECTIONS.md",
                                 "本轮三项更正记录（与 round2 根目录同内容）"),
            "figures": {
                "path": str((seal_dir or run_dir) / "figures"),
                "exists": ((seal_dir or run_dir) / "figures").is_dir(),
                "count": len(list(((seal_dir or run_dir) / "figures").glob("*.png"))),
            },
            "figures_manifest": entry((seal_dir or run_dir) / "figures_manifest.json"),
            "report_data": entry((seal_dir or run_dir) / "report_data.json"),
            "stage16_verification": entry(STAGE16 / "verification.json"),
            "stage17_verification": entry(STAGE17 / "verification.json"),
            "stage18_verification": entry(run_dir / "verification.json"),
        },
        "replay": {
            "stage17_replay": str(REPLAY17),
            "stage17_exists": REPLAY17.is_dir(),
            "stage18_replay": str(REPLAY18),
            "stage18_exists": REPLAY18.is_dir(),
            "verification": entry((seal_dir or run_dir) / "replay_verification.json"),
        },
        "history_preserved": {
            "stage11": str(STAGE11),
            "stage12": str(STAGE12),
            "stage13": str(STAGE13),
            "stage14_old": str(STAGE14),
            "stage15_sealed": str(OUTPUT_ROOT / "stage15" / "20261115-seal-v1"),
            "stage10": str(HISTORY / "stage10" / "20261114-attribution-v1"),
            "fixed_run": str(HISTORY / "20261114-fixed"),
            "main_run": str(HISTORY / "20261114-main"),
            "note": "旧运行只读保留；更正版写入新目录，不覆盖任何历史结果",
        },
        "claims": {
            "real_labels_not_fabricated": (
                "同日跌倒整日剔除，阳性从 1/2/4 降为 0/1/3；未改写任何真实跌倒日期"),
            "alarm_window_fixed": (
                "阈值与误报只使用监测第 1–30 天；第 31–37 天只判定 7 天结局"),
            "ablation_truncated": (
                "no_direction_cumulative 的 D 与 C 只用截至 t 的观测"),
            "research_only": "所有模型 research_only=true、calibrated=false；临床概率出口未变",
            "synthetic_not_real": "S 分支为半合成评价，不称为真实病例或临床验证",
        },
        "reproduction": [
            "cd F:/jiebang_team/anti-fall/elder-detect-main",
            "D:/AnacondaData/envs_dirs/py3.13/python.exe scripts/round2_stage16_keys.py --run-id 20261115-keys-v2",
            "D:/AnacondaData/envs_dirs/py3.13/python.exe scripts/round2_stage17_refreeze.py --run-id 20261115-refreeze-v1",
            "D:/AnacondaData/envs_dirs/py3.13/python.exe scripts/round2_stage18_evaluate.py --run-id 20261115-corrected-eval-v1",
            "D:/AnacondaData/envs_dirs/py3.13/python.exe scripts/verify_round2_stage16.py",
            "D:/AnacondaData/envs_dirs/py3.13/python.exe scripts/verify_round2_stage17.py",
            "D:/AnacondaData/envs_dirs/py3.13/python.exe scripts/verify_round2_stage18.py",
        ],
    }
    return index


def run(args: argparse.Namespace) -> int:
    run_dir = Path(args.out_root) / f"stage18/{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    stage18 = Path(args.stage18)

    figures = []
    for builder in (figure_primary, figure_augmentation, figure_ablation,
                    figure_detection, figure_real, figure_difference):
        try:
            figures.append(builder(stage18, figures_dir))
        except Exception as error:  # noqa: BLE001
            figures.append({"file": None, "error": f"{type(error).__name__}: {error}"})
    _write_json(run_dir / "figures_manifest.json", figures)

    replay = _replay_payload()
    _write_json(run_dir / "replay_verification.json", replay)
    data = _report_data(stage18)
    _write_json(run_dir / "report_data.json", data)
    index = build_index(stage18, run_dir)
    _write_json(run_dir / "round2_delivery_index.json", index)
    print(json.dumps({
        "figures": [item.get("file") for item in figures],
        "replay": replay,
        "index_entries": sum(
            1 for value in index.values() if isinstance(value, dict) and "path" in value),
    }, ensure_ascii=False, indent=1)[:3000], flush=True)
    print(f"[stage18-seal] 完成：{run_dir}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 18 封存与索引")
    parser.add_argument("--run-id", default="20261115-corrected-seal-v1")
    parser.add_argument("--out-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--stage18", default=str(STAGE18))
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
