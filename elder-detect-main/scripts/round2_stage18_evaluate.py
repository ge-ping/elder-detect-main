"""阶段 18：复算更正版结果并封存论文交付包。

与阶段 14 的区别只在输入与三处更正，效果量与可估规则不变：

* 测试键：阶段 16 的 171 个真实键（同日跌倒整日剔除）与剔除 117 条同日事件
  后的半合成键；
* 报警窗口：阈值只由监测第 1–30 天标定，误报与检出按 ``round2_time`` 的
  台账计算（变化前报警单记、变化后首次报警计检出、监测结束记截尾）；
* 消融：``no_direction_cumulative`` 只用截至 t 的 D 与 C。

额外输出 ``difference_table.csv``（旧/新逐格差异）与 ``sample_counts.csv``
（实际样本与阳性数），供最终报告直接引用。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import md5
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import round2_stage14_evaluate as ev  # noqa: E402
from longitudinal.cox_horizon import fit_horizon_cox, joint_test_coefficients  # noqa: E402
from mobilise_d.round2 import R2_FEATURES  # noqa: E402
from mobilise_d.round2_time import (  # noqa: E402
    ALARM_WINDOW_DAYS, HORIZONS_DAYS, detection_ledger_from_frame,
)

OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise" / "round2"
STAGE11 = OUTPUT_ROOT / "stage11" / "20261115-protocol-v1"
STAGE14 = OUTPUT_ROOT / "stage14" / "20261115-eval-v1"
STAGE16 = OUTPUT_ROOT / "stage16" / "20261115-keys-v2"
STAGE17 = OUTPUT_ROOT / "stage17" / "20261115-refreeze-v1"
SEED = 20261115
BOOTSTRAP = 2000
SCORING_VERSION = "round2-scoring-v2-truncated"

LEVEL_COLUMNS = ev.LEVEL_COLUMNS
CHANGE_COLUMNS = ev.CHANGE_COLUMNS
LADDERS = ev.LADDERS
ABLATIONS = {
    "full": ("T", "C"),
    "no_precision": ("T_none", "C_none"),
    "direction_only": ("T",),
    # 更正：不再读取尾部 7 天的 omnibus D/C，改用截至 t 的 D 与截断 C
    "no_direction_cumulative": ("D_snapshot", "C_omnibus_truncated"),
}
CONDITIONS = ev.CONDITIONS
MECHANISMS = ev.MECHANISMS
NOISE_FAMILIES = ev.NOISE_FAMILIES
_metrics = ev._metrics
_bootstrap_delta = ev._bootstrap_delta
conclusion_state = ev.conclusion_state
_fit_and_predict = ev._fit_and_predict

CODE_SNAPSHOT_FILES = (
    "scripts/round2_stage18_evaluate.py",
    "scripts/verify_round2_stage18.py",
    "mobilise_d/round2_time.py",
    "mobilise_d/round2_scoring.py",
    "longitudinal/cox_horizon.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 18：复算与交付")
    parser.add_argument("--run-id", default="20261115-corrected-eval-v1")
    parser.add_argument("--out-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--stage11", default=str(STAGE11))
    parser.add_argument("--stage16", default=str(STAGE16))
    parser.add_argument("--stage17", default=str(STAGE17))
    parser.add_argument("--stage14-old", default=str(STAGE14))
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    parser.add_argument("--only", default="all",
                        choices=("all", "core", "augmentation", "ablation"))
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")


def _md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _training_age_params(diagnostics: Mapping[str, Any]) -> dict[int, tuple[float, float]]:
    return {
        int(fold): (float(value["real"]["age_mean"]), float(value["real"]["age_sd"]))
        for fold, value in diagnostics["folds"].items()
    }


def _real_rows(stage17: Path, stage11: Path, variants: pd.DataFrame) -> pd.DataFrame:
    scores = pd.read_csv(stage17 / "real_scores.csv", dtype={"participantid": str})
    candidates = pd.read_csv(stage11 / "real_candidates.csv", dtype={"participantid": str})
    splits = pd.read_csv(stage11 / "splits.csv", dtype={"participantid": str})
    frame = scores.loc[scores["status"] == "scored"].copy()
    frame["participantid"] = frame["participantid"].astype(str)
    frame["landmark_day"] = pd.to_datetime(frame["landmark_day"]).dt.strftime("%Y-%m-%d")
    frame = frame.merge(
        candidates[[
            "participantid", "landmark_day", "first_event_7d", "y_7d_num",
        ]].rename(columns={"y_7d_num": "y_7d_checked"}),
        on=["participantid", "landmark_day"], how="left", validate="one_to_one",
    )
    landmark = pd.to_datetime(frame["landmark_day"])
    gap = (pd.to_datetime(frame["first_event_7d"]) - landmark).dt.days
    frame["event_gap"] = np.where(pd.notna(frame["first_event_7d"]), gap, np.nan)
    frame["duration"] = np.where((frame["event_gap"] > 0) & (frame["event_gap"] <= 7),
                                 frame["event_gap"], 7.0)
    frame["event"] = ((frame["event_gap"] > 0) & (frame["event_gap"] <= 7)).astype(float)
    frame = frame.merge(splits[["participantid", "fold"]], on="participantid", how="left")
    frame["key"] = frame["participantid"] + "|" + frame["landmark_day"]
    variant = variants.loc[variants["test_set"] == "real_test"].copy()
    variant["participantid"] = variant["participantid"].astype(str)
    variant["landmark_day"] = pd.to_datetime(variant["landmark_day"]).dt.strftime("%Y-%m-%d")
    variant["key"] = variant["participantid"] + "|" + variant["landmark_day"]
    frame = frame.merge(
        variant[["key", "T", "C", "T_none", "C_none", "D_snapshot", "C_omnibus_truncated",
                 "status_full", "status_none"]],
        on="key", how="left", suffixes=("", "_variant"), validate="one_to_one",
    )
    frame["branch"] = "R"
    frame["test_set"] = "real_test"
    frame["cohort"] = "pd"
    frame["scenario"] = "pd_real"
    frame["scoring_version"] = SCORING_VERSION
    return frame


def _synthetic_rows(stage17: Path, variants: pd.DataFrame) -> pd.DataFrame:
    test = pd.read_csv(stage17 / "synthetic_test_scores.csv", low_memory=False)
    train = pd.read_csv(stage17 / "synthetic_train_scores.csv", low_memory=False)
    frame = pd.concat([train.assign(batch="train"), test.assign(batch="test")],
                      ignore_index=True, sort=False)
    frame = frame.loc[frame["status"] == "scored"].copy()
    gap = pd.to_numeric(frame["event_gap_from_snapshot"], errors="coerce")
    frame["event_gap"] = gap
    frame["duration"] = np.where((gap > 0) & (gap <= 7), gap, 7.0)
    frame["event"] = ((gap > 0) & (gap <= 7)).astype(float)
    frame["key"] = frame["series_id"].astype(str)
    variant = variants.loc[variants["test_set"].str.startswith("synthetic")].copy()
    variant = variant.rename(columns={"series_id": "key"})
    variant["key"] = variant["key"].astype(str)
    frame = frame.merge(
        variant[["key", "T", "C", "T_none", "C_none", "D_snapshot", "C_omnibus_truncated",
                 "status_full", "status_none"]],
        on="key", how="left", suffixes=("", "_variant"), validate="one_to_one",
    )
    frame["branch"] = "S"
    frame["test_set"] = np.where(frame["batch"] == "test", "synthetic_test", "synthetic_train")
    frame["cohort"] = "synthetic_pd_like"
    frame["scenario"] = frame["mechanism"] + "|" + frame["noise_family"]
    frame["scoring_version"] = SCORING_VERSION
    return frame


def _alarm_tables(detection: pd.DataFrame, thresholds: Mapping[str, Any],
                  old_thresholds: Mapping[str, Any],
                  excluded_series: set[str] | None = None,
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """误报/检出表与逐序列台账（监测第 1–30 天）。"""
    frame = detection.copy()
    if excluded_series:
        frame = frame.loc[~frame["series_id"].astype(str).isin(excluded_series)].copy()
    frame["fold"] = frame["fold"].astype(int)
    for method, column in (("ds_pcd", "C"), ("rms", "rms")):
        frame[f"{method}_threshold"] = frame["fold"].map(
            lambda fold, m=method: thresholds["per_fold"][str(int(fold))]["methods"][m]["threshold"]
        )
        frame[f"{method}_old_threshold"] = frame["fold"].map(
            lambda fold, m=method: old_thresholds["per_fold"][str(int(fold))]["methods"][m]["threshold"]
        )
        frame[f"{method}_alarm"] = frame[column] >= frame[f"{method}_threshold"]
        frame[f"{method}_alarm_old"] = frame[column] >= frame[f"{method}_old_threshold"]
    frame["monitoring_offset"] = pd.to_numeric(frame["monitoring_offset"], errors="coerce")
    ledgers = []
    for method in ("ds_pcd", "rms"):
        ledger = detection_ledger_from_frame(
            frame, alarm_column=f"{method}_alarm",
            group_columns=("series_id", "fold", "trajectory", "mechanism", "noise_family"),
            onset_column="onset_offset", monitoring_days=ALARM_WINDOW_DAYS,
        )
        ledger["method"] = method
        ledgers.append(ledger)
    ledger_frame = pd.concat(ledgers, ignore_index=True)

    # 旧阈值 30 天诊断（同一批稳定测试序列）
    diagnostic = []
    for method in ("ds_pcd", "rms"):
        stable = frame.loc[frame["trajectory"] == "stable"]
        within = stable.loc[stable["monitoring_offset"] <= ALARM_WINDOW_DAYS]
        for label, column in (("new_threshold_30d", f"{method}_alarm"),
                              ("old_threshold_30d", f"{method}_alarm_old")):
            per_series = within.groupby("series_id")[column].any()
            diagnostic.append({
                "method": method, "definition": label,
                "n_series": int(per_series.size),
                "false_alarm_rate": float(per_series.mean()),
            })
        all_days = stable.dropna(subset=[f"{method}_alarm"])
        per_series_37 = all_days.groupby("series_id")[f"{method}_alarm_old"].any()
        diagnostic.append({
            "method": method, "definition": "old_threshold_37d",
            "n_series": int(per_series_37.size),
            "false_alarm_rate": float(per_series_37.mean()),
        })
    diagnostic_frame = pd.DataFrame(diagnostic)

    rows: list[dict] = []
    stable = ledger_frame.loc[ledger_frame["trajectory"] == "stable"]
    for method in ("ds_pcd", "rms"):
        subset = stable.loc[stable["method"] == method]
        rate = float(subset["false_alarm_within_monitoring"].mean())
        rows.append({
            "table": "stable_null", "method": method, "subset": "stable",
            "n_series": int(len(subset)), "false_alarm_30d": rate,
            "false_alarm_ci": [
                float(rate - 1.96 * math.sqrt(rate * (1 - rate) / max(len(subset), 1))),
                float(rate + 1.96 * math.sqrt(rate * (1 - rate) / max(len(subset), 1))),
            ],
            "alarms_after_monitoring_share": float(
                (subset["alarms_after_monitoring"] > 0).mean()),
            "note": "监测第 1–30 天内至少一次越界；第 31–37 天不计入",
        })
    cell_rows: list[dict] = []
    for method in ("ds_pcd", "rms"):
        subset = stable.loc[stable["method"] == method]
        for (mechanism, noise), group in subset.groupby(["mechanism", "noise_family"]):
            rate = float(group["false_alarm_within_monitoring"].mean())
            cell_rows.append({
                "method": method, "mechanism": mechanism, "noise_family": noise,
                "n_series": int(len(group)), "false_alarm_30d": rate,
            })
    cell_frame = pd.DataFrame(cell_rows)
    changed = ledger_frame.loc[
        ledger_frame["trajectory"].isin(("slow_linear", "rapid_step"))
        & ledger_frame["onset_offset"].notna()
    ]
    for method in ("ds_pcd", "rms"):
        subset = changed.loc[changed["method"] == method]
        for window in HORIZONS_DAYS:
            detected = subset[f"detected_{window}d"].astype(bool)
            extended = subset[f"detected_extended_{window}d"].astype(bool)
            latency = pd.to_numeric(subset[f"latency_days_{window}d"], errors="coerce")
            censored = subset[f"censored_{window}d"].astype(bool)
            has_post = subset["first_post_change_offset"].notna()
            late = has_post & ~detected
            never = ~has_post
            rows.append({
                "table": "known_change", "method": method,
                "subset": "slow_linear_rapid_step", "window_days": window,
                "n_change_series": int(len(subset)),
                "detected_within_window": float(detected.mean()),
                "detected_late": float(late.mean()),
                "undetected_share": float(never.mean()),
                "pre_change_alarm_share": float(subset["pre_change_alarm"].mean()),
                "censored_share": float(censored.mean()),
                "observable_days_mean": float(
                    pd.to_numeric(subset[f"observable_days_{window}d"]).mean()),
                "median_latency_days": float(latency.median()) if latency.notna().any() else None,
                "detected_extended_share": float(extended.mean()),
                "note": (
                    "变化后首次报警计检出；窗口为 [u, u+H)，"
                    "监测第 30 天后越界记为截尾；extended 为到第 37 天的诊断口径"
                ),
            })
    table = pd.DataFrame(rows)
    return table, ledger_frame, diagnostic_frame, cell_frame


def run(args: argparse.Namespace) -> int:
    run_dir = Path(args.out_root) / f"stage18/{args.run_id}"
    if run_dir.exists() and any(run_dir.iterdir()):
        if args.only == "all" and (run_dir / "comparison.csv").is_file():
            print(f"[stage18] 复用已存在的比较表：{run_dir}", flush=True)
        else:
            raise SystemExit(f"运行目录已存在且非空，拒绝覆盖：{run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    stage11 = Path(args.stage11)
    stage16 = Path(args.stage16)
    stage17 = Path(args.stage17)
    stage14_old = Path(args.stage14_old)

    print("[stage18] 读取冻结输入", flush=True)
    frozen_run = pd.read_csv(stage17 / "oof_predictions.csv", low_memory=False)
    variants = pd.read_csv(stage17 / "variant_scores.csv", dtype={"participantid": str})
    variants["participantid"] = variants["participantid"].astype(str)
    diagnostics = json.loads((stage17 / "training_diagnostics.json").read_text(encoding="utf-8"))
    age_params = _training_age_params(diagnostics)
    thresholds = json.loads((stage17 / "thresholds.json").read_text(encoding="utf-8"))
    old_thresholds = json.loads(
        (OUTPUT_ROOT / "stage12" / "20261115-supplement-v1" / "threshold_calibration.json")
        .read_text(encoding="utf-8"))
    detection = pd.read_csv(stage17 / "synthetic_detection_daily.csv", low_memory=False)
    keys = pd.read_csv(stage16 / "real_keys.csv", dtype={"participantid": str})

    real = _real_rows(stage17, stage11, variants)
    synthetic = _synthetic_rows(stage17, variants)
    folds = sorted(real["fold"].dropna().unique())

    # 已冻结的 M0–M3 预测
    frozen = frozen_run.copy()
    frozen["participantid"] = frozen["participantid"].astype(str)
    frozen["key"] = np.where(
        frozen["test_set"] == "real_test",
        frozen["participantid"] + "|" + frozen["landmark_day"].astype(str),
        frozen["series_id"].astype(str),
    )
    frozen["scenario"] = np.where(
        frozen["test_set"] == "real_test", "pd_real",
        frozen["mechanism"].astype(str) + "|" + frozen["noise_family"].astype(str))
    frozen["cohort"] = np.where(frozen["test_set"] == "real_test", "pd", "synthetic_pd_like")
    frozen["branch"] = np.where(frozen["test_set"] == "real_test", "R", "S")
    frozen["scoring_version"] = SCORING_VERSION
    paired_rows: list[pd.DataFrame] = []
    for horizon in HORIZONS_DAYS:
        column = f"p_{horizon}d"
        paired_rows.append(pd.DataFrame({
            "branch": frozen["branch"],
            "test_set": frozen["test_set"],
            "key": frozen["key"],
            "fold": frozen["fold"],
            "model": frozen["model"],
            "condition": frozen["condition"],
            "ablation": "full",
            "horizon_days": horizon,
            "scenario": frozen["scenario"],
            "mechanism": frozen.get("mechanism", ""),
            "noise_family": frozen.get("noise_family", ""),
            "trajectory": frozen.get("trajectory", ""),
            "cohort": frozen["cohort"],
            "scoring_version": SCORING_VERSION,
            "prediction": frozen[column],
            "label": frozen[f"y_{horizon}d"],
        }))

    # 消融：同折、同条件、同训练行，只换 DS-PCD 组件
    print("[stage18] 消融变体模型", flush=True)
    ablation_records: list[dict] = []
    for fold in folds:
        fold = int(fold)
        mean, sd = age_params[fold]
        train_real = real.loc[real["fold"] != fold]
        train_synthetic = synthetic.loc[
            (synthetic["fold"] == fold) & (synthetic["batch"] == "train")]
        test_real = real.loc[real["fold"] == fold]
        test_synthetic = synthetic.loc[
            (synthetic["fold"] == fold) & (synthetic["batch"] == "test")]
        for condition in CONDITIONS:
            for variant, extra in ABLATIONS.items():
                if variant == "full":
                    continue
                features = LADDERS["M2"] + extra
                for name, train_real_part, test_part, branch, test_set in (
                    ("real", train_real, test_real, "R", "real_test"),
                    ("synthetic", train_synthetic, test_synthetic, "S", "synthetic_test"),
                ):
                    if not len(test_part):
                        continue
                    bundle = f"round2-abl-fold{fold}-{condition}-{variant}-{name}"
                    fit, predictions, info = _fit_and_predict(
                        train_real=train_real_part, train_synthetic=train_synthetic,
                        test=test_part, features=features, condition=condition,
                        fold=fold, bundle=bundle, age_mean=mean, age_sd=sd)
                    ablation_records.append({
                        "bundle": bundle, "fold": fold, "condition": condition,
                        "variant": variant, "branch": branch,
                        "status": fit.status, "features": list(features), **info,
                    })
                    for horizon in HORIZONS_DAYS:
                        paired_rows.append(pd.DataFrame({
                            "branch": branch, "test_set": test_set,
                            "key": test_part["key"], "fold": fold,
                            "model": f"M3_{variant}", "condition": condition,
                            "ablation": variant, "horizon_days": horizon,
                            "scenario": test_part["scenario"],
                            "mechanism": test_part.get("mechanism", ""),
                            "noise_family": test_part.get("noise_family", ""),
                            "trajectory": test_part.get("trajectory", ""),
                            "cohort": test_part["cohort"],
                            "scoring_version": SCORING_VERSION,
                            "prediction": predictions[horizon],
                            "label": test_part[f"y_{horizon}d"],
                        }))
    paired = pd.concat(paired_rows, ignore_index=True, sort=False)
    paired.to_csv(run_dir / "paired_predictions.csv", index=False, encoding="utf-8")

    # 指标表
    print("[stage18] 指标与配对比较", flush=True)
    metric_rows: list[dict] = []
    for keys_group, group in paired.groupby(
        ["branch", "test_set", "scenario", "condition", "ablation", "model", "horizon_days"]
    ):
        branch, test_set, scenario, condition, ablation, model, horizon = keys_group
        metrics = _metrics(group["label"].to_numpy(), group["prediction"].to_numpy())
        metric_rows.append({
            "branch": branch, "test_set": test_set, "scenario": scenario,
            "condition": condition, "ablation": ablation, "model": model,
            "horizon_days": horizon, "scoring_version": SCORING_VERSION, **metrics,
        })
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(run_dir / "metrics_by_cell.csv", index=False, encoding="utf-8")

    def paired_delta(frame: pd.DataFrame, model_a: str, model_b: str, *,
                     cluster: str, seed: int) -> dict:
        wide = frame.pivot_table(index=["key", "label"], columns="model",
                                 values="prediction", aggfunc="first").reset_index()
        if model_a not in wide.columns or model_b not in wide.columns:
            return {"status": "not_paired"}
        wide = wide.dropna(subset=[model_a, model_b, "label"])
        if not len(wide):
            return {"status": "no_rows"}
        if cluster == "participantid":
            wide["participantid"] = wide["key"].astype(str).str.split("|").str[0]
        return _bootstrap_delta(
            wide["label"].to_numpy(), wide[model_a].to_numpy(), wide[model_b].to_numpy(),
            wide[cluster].astype(str).to_numpy(), samples=args.bootstrap, seed=seed)

    comparison_rows: list[dict] = []
    comparison_path = run_dir / "comparison.csv"
    reuse_comparison = comparison_path.is_file() and args.only in ("all", "augmentation", "ablation")
    for (branch, condition, horizon), group in ([] if reuse_comparison else paired.loc[
        paired["ablation"] == "full"
    ].groupby(["branch", "condition", "horizon_days"])):
        cluster = "participantid" if branch == "R" else "key"
        for scenario, subset in group.groupby("scenario"):
            delta = paired_delta(subset, "M2", "M3", cluster=cluster,
                                 seed=SEED + int(horizon))
            positives = int(subset.loc[subset["model"] == "M2", "label"].sum())
            row = {
                "branch": branch, "condition": condition, "horizon_days": int(horizon),
                "scenario": scenario,
                "n": int(subset.loc[subset["model"] == "M2", "key"].nunique()),
                "positives": positives, **delta,
            }
            row.update(conclusion_state(row))
            comparison_rows.append(row)
    comparison = (pd.read_csv(comparison_path) if reuse_comparison
                  else pd.DataFrame(comparison_rows))
    if not reuse_comparison:
        comparison.to_csv(comparison_path, index=False, encoding="utf-8")
    comparison.loc[(comparison["branch"] == "S") & (comparison["horizon_days"] == 7)].to_csv(
        run_dir / "primary_7d.csv", index=False, encoding="utf-8")
    comparison.loc[(comparison["branch"] == "S") & (comparison["horizon_days"] != 7)].to_csv(
        run_dir / "secondary_1_3d.csv", index=False, encoding="utf-8")

    # 等权综合
    composite_rows = []
    for (branch, condition, horizon), group in comparison.groupby(
        ["branch", "condition", "horizon_days"]
    ):
        if branch != "S":
            continue
        by_mechanism = group.loc[group["scenario"].str.startswith(tuple(MECHANISMS)), :].copy()
        by_mechanism["mechanism"] = by_mechanism["scenario"].str.split("|").str[0]
        means = by_mechanism.groupby("mechanism")[
            ["delta_brier", "delta_auroc", "positives", "n"]
        ].mean(numeric_only=True)
        composite_rows.append({
            "branch": branch, "condition": condition, "horizon_days": int(horizon),
            "mechanisms": int(len(means)),
            "delta_brier_equal_weight": float(means["delta_brier"].mean()),
            "delta_auroc_equal_weight": float(means["delta_auroc"].mean()),
            "positives_mean": float(means["positives"].mean()),
            "note": "三种机制等权平均；压力测试与噪声族不进入综合指标",
        })
    composite = pd.DataFrame(composite_rows)
    composite.to_csv(run_dir / "composite_equal_weight.csv", index=False, encoding="utf-8")

    # 训练增强
    print("[stage18] 训练增强与组件消融", flush=True)
    augmentation_rows = []
    full = paired.loc[paired["ablation"] == "full"]
    augmentation_path = run_dir / "training_augmentation.csv"
    reuse_augmentation = augmentation_path.is_file() and args.only in ("all", "ablation")
    for (branch, horizon, model, scenario), group in ([] if reuse_augmentation else full.groupby(
        ["branch", "horizon_days", "model", "scenario"]
    )):
        pivot = group.pivot_table(index=["key", "label"], columns="condition",
                                  values="prediction", aggfunc="first").reset_index()
        if not {"R-only", "R+S"} <= set(pivot.columns):
            continue
        pivot = pivot.dropna(subset=["R-only", "R+S", "label"])
        cluster = (pivot["key"].astype(str).str.split("|").str[0]
                   if branch == "R" else pivot["key"].astype(str))
        delta = _bootstrap_delta(
            pivot["label"].to_numpy(), pivot["R-only"].to_numpy(),
            pivot["R+S"].to_numpy(), np.asarray(cluster, dtype=str),
            samples=max(200, args.bootstrap // 4), seed=SEED + int(horizon))
        augmentation_rows.append({
            "branch": branch, "horizon_days": int(horizon), "model": model,
            "scenario": scenario, "n": int(len(pivot)),
            "positives": int(pivot["label"].sum()), **delta,
        })
    augmentation = (pd.read_csv(augmentation_path) if reuse_augmentation
                    else pd.DataFrame(augmentation_rows))
    if not reuse_augmentation:
        augmentation.to_csv(augmentation_path, index=False, encoding="utf-8")

    # 组件消融（相对 M2）
    ablation_rows: list[dict] = []
    base_rows = paired.loc[(paired["ablation"] == "full") & (paired["model"] == "M2")].copy()
    ablation_path = run_dir / "ablation.csv"
    reuse_ablation = ablation_path.is_file() and args.only == "all"
    for (branch, condition, horizon, scenario), variant_group in ([] if reuse_ablation else paired.loc[
        paired["ablation"] != "full"
    ].groupby(["branch", "condition", "horizon_days", "scenario"])):
        base = base_rows.loc[
            (base_rows["branch"] == branch)
            & (base_rows["condition"] == condition)
            & (base_rows["horizon_days"] == horizon)
            & (base_rows["scenario"] == scenario)]
        for variant, group in variant_group.groupby("ablation"):
            model = f"M3_{variant}"
            model_rows = group.loc[group["model"] == model]
            if not len(model_rows) or not len(base):
                continue
            subset = pd.concat([base, model_rows], ignore_index=True, sort=False)
            cluster = "participantid" if branch == "R" else "key"
            delta = paired_delta(subset, "M2", model, cluster=cluster,
                                 seed=SEED + int(horizon))
            ablation_rows.append({
                "branch": branch, "condition": condition, "horizon_days": int(horizon),
                "variant": variant, "scenario": scenario,
                "n": int(subset.loc[subset["model"] == "M2", "key"].nunique()),
                "positives": int(subset.loc[subset["model"] == "M2", "label"].sum()),
                **delta,
            })
        full_rows = paired.loc[
            (paired["ablation"] == "full") & (paired["model"] == "M3")
            & (paired["branch"] == branch) & (paired["condition"] == condition)
            & (paired["horizon_days"] == horizon) & (paired["scenario"] == scenario)]
        if len(full_rows) and len(base):
            subset = pd.concat([base, full_rows], ignore_index=True, sort=False)
            cluster = "participantid" if branch == "R" else "key"
            delta = paired_delta(subset, "M2", "M3", cluster=cluster,
                                 seed=SEED + int(horizon))
            ablation_rows.append({
                "branch": branch, "condition": condition, "horizon_days": int(horizon),
                "variant": "full", "scenario": scenario,
                "n": int(subset.loc[subset["model"] == "M2", "key"].nunique()),
                "positives": int(subset.loc[subset["model"] == "M2", "label"].sum()),
                **delta,
            })
    ablation = (pd.read_csv(ablation_path) if reuse_ablation else pd.DataFrame(ablation_rows))
    if not reuse_ablation:
        ablation.to_csv(ablation_path, index=False, encoding="utf-8")

    # 报警表与台账
    print("[stage18] 报警窗口与检出", flush=True)
    excluded_series = set(pd.read_csv(stage16 / "synthetic_exclusions.csv")["series_id"].astype(str))
    alarm_table, ledger, alarm_diagnostic, alarm_by_cell = _alarm_tables(
        detection, thresholds, old_thresholds, excluded_series=excluded_series)
    alarm_table.to_csv(run_dir / "detection_table.csv", index=False, encoding="utf-8")
    alarm_by_cell.to_csv(run_dir / "alarm_by_cell.csv", index=False, encoding="utf-8")
    ledger.to_csv(run_dir / "alarm_ledger.csv", index=False, encoding="utf-8")
    alarm_diagnostic.to_csv(run_dir / "alarm_diagnostic.csv", index=False, encoding="utf-8")

    # T/C 联合检验
    joint_tests: dict[str, Any] = {}
    for branch, condition in (("R", "R-only"), ("R", "R+S"), ("S", "S-only")):
        fold = int(folds[0])
        mean, sd = age_params[fold]
        train_real = real.loc[real["fold"] != fold]
        train_synthetic = synthetic.loc[
            (synthetic["fold"] == fold) & (synthetic["batch"] == "train")]
        test = (synthetic.loc[(synthetic["fold"] == fold) & (synthetic["batch"] == "test")]
                if branch == "S" else real.loc[real["fold"] == fold])
        features = LADDERS["M3"]
        fit, _, _ = _fit_and_predict(
            train_real=train_real, train_synthetic=train_synthetic, test=test,
            features=features, condition=condition, fold=fold,
            bundle=f"joint-{branch}-{condition}", age_mean=mean, age_sd=sd)
        test = test.copy()
        test["age_z"] = (test["age"].astype(float) - mean) / sd
        test = test.loc[np.isfinite(test[list(features)].to_numpy(dtype=float)).all(axis=1)]
        clusters = (test["participantid"].astype(str) if branch == "R"
                    else test["series_id"].astype(str))
        joint_tests[f"{branch}|{condition}|fold{fold}"] = joint_test_coefficients(
            fit,
            test["duration"].to_numpy(dtype=float),
            test["event"].to_numpy(dtype=int),
            test[list(features)].to_numpy(dtype=float),
            clusters.to_numpy(), ["T", "C"])
    _write_json(run_dir / "joint_tests.json", joint_tests)

    # 实际样本与阳性数
    corrected = keys.loc[keys["corrected_key"].astype(bool)]
    sample_rows: list[dict] = []
    for horizon in HORIZONS_DAYS:
        sample_rows.append({
            "branch": "R", "test_set": "real_test", "horizon_days": horizon,
            "n": int(len(real)), "people": int(real["participantid"].nunique()),
            "positives": int(real[f"y_{horizon}d"].sum()),
            "positive_people": int(real.loc[real[f"y_{horizon}d"].eq(1),
                                            "participantid"].nunique()),
            "note": "同日跌倒整日剔除后的 171 键；AUROC 需 ≥5 个阳性",
        })
    for horizon in HORIZONS_DAYS:
        test_syn = synthetic.loc[synthetic["batch"] == "test"]
        sample_rows.append({
            "branch": "S", "test_set": "synthetic_test", "horizon_days": horizon,
            "n": int(len(test_syn)), "people": int(test_syn["series_id"].nunique()),
            "positives": int(test_syn[f"y_{horizon}d"].sum()),
            "positive_people": int(test_syn.loc[test_syn[f"y_{horizon}d"].eq(1),
                                                "series_id"].nunique()),
            "note": "剔除同日事件后的独立虚拟个体；不混入真实测试集",
        })
    sample_counts = pd.DataFrame(sample_rows)
    sample_counts.to_csv(run_dir / "sample_counts.csv", index=False, encoding="utf-8")

    # 旧/新差异表
    difference_rows: list[dict] = []
    old_comparison = pd.read_csv(stage14_old / "comparison.csv")
    merged = old_comparison.merge(
        comparison, on=["branch", "condition", "horizon_days", "scenario"],
        how="outer", suffixes=("_old", "_new"), indicator=True).rename(
        columns={"_merge": "merge_flag"})
    for row in merged.itertuples(index=False):
        difference_rows.append({
            "branch": row.branch, "condition": row.condition,
            "horizon_days": row.horizon_days, "scenario": row.scenario,
            "matched": row.merge_flag == "both",
            "n_old": getattr(row, "n_old", None), "n_new": getattr(row, "n_new", None),
            "positives_old": getattr(row, "positives_old", None),
            "positives_new": getattr(row, "positives_new", None),
            "delta_brier_old": getattr(row, "delta_brier_old", None),
            "delta_brier_new": getattr(row, "delta_brier_new", None),
            "delta_brier_change": (
                getattr(row, "delta_brier_new", None) - getattr(row, "delta_brier_old", None)
                if pd.notna(getattr(row, "delta_brier_new", None))
                and pd.notna(getattr(row, "delta_brier_old", None)) else None),
            "delta_auroc_old": getattr(row, "delta_auroc_old", None),
            "delta_auroc_new": getattr(row, "delta_auroc_new", None),
            "state_old": getattr(row, "state_old", None),
            "state_new": getattr(row, "state_new", None),
        })
    difference = pd.DataFrame(difference_rows)
    difference.to_csv(run_dir / "difference_table.csv", index=False, encoding="utf-8")

    # 三项更正的汇总
    old_detection = pd.read_csv(stage14_old / "detection_table.csv")
    corrections = {
        "same_day": {
            "old_real_test_rows": 172, "new_real_test_rows": int(len(real)),
            "old_real_test_people": 134,
            "new_real_test_people": int(real["participantid"].nunique()),
            "old_positives_1_3_7": [1, 2, 4],
            "new_positives_1_3_7": [int(real[f"y_{h}d"].sum()) for h in HORIZONS_DAYS],
            "synthetic_test_same_day_removed": 117,
            "synthetic_train_same_day_removed": 37,
        },
        "alarm_window": {
            "new_thresholds": {
                str(fold): {
                    "ds_pcd": thresholds["per_fold"][str(fold)]["methods"]["ds_pcd"]["threshold"],
                    "rms": thresholds["per_fold"][str(fold)]["methods"]["rms"]["threshold"],
                } for fold in (0, 1, 2)
            },
            "old_thresholds": {
                str(fold): {
                    "ds_pcd": old_thresholds["per_fold"][str(fold)]["methods"]["ds_pcd"]["threshold"],
                    "rms": old_thresholds["per_fold"][str(fold)]["methods"]["rms"]["threshold"],
                } for fold in (0, 1, 2)
            },
            "achieved_rate": {
                str(fold): {
                    "ds_pcd": thresholds["per_fold"][str(fold)]["methods"]["ds_pcd"]["achieved_rate"],
                    "rms": thresholds["per_fold"][str(fold)]["methods"]["rms"]["achieved_rate"],
                } for fold in (0, 1, 2)
            },
            "old_table_false_alarm": old_detection.loc[
                old_detection["subset"] == "stable_null",
                ["method", "false_alarm_30d"]].to_dict("records"),
            "corrected_false_alarm": alarm_table.loc[
                alarm_table["table"] == "stable_null",
                ["method", "false_alarm_30d"]].to_dict("records"),
            "old_threshold_30d_diagnostic": alarm_diagnostic.to_dict("records"),
        },
        "ablation_truncation": {
            "note": "旧 no_direction_cumulative 读取尾部 7 天的 D 与 C；新版只用截至 t 的值",
            "changed": True,
        },
    }
    _write_json(run_dir / "corrections_summary.json", corrections)

    # 结论
    conclusions = {"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for branch in ("S", "R"):
        subset = comparison.loc[(comparison["branch"] == branch)
                                & (comparison["horizon_days"] == 7)]
        if len(subset):
            conclusions[branch] = {
                "by_condition_scenario": subset[[
                    "condition", "scenario", "n", "positives", "delta_brier",
                    "delta_brier_ci", "delta_auroc", "delta_auroc_ci", "state",
                ]].to_dict("records"),
            }
    conclusions["composite_7d"] = composite.loc[
        composite["horizon_days"] == 7].to_dict("records") if len(composite) else []
    conclusions["alarm_table"] = alarm_table.to_dict("records")
    _write_json(run_dir / "conclusions.json", conclusions)
    _write_json(run_dir / "ablation_fit_status.json", {"models": ablation_records})
    _write_json(run_dir / "counts.json", {
        "paired_rows": int(len(paired)),
        "keys": {
            "real": int(real["key"].nunique()),
            "synthetic_test": int(synthetic.loc[synthetic["batch"] == "test", "key"].nunique()),
            "synthetic_train": int(synthetic.loc[synthetic["batch"] == "train", "key"].nunique()),
        },
        "comparisons": int(len(comparison)),
        "bootstrap": args.bootstrap,
        "scoring_version": SCORING_VERSION,
    })

    snapshot = run_dir / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for relative in CODE_SNAPSHOT_FILES:
        source = PROJECT_ROOT / relative
        if source.is_file():
            target = snapshot / Path(relative).name
            shutil.copy2(source, target)
    manifest = {
        "stage": 18,
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "upstream": {
            str(stage16): {name: _md5(stage16 / name) for name in (
                "real_keys.csv", "window_definitions.json", "synthetic_exclusions.csv")},
            str(stage17): {name: _md5(stage17 / name) for name in (
                "oof_predictions.csv", "variant_scores.csv", "thresholds.json",
                "freeze_manifest.json", "synthetic_test_scores.csv")},
            str(stage14_old): {name: _md5(stage14_old / name) for name in (
                "comparison.csv", "detection_table.csv")},
        },
        "old_run": str(stage14_old),
        "environment": {"python_version": __import__("platform").python_version(),
                        "executable": sys.executable},
        "seed": SEED,
        "bootstrap": args.bootstrap,
        "outputs": sorted(str(path.relative_to(run_dir))
                          for path in run_dir.rglob("*") if path.is_file()),
    }
    _write_json(run_dir / "manifest.json", manifest)
    print(f"[stage18] 完成：{run_dir}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
