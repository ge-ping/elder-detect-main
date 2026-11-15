"""阶段 11：固定数据补充范围与再实验协议。

只读原包与历史产物，输出数据缺口、队列流程、真实候选表、人员划分与
``protocol.json``。本阶段不训练 DS-PCD 风险模型、不生成批量增强数据。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import md5
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mobilise_d import sources  # noqa: E402
from mobilise_d.clinical import load_clinical  # noqa: E402
from mobilise_d.daily import load_daily, unit_consistency_checks  # noqa: E402
from mobilise_d.round2 import (  # noqa: E402
    COHORTS, DATED_COHORTS, EXTENDED_DIRECTIONS, EXTENDED_FEATURES, HISTORY_DIRECTIONS,
    HISTORY_FEATURES, HORIZONS_DAYS, PRIMARY_HORIZON_DAYS, R2_BASELINE_COLUMNS,
    R2_DIRECTIONS, R2_FEATURES, R2_SOURCE_COLUMN, Round2Config,
    assign_person_folds, build_round2_cohort, daily_field_audit, data_gap_matrix,
    load_baseline_fall_history, load_fall_history, nested_label_audit,
    real_candidate_table, short_window_table,
)

OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise" / "round2"
HISTORICAL_RUN = PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise" / "20261114-fixed"
HISTORICAL_STAGE3 = PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise" / "stage3"
HISTORICAL_STAGE10 = (
    PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise" / "stage10" / "20261114-attribution-v1"
)
SEED = 20261115

CODE_SNAPSHOT_FILES = (
    "mobilise_d/round2.py",
    "mobilise_d/cohort.py",
    "mobilise_d/clinical.py",
    "mobilise_d/daily.py",
    "mobilise_d/sources.py",
    "longitudinal/ds_pcd.py",
    "longitudinal/cox_fit.py",
    "scripts/round2_stage11_protocol.py",
    "scripts/verify_round2_stage11.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 11：数据补充范围与再实验协议")
    parser.add_argument("--run-id", default="20261115-protocol-v1")
    parser.add_argument("--out-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--cache-dir", default=None)
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(repo: Path) -> dict:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return {"commit": commit, "dirty": bool(status), "status_lines": len(status.splitlines())}
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"}


def _environment() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def cohort_flow_table(cohort) -> pd.DataFrame:
    frame = cohort.flow.copy()
    # 各队列复用同一筛选函数，步骤名按队列改写，避免把 MS/PFF/COPD 叫成 PD。
    if "cohort" in frame.columns:
        frame["step"] = [
            step.replace("older_pd", f"older_{name}")
            for step, name in zip(frame["step"], frame["cohort"])
        ]
        frame["detail"] = [
            detail.replace("cohort pd == T", f"cohort {name} == T")
            for detail, name in zip(frame["detail"], frame["cohort"])
        ]
    frame["evidence_type"] = "real"
    return frame


def _short_counts(short: pd.DataFrame) -> dict:
    out = {}
    for horizon in HORIZONS_DAYS:
        column = f"y_{horizon}d_num"
        out[f"{horizon}d"] = {
            "rows": int(len(short)),
            "known": int(short[column].notna().sum()),
            "positive_landmarks": int((short[column] == 1).sum()),
            "positive_people": int(short.loc[short[column] == 1, "participantid"].nunique()),
            "unknown": int(short[column].isna().sum()),
        }
    return out


def _add_durations(real: pd.DataFrame) -> pd.DataFrame:
    """从首次事件日推出截断随访时间与事件指示。

    7 天窗口用同一截断资料拟合，再从同一生存曲线查询 1/3/7 天；30 天仅用于
    已查看过的历史阳性对照，不进入本轮主分析。
    """
    frame = real.copy()
    landmark = pd.to_datetime(frame["landmark_day"])
    for horizon in HORIZONS_DAYS:
        first = frame.get(f"first_event_{horizon}d")
        gap = (pd.to_datetime(first) - landmark).dt.days
        event = frame[f"y_{horizon}d_num"]
        duration = np.where(event == 1, gap, float(horizon))
        frame[f"duration_{horizon}d"] = np.where(event.isna(), np.nan, duration)
        frame[f"event_{horizon}d"] = event
    historical = pd.read_csv(
        HISTORICAL_RUN / "data/landmark_windows.csv", dtype={"participantid": str}
    )
    historical = historical.loc[historical["horizon_days"] == 30].copy()
    historical["landmark_day"] = pd.to_datetime(historical["landmark_day"]).dt.strftime("%Y-%m-%d")
    historical["event_30d"] = historical["event"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    gap30 = (
        pd.to_datetime(historical["first_event_day"]) - pd.to_datetime(historical["landmark_day"])
    ).dt.days
    historical["duration_30d"] = np.where(historical["event_30d"] == 1, gap30, 30.0)
    frame = frame.merge(
        historical[["participantid", "landmark_day", "event_30d", "duration_30d"]],
        on=["participantid", "landmark_day"], how="left", validate="one_to_one",
    )
    return frame


def _clinical_reference(
    real: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    seed: int = SEED,
) -> dict:
    """临床参照诊断：既往跌倒史 / +年龄 / +绝对步速。

    只使用已查看过的 PD 范围；结果不改变历史主表，也不作为流程正确门槛。
    短窗阳性过少时按预先声明的规则标 ``not_estimable``。
    """
    from longitudinal.cox_horizon import fit_horizon_cox, predict_horizon_risk
    from sklearn.metrics import roc_auc_score

    frame = _add_durations(real.loc[real["r_branch_eligible"]].copy())
    frame = frame.merge(folds[["participantid", "fold"]], on="participantid", how="left")
    frame = frame.loc[frame["fold"].notna()].copy()
    frame["fold"] = frame["fold"].astype(int)
    frame["age_z"] = (frame["age"] - frame["age"].mean()) / frame["age"].std(ddof=0)
    speed = frame["landmark_walking_speed_m_s"].astype(float)
    frame["speed_z"] = (speed - speed.mean()) / speed.std(ddof=0)
    models = {
        "fall_history": ["prior_fall_any"],
        "fall_history_age": ["prior_fall_any", "age_z"],
        "fall_history_age_speed": ["prior_fall_any", "age_z", "speed_z"],
    }
    out: dict[str, Any] = {
        "scope": "real PD short windows (already viewed historically)",
        "rows": int(len(frame)),
        "people": int(frame["participantid"].nunique()),
        "horizons": {},
    }
    plan = {1: (1, 7), 3: (3, 7), 7: (7, 7), 30: (30, 30)}
    for horizon, (label_horizon, fit_horizon) in plan.items():
        # 30 天标签来自已封存的历史运行，不是本轮短窗标签。
        label = "event_30d" if horizon == 30 else f"y_{label_horizon}d_num"
        if horizon == 30:
            frame["event_30d"] = pd.to_numeric(frame.get("event_30d"), errors="coerce")
        duration_column = f"duration_{fit_horizon}d"
        event_column = f"event_{fit_horizon}d"
        if label not in frame.columns or duration_column not in frame.columns:
            continue
        window = frame.loc[frame[label].notna() & frame[duration_column].notna()].copy()
        result = {
            "rows": int(len(window)),
            "positives": int((window[label] == 1).sum()),
            "positive_people": int(window.loc[window[label] == 1, "participantid"].nunique()),
            "fit_horizon_days": fit_horizon,
            "query_horizon_days": horizon,
            "models": {},
        }
        for name, covariates in models.items():
            preds = pd.Series(np.nan, index=window.index, dtype=float)
            status = []
            for fold in sorted(window["fold"].unique()):
                train = window.loc[window["fold"] != fold]
                test = window.loc[window["fold"] == fold]
                if not len(test):
                    continue
                fit = fit_horizon_cox(
                    train[duration_column].to_numpy(dtype=float),
                    (train[event_column] == 1).to_numpy(dtype=int),
                    train[covariates].to_numpy(dtype=float),
                    covariates,
                    horizon_days=float(fit_horizon),
                )
                status.append(fit.status)
                if fit.status != "ok":
                    continue
                preds.loc[test.index] = predict_horizon_risk(
                    fit, test[covariates].to_numpy(dtype=float), horizon
                )
            scored = preds.notna()
            entry = {
                "status": sorted(set(status)),
                "scored_rows": int(scored.sum()),
                "features": list(covariates),
            }
            if scored.sum() >= 5 and window.loc[scored, label].nunique() > 1:
                entry["auroc"] = float(roc_auc_score(window.loc[scored, label], preds[scored]))
                entry["brier"] = float(
                    ((preds[scored] - window.loc[scored, label]) ** 2).mean()
                )
                entry["estimable"] = True
            else:
                entry["estimable"] = False
                entry["reason"] = "too_few_labels_or_single_class"
            result["models"][name] = entry
        result["auroc_estimable"] = result["positives"] >= 5
        if not result["auroc_estimable"]:
            result["not_estimable_reason"] = (
                f"窗口 {horizon} 天只有 {result['positives']} 个阳性预测日，"
                "少于预先声明的 5 个，区分度不可估"
            )
        out["horizons"][f"{horizon}d"] = result
    return out


def build_protocol(
    *,
    cohort,
    real_candidates: pd.DataFrame,
    folds: pd.DataFrame,
    config: Round2Config,
    reference: dict,
    history_gap: dict,
) -> dict:
    """本轮唯一冻结的实验协议（C1–C10 的可执行取值）。"""
    real = real_candidates.loc[real_candidates["cohort"].isin(DATED_COHORTS)].copy()
    eligible = real.loc[real["baseline_eligible"].astype(bool)]
    r_persons = int(eligible["participantid"].nunique())
    short = cohort.short_windows
    short = short.loc[short["horizon_days"].isin(HORIZONS_DAYS)].copy()
    short_labels = short_window_table(short)
    return {
        "protocol_version": "round2-v1",
        "stage": 11,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "horizons_days": list(HORIZONS_DAYS),
        "primary_horizon_days": PRIMARY_HORIZON_DAYS,
        "secondary_horizons_days": [1, 3],
        "time_definitions": {
            "baseline_window": "个人既往参考：预测日前最早 14 个主特征有效日，严格早于预测日",
            "monitoring_window": "两个预测时点之间的实际有效观测日；S 固定 30 天",
            "prediction_horizon_days": list(HORIZONS_DAYS),
            "landmark_definition": "该次采集块最后一天的观测结束后评估；同日跌倒不计入",
            "nested_labels": "同一事件时间导出 y_1d<=y_3d<=y_7d；未知标签保持空值",
            "monitoring_false_alarm_window_days": 30,
            "reset_rule": "真实分支沿用 >7 天空档重置 CUSUM；S 连续 30 天不重置",
        },
        "branches": {
            "R": {
                "description": "真实结局内部验证/探索：PD >=65 岁，逐次跌倒日期",
                "cohorts": list(DATED_COHORTS),
                "persons_with_baseline_and_outcome": r_persons,
                "notes": "MS 只有半年总数、PFF/COPD 只有半年回忆，均无逐次日期，不进入短窗监督测试",
            },
            "S": {
                "description": "半合成完整流程：真实训练数据约束分布与噪声，独立生成person与事件时间",
                "mechanisms": ["level_dominant", "sustained_change", "change_no_effect"],
                "noise_families": ["gaussian_ar1", "heavy_tail_spike"],
                "stress_variant": "measurement_bias_drift",
                "trajectories": {
                    "level_dominant": ["stable", "slow_linear"],
                    "sustained_change": ["slow_linear", "rapid_step"],
                    "change_no_effect": ["transient", "reverse"],
                },
                "scenarios_auxiliary": [
                    "stable_null", "gradual_prior_direction", "sudden_change",
                    "reverse_change", "transient_noise_recovery", "heterogeneous_precision",
                ],
                "person_budget": {
                    "train_per_fold_per_mechanism_per_noise": 600,
                    "test_per_fold_per_mechanism_per_noise": 2000,
                    "calibration_stable_per_fold": 800,
                    "stress_test_per_fold": 1000,
                    "total_test_sequences": 36000,
                    "total_train_sequences": 10800,
                    "justification": "7 天事件率按真实训练折 30 天风险的常风险折算约 2%，2000×3 折可给出约 120 个独立事件；不是为提高短期事件率而调参",
                },
                "noise_fit_source": "per-fold training donors only (C3)",
                "training_conditions": ["R-only", "R+S", "S-only"],
                "simulation_parameters": {
                    "latent_health_units": "个人基线标准差（z 单位）",
                    "loadings": {
                        "gait_speed_mean": -1.0, "stride_length_mean_cm": -1.0,
                        "stride_duration_bout_cv": 1.0,
                    },
                    "fitted_from_training_donors": [
                        "个人基线均值与个人标准差分布", "日间波动与滞后 1 自相关",
                        "会话级测量标准误分布与段数关系", "相邻采集块变化幅度分布",
                        "年龄分布", "既往跌倒比例", "30 天事件率锚点",
                    ],
                    "declared_assumptions": {
                        "beta_level_per_sd": 0.5,
                        "beta_change_per_sd_mechanism_B": 0.8,
                        "beta_age_per_sd": 0.3,
                        "beta_prior_fall": 0.8,
                        "trajectory_timing_days": {"rapid_step_window": [10, 25], "transient_length": 7},
                        "note": "只有训练数据支持的参数才拟合，其余明确为研究假设（不调参）",
                    },
                    "event_rate_anchor": "训练折真实 30 天观测事件比例（常风险换算校验 7 天比例）",
                    "seeds": {"base": 20261115, "per_fold_offset": 1000, "per_mechanism_offset": 10},
                },
            },
        },
        "features": {
            "main": {
                "names": list(R2_FEATURES),
                "directions": dict(R2_DIRECTIONS),
                "source_columns": dict(R2_SOURCE_COLUMN),
                "role": "DS-PCD 方向统计量与绝对水平、朴素变化",
                "frozen_before_performance": True,
            },
            "extended_sensitivity": {
                "names": list(EXTENDED_FEATURES),
                "directions": dict(EXTENDED_DIRECTIONS),
                "note": "活动量受佩戴时长控制、缺方向依据，只作双侧通道",
            },
            "historical_ablation": {
                "names": list(HISTORY_FEATURES),
                "directions": dict(HISTORY_DIRECTIONS),
                "note": "历史三项特征；步频与跨步周期近似互为倒数，方向检验退化为变慢",
            },
            "baseline_columns": list(R2_BASELINE_COLUMNS),
            "variability_source": "逐段资料重算段间跨步周期变异；日表 *_var_d 单位未核验，不进入主特征",
            "precision_modes": ["session_se", "none"],
        },
        "ds_pcd": {
            "core_module": "longitudinal/ds_pcd.py",
            "min_baseline_days": 14,
            "min_baseline_days_after_outlier": 10,
            "monitoring_days": 30,
            "k_ref": 0.5,
            "shrink_lambda": 0.5,
            "null_samples": 20000,
            "quantile_points": 1000,
            "q_min": -4.0,
            "q_max": 6.0,
            "outlier_z": 4.0,
            "min_features": 2,
            "score_snapshot": "监测结束日 t 的 T 与 C（主比较）；逐日刷新只作动态评价",
            "ablation_configs": {
                "full": {"precision_mode": "session_se", "use_change": True, "direction": True},
                "no_precision": {"precision_mode": "none", "use_change": True, "direction": True},
                "direction_only": {"precision_mode": "session_se", "use_change": False, "direction": True},
                "cumulative_only": {"precision_mode": "session_se", "use_change": True, "direction": False},
            },
            "two_sided_channel": "omnibus D 保留，仅用于观察先验方向外变化",
        },
        "models": {
            "ladder": {
                "M0": ["age_z", "prior_fall_any"],
                "M1": ["M0", "absolute_level(feature means in monitoring window)"],
                "M2": ["M1", "naive_change(monitoring mean - baseline mean)"],
                "M3": ["M2", "ds_pcd_T", "ds_pcd_C"],
            },
            "primary_comparison": "M3 - M2",
            "secondary_comparisons": ["M1 - M0", "M2 - M1"],
            "training_conditions": ["R-only", "R+S", "S-only"],
            "family": "Landmark Cox, Breslow ties, fixed ridge, full baseline survival ladder",
            "horizon_query": "1/3/7 天累计风险由同一生存曲线查询，概率单调",
            "weights": {
                "synthetic_total_weight_share": 0.5,
                "normalisation": "按源人/生成家族归一，复制不增加独立权重",
                "real_rows_weight": 1.0,
            },
            "age_centering": "训练折均值/标准差",
            "research_only": True,
            "calibrated": False,
        },
        "splits": {
            "unit": "participantid",
            "folds": 3,
            "seed": SEED,
            "stratified_by": ["any_event_7d", "cohort"],
            "frozen_before_derivation": True,
            "real_persons": int(len(folds)),
            "fold_sizes": folds["fold"].value_counts().sort_index().to_dict(),
            "fold_positives_7d": (
                folds.merge(
                    short_labels.groupby("participantid", as_index=False)["y_7d_num"].max(),
                    on="participantid", how="left",
                ).groupby("fold")["y_7d_num"].apply(lambda s: int((s == 1).sum())).to_dict()
            ),
            "fold_positive_rows_7d": (
                short_labels.merge(folds[["participantid", "fold"]], on="participantid", how="inner")
                .groupby("fold")["y_7d_num"].apply(lambda s: int((s == 1).sum())).to_dict()
            ),
            "historical_folds_reused": True,
        },
        "statistics": {
            "primary_effect": "Delta Brier (M3-M2, 7d, negative = improvement)",
            "metrics": ["Delta Brier", "Delta AUROC", "AUPRC", "calibration slope/intercept",
                        "effective N", "independent positives"],
            "interval": "person-clustered paired bootstrap (R); source-family clustered + Monte Carlo batches (S)",
            "bootstrap_samples": 2000,
            "multiple_testing": "1/3 天按预设次要窗口报告，不挑选最优窗口改主终点",
            "estimability_rule": {
                "auroc_min_positives": 5,
                "brier": "标签完整即可计算，需与不可估的区分度分开报告",
                "delta": "仅在同一测试键、同一 M 对的配对行上计算",
            },
            "joint_test": "人员聚类稳健的 T/C 联合检验；不可估时保留效果量与原因",
            "no_lrt_on_penalised_or_synthetic_rows": True,
            "conclusion_states": [
                "incremental_evidence", "no_clear_increment_wide_interval",
                "no_practical_increment_or_decline",
            ],
            "significance_is_not_completion_gate": True,
        },
        "calibration": {
            "target": "30 个监测日内至少一次误报的概率 10%",
            "methods": ["RMS", "DS-PCD CUSUM"],
            "calibration_data": "独立标定部分的稳定虚拟序列（不进入训练、不参与测试评价）",
            "test_false_alarm_reported_stage": 14,
            "not_a_real_population_rate": True,
        },
        "paths": {
            "output_root": "output/ds_pcd_mobilise/round2",
            "raw_packages": "anti-fall/_datasets/Mobilise-D-CVS/21060991",
            "historical_fixed_run": "output/ds_pcd_mobilise/20261114-fixed",
            "historical_stage3": "output/ds_pcd_mobilise/stage3",
            "historical_stage10": "output/ds_pcd_mobilise/stage10/20261114-attribution-v1",
        },
        "preserved_history": {
            "stage10_main": {
                "rows": 172, "people": 134, "events_30d": 16, "positive_people": 14,
                "delta_auroc": 0.008012820512820484,
                "delta_brier": -0.0007249743316076612,
                "ci_delta_auroc": [-0.1370749745196013, 0.12621808488483482],
                "judgment": "insufficient_evidence",
            },
            "do_not_overwrite": ["20261114-main", "20261114-fixed", "stage10/20261114-attribution-v1"],
        },
        "real_reference_diagnostic": reference,
        "data_gap_summary": history_gap,
        "verification_plan": {
            "stage11": ["短窗连接核验", "人员划分无交集", "协议主比较与模拟标签来源完整"],
            "stage12": ["时间线与来源核验", "真实/虚拟共享核心回放一致", "阈值标定记录"],
            "stage13": ["模型隔离与短窗数值核验", "冻结清单"],
            "stage14": ["逐条预测独立重算主表", "不可估项入汇总"],
            "stage15": ["冻结主流程一次复现", "封存检查"],
        },
        "explicit_non_scope": [
            "新数据下载", "依赖安装", "D1 视频自采", "临床部署", "半年/30 天终点重跑",
        ],
    }


def run(args: argparse.Namespace) -> int:
    run_dir = Path(args.out_root) / f"stage11/{args.run_id}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"运行目录已存在且非空，拒绝覆盖：{run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir) if args.cache_dir else sources.default_cache_dir()
    resolved = sources.ensure_files(cache, sources.COHORT_FILES, offline=True)

    print("[stage11] 读取临床与日表", flush=True)
    clinical = load_clinical(resolved["Clinical Data.zip"])
    daily_data = load_daily(resolved["Aggregated DMO Data.zip"])
    fall_history = load_fall_history(resolved["Clinical Data.zip"])
    baseline_history = load_baseline_fall_history(resolved["Clinical Data.zip"])

    config = Round2Config()
    print("[stage11] 装配四队列", flush=True)
    cohort = build_round2_cohort(
        clinical, daily_data.frame, config,
        fall_history=fall_history, baseline_history=baseline_history,
    )

    short = short_window_table(cohort.short_windows)
    real = real_candidate_table(cohort)
    real.to_csv(run_dir / "real_candidates.csv", index=False, encoding="utf-8")

    # 人群与队列
    cohort.participants.to_csv(run_dir / "participants.csv", index=False, encoding="utf-8")
    cohort.landmarks.to_csv(run_dir / "landmarks.csv", index=False, encoding="utf-8")
    cohort.baseline_days.to_csv(run_dir / "baseline_days.csv", index=False, encoding="utf-8")
    cohort.windows.to_csv(run_dir / "landmark_windows.csv", index=False, encoding="utf-8")
    short.to_csv(run_dir / "short_window_labels.csv", index=False, encoding="utf-8")
    cohort_flow_table(cohort).to_csv(run_dir / "cohort_flow.csv", index=False, encoding="utf-8")
    daily_field_audit(cohort.daily).to_csv(run_dir / "daily_field_audit.csv", index=False, encoding="utf-8")

    audit = nested_label_audit(short)
    _write_json(run_dir / "short_window_audit.json", audit)

    gap = data_gap_matrix(cohort, config)
    gap.to_csv(run_dir / "data_gap_matrix.csv", index=False, encoding="utf-8")

    # 人员划分：真实分支沿用历史 stage3 三折（已看过 PD 范围）
    historical_folds = pd.read_csv(HISTORICAL_STAGE3 / "folds.csv", dtype={"participantid": str})
    historical_folds = historical_folds.rename(columns={"fold": "fold"})
    real_persons = sorted(
        real.loc[real["r_branch_eligible"], "participantid"].unique()
    )
    fold_map = historical_folds.set_index("participantid")["fold"]
    missing = [person for person in real_persons if person not in fold_map.index]
    if missing:
        raise SystemExit(f"历史三折未覆盖 {len(missing)} 名真实候选受试者：{missing[:5]}")
    folds = (
        pd.DataFrame({"participantid": real_persons})
        .assign(fold=lambda frame: frame["participantid"].map(fold_map).astype(int))
        .merge(
            real.groupby("participantid", as_index=False).agg(
                cohort=("cohort", "first"),
                n_landmarks=("landmark_day", "size"),
            ),
            on="participantid", how="left", validate="one_to_one",
        )
        .merge(
            real.groupby("participantid", as_index=False)["y_7d_num"].max().rename(
                columns={"y_7d_num": "any_event_7d"}
            ),
            on="participantid", how="left", validate="one_to_one",
        )
    )
    folds["stratum"] = folds["fold"].astype(str)
    folds.to_csv(run_dir / "splits.csv", index=False, encoding="utf-8")

    # S 的 donor 划分：同一组真实人员，反向标注训练/测试折
    donor_rows = []
    for fold in sorted(folds["fold"].unique()):
        for _, row in folds.iterrows():
            donor_rows.append({
                "participantid": row["participantid"],
                "fold": int(fold),
                "role": "test" if int(row["fold"]) == int(fold) else "train",
                "any_event_7d": row["any_event_7d"],
            })
    pd.DataFrame(donor_rows).to_csv(run_dir / "splits_donors.csv", index=False, encoding="utf-8")

    # 人群中位特征水平（供 S 生成器参照，仅训练侧在阶段 12 重新估计）
    reference = _clinical_reference(real, folds)
    _write_json(run_dir / "clinical_reference.json", reference)

    counts = {
        "real": {
            "candidate_landmarks": int(len(real)),
            "baseline_ready": int(real["baseline_eligible"].astype(bool).sum()),
            "r_branch_eligible": int(real["r_branch_eligible"].sum()),
            "persons": int(len(folds)),
            "short_window": _short_counts(real),
            "by_cohort": {
                name: {
                    "landmarks": int((real["cohort"] == name).sum()),
                    "baseline_ready": int(
                        ((real["cohort"] == name) & real["baseline_eligible"].astype(bool)).sum()
                    ),
                }
                for name in COHORTS
            },
        },
        "cohort_counts": cohort.counts,
        "simulated": {"sequences": 0, "events": 0, "note": "阶段 11 不生成数据"},
    }
    _write_json(run_dir / "counts.json", counts)

    unit_checks = unit_consistency_checks(cohort.daily)
    _write_json(run_dir / "unit_checks.json", unit_checks)

    history_gap = {
        "historical_172": {
            "rows": 172, "people": 134, "events_7d": 4, "events_30d": 16,
            "source": str(HISTORICAL_RUN),
        },
        "historical_folds": {"people": int(len(historical_folds)), "source": str(HISTORICAL_STAGE3)},
        "reused": ["人员三折", "PD 队列口径", "DS-PCD 核心", "Landmark Cox 家族"],
        "not_reused": ["30 天终点", "半年终点", "旧 3 特征主规格"],
    }
    protocol = build_protocol(
        cohort=cohort, real_candidates=real, folds=folds, config=config,
        reference=reference, history_gap=history_gap,
    )
    _write_json(run_dir / "protocol.json", protocol)

    # 代码快照与清单
    snapshot = run_dir / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for relative in CODE_SNAPSHOT_FILES:
        source_path = PROJECT_ROOT / relative
        if source_path.is_file():
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
    manifest = {
        "stage": 11,
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "release": sources.release_manifest(sources.COHORT_FILES),
        "inputs": {
            name: {"path": str(path), "md5": _md5(path), "bytes": path.stat().st_size}
            for name, path in resolved.items()
        },
        "historical_inputs": {
            "stage3_folds": {"path": str(HISTORICAL_STAGE3 / "folds.csv"),
                             "md5": _md5(HISTORICAL_STAGE3 / "folds.csv")},
            "fixed_run": str(HISTORICAL_RUN),
            "stage10": str(HISTORICAL_STAGE10),
        },
        "environment": _environment(),
        "git": _git_state(PROJECT_ROOT.parent),
        "code_snapshot": list(CODE_SNAPSHOT_FILES),
        "config": config.to_dict(),
        "seed": SEED,
        "commands": [
            "python scripts/round2_stage11_protocol.py --run-id " + args.run_id,
            "python scripts/verify_round2_stage11.py --run <run_dir>",
        ],
        "outputs": sorted(
            str(path.relative_to(run_dir)) for path in run_dir.rglob("*") if path.is_file()
        ),
    }
    _write_json(run_dir / "manifest.json", manifest)
    print(f"[stage11] 完成：{run_dir}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
