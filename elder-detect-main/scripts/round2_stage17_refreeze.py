"""阶段 17：修正评分与报警窗口，重拟合并冻结模型。

三项更正的落地：

1. **同日事件**：真实键只保留阶段 16 的 171 个合格键；半合成训练/测试剔除
   事件恰在预测日的序列（测试 117、训练 37）。训练侧锚点、生成器与标准化
   参数按剔หลัง数据重算，未受影响的部分按版本引用并做等价性核验。
2. **30 天报警窗**：RMS 与 DS-PCD 只用独立稳定标定集的监测第 1–30 天
   “至少一次越界”标定同一 10% 目标，冻结阈值后再用于测试；第 31–37 天只
   用于判定 7 天结局。
3. **截断到 t 的消融**：所有主模型与四种消融只接收日期 ≤ t 的输入；“不分
   方向但同样累积”的 C 改由截至 t 的逐日 omnibus q 累积（旧实现读入尾部
   7 天）。

输出：新阈值、重算生成器、半合成序列、四规格评分、36 个配对模型、折外预测
与冻结清单。本阶段不计算任何外层测试指标。
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
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import round2_stage13_train_freeze as freeze  # noqa: E402
import round2_stage12_supplement as supplement  # noqa: E402

from longitudinal.cox_horizon import DEFAULT_HORIZONS, predict_horizon_risk  # noqa: E402
from longitudinal.ds_pcd import normal_score  # noqa: E402
from mobilise_d.round2 import R2_DIRECTIONS, R2_FEATURES  # noqa: E402
from mobilise_d.round2_scoring import (  # noqa: E402
    PersonSeries, calibrate_alarm_thresholds, first_alarm_day, level_and_change,
    r2_ds_pcd_config, score_person,
)
from mobilise_d.round2_synth import (  # noqa: E402
    GeneratorConfig, GeneratorParams, calibrate_lambda0, fit_generator_params,
    person_records, simulate_batch,
)
from mobilise_d.round2_time import (  # noqa: E402
    ALARM_WINDOW_DAYS, KEYS_VERSION, OUTCOME_TAIL_DAYS, SCORING_VERSION,
    THRESHOLD_VERSION,
)

OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise" / "round2"
STAGE11 = OUTPUT_ROOT / "stage11" / "20261115-protocol-v1"
STAGE12 = OUTPUT_ROOT / "stage12" / "20261115-supplement-v1"
STAGE13 = OUTPUT_ROOT / "stage13" / "20261115-freeze-v2"
STAGE14 = OUTPUT_ROOT / "stage14" / "20261115-eval-v1"
STAGE16 = OUTPUT_ROOT / "stage16" / "20261115-keys-v2"
SEED = 20261115
MECHANISMS = ("level_dominant", "sustained_change", "change_no_effect")
NOISE_FAMILIES = ("gaussian_ar1", "heavy_tail_spike")
BUDGET_INIT = {"train": 600, "test": 2000, "calibration": 800, "stress": 1000,
               "detection_per_cell": 300}
BUDGET = dict(BUDGET_INIT)
CONDITIONS = freeze.CONDITIONS
LADDERS = freeze.LADDERS
ABLATIONS = {
    "full": ("T", "C"),
    "no_precision": ("T_none", "C_none"),
    "direction_only": ("T",),
    "no_direction_cumulative": ("D_snapshot", "C_omnibus_truncated"),
}

CODE_SNAPSHOT_FILES = (
    "mobilise_d/round2_time.py",
    "mobilise_d/round2_scoring.py",
    "mobilise_d/round2_synth.py",
    "longitudinal/cox_horizon.py",
    "longitudinal/ds_pcd.py",
    "scripts/round2_stage17_refreeze.py",
    "scripts/verify_round2_stage17.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 17：重拟合并冻结")
    parser.add_argument("--run-id", default="20261115-refreeze-v1")
    parser.add_argument("--out-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--stage11", default=str(STAGE11))
    parser.add_argument("--stage12", default=str(STAGE12))
    parser.add_argument("--stage13", default=str(STAGE13))
    parser.add_argument("--stage14", default=str(STAGE14))
    parser.add_argument("--stage16", default=str(STAGE16))
    parser.add_argument("--regenerate-folds", default="0,1,2",
                        help="需要重新生成的折（默认全部，逐折记录是否与旧值一致）")
    parser.add_argument("--skip-variants", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--limit", type=float, default=1.0,
                        help="仅用于开发冒烟：按比例缩小样本预算")
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_code(run_dir: Path) -> list[str]:
    target = run_dir / "code_snapshot"
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for relative in CODE_SNAPSHOT_FILES:
        source = PROJECT_ROOT / relative
        if not source.is_file():
            continue
        destination = target / Path(relative).name
        shutil.copy2(source, destination)
        written.append(str(destination))
    return written


# ---------------------------------------------------------------- 生成器


def fit_and_calibrate(
    *,
    fold: int,
    donors: list[str],
    person_daily: pd.DataFrame,
    landmarks: pd.DataFrame,
    config: GeneratorConfig,
) -> tuple[GeneratorParams, dict]:
    """与阶段 12 完全相同的拟合规格（同一函数、同一目标率、同一种子）。"""
    params = fit_generator_params(
        person_daily=person_daily,
        landmarks=landmarks,
        donors=donors,
        feature_columns={name: supplement.R2_SOURCE_COLUMN[name] for name in R2_FEATURES},
        config=config,
        provenance={
            "fold": fold,
            "donor_count": len(donors),
            "seed": SEED + 1000 * fold,
            "source": "real PD training-fold daily gait + bout-derived features",
            "correction": "same-day fall landmarks removed from training anchors",
        },
    )
    calibration: dict[str, Any] = {"fold": fold, "cells": {}}
    lambda0: dict[str, float] = {}
    for mechanism in MECHANISMS:
        for noise in NOISE_FAMILIES:
            value, diagnostics = calibrate_lambda0(
                params, mechanism=mechanism, noise_family=noise,
                target_rate=params.event_rate_anchor_7d,
                config=config, seed=SEED + 1000 * fold + 17,
            )
            lambda0[f"{mechanism}|{noise}"] = float(value)
            calibration["cells"][f"{mechanism}|{noise}"] = {
                "lambda0": float(value), **diagnostics,
            }
    lambda0["default"] = float(np.mean(list(lambda0.values())))
    calibration["event_rate_anchor_7d"] = params.event_rate_anchor_7d
    calibration["event_rate_anchor_30d"] = params.event_rate_anchor_30d
    return GeneratorParams(**{**params.__dict__, "lambda0": lambda0}), calibration


# ---------------------------------------------------------------- 评分批次


def score_batch(
    *,
    params: GeneratorParams,
    fold: int,
    mechanism: str,
    noise: str,
    budget: int,
    seed: int,
    config: GeneratorConfig,
    ds_config,
    prefix: str,
    trajectory_override: str | None = None,
    precision_scale: float = 1.0,
    bias_drift_sd: float = 0.0,
    alarm_threshold: float | None = None,
    rms_threshold: float | None = None,
    daily_limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """评分一批虚拟序列。

    返回（逐序列记录、逐日聚合、逐日明细）；逐日聚合只保留
    监测第 1–30 天的阈值输入与报警归属，逐日明细只保留前 ``daily_limit`` 条序列。
    """
    frame, series = simulate_batch(
        params, count=budget, fold=fold, mechanism=mechanism, noise_family=noise,
        seed=seed, config=config, prefix=prefix, keep_series=True,
        trajectory_override=trajectory_override, precision_scale=precision_scale,
        lambda0=params.lambda0.get(f"{mechanism}|{noise}"), bias_drift_sd=bias_drift_sd,
    )
    records: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    monitoring = config.monitoring_days
    for position, person in enumerate(series):
        records_in = person_records(person, R2_FEATURES)
        observed = np.asarray(person["observed"], dtype=float)
        series_obj = PersonSeries(
            person_id=person["series_id"],
            days=tuple(record.date for record in records_in),
            features={name: tuple(observed[:, index]) for index, name in enumerate(R2_FEATURES)},
            feature_se={
                name: tuple(np.asarray(person["se"], dtype=float)[:, index])
                for index, name in enumerate(R2_FEATURES)
            },
            source="synthetic",
            meta={"fold": fold, "mechanism": mechanism, "noise_family": noise,
                  "trajectory": person["trajectory"], "series_id": person["series_id"]},
        )
        snapshot_day = records_in[config.baseline_days + monitoring - 1].date
        result = score_person(
            series_obj, ds_config, baseline_days=config.baseline_days,
            snapshot_day=snapshot_day, alarm_threshold=alarm_threshold,
            rms_threshold=rms_threshold,
            baseline_dates=[record.date for record in records_in[: config.baseline_days]],
        )
        daily = result.pop("daily", [])
        baseline_indices = list(range(config.baseline_days))
        monitoring_indices = list(range(config.baseline_days, config.baseline_days + monitoring))
        covariates = level_and_change(
            series_obj, baseline_indices, monitoring_indices, features=R2_FEATURES
        )
        # 监测第 1–30 天按自然日识别（个别序列会多出基线末日一行）
        monitoring_start_day = records_in[config.baseline_days].date
        tail_end_day = records_in[
            config.baseline_days + monitoring + OUTCOME_TAIL_DAYS - 1
        ].date
        window = [row for row in daily if monitoring_start_day <= row["day"] <= snapshot_day]
        tail = [row for row in daily if snapshot_day < row["day"] <= tail_end_day]
        c_values = [float(row["C"]) for row in window if row.get("C") is not None]
        rms_values = [float(row["rms"]) for row in window if row.get("rms") is not None]
        c_all = [float(row["C"]) for row in daily if row.get("C") is not None]
        rms_all = [float(row["rms"]) for row in daily if row.get("rms") is not None]
        aggregate = {
            "series_id": person["series_id"],
            "fold": fold,
            "mechanism": mechanism,
            "noise_family": noise,
            "trajectory": person["trajectory"],
            "monitoring_days_observed": len(window),
            "tail_days_observed": len(tail),
            "max_C_30d": max(c_values) if c_values else None,
            "rms_max_30d": max(rms_values) if rms_values else None,
            "max_C_37d": max(c_all) if c_all else None,
            "rms_max_37d": max(rms_all) if rms_all else None,
        }
        records.append({
            **{key: person[key] for key in (
                "series_id", "fold", "cohort", "mechanism", "noise_family", "trajectory",
                "source", "age", "age_z", "prior_fall_any", "delta", "h_snapshot",
                "h_max", "event_day", "event_gap_from_snapshot",
                "first_event_offset_from_monitoring", "y_1d", "y_3d", "y_7d",
            )},
            **{key: value for key, value in result.items() if key != "daily"},
            **covariates,
            **{key: value for key, value in aggregate.items()
               if key not in ("series_id", "fold", "mechanism", "noise_family", "trajectory")},
            "batch": prefix,
            "precision_scale": precision_scale,
            "bias_drift_sd": bias_drift_sd,
        })
        aggregates.append(aggregate)
        if daily_limit is not None and position < daily_limit:
            h = np.asarray(person["h_trajectory"], dtype=float)
            baseline_mean = float(h[: config.baseline_days].mean())
            threshold = 0.25 * abs(float(person["delta"]))
            onset = None
            for index in range(config.baseline_days, len(h)):
                if abs(h[index] - baseline_mean) > threshold:
                    onset = int(index - config.baseline_days + 1)
                    break
            day_offsets = {
                day: index + 1 for index, day in enumerate(
                    [record.date for record in records_in[
                        config.baseline_days:
                        config.baseline_days + monitoring + OUTCOME_TAIL_DAYS
                    ]]
                )
            }
            for item in daily:
                offset = day_offsets.get(item["day"])
                if offset is None:
                    continue
                details.append({
                    "series_id": person["series_id"],
                    "fold": fold,
                    "mechanism": mechanism,
                    "noise_family": noise,
                    "trajectory": person["trajectory"],
                    "day": item["day"],
                    "monitoring_offset": int(offset),
                    "within_monitoring": bool(item["day"] <= snapshot_day),
                    "q": item["q"], "C": item["C"], "T": item["T"], "rms": item["rms"],
                    "onset_offset": onset,
                    "ds_alarm": (
                        None if alarm_threshold is None or item["C"] is None
                        else bool(item["C"] >= alarm_threshold)
                    ),
                    "rms_alarm": (
                        None if rms_threshold is None or item["rms"] is None
                        else bool(item["rms"] >= rms_threshold)
                    ),
                })
    return pd.DataFrame(records), pd.DataFrame(aggregates), pd.DataFrame(details)


def truncate_omnibus(
    daily: Sequence[dict], k_ref: float, snapshot_day: str,
) -> tuple[float | None, float | None, float | None]:
    """用截至预测日 t（含当天）的 omnibus q 重放 CUSUM。"""
    q_values = []
    last = None
    for row in daily:
        if str(row.get("day")) > str(snapshot_day):
            continue
        p = row.get("omnibus_p")
        if p is None or not math.isfinite(float(p)):
            continue
        q_values.append(float(normal_score(float(p), -4.0, 6.0)))
        last = row
    if not q_values:
        return None, None, None
    current = 0.0
    for q in q_values:
        current = max(0.0, current + q - k_ref)
    return float(current), float(q_values[-1]), (
        float(last.get("omnibus_D")) if last else None
    )


# ---------------------------------------------------------------- 变体评分


def variant_rows_for_synthetic(
    *, params: GeneratorParams, fold: int, mechanism: str, noise: str,
    budget: int, seed: int, config: GeneratorConfig, full_config, none_config,
    prefix: str, k_ref: float,
) -> list[dict]:
    frame, series = simulate_batch(
        params, count=budget, fold=fold, mechanism=mechanism, noise_family=noise,
        seed=seed, config=config, prefix=prefix, keep_series=True,
        lambda0=params.lambda0.get(f"{mechanism}|{noise}"),
    )
    rows: list[dict] = []
    for person in series:
        records_in = person_records(person, R2_FEATURES)
        observed = np.asarray(person["observed"], dtype=float)
        se = np.asarray(person["se"], dtype=float)
        series_obj = PersonSeries(
            person_id=person["series_id"],
            days=tuple(record.date for record in records_in),
            features={name: tuple(observed[:, index]) for index, name in enumerate(R2_FEATURES)},
            feature_se={name: tuple(se[:, index]) for index, name in enumerate(R2_FEATURES)},
            source="synthetic",
            meta={"fold": fold, "mechanism": mechanism, "noise_family": noise,
                  "trajectory": person["trajectory"], "series_id": person["series_id"]},
        )
        snapshot_day = records_in[config.baseline_days + config.monitoring_days - 1].date
        baseline_dates = [record.date for record in records_in[: config.baseline_days]]
        full = score_person(series_obj, full_config, snapshot_day=snapshot_day,
                            baseline_dates=baseline_dates)
        none = score_person(series_obj, none_config, snapshot_day=snapshot_day,
                            baseline_dates=baseline_dates)
        c_omni, q_omni, d_snap = truncate_omnibus(full.get("daily", []), k_ref, snapshot_day)
        c_omni_full, _, d_snap_full = truncate_omnibus(
            full.get("daily", []), k_ref, "9999-12-31")
        rows.append({
            "branch": "S", "test_set": f"synthetic_{prefix}", "series_id": person["series_id"],
            "fold": fold, "mechanism": mechanism, "noise_family": noise,
            "trajectory": person["trajectory"],
            "status_full": full.get("status"), "status_none": none.get("status"),
            "T": full.get("T"), "C": full.get("C"), "q": full.get("q"),
            "T_none": none.get("T"), "C_none": none.get("C"), "q_none": none.get("q"),
            "D_snapshot": d_snap, "D_snapshot_full": d_snap_full,
            "C_omnibus_truncated": c_omni,
            "C_omnibus_full": c_omni_full, "q_omnibus": q_omni,
            "mask_full": full.get("mask"), "mask_none": none.get("mask"),
            "baseline_days_used": full.get("baseline_days_used"),
        })
    return rows


def variant_rows_for_real(
    *, keys: pd.DataFrame, person_daily: pd.DataFrame, baseline_days: pd.DataFrame,
    full_config, none_config, k_ref: float, real_scores: pd.DataFrame,
) -> list[dict]:
    fold_map = dict(zip(keys["participantid"], keys["fold"].astype(float)))
    scored = real_scores.loc[real_scores["status"] == "scored"].copy()
    scored["participantid"] = scored["participantid"].astype(str)
    keep = set(zip(keys["participantid"], keys["landmark_day"]))
    rows: list[dict] = []
    for row in scored.itertuples(index=False):
        person = str(row.participantid)
        landmark = str(row.landmark_day)
        if (person, landmark) not in keep:
            continue
        baseline_list = sorted(
            baseline_days.loc[
                (baseline_days["participantid"] == person)
                & (baseline_days["landmark_day"] == landmark),
                "baseline_day",
            ].astype(str)
        )
        person_rows = person_daily.loc[person_daily["participantid"] == person]
        window = person_rows.loc[
            person_rows["day"].astype(str).isin(baseline_list)
            | ((person_rows["day"] >= str(row.baseline_end)) & (person_rows["day"] <= landmark))
        ]
        series = supplement.series_from_frame(window, person, features=R2_FEATURES,
                                              source="observed")
        full = score_person(series, full_config, snapshot_day=landmark,
                            baseline_dates=baseline_list)
        none = score_person(series, none_config, snapshot_day=landmark,
                            baseline_dates=baseline_list)
        c_omni, q_omni, d_snap = truncate_omnibus(full.get("daily", []), k_ref, landmark)
        c_omni_full, _, d_snap_full = truncate_omnibus(
            full.get("daily", []), k_ref, "9999-12-31")
        rows.append({
            "branch": "R", "test_set": "real_test", "series_id": f"real-{person}-{landmark}",
            "participantid": person, "landmark_day": landmark,
            "fold": int(fold_map.get(person, -1)),
            "status_full": full.get("status"), "status_none": none.get("status"),
            "T": full.get("T"), "C": full.get("C"), "q": full.get("q"),
            "T_none": none.get("T"), "C_none": none.get("C"), "q_none": none.get("q"),
            "D_snapshot": d_snap, "D_snapshot_full": d_snap_full,
            "C_omnibus_truncated": c_omni,
            "C_omnibus_full": c_omni_full, "q_omnibus": q_omni,
            "mask_full": full.get("mask"), "mask_none": none.get("mask"),
            "baseline_days_used": full.get("baseline_days_used"),
        })
    return rows


# ---------------------------------------------------------------- 主流程（可恢复分阶段）


def _parts(run_dir: Path) -> Path:
    path = run_dir / "_parts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _phase_generators(run_dir: Path, stage11: Path, stage12: Path, stage16: Path) -> dict:
    """生成器（含阶段 12 等价性重放）、独立标定集与 30 天阈值。"""
    keys_all = pd.read_csv(stage16 / "real_keys.csv", dtype={"participantid": str})
    splits = pd.read_csv(stage11 / "splits.csv", dtype={"participantid": str})
    candidates_old = pd.read_csv(stage11 / "real_candidates.csv",
                                 dtype={"participantid": str})
    person_daily = pd.read_csv(stage12 / "person_daily_features.csv",
                               dtype={"participantid": str}, low_memory=False)
    config = GeneratorConfig()
    full_config = r2_ds_pcd_config(precision_mode="session_se")
    same_day = keys_all.loc[keys_all["same_day_fall"].astype(bool),
                            ["participantid", "landmark_day"]]
    same_day_frame = same_day.copy()
    same_day_frame["landmark_day"] = pd.to_datetime(
        same_day_frame["landmark_day"]).dt.strftime("%Y-%m-%d")
    same_day_frame["_drop"] = True
    landmarks = candidates_old.copy()
    landmarks["landmark_day"] = pd.to_datetime(landmarks["landmark_day"]).dt.strftime("%Y-%m-%d")
    landmarks = landmarks.merge(same_day_frame, on=["participantid", "landmark_day"], how="left")
    landmarks = landmarks.loc[landmarks["_drop"].isna()].drop(columns="_drop")
    landmarks_correction = {
        "old_landmarks": int(len(candidates_old)),
        "removed_same_day": int(len(candidates_old) - len(landmarks)),
        "new_landmarks": int(len(landmarks)),
        "population": "stage11 real_candidates（与阶段 12 生成器锚点人口一致）",
    }
    folds = sorted(splits["fold"].unique())
    generator_records: list[dict] = []
    for fold in folds:
        fold = int(fold)
        old_payload = json.loads(
            (stage12 / f"generator_params_fold{fold}.json").read_text(encoding="utf-8"))
        old_params = old_payload["params"]
        donors = sorted(splits.loc[splits["fold"] != fold, "participantid"])
        same_day_persons = {str(value) for value in same_day["participantid"]}
        donor_overlap = sorted(same_day_persons & {str(value) for value in donors})
        params_replay, _ = fit_and_calibrate(
            fold=fold, donors=donors, person_daily=person_daily,
            landmarks=candidates_old.assign(
                landmark_day=pd.to_datetime(candidates_old["landmark_day"]).dt.strftime("%Y-%m-%d")),
            config=config)
        replay_payload = params_replay.to_dict()
        replay_diff = {}
        for name, value in replay_payload.items():
            if name in ("lambda0", "provenance", "feature_names"):
                continue
            old_value = old_params.get(name)
            if old_value is None or isinstance(old_value, str):
                continue
            if isinstance(old_value, (list, tuple)) and old_value and isinstance(old_value[0], str):
                continue
            if isinstance(old_value, (list, tuple)):
                replay_diff[name] = float(np.max(np.abs(
                    np.asarray(old_value, dtype=float) - np.asarray(value, dtype=float))))
            elif isinstance(old_value, (int, float)):
                replay_diff[name] = float(abs(float(old_value) - float(value)))
        lambda0_replay = {
            key: float(abs(float(value) - float(old_params.get("lambda0", {}).get(key, value))))
            for key, value in replay_payload["lambda0"].items()}
        params, calibration = fit_and_calibrate(
            fold=fold, donors=donors, person_daily=person_daily,
            landmarks=landmarks, config=config)
        payload = params.to_dict()
        comparable = ("population_mean", "between_person_sd", "within_person_sd", "ar1_phi",
                      "age_mean", "age_sd", "prior_fall_rate", "event_rate_anchor_7d",
                      "event_rate_anchor_30d", "change_magnitude")
        diff = {}
        for name in comparable:
            old_value = old_params.get(name)
            new_value = payload.get(name)
            if isinstance(old_value, (list, tuple)):
                diff[name] = float(np.max(np.abs(
                    np.asarray(old_value, dtype=float) - np.asarray(new_value, dtype=float))))
            else:
                diff[name] = float(abs(float(old_value) - float(new_value)))
        _write_json(run_dir / f"generator_params_fold{fold}.json", {
            "params": payload,
            "calibration": calibration,
            "generator_version": "round2-synth-v1",
            "config": config.to_dict(),
            "reproduction_of_stage12": {
                "max_abs_parameter_diff": replay_diff,
                "max_abs_lambda0_diff": lambda0_replay,
                "note": "同一代码路径、同一锚点人口（未剔除同日键）重跑阶段 12 参数",
            },
            "correction": {
                "keys_version": KEYS_VERSION,
                "same_day_removed_from_training_anchors": True,
                "donors_include_removed_person": bool(donor_overlap),
                "donor_overlap_same_day_persons": donor_overlap,
                "old_parameter_max_abs_diff": diff,
            },
        })
        generator_records.append({
            "fold": fold,
            "path": f"generator_params_fold{fold}.json",
            "donor_count": len(donors),
            "donors_include_removed_person": bool(donor_overlap),
            "donor_overlap_same_day_persons": donor_overlap,
            "event_rate_anchor_7d": params.event_rate_anchor_7d,
            "old_event_rate_anchor_7d": old_params.get("event_rate_anchor_7d"),
            "lambda0": params.lambda0,
            "old_lambda0": old_payload["params"].get("lambda0"),
            "max_abs_parameter_diff": diff,
            "unchanged": bool(max(diff.values()) < 1e-12),
            "reproduces_stage12_with_old_landmarks": bool(
                max(list(replay_diff.values()) + list(lambda0_replay.values()) or [0.0]) < 1e-12),
            "stage12_reproduction_max_abs_diff": max(
                list(replay_diff.values()) + list(lambda0_replay.values()) or [0.0]),
        })
        print(f"[stage17] fold {fold} 生成器完成（overlap={donor_overlap}）", flush=True)
    _write_json(run_dir / "generator_manifest.json", {
        "generator_version": "round2-synth-v1",
        "config": config.to_dict(),
        "keys_version": KEYS_VERSION,
        "landmarks_correction": landmarks_correction,
        "same_day_persons": sorted({str(value) for value in same_day["participantid"]}),
        "folds": generator_records,
        "note": "参数只由该折训练 donor 估计；同日跌倒的 landmark 已从训练锚点剔除",
    })
    params_by_fold = {
        int(fold): GeneratorParams.from_dict(
            _load(run_dir / f"generator_params_fold{fold}.json")["params"])
        for fold in folds
    }

    print("[stage17] 重建独立稳定标定集（监测第 1–30 天）", flush=True)
    calibration_rows, aggregates, details = [], [], []
    for fold in folds:
        fold = int(fold)
        frame, aggregate, detail = score_batch(
            params=params_by_fold[fold], fold=fold, mechanism="level_dominant",
            noise="gaussian_ar1", budget=BUDGET["calibration"],
            seed=SEED + 1000 * fold + 31, config=config, ds_config=full_config,
            prefix="calib", trajectory_override="stable",
            daily_limit=BUDGET["calibration"])
        frame["batch"] = "calibration_stable"
        aggregate["batch"] = "calibration_stable"
        calibration_rows.append(frame)
        aggregates.append(aggregate)
        if len(detail):
            detail["batch"] = "calibration_stable"
            details.append(detail)
    calibration_frame = pd.concat(calibration_rows, ignore_index=True)
    calibration_aggregate = pd.concat(aggregates, ignore_index=True)
    old_thresholds = json.loads(
        (stage12 / "threshold_calibration.json").read_text(encoding="utf-8"))
    thresholds: dict[str, Any] = {"target_rate": 0.10, "window_days": ALARM_WINDOW_DAYS,
                                  "version": THRESHOLD_VERSION, "per_fold": {}}
    for fold in folds:
        fold = int(fold)
        subset = calibration_aggregate.loc[calibration_aggregate["fold"] == fold]
        rows = subset.rename(columns={"max_C_30d": "max_C", "rms_max_30d": "rms_max"})
        entry = calibrate_alarm_thresholds(rows.to_dict("records"), target_rate=0.10)
        old = old_thresholds["per_fold"][str(fold)]["methods"]
        entry["old_thresholds"] = {"ds_pcd": old["ds_pcd"]["threshold"],
                                   "rms": old["rms"]["threshold"]}
        entry["prefix_window_days"] = ALARM_WINDOW_DAYS
        entry["n_series"] = int(len(subset))
        entry["max_C_37d_p90"] = float(np.nanquantile(subset["max_C_37d"], 0.9))
        entry["rms_max_37d_p90"] = float(np.nanquantile(subset["rms_max_37d"], 0.9))
        thresholds["per_fold"][str(fold)] = entry
    _write_json(run_dir / "thresholds.json", thresholds)
    calibration_frame["threshold_version"] = THRESHOLD_VERSION
    calibration_frame.to_csv(run_dir / "calibration_stable_scores.csv", index=False,
                             encoding="utf-8")
    calibration_aggregate.to_csv(run_dir / "calibration_prefix_summary.csv", index=False,
                                 encoding="utf-8")
    if details:
        pd.concat(details, ignore_index=True).to_csv(
            run_dir / "calibration_prefix_daily.csv", index=False, encoding="utf-8")
    print(f"[stage17] 阈值：{json.dumps(thresholds['per_fold']['0']['methods'], ensure_ascii=False)[:260]}",
          flush=True)
    return {"thresholds": thresholds, "generators": generator_records,
            "landmarks_correction": landmarks_correction}


def _phase_build(run_dir: Path, stage11: Path, stage12: Path, stage16: Path, fold: int) -> dict:
    """一折的训练/测试/压力序列与逐日报警明细（同名文件存在则跳过）。"""
    parts = _parts(run_dir)
    targets = {
        "train": parts / f"synthetic_train_fold{fold}.csv",
        "test": parts / f"synthetic_test_fold{fold}.csv",
        "stress": parts / f"synthetic_stress_fold{fold}.csv",
        "aggregate": parts / f"synthetic_aggregate_fold{fold}.csv",
        "detection": parts / f"synthetic_detection_fold{fold}.csv",
        "report": parts / f"regeneration_fold{fold}.json",
    }
    if all(path.is_file() for path in targets.values()):
        print(f"[stage17] fold {fold} 序列已存在，跳过", flush=True)
        return _load(targets["report"])
    config = GeneratorConfig()
    full_config = r2_ds_pcd_config(precision_mode="session_se")
    thresholds = _load(run_dir / "thresholds.json")
    params = GeneratorParams.from_dict(
        _load(run_dir / f"generator_params_fold{fold}.json")["params"])
    synthetic_excluded = pd.read_csv(stage16 / "synthetic_exclusions.csv")
    stage12_train = pd.read_csv(stage12 / "synthetic_train_scores.csv", low_memory=False)
    methods = thresholds["per_fold"][str(fold)]["methods"]
    ds_threshold = methods["ds_pcd"]["threshold"]
    rms_threshold = methods["rms"]["threshold"]
    train_frames, test_frames = [], []
    aggregates, detection_frames = [], []
    for mechanism in MECHANISMS:
        for noise in NOISE_FAMILIES:
            offset = MECHANISMS.index(mechanism) * 10 + NOISE_FAMILIES.index(noise)
            train_frame, train_agg, _ = score_batch(
                params=params, fold=fold, mechanism=mechanism, noise=noise,
                budget=BUDGET["train"], seed=SEED + 1000 * fold + 100 + offset,
                config=config, ds_config=full_config, prefix="train",
                alarm_threshold=ds_threshold, rms_threshold=rms_threshold)
            train_frame["batch"] = "train"
            train_frames.append(train_frame)
            aggregates.append(train_agg.assign(batch="train",
                                               cells=f"{mechanism}|{noise}"))
            test_frame, test_agg, detection = score_batch(
                params=params, fold=fold, mechanism=mechanism, noise=noise,
                budget=BUDGET["test"], seed=SEED + 1000 * fold + 200 + offset,
                config=config, ds_config=full_config, prefix="test",
                alarm_threshold=ds_threshold, rms_threshold=rms_threshold,
                daily_limit=BUDGET["detection_per_cell"])
            test_frame["batch"] = "test"
            test_frames.append(test_frame)
            aggregates.append(test_agg.assign(batch="test",
                                              cells=f"{mechanism}|{noise}"))
            if len(detection):
                detection_frames.append(detection)
    stress_frame, stress_agg, _ = score_batch(
        params=params, fold=fold, mechanism="level_dominant", noise="gaussian_ar1",
        budget=BUDGET["stress"], seed=SEED + 1000 * fold + 300, config=config,
        ds_config=full_config, prefix="stress", trajectory_override="stable",
        bias_drift_sd=0.5, alarm_threshold=ds_threshold, rms_threshold=rms_threshold)
    stress_frame["batch"] = "stress_measurement_bias_drift"
    aggregates.append(stress_agg.assign(batch="stress", cells="stress|stable"))
    train_all = pd.concat(train_frames, ignore_index=True)
    test_all = pd.concat(test_frames, ignore_index=True)
    old_fold = stage12_train.loc[stage12_train["fold"] == fold]
    entry: dict[str, Any] = {"fold": fold, "generated": True,
                             "train_rows_old": int(len(old_fold)),
                             "train_rows_new": int(len(train_all))}
    if len(old_fold) == len(train_all):
        numeric = ["T", "C", "q", "T_none", "C_none", "q_none", "level_gait_speed_mean",
                   "change_gait_speed_mean", "level_stride_length_mean_cm",
                   "change_stride_length_mean_cm", "level_stride_duration_bout_cv",
                   "change_stride_duration_bout_cv", "y_1d", "y_3d", "y_7d"]
        merged = old_fold.merge(train_all, on="series_id", suffixes=("_old", "_new"))
        entry["train_max_abs_diff"] = {}
        for name in numeric:
            old_column = f"{name}_old" if f"{name}_old" in merged.columns else name
            new_column = f"{name}_new" if f"{name}_new" in merged.columns else name
            if old_column not in merged.columns or new_column not in merged.columns:
                continue
            delta = (pd.to_numeric(merged[old_column], errors="coerce")
                     - pd.to_numeric(merged[new_column], errors="coerce")).abs()
            entry["train_max_abs_diff"][name] = (
                float(delta.max()) if delta.notna().any() else None)
    else:
        entry["train_max_abs_diff"] = None
    excluded_train = set(synthetic_excluded.loc[synthetic_excluded["batch"] == "train",
                                               "series_id"])
    excluded_test = set(synthetic_excluded.loc[synthetic_excluded["batch"] == "test",
                                              "series_id"])
    entry["same_day_removed_train"] = int(train_all["series_id"].isin(excluded_train).sum())
    entry["same_day_removed_test"] = int(test_all["series_id"].isin(excluded_test).sum())
    train_all = train_all.loc[~train_all["series_id"].isin(excluded_train)]
    test_all = test_all.loc[~test_all["series_id"].isin(excluded_test)]
    train_all.to_csv(targets["train"], index=False, encoding="utf-8")
    test_all.to_csv(targets["test"], index=False, encoding="utf-8")
    stress_frame.to_csv(targets["stress"], index=False, encoding="utf-8")
    pd.concat(aggregates, ignore_index=True).to_csv(targets["aggregate"], index=False,
                                                   encoding="utf-8")
    if detection_frames:
        pd.concat(detection_frames, ignore_index=True).to_csv(
            targets["detection"], index=False, encoding="utf-8")
    _write_json(targets["report"], entry)
    print(f"[stage17] fold {fold} 序列完成（train {len(train_all)} / test {len(test_all)}）",
          flush=True)
    return entry


def _phase_detect(run_dir: Path, stage16: Path, fold: int) -> None:
    """重算一折测试序列的逐日报警明细（全部序列，不只前 300 条）。"""
    target = _parts(run_dir) / f"synthetic_detection_fold{fold}.csv"
    marker = _parts(run_dir) / f"detection_full_fold{fold}.done"
    if marker.is_file():
        print(f"[stage17] fold {fold} 报警明细已完整，跳过", flush=True)
        return
    config = GeneratorConfig()
    full_config = r2_ds_pcd_config(precision_mode="session_se")
    thresholds = _load(run_dir / "thresholds.json")
    params = GeneratorParams.from_dict(
        _load(run_dir / f"generator_params_fold{fold}.json")["params"])
    methods = thresholds["per_fold"][str(fold)]["methods"]
    ds_threshold = methods["ds_pcd"]["threshold"]
    rms_threshold = methods["rms"]["threshold"]
    frames: list[pd.DataFrame] = []
    for mechanism in MECHANISMS:
        for noise in NOISE_FAMILIES:
            offset = MECHANISMS.index(mechanism) * 10 + NOISE_FAMILIES.index(noise)
            _, _, detection = score_batch(
                params=params, fold=fold, mechanism=mechanism, noise=noise,
                budget=BUDGET["test"], seed=SEED + 1000 * fold + 200 + offset,
                config=config, ds_config=full_config, prefix="test",
                alarm_threshold=ds_threshold, rms_threshold=rms_threshold,
                daily_limit=BUDGET["test"])
            if len(detection):
                frames.append(detection)
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(target, index=False, encoding="utf-8")
    marker.write_text("done", encoding="utf-8")
    print(f"[stage17] fold {fold} 报警明细完成：{frame['series_id'].nunique()} 序列", flush=True)


def _phase_variants_real(run_dir: Path, stage11: Path, stage12: Path, stage16: Path) -> None:
    target = _parts(run_dir) / "variants_real.csv"
    if target.is_file():
        print("[stage17] 真实变体已存在，跳过", flush=True)
        return
    full_config = r2_ds_pcd_config(precision_mode="session_se")
    none_config = r2_ds_pcd_config(precision_mode="none")
    keys = pd.read_csv(stage16 / "real_keys.csv", dtype={"participantid": str})
    keys = keys.loc[keys["corrected_key"].astype(bool)].copy()
    keys["fold"] = keys["fold"].astype(int)
    person_daily = pd.read_csv(stage12 / "person_daily_features.csv",
                               dtype={"participantid": str}, low_memory=False)
    baseline_days = pd.read_csv(stage11 / "baseline_days.csv", dtype={"participantid": str})
    real_scores_old = pd.read_csv(stage12 / "real_scores.csv", dtype={"participantid": str})
    rows = variant_rows_for_real(
        keys=keys, person_daily=person_daily, baseline_days=baseline_days,
        full_config=full_config, none_config=none_config, k_ref=full_config.k_ref,
        real_scores=real_scores_old)
    pd.DataFrame(rows).to_csv(target, index=False, encoding="utf-8")
    print(f"[stage17] 真实变体完成：{len(rows)} 键", flush=True)


def _phase_variants_fold(run_dir: Path, stage12: Path, fold: int) -> None:
    target = _parts(run_dir) / f"variants_fold{fold}.csv"
    if target.is_file():
        print(f"[stage17] fold {fold} 变体已存在，跳过", flush=True)
        return
    config = GeneratorConfig()
    full_config = r2_ds_pcd_config(precision_mode="session_se")
    none_config = r2_ds_pcd_config(precision_mode="none")
    params = GeneratorParams.from_dict(
        _load(run_dir / f"generator_params_fold{fold}.json")["params"])
    rows: list[dict] = []
    for mechanism in MECHANISMS:
        for noise in NOISE_FAMILIES:
            offset = MECHANISMS.index(mechanism) * 10 + NOISE_FAMILIES.index(noise)
            for prefix, budget, seed_base in (("train", BUDGET["train"], 100),
                                              ("test", BUDGET["test"], 200)):
                rows.extend(variant_rows_for_synthetic(
                    params=params, fold=fold, mechanism=mechanism, noise=noise,
                    budget=budget, seed=SEED + 1000 * fold + seed_base + offset,
                    config=config, full_config=full_config, none_config=none_config,
                    prefix=prefix, k_ref=full_config.k_ref))
    frame = pd.DataFrame(rows)
    excluded = pd.read_csv(STAGE16 / "synthetic_exclusions.csv")
    frame = frame.loc[~frame["series_id"].isin(set(excluded["series_id"]))].copy()
    frame.to_csv(target, index=False, encoding="utf-8")
    print(f"[stage17] fold {fold} 变体完成：{len(frame)} 条", flush=True)


def _phase_models(run_dir: Path, stage11: Path, stage16: Path) -> dict:
    if (run_dir / "oof_predictions.csv").is_file():
        print("[stage17] 模型已存在，跳过", flush=True)
        return {"skipped": True}
    keys = pd.read_csv(stage16 / "real_keys.csv", dtype={"participantid": str})
    keys = keys.loc[keys["corrected_key"].astype(bool)].copy()
    keys["fold"] = keys["fold"].astype(int)
    splits = pd.read_csv(stage11 / "splits.csv", dtype={"participantid": str})
    train_frame = pd.read_csv(run_dir / "synthetic_train_scores.csv", low_memory=False)
    test_frame = pd.read_csv(run_dir / "synthetic_test_scores.csv", low_memory=False)
    real_scores_old = pd.read_csv(STAGE12 / "real_scores.csv", dtype={"participantid": str})
    corrected_candidates = keys.copy()
    for horizon in (1, 3, 7):
        corrected_candidates[f"y_{horizon}d_num"] = corrected_candidates[
            f"y_{horizon}d"].map({True: 1.0, False: 0.0})
    corrected_scores = real_scores_old.loc[real_scores_old["status"].eq("scored")].copy()
    keep = set(zip(keys["participantid"], keys["landmark_day"]))
    corrected_scores = corrected_scores.loc[
        [(str(person), str(day)) in keep for person, day in zip(
            corrected_scores["participantid"], corrected_scores["landmark_day"])]
    ].copy()
    corrected_scores.to_csv(run_dir / "real_scores.csv", index=False, encoding="utf-8")
    folds = sorted(splits["fold"].unique())
    model_records: list[dict] = []
    predictions: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {"folds": {}, "imputation": {}}
    for fold in folds:
        fold = int(fold)
        real_frame, real_diag = freeze.build_real_matrix(
            corrected_scores, corrected_candidates, splits, fold=fold)
        real_train_all = real_frame.loc[real_frame["fold"] != fold].copy()
        real_test_all = real_frame.loc[real_frame["fold"] == fold].copy()
        real_test_common = freeze._complete(real_test_all, LADDERS["M3"])
        real_train = real_train_all
        real_test = real_test_common
        age_mean, age_sd = real_diag["age_mean"], real_diag["age_sd"]
        synthetic_fold = freeze.build_synthetic_matrix(
            train_frame, fold=fold, age_mean=age_mean, age_sd=age_sd)
        synthetic_test_fold = freeze.build_synthetic_matrix(
            test_frame, fold=fold, age_mean=age_mean, age_sd=age_sd)
        synthetic_test_common = freeze._complete(synthetic_test_fold, LADDERS["M3"])
        diagnostics["folds"][str(fold)] = {
            "real": real_diag,
            "synthetic_train_rows": int(len(synthetic_fold)),
            "real_test_rows_all": int(len(real_test_all)),
            "real_test_rows_common": int(len(real_test_common)),
            "synthetic_test_rows_all": int(len(synthetic_test_fold)),
            "synthetic_test_rows_common": int(len(synthetic_test_common)),
            "real_test_positives": {
                f"{horizon}d": int(real_test_common[f"y_{horizon}d"].sum())
                for horizon in (1, 3, 7)},
        }
        for model, features in LADDERS.items():
            for condition in CONDITIONS:
                train_complete = freeze._complete(real_train, features)
                synthetic_train_complete = freeze._complete(synthetic_fold, features)
                frame, weight_info = freeze._weights_for_condition(
                    train_complete, synthetic_train_complete, condition)
                bundle_id = f"round2-fold{fold}-{model}-{condition}"
                fit, metadata = freeze._fit_one(
                    frame, features, fold=fold, model=model, condition=condition,
                    bundle_id=bundle_id)
                model_dir = run_dir / "fold_models" / f"fold{fold}" / condition / model
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / "model.json").write_text(
                    json.dumps(fit.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
                model_records.append({
                    "bundle_id": bundle_id, "fold": fold, "model": model,
                    "condition": condition, "status": fit.status,
                    "converged": bool(fit.converged), "rows": int(len(frame)),
                    "events": int(frame["event"].sum()),
                    "effective_events": fit.effective_events,
                    "total_weight": fit.total_weight,
                    "active_features": list(fit.active_features),
                    "coefficients": dict(zip(fit.feature_names, fit.coefficients)),
                    "baseline_times": list(fit.baseline_times),
                    "baseline_survival_tail": (
                        fit.baseline_survival[-1] if fit.baseline_survival else None),
                    "degenerate_horizons": fit.notes.get("degenerate_horizons_days"),
                    "weights": weight_info,
                    "model_path": str((model_dir / "model.json").relative_to(run_dir)),
                })
                for label, subset in (("real_test", real_test),
                                       ("synthetic_test", synthetic_test_common)):
                    if not len(subset):
                        continue
                    usable = subset.copy()
                    if usable[list(features)].isna().any(axis=1).any():
                        usable = usable.loc[~usable[list(features)].isna().any(axis=1)]
                    if not len(usable):
                        continue
                    matrix = freeze._model_matrix(usable, features)
                    out = pd.DataFrame({
                        "bundle_id": bundle_id, "fold": fold, "model": model,
                        "condition": condition,
                        "branch": "R" if label == "real_test" else "S",
                        "test_set": label,
                        **{key: usable[key].to_numpy() for key in (
                            "participantid", "landmark_day", "series_id", "age",
                            "prior_fall_any", "T", "C", "y_1d", "y_3d", "y_7d",
                            "event_gap", "event_gap_from_snapshot", "mechanism",
                            "noise_family", "trajectory") if key in usable.columns},
                        **{f"p_{horizon}d": predict_horizon_risk(fit, matrix, horizon)
                           for horizon in DEFAULT_HORIZONS},
                        "score_rows": int(len(usable)),
                        "dropped_missing_covariate": int(len(subset) - len(usable)),
                    })
                    predictions.append(out)
        print(f"[stage17] fold {fold} 模型完成", flush=True)
    prediction_frame = pd.concat(predictions, ignore_index=True, sort=False)
    prediction_frame.to_csv(run_dir / "oof_predictions.csv", index=False, encoding="utf-8")
    _write_json(run_dir / "model_registry.json", {
        "models": model_records,
        "ladders": {model: list(features) for model, features in LADDERS.items()},
        "conditions": list(CONDITIONS),
        "primary_comparison": "M3-M2",
        "ridge": freeze.RIDGE,
        "weight_share": freeze.SYNTHETIC_WEIGHT_SHARE,
        "horizons_days": list(DEFAULT_HORIZONS),
    })
    _write_json(run_dir / "training_diagnostics.json", diagnostics)
    status_counts = pd.Series([record["status"] for record in model_records]).value_counts()
    freeze_payload = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "outcome_metrics_computed": False,
        "keys_version": KEYS_VERSION,
        "scoring_version": SCORING_VERSION,
        "threshold_version": THRESHOLD_VERSION,
        "model_slots": int(len(model_records)),
        "status_counts": status_counts.to_dict(),
        "real_test_keys": int(prediction_frame.loc[
            prediction_frame["test_set"] == "real_test",
            ["participantid", "landmark_day"]].drop_duplicates().shape[0]),
        "synthetic_test_keys": int(prediction_frame.loc[
            prediction_frame["test_set"] == "synthetic_test", "series_id"].nunique()),
        "note": "阶段 17 只拟合并保存折外预测，未计算任何外层测试指标",
    }
    _write_json(run_dir / "freeze_manifest.json", freeze_payload)
    return freeze_payload


def _phase_variant_compare(run_dir: Path, stage14: Path) -> dict:
    parts = _parts(run_dir)
    if not (parts / "variants_real.csv").is_file():
        return {}
    real_new = pd.read_csv(parts / "variants_real.csv", dtype={"participantid": str})
    frames = [real_new]
    for fold in (0, 1, 2):
        path = parts / f"variants_fold{fold}.csv"
        if path.is_file():
            frames.append(pd.read_csv(path, low_memory=False))
    variant_frame = pd.concat(frames, ignore_index=True, sort=False)
    variant_frame.to_csv(run_dir / "variant_scores.csv", index=False, encoding="utf-8")
    old_variants = pd.read_csv(stage14 / "variant_scores.csv", low_memory=False)

    def _norm_person(value: Any) -> str:
        text = str(value)
        return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text

    real_old = old_variants.loc[old_variants["branch"] == "R"].copy()
    real_new = variant_frame.loc[variant_frame["branch"] == "R"].copy()
    for frame in (real_old, real_new):
        frame["participantid"] = frame["participantid"].map(_norm_person)
        frame["landmark_day"] = pd.to_datetime(frame["landmark_day"]).dt.strftime("%Y-%m-%d")
    merged = real_old.merge(real_new, on=["participantid", "landmark_day"],
                            suffixes=("_old", "_new"))
    comparison: dict[str, Any] = {
        "real_keys_old": int(len(real_old)), "real_keys_new": int(len(real_new)),
        "real_keys_matched": int(len(merged))}
    for name in ("T", "C", "T_none", "C_none", "D_snapshot"):
        delta = (pd.to_numeric(merged[f"{name}_old"], errors="coerce")
                 - pd.to_numeric(merged[f"{name}_new"], errors="coerce")).abs()
        comparison[f"real_{name}_max_abs_diff"] = (
            float(delta.max()) if delta.notna().any() else None)
    old_c = "C_omnibus_old" if "C_omnibus_old" in merged.columns else "C_omnibus"
    full_c = "C_omnibus_full_new" if "C_omnibus_full_new" in merged.columns else "C_omnibus_full"
    trunc_c = ("C_omnibus_truncated_new" if "C_omnibus_truncated_new" in merged.columns
               else "C_omnibus_truncated")
    if old_c in merged.columns and full_c in merged.columns:
        comparison["real_C_omnibus_full_max_abs_diff"] = float(
            (pd.to_numeric(merged[old_c], errors="coerce")
             - pd.to_numeric(merged[full_c], errors="coerce")).abs().max())
        comparison["real_C_omnibus_truncated_max_abs_diff"] = float(
            (pd.to_numeric(merged[old_c], errors="coerce")
             - pd.to_numeric(merged[trunc_c], errors="coerce")).abs().max())
        comparison["real_C_omnibus_truncated_changed_keys"] = int(
            ((pd.to_numeric(merged[old_c], errors="coerce")
              - pd.to_numeric(merged[trunc_c], errors="coerce")).abs() > 1e-9).sum())

    synthetic_all = variant_frame.loc[variant_frame["branch"] == "S"].copy()
    both = synthetic_all.dropna(subset=["C_omnibus_truncated", "C_omnibus_full"])
    delta_all = both["C_omnibus_full"] - both["C_omnibus_truncated"]
    comparison["synthetic_series"] = int(len(synthetic_all))
    comparison["synthetic_truncation_compared"] = int(len(both))
    comparison["synthetic_truncation_changed_series"] = int((delta_all.abs() > 1e-9).sum())
    comparison["synthetic_truncation_max_abs_diff"] = (
        float(delta_all.abs().max()) if len(both) else None)
    comparison["synthetic_truncation_mean_abs_diff"] = (
        float(delta_all.abs().mean()) if len(both) else None)

    # 代码路径等价性：用阶段 12 参数重算一个测试单元
    old_params = GeneratorParams.from_dict(
        json.loads((STAGE12 / "generator_params_fold0.json")
                   .read_text(encoding="utf-8"))["params"])
    config = GeneratorConfig()
    full_config = r2_ds_pcd_config(precision_mode="session_se")
    none_config = r2_ds_pcd_config(precision_mode="none")
    reproduction = variant_rows_for_synthetic(
        params=old_params, fold=0, mechanism="level_dominant", noise="gaussian_ar1",
        budget=min(BUDGET_INIT["test"], 200), seed=SEED + 200, config=config,
        full_config=full_config, none_config=none_config, prefix="test",
        k_ref=full_config.k_ref)
    reproduction_frame = pd.DataFrame(reproduction)
    old_cell = old_variants.loc[
        (old_variants["branch"] == "S") & (old_variants["test_set"] == "synthetic_test")
        & (old_variants["mechanism"] == "level_dominant")
        & (old_variants["noise_family"] == "gaussian_ar1")].copy()
    merged_rep = old_cell.merge(reproduction_frame, on="series_id",
                                suffixes=("_old", "_new"))
    comparison["reproduction_series_matched"] = int(len(merged_rep))
    for name in ("T", "C", "T_none", "C_none", "D_snapshot"):
        delta = (pd.to_numeric(merged_rep[f"{name}_old"], errors="coerce")
                 - pd.to_numeric(merged_rep[f"{name}_new"], errors="coerce")).abs()
        comparison[f"reproduction_{name}_max_abs_diff"] = (
            float(delta.max()) if delta.notna().any() else None)
    rep_old_c = "C_omnibus_old" if "C_omnibus_old" in merged_rep.columns else "C_omnibus"
    if rep_old_c in merged_rep.columns:
        for label, name in (("full", "C_omnibus_full"), ("truncated", "C_omnibus_truncated")):
            column = f"{name}_new" if f"{name}_new" in merged_rep.columns else name
            if column in merged_rep.columns:
                delta = (pd.to_numeric(merged_rep[rep_old_c], errors="coerce")
                         - pd.to_numeric(merged_rep[column], errors="coerce")).abs()
                comparison[f"reproduction_C_omnibus_{label}_max_abs_diff"] = float(delta.max())
                comparison[f"reproduction_C_omnibus_{label}_changed_series"] = int(
                    (delta > 1e-9).sum())
    rep_old_d = "D_snapshot_old" if "D_snapshot_old" in merged_rep.columns else "D_snapshot"
    if rep_old_d in merged_rep.columns:
        for label, name in (("full", "D_snapshot_full"), ("truncated", "D_snapshot")):
            column = f"{name}_new" if f"{name}_new" in merged_rep.columns else name
            if column in merged_rep.columns:
                comparison[f"reproduction_D_snapshot_{label}_max_abs_diff"] = float(
                    (pd.to_numeric(merged_rep[rep_old_d], errors="coerce")
                     - pd.to_numeric(merged_rep[column], errors="coerce")).abs().max())
    comparison["reproduction_note"] = (
        "用阶段 12 保存的生成器参数在同一单元格重算 200 条测试序列；"
        "T/C/none/D 以及未截断的 omnibus D、C 应与旧值完全一致")
    _write_json(run_dir / "variant_comparison.json", comparison)
    return comparison


def _phase_finalize(run_dir: Path, stage11: Path, stage12: Path, stage16: Path) -> int:
    parts = _parts(run_dir)
    for name, target in (
        ("train", "synthetic_train_scores.csv"),
        ("test", "synthetic_test_scores.csv"),
        ("stress", "synthetic_stress_scores.csv"),
        ("aggregate", "synthetic_daily_summary.csv"),
        ("detection", "synthetic_detection_daily.csv"),
    ):
        frames = []
        for fold in (0, 1, 2):
            path = parts / f"synthetic_{name}_fold{fold}.csv"
            if path.is_file():
                frames.append(pd.read_csv(path, low_memory=False))
        if frames:
            pd.concat(frames, ignore_index=True, sort=False).to_csv(
                run_dir / target, index=False, encoding="utf-8")
    reports = []
    same_day_removed = {"train": {}, "test": {}}
    for fold in (0, 1, 2):
        path = parts / f"regeneration_fold{fold}.json"
        if path.is_file():
            entry = _load(path)
            reports.append(entry)
            same_day_removed["train"][str(fold)] = entry.get("same_day_removed_train")
            same_day_removed["test"][str(fold)] = entry.get("same_day_removed_test")
    _write_json(run_dir / "regeneration_check.json", {
        "folds": reports, "same_day_removed": same_day_removed,
        "note": "生成器与序列逐折保存；fold 0 的参数与阶段 12 相同，fold 1/2 按剔后锚点重算"})
    train = pd.read_csv(run_dir / "synthetic_train_scores.csv", low_memory=False)
    test = pd.read_csv(run_dir / "synthetic_test_scores.csv", low_memory=False)
    stress = pd.read_csv(run_dir / "synthetic_stress_scores.csv", low_memory=False)
    detection = run_dir / "synthetic_detection_daily.csv"
    calibration = pd.read_csv(run_dir / "calibration_stable_scores.csv", low_memory=False)
    keys = pd.read_csv(stage16 / "real_keys.csv", dtype={"participantid": str})
    _write_json(run_dir / "counts.json", {
        "real_keys": int(keys["corrected_key"].astype(bool).sum()),
        "real_people": int(keys.loc[keys["corrected_key"].astype(bool),
                                    "participantid"].nunique()),
        "synthetic_train_rows": int(len(train)),
        "synthetic_test_rows": int(len(test)),
        "synthetic_stress_rows": int(len(stress)),
        "synthetic_train_same_day_removed": int(sum(
            value or 0 for value in same_day_removed["train"].values())),
        "synthetic_test_same_day_removed": int(sum(
            value or 0 for value in same_day_removed["test"].values())),
        "synthetic_train_scored": int((train["status"] == "scored").sum()),
        "synthetic_test_scored": int((test["status"] == "scored").sum()),
        "calibration_series": int(len(calibration)),
        "detection_daily_rows": int(len(pd.read_csv(detection, low_memory=False)))
        if detection.is_file() else 0,
        "threshold_version": THRESHOLD_VERSION,
        "scoring_version": SCORING_VERSION,
        "keys_version": KEYS_VERSION,
    })
    code_files = _snapshot_code(run_dir)
    definitions = json.loads((stage16 / "window_definitions.json").read_text(encoding="utf-8"))
    freeze_payload = _load(run_dir / "freeze_manifest.json") if (
        run_dir / "freeze_manifest.json").is_file() else {}
    manifest = {
        "stage": 17,
        "run_id": run_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "upstream": {
            str(stage16): {name: _md5(stage16 / name) for name in (
                "real_keys.csv", "window_definitions.json", "synthetic_exclusions.csv")},
            str(stage11): {name: _md5(stage11 / name) for name in (
                "splits.csv", "real_candidates.csv", "baseline_days.csv", "protocol.json")},
            str(stage12): {name: _md5(stage12 / name) for name in (
                "person_daily_features.csv", "real_scores.csv", "threshold_calibration.json",
                "generator_params_fold0.json", "generator_params_fold1.json",
                "generator_params_fold2.json")},
        },
        "keys_version": KEYS_VERSION,
        "scoring_version": SCORING_VERSION,
        "threshold_version": THRESHOLD_VERSION,
        "window_definitions": definitions,
        "environment": {"python_version": __import__("platform").python_version(),
                        "executable": sys.executable},
        "code_snapshot": code_files,
        "freeze": freeze_payload,
    }
    _write_json(run_dir / "manifest.json", manifest)
    return 0


def run(args: argparse.Namespace) -> int:
    run_dir = Path(args.out_root) / f"stage17/{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stage11, stage12 = Path(args.stage11), Path(args.stage12)
    stage16 = Path(args.stage16)
    if args.limit != 1.0:
        for key in BUDGET:
            BUDGET[key] = max(8, int(round(BUDGET[key] * float(args.limit))))
        print(f"[stage17] 冒烟预算：{BUDGET}", flush=True)
    phases = [args.phase] if args.phase != "all" else [
        "generators", "build", "detect", "variants", "models", "finalize"]
    folds = [int(value) for value in str(args.folds).split(",")] if args.folds else [0, 1, 2]
    for phase in phases:
        if phase == "generators":
            if (run_dir / "thresholds.json").is_file():
                print("[stage17] 生成器与阈值已存在，跳过", flush=True)
            else:
                _phase_generators(run_dir, stage11, stage12, stage16)
        elif phase == "build":
            for fold in folds:
                _phase_build(run_dir, stage11, stage12, stage16, fold)
        elif phase == "detect":
            for fold in folds:
                _phase_detect(run_dir, stage16, fold)
        elif phase == "variants":
            _phase_variants_real(run_dir, stage11, stage12, stage16)
            for fold in folds:
                _phase_variants_fold(run_dir, stage12, fold)
            _phase_variant_compare(run_dir, Path(args.stage14))
        elif phase == "models":
            _phase_models(run_dir, stage11, stage16)
        elif phase == "finalize":
            _phase_finalize(run_dir, stage11, stage12, stage16)
        else:
            raise SystemExit(f"未知阶段：{phase}")
    print(f"[stage17] 阶段 {args.phase} 完成：{run_dir}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.add_argument("--phase", default="all",
                        choices=("all", "generators", "build", "detect", "variants",
                                 "models", "finalize"))
    parser.add_argument("--folds", default="0,1,2")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
