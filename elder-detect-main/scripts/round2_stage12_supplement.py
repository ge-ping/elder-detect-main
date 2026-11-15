"""阶段 12：形成可追溯的补充数据与 DS-PCD 评分。

步骤：

1. 逐段派生真实 PD 的会话标准误与段间跨步周期变异，并保留因果回退来源；
2. 在每个训练折上用**训练 donor** 拟合生成器参数与事件率锚点；
3. 生成半合成训练/标定/测试序列（两种噪声族、三种机制、压力测试），只用
   共享的 RMS/DS-PCD 评分入口打分；
4. 在独立标定序列上给 RMS 与 DS-PCD 同等机会标定 30 天误报 10% 阈值；
5. 保存生成器清单、来源链、特征来源、分数与分布诊断。

不拟合最终风险模型、不查看真实测试预测性能。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import md5
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.ds_pcd import config_fingerprint  # noqa: E402
from mobilise_d.round2 import (  # noqa: E402
    R2_FEATURES, R2_SOURCE_COLUMN, Round2Config,
)
from mobilise_d.round2_features import (  # noqa: E402
    BoutDerivedConfig, daily_bout_features, scan_pd_bouts, validate_against_daily,
)
from mobilise_d.round2_scoring import (  # noqa: E402
    PersonSeries, calibrate_alarm_thresholds, first_alarm_day, level_and_change,
    r2_ds_pcd_config, score_person, scoring_fingerprint, series_from_frame,
)
from mobilise_d.round2_synth import (  # noqa: E402
    GeneratorConfig, GeneratorParams, MECHANISMS, NOISE_FAMILIES, SCENARIO_TRAJECTORY,
    TRAJECTORY_MIX, calibrate_lambda0, fit_generator_params, simulate_batch,
)

OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise" / "round2"
STAGE11 = OUTPUT_ROOT / "stage11" / "20261115-protocol-v1"
SEED = 20261115

CODE_SNAPSHOT_FILES = (
    "mobilise_d/round2.py",
    "mobilise_d/round2_features.py",
    "mobilise_d/round2_scoring.py",
    "mobilise_d/round2_synth.py",
    "mobilise_d/precision.py",
    "mobilise_d/observations.py",
    "longitudinal/ds_pcd.py",
    "longitudinal/deviation.py",
    "longitudinal/baseline.py",
    "longitudinal/uncertainty.py",
    "scripts/round2_stage12_supplement.py",
    "scripts/verify_round2_stage12.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 12：补充数据与评分")
    parser.add_argument("--run-id", default="20261115-supplement-v1")
    parser.add_argument("--out-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--stage11", default=str(STAGE11))
    parser.add_argument("--limit", type=float, default=1.0,
                        help="仅用于开发冒烟：按比例缩小样本预算")
    parser.add_argument("--skip-bout-scan", action="store_true",
                        help="复用运行目录中已有的逐段派生表")
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


def _person_frame(daily: pd.DataFrame, person: str) -> pd.DataFrame:
    frame = daily.loc[daily["participantid"] == str(person)]
    return frame.sort_values("day", kind="stable")


def score_real_candidates(
    real: pd.DataFrame,
    daily: pd.DataFrame,
    baseline_days: pd.DataFrame,
    *,
    config,
    features: Sequence[str] = R2_FEATURES,
) -> tuple[pd.DataFrame, dict]:
    """对每个真实候选预测日评分；预测日必须本身可评分。"""
    baseline_lookup: dict[tuple[str, str], list[str]] = {}
    for (person, landmark), group in baseline_days.groupby(
        ["participantid", "landmark_day"], sort=False
    ):
        baseline_lookup[(str(person), str(landmark))] = sorted(
            pd.to_datetime(group["baseline_day"]).dt.strftime("%Y-%m-%d")
        )
    rows: list[dict[str, Any]] = []
    statuses: dict[str, int] = {}
    for row in real.loc[real["r_branch_eligible"]].itertuples(index=False):
        person = str(row.participantid)
        landmark = pd.Timestamp(row.landmark_day).strftime("%Y-%m-%d")
        baseline_list = baseline_lookup.get((person, landmark), [])
        person_daily = _person_frame(daily, person)
        monitoring_start = pd.Timestamp(row.monitoring_start_day).strftime("%Y-%m-%d")
        window = person_daily.loc[
            (
                person_daily["day"].astype(str).isin(baseline_list)
                | (
                    (person_daily["day"] >= monitoring_start)
                    & (person_daily["day"] <= landmark)
                )
            )
        ].copy()
        if len(window) < 15:
            statuses["insufficient_window_rows"] = statuses.get("insufficient_window_rows", 0) + 1
            rows.append({
                "participantid": person, "landmark_day": landmark, "status": "insufficient_window_rows",
                "window_rows": int(len(window)),
            })
            continue
        series = series_from_frame(
            window, person, features=features, source="observed",
            meta={"cohort": row.cohort, "visit_number": row.visit_number},
        )
        result = score_person(
            series, config, baseline_days=14, snapshot_day=landmark,
            baseline_dates=baseline_list,
        )
        baseline_indices = [index for index, day in enumerate(series.days) if day in set(baseline_list)]
        monitoring_indices = [
            index for index, day in enumerate(series.days)
            if day >= monitoring_start and day <= landmark
        ]
        covariates = level_and_change(
            series, baseline_indices, monitoring_indices, features=features
        )
        statuses[result.get("status", "unknown")] = statuses.get(result.get("status", "unknown"), 0) + 1
        rows.append({
            "participantid": person,
            "cohort": row.cohort,
            "visit_number": row.visit_number,
            "landmark_day": landmark,
            "age": row.age,
            "prior_fall_any": row.prior_fall_any,
            "prior_fall_source": row.prior_fall_source,
            "y_1d": row.y_1d_num,
            "y_3d": row.y_3d_num,
            "y_7d": row.y_7d_num,
            "n_baseline_days": len(baseline_list),
            "n_monitoring_days": len(monitoring_indices),
            **{key: value for key, value in result.items() if key != "daily"},
            **covariates,
        })
    frame = pd.DataFrame(rows)
    return frame, {"status_counts": statuses, "rows": int(len(frame))}


def _fit_and_calibrate(
    *,
    fold: int,
    donors: list[str],
    person_daily: pd.DataFrame,
    landmarks: pd.DataFrame,
    config: GeneratorConfig,
    budgets: dict[str, int],
) -> tuple[GeneratorParams, dict]:
    params = fit_generator_params(
        person_daily=person_daily,
        landmarks=landmarks,
        donors=donors,
        feature_columns={name: R2_SOURCE_COLUMN[name] for name in R2_FEATURES},
        config=config,
        provenance={
            "fold": fold,
            "donor_count": len(donors),
            "seed": SEED + 1000 * fold,
            "source": "real PD training-fold daily gait + bout-derived features",
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


def _score_batch(
    _unused: pd.DataFrame | None = None,
    *,
    params: GeneratorParams,
    fold: int,
    mechanism: str,
    noise: str,
    budget: int,
    seed: int,
    config: GeneratorConfig,
    ds_config,
    keep_daily: bool,
    prefix: str,
    trajectory_override: str | None = None,
    precision_scale: float = 1.0,
    bias_drift_sd: float = 0.0,
    alarm_threshold: float | None = None,
    rms_threshold: float | None = None,
    detection_limit: int | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frame, series = simulate_batch(
        params, count=budget, fold=fold, mechanism=mechanism, noise_family=noise,
        seed=seed, config=config, prefix=prefix, keep_series=True,
        trajectory_override=trajectory_override, precision_scale=precision_scale,
        lambda0=params.lambda0.get(f"{mechanism}|{noise}"), bias_drift_sd=bias_drift_sd,
    )
    out: list[dict[str, Any]] = []
    detection: list[dict[str, Any]] = []
    for position, person in enumerate(series):
        from mobilise_d.round2_synth import person_records

        records = person_records(person, R2_FEATURES)
        observed = np.asarray(person["observed"], dtype=float)
        series_obj = PersonSeries(
            person_id=person["series_id"],
            days=tuple(record.date for record in records),
            features={name: tuple(observed[:, index]) for index, name in enumerate(R2_FEATURES)},
            feature_se={
                name: tuple(np.asarray(person["se"], dtype=float)[:, index])
                for index, name in enumerate(R2_FEATURES)
            },
            source="synthetic",
            meta={
                "fold": fold, "mechanism": mechanism, "noise_family": noise,
                "trajectory": person["trajectory"], "series_id": person["series_id"],
            },
        )
        snapshot_day = records[config.baseline_days + config.monitoring_days - 1].date
        result = score_person(
            series_obj, ds_config, baseline_days=config.baseline_days,
            snapshot_day=snapshot_day, alarm_threshold=alarm_threshold,
            rms_threshold=rms_threshold,
            baseline_dates=[record.date for record in records[: config.baseline_days]],
        )
        daily = result.pop("daily", [])
        baseline_indices = list(range(config.baseline_days))
        monitoring_indices = list(
            range(config.baseline_days, config.baseline_days + config.monitoring_days)
        )
        covariates = level_and_change(
            series_obj, baseline_indices, monitoring_indices, features=R2_FEATURES
        )
        first_ds_alarm = first_alarm_day(daily, method="ds_pcd", threshold=alarm_threshold)
        first_rms = first_alarm_day(daily, method="rms", threshold=rms_threshold)
        record = {
            **{key: person[key] for key in (
                "series_id", "fold", "cohort", "mechanism", "noise_family", "trajectory",
                "source", "age", "age_z", "prior_fall_any", "delta", "h_snapshot",
                "h_max", "event_day", "event_gap_from_snapshot",
                "first_event_offset_from_monitoring", "y_1d", "y_3d", "y_7d",
            )},
            **{key: value for key, value in result.items() if key != "daily"},
            **covariates,
            "ds_pcd_first_alarm_day": first_ds_alarm,
            "rms_first_alarm_day_scored": first_rms,
            "batch": prefix,
            "precision_scale": precision_scale,
            "bias_drift_sd": bias_drift_sd,
        }
        out.append(record)
        if keep_daily and (detection_limit is None or position < detection_limit):
            h = np.asarray(person["h_trajectory"], dtype=float)
            baseline_mean = float(h[: config.baseline_days].mean())
            threshold = 0.25 * abs(float(person["delta"]))
            onset = None
            for index in range(config.baseline_days, len(h)):
                if abs(h[index] - baseline_mean) > threshold:
                    onset = int(index - config.baseline_days + 1)
                    break
            for item in daily:
                detection.append({
                    "series_id": person["series_id"],
                    "fold": fold,
                    "mechanism": mechanism,
                    "noise_family": noise,
                    "trajectory": person["trajectory"],
                    "day": item["day"],
                    "q": item["q"],
                    "C": item["C"],
                    "T": item["T"],
                    "rms": item["rms"],
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
    return pd.DataFrame(out), detection


def _distribution_diagnostics(
    real_daily: pd.DataFrame,
    synthetic_daily: pd.DataFrame,
    *,
    features: Sequence[str] = R2_FEATURES,
) -> dict:
    """真实与虚拟输入的分布、缺测与精度对照。"""
    summary: dict[str, Any] = {}
    for name in features:
        real = pd.to_numeric(real_daily.get(name), errors="coerce")
        sim = pd.to_numeric(synthetic_daily.get(name), errors="coerce")
        summary[name] = {
            "real": {
                "n": int(real.notna().sum()),
                "missing_share": float(real.isna().mean()),
                "mean": float(real.mean()),
                "sd": float(real.std(ddof=1)),
                "p10": float(real.quantile(0.1)),
                "p50": float(real.median()),
                "p90": float(real.quantile(0.9)),
            },
            "synthetic": {
                "n": int(sim.notna().sum()),
                "missing_share": float(sim.isna().mean()),
                "mean": float(sim.mean()),
                "sd": float(sim.std(ddof=1)),
                "p10": float(sim.quantile(0.1)),
                "p50": float(sim.median()),
                "p90": float(sim.quantile(0.9)),
            },
        }
    return summary


def run(args: argparse.Namespace) -> int:
    run_dir = Path(args.out_root) / f"stage12/{args.run_id}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"运行目录已存在且非空，拒绝覆盖：{run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    stage11 = Path(args.stage11)
    protocol = json.loads((stage11 / "protocol.json").read_text(encoding="utf-8"))
    real = pd.read_csv(stage11 / "real_candidates.csv", dtype={"participantid": str})
    baseline_days = pd.read_csv(stage11 / "baseline_days.csv", dtype={"participantid": str})
    splits = pd.read_csv(stage11 / "splits.csv", dtype={"participantid": str})
    landmarks = pd.read_csv(stage11 / "landmarks.csv", dtype={"participantid": str})
    # 日表由同一装配函数重建（阶段 11 只封存派生表，不重复保存 28k 行日表）
    from mobilise_d.daily import load_daily
    from mobilise_d.sources import BOUTS, COHORT_FILES, default_cache_dir, ensure_files

    resolved = ensure_files(default_cache_dir(), COHORT_FILES, offline=True)
    bouts_path = default_cache_dir() / BOUTS.name
    daily_frame = load_daily(resolved["Aggregated DMO Data.zip"]).frame
    daily_frame["participantid"] = daily_frame["participantid"].astype(str)
    daily_frame["visit_number"] = daily_frame["visit_number"].astype(str).str.lower()
    daily_frame["day"] = pd.to_datetime(daily_frame["day"]).dt.normalize()
    daily_frame.to_csv(run_dir / "real_daily_gait.csv", index=False, encoding="utf-8")

    # 1. 逐段派生特征与精度（只对 PD）
    pd_people = sorted(
        pd.read_csv(stage11 / "participants.csv", dtype={"participantid": str})
        .loc[lambda frame: frame["cohort"] == "pd", "participantid"].unique()
    )
    if args.skip_bout_scan and (run_dir / "bout_features_pd.csv").exists():
        bout_daily = pd.read_csv(run_dir / "bout_features_pd.csv", dtype={"participantid": str})
        bout_daily["day"] = pd.to_datetime(bout_daily["day"]).dt.strftime("%Y-%m-%d")
        scan_notes = {"reused": True}
    else:
        print(f"[stage12] 扫描 PD 步行段（{len(pd_people)} 人）", flush=True)
        bout_daily, scan_notes = scan_pd_bouts(
            pd_people, config=BoutDerivedConfig()
        )
        bout_daily["day"] = pd.to_datetime(bout_daily["day"]).dt.strftime("%Y-%m-%d")
        bout_daily.to_csv(run_dir / "bout_features_pd.csv", index=False, encoding="utf-8")
    _write_json(run_dir / "bout_scan_notes.json", scan_notes)

    daily_frame["day"] = daily_frame["day"].dt.strftime("%Y-%m-%d")
    merged = daily_frame.merge(
        bout_daily,
        on=["participantid", "visit_number", "day"], how="left",
        suffixes=("", "_bout"), validate="one_to_one",
    )
    merged["gait_speed_mean"] = merged["gait_speed_bout_mean"]
    merged["stride_length_mean_cm"] = merged["stride_length_bout_mean_cm"]
    merged["stride_duration_bout_cv"] = merged["stride_duration_bout_cv"]
    for name in R2_FEATURES:
        merged[name] = pd.to_numeric(merged[name], errors="coerce")
    merged = merged.sort_values(["participantid", "visit_number", "day"], kind="stable")
    merged.to_csv(run_dir / "person_daily_features.csv", index=False, encoding="utf-8")
    unit_check = validate_against_daily(daily_frame.assign(day=daily_frame["day"]), bout_daily)
    _write_json(run_dir / "bout_vs_daily_check.json", unit_check)

    feature_provenance = {
        "features": {
            "gait_speed_mean": {
                "source": "Walking Bout DMO: mean(averagestridespeed | duration>30s)",
                "unit": "m/s",
                "precision": "bout bootstrap / analytic iid + causal pooled fallback",
                "verified_against": "ws_30_avg_d",
            },
            "stride_length_mean_cm": {
                "source": "Walking Bout DMO: 100*mean(averagestridelength | duration>30s)",
                "unit": "cm",
                "precision": "bout bootstrap / analytic iid + causal pooled fallback",
                "verified_against": "strlen_30_avg_d",
            },
            "stride_duration_bout_cv": {
                "source": "Walking Bout DMO: sd(averagestrideduration)/mean over day's bouts",
                "unit": "1",
                "precision": "bout bootstrap",
                "note": "段间变异，不冒称逐步变异；日表 strdur_all_var_d 已核验不可用于此口径",
            },
        },
        "precision_modes": ["session_se", "none"],
        "fallback_counts": {
            "gait_speed_mean": bout_daily["se_source_gait_speed_mean"].value_counts().to_dict()
            if "se_source_gait_speed_mean" in bout_daily.columns else {},
            "stride_length_mean_cm": bout_daily["se_source_stride_length_mean_cm"].value_counts().to_dict()
            if "se_source_stride_length_mean_cm" in bout_daily.columns else {},
            "stride_duration_bout_cv": bout_daily["method_stride_duration_bout_cv"].value_counts().to_dict()
            if "method_stride_duration_bout_cv" in bout_daily.columns else {},
        },
        "imputation": {
            "rule": "训练折内同人相邻观测日线性插值，仅填补 <7 天空档的零星缺测",
            "applies_to": "training folds only",
        },
    }
    _write_json(run_dir / "feature_provenance.json", feature_provenance)

    # 2. 真实候选评分
    ds_config = r2_ds_pcd_config(precision_mode="session_se")
    print("[stage12] 真实候选评分", flush=True)
    real_scores, real_status = score_real_candidates(
        real, merged, baseline_days, config=ds_config
    )
    real_scores.to_csv(run_dir / "real_scores.csv", index=False, encoding="utf-8")
    _write_json(run_dir / "real_scoring_status.json", real_status)

    # 3. 每折生成器
    config = GeneratorConfig()
    limit = float(args.limit)
    budget = {
        "train": max(20, int(round(600 * limit))),
        "test": max(20, int(round(2000 * limit))),
        "calibration": max(50, int(round(800 * limit))),
        "stress": max(20, int(round(1000 * limit))),
        "detection_per_cell": max(20, int(round(300 * limit))),
    }
    folds = sorted(splits["fold"].unique())
    generator_records: list[dict] = []
    thresholds_by_fold: dict[int, dict] = {}
    calibration_rows: list[dict] = []
    all_train: list[pd.DataFrame] = []
    all_test: list[pd.DataFrame] = []
    all_stress: list[pd.DataFrame] = []
    all_detection: list[pd.DataFrame] = []
    synthetic_daily_sample: list[pd.DataFrame] = []
    for fold in folds:
        donors = sorted(splits.loc[splits["fold"] != fold, "participantid"])
        print(f"[stage12] fold {fold}：{len(donors)} 名训练 donor", flush=True)
        params, calibration = _fit_and_calibrate(
            fold=int(fold), donors=donors, person_daily=merged, landmarks=real,
            config=config, budgets=budget,
        )
        params_path = run_dir / f"generator_params_fold{fold}.json"
        _write_json(params_path, {
            "params": params.to_dict(),
            "calibration": calibration,
            "generator_version": "round2-synth-v1",
            "config": config.to_dict(),
        })
        generator_records.append({
            "fold": int(fold),
            "path": params_path.name,
            "donor_count": len(donors),
            "donor_ids": donors[:10] + ["..."],
            "lambda0": params.lambda0,
            "event_rate_anchor_7d": params.event_rate_anchor_7d,
            "event_rate_anchor_30d": params.event_rate_anchor_30d,
            "ar1_phi": params.ar1_phi,
            "within_person_sd": list(params.within_person_sd),
            "between_person_sd": list(params.between_person_sd),
            "change_magnitude": list(params.change_magnitude),
        })
        # 标定：稳定零假设序列
        calibration_frame, _ = _score_batch(
            params=params, fold=int(fold), mechanism="level_dominant",
            noise="gaussian_ar1", budget=budget["calibration"],
            seed=SEED + 1000 * int(fold) + 31, config=config, ds_config=ds_config,
            keep_daily=False, prefix="calib", trajectory_override="stable",
        )
        calibration_frame["batch"] = "calibration_stable"
        calibration_rows.append(calibration_frame)
        thresholds = calibrate_alarm_thresholds(
            calibration_frame.to_dict("records"), target_rate=0.10
        )
        thresholds_by_fold[int(fold)] = thresholds
        ds_threshold = thresholds["methods"]["ds_pcd"]["threshold"]
        rms_threshold = thresholds["methods"]["rms"]["threshold"]
        # 训练与测试序列（三种机制 × 两种噪声族）
        for mechanism in MECHANISMS:
            for noise in NOISE_FAMILIES:
                offset = (MECHANISMS.index(mechanism) * 10 + NOISE_FAMILIES.index(noise))
                train_frame, _ = _score_batch(
                    params=params, fold=int(fold), mechanism=mechanism,
                    noise=noise, budget=budget["train"],
                    seed=SEED + 1000 * int(fold) + 100 + offset, config=config,
                    ds_config=ds_config, keep_daily=False, prefix="train",
                    alarm_threshold=ds_threshold, rms_threshold=rms_threshold,
                )
                train_frame["batch"] = "train"
                all_train.append(train_frame)
                test_frame, detection = _score_batch(
                    params=params, fold=int(fold), mechanism=mechanism,
                    noise=noise, budget=budget["test"],
                    seed=SEED + 1000 * int(fold) + 200 + offset, config=config,
                    ds_config=ds_config, keep_daily=True, prefix="test",
                    alarm_threshold=ds_threshold, rms_threshold=rms_threshold,
                )
                test_frame["batch"] = "test"
                all_test.append(test_frame)
                if detection:
                    all_detection.append(pd.DataFrame(detection))
                if fold == folds[0] and noise == NOISE_FAMILIES[0]:
                    synthetic_daily_sample.append(_synthetic_daily_frame(params, config, mechanism, noise))
        # 压力测试：测量偏差漂移
        stress_frame, _ = _score_batch(
            params=params, fold=int(fold), mechanism="level_dominant",
            noise="gaussian_ar1", budget=budget["stress"],
            seed=SEED + 1000 * int(fold) + 300, config=config, ds_config=ds_config,
            keep_daily=False, prefix="stress", trajectory_override="stable",
            precision_scale=1.0, bias_drift_sd=0.5,
            alarm_threshold=ds_threshold, rms_threshold=rms_threshold,
        )
        stress_frame["batch"] = "stress_measurement_bias_drift"
        all_stress.append(stress_frame)

    train_frame = pd.concat(all_train, ignore_index=True)
    test_frame = pd.concat(all_test, ignore_index=True)
    stress_frame = pd.concat(all_stress, ignore_index=True)
    calibration_frame = pd.concat(calibration_rows, ignore_index=True)
    train_frame.to_csv(run_dir / "synthetic_train_scores.csv", index=False, encoding="utf-8")
    test_frame.to_csv(run_dir / "synthetic_test_scores.csv", index=False, encoding="utf-8")
    stress_frame.to_csv(run_dir / "synthetic_stress_scores.csv", index=False, encoding="utf-8")
    calibration_frame.to_csv(run_dir / "calibration_stable_scores.csv", index=False, encoding="utf-8")
    if all_detection:
        pd.concat(all_detection, ignore_index=True).to_csv(
            run_dir / "synthetic_detection_daily.csv", index=False, encoding="utf-8"
        )
    if synthetic_daily_sample:
        pd.concat(synthetic_daily_sample, ignore_index=True).to_csv(
            run_dir / "synthetic_daily_sample.csv", index=False, encoding="utf-8"
        )
    _write_json(run_dir / "generator_manifest.json", {
        "generator_version": "round2-synth-v1",
        "config": config.to_dict(),
        "folds": generator_records,
        "note": "参数只由该折训练 donor 估计；测试折 donor 不参与",
    })
    _write_json(run_dir / "threshold_calibration.json", {
        "target_rate": 0.10,
        "window_days": 30,
        "per_fold": {str(fold): value for fold, value in thresholds_by_fold.items()},
        "note": "阈值属于对应生成分布，不能写成真实老人 30 天误报率",
    })

    diagnostics = {
        "distributions": _distribution_diagnostics(
            merged.loc[merged["participantid"].isin(set(
                splits.loc[splits["fold"] == folds[0], "participantid"]
            ))],
            pd.concat(synthetic_daily_sample, ignore_index=True) if synthetic_daily_sample else pd.DataFrame(),
        ),
        "event_rates": {
            "synthetic_train": _event_rate_table(train_frame),
            "synthetic_test": _event_rate_table(test_frame),
            "synthetic_test_by_fold": _event_rate_table(test_frame, by=["fold"]),
            "real_training_anchor": {
                str(fold): float(
                    real.loc[
                        real["participantid"].isin(
                            set(splits.loc[splits["fold"] != fold, "participantid"])
                        ) & real["r_branch_eligible"],
                        "y_7d_num",
                    ].mean()
                )
                for fold in folds
            },
        },
        "scores": {
            "real": {
                "T_median": float(real_scores["T"].median()),
                "C_median": float(real_scores["C"].median()),
                "scored": int((real_scores["status"] == "scored").sum()),
            },
            "synthetic_test": {
                "T_median": float(test_frame["T"].median()),
                "C_median": float(test_frame["C"].median()),
                "scored": int((test_frame["status"] == "scored").sum()),
            },
        },
        "precision": {
            "real_missing_se_share": float(
                (bout_daily.get("se_gait_speed_mean", pd.Series(dtype=float)).isna()).mean()
            ),
        },
    }
    _write_json(run_dir / "distribution_diagnostics.json", diagnostics)

    lineage = pd.DataFrame({
        "series_id": test_frame["series_id"],
        "fold": test_frame["fold"],
        "mechanism": test_frame["mechanism"],
        "noise_family": test_frame["noise_family"],
        "trajectory": test_frame["trajectory"],
        "generator_params": [
            f"generator_params_fold{int(fold)}.json" for fold in test_frame["fold"]
        ],
        "source": "synthetic",
        "parent_donor_pool": [
            f"fold{int(fold)}_train_donors" for fold in test_frame["fold"]
        ],
        "seed_family": [f"SEED+1000*{int(fold)}+200" for fold in test_frame["fold"]],
    })
    lineage.to_csv(run_dir / "lineage.csv", index=False, encoding="utf-8")

    counts = {
        "real": {
            "candidates_scored": int(len(real_scores)),
            "status": real_status["status_counts"],
            "true_events_7d": int(real_scores["y_7d"].sum()),
        },
        "synthetic": {
            "train": int(len(train_frame)),
            "test": int(len(test_frame)),
            "calibration": int(len(calibration_frame)),
            "stress": int(len(stress_frame)),
            "by_mechanism": test_frame.groupby("mechanism").size().to_dict(),
            "by_noise_family": test_frame.groupby("noise_family").size().to_dict(),
        },
        "budgets": budget,
        "limit": limit,
    }
    _write_json(run_dir / "counts.json", counts)

    snapshot = run_dir / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for relative in CODE_SNAPSHOT_FILES:
        source = PROJECT_ROOT / relative
        if source.is_file():
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    manifest = {
        "stage": 12,
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "upstream": {
            "stage11": str(stage11),
            "protocol": str(stage11 / "protocol.json"),
            "protocol_md5": _md5(stage11 / "protocol.json"),
        },
        "inputs": {
            name: {"path": str(path), "md5": _md5(path)}
            for name, path in {
                "real_candidates": stage11 / "real_candidates.csv",
                "baseline_days": stage11 / "baseline_days.csv",
                "splits": stage11 / "splits.csv",
                "landmarks": stage11 / "landmarks.csv",
                "clinical": resolved["Clinical Data.zip"],
                "daily": resolved["Aggregated DMO Data.zip"],
                "bouts": bouts_path,
            }.items()
        },
        "scoring": {
            "config_fingerprint": config_fingerprint(ds_config),
            "scoring_fingerprint": scoring_fingerprint(ds_config),
            "precision_mode": ds_config.precision_mode,
            "features": list(ds_config.features),
            "directions": dict(ds_config.directions),
        },
        "generator_version": "round2-synth-v1",
        "seed": SEED,
        "commands": [
            "python scripts/round2_stage12_supplement.py --run-id " + args.run_id,
            "python scripts/verify_round2_stage12.py --run " + str(run_dir),
        ],
        "outputs": sorted(
            str(path.relative_to(run_dir)) for path in run_dir.rglob("*") if path.is_file()
        ),
    }
    _write_json(run_dir / "manifest.json", manifest)
    print(f"[stage12] 完成：{run_dir}", flush=True)
    return 0


def _event_rate_table(frame: pd.DataFrame, by: Sequence[str] | None = None) -> dict:
    """事件率表：以 ``机制|噪声[|折]`` 为字符串键。"""
    keys = ["mechanism", "noise_family"] + list(by or [])
    table = frame.groupby(keys)[["y_1d", "y_3d", "y_7d"]].mean()
    counts = frame.groupby(keys).size()
    out: dict[str, Any] = {}
    for index, row in table.iterrows():
        label = "|".join(str(part) for part in (index if isinstance(index, tuple) else (index,)))
        out[label] = {
            "n": int(counts.loc[index]),
            "y_1d": round(float(row["y_1d"]), 5),
            "y_3d": round(float(row["y_3d"]), 5),
            "y_7d": round(float(row["y_7d"]), 5),
        }
    return out


def _synthetic_daily_frame(
    params: GeneratorParams, config: GeneratorConfig, mechanism: str, noise: str
) -> pd.DataFrame:
    """少量虚拟逐日长表，仅用于分布诊断。"""
    from mobilise_d.round2_synth import simulate_person

    rng = np.random.default_rng(SEED + 991)
    rows = []
    for index in range(60):
        trajectory = TRAJECTORY_MIX[mechanism][index % len(TRAJECTORY_MIX[mechanism])]
        person = simulate_person(
            params, series_id=f"diag-{index}", mechanism=mechanism, noise_family=noise,
            trajectory=trajectory, rng=rng, config=config,
            forcing_lambda0=params.lambda0.get(f"{mechanism}|{noise}"),
        )
        observed = np.asarray(person["observed"], dtype=float)
        se = np.asarray(person["se"], dtype=float)
        for day in range(observed.shape[0]):
            rows.append({
                "series_id": person["series_id"],
                "mechanism": mechanism,
                "noise_family": noise,
                "day_index": day,
                "gait_speed_mean": observed[day, 0],
                "stride_length_mean_cm": observed[day, 1],
                "stride_duration_bout_cv": observed[day, 2],
                "se_gait_speed_mean": se[day, 0],
                "se_stride_length_mean_cm": se[day, 1],
                "se_stride_duration_bout_cv": se[day, 2],
                "h": person["h_snapshot"] if day == config.baseline_days + config.monitoring_days - 1 else np.nan,
            })
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
