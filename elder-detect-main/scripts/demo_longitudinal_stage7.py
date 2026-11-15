"""阶段 7 演示：真实穿戴观测前缀 → 纵向服务 → 研究折模型 → 三层输出。

个人步态变化（RMS/DS-PCD）、内部折外研究结果（calibrated=false）、
经验证的临床概率（当前没有，服务概率保持为空）。
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.baseline import build_baseline  # noqa: E402
from longitudinal.config import LongitudinalConfig  # noqa: E402
from longitudinal.ds_pcd import DayRecord, DsPcdConfig, fit_baseline  # noqa: E402
from longitudinal.research import research_prediction  # noqa: E402
from longitudinal.schema import DailyObservation  # noqa: E402
from longitudinal.service import LongitudinalRiskService  # noqa: E402
from longitudinal.storage import ProfileStore  # noqa: E402
from longitudinal.survival import (  # noqa: E402
    CoxModelArtifact,
    predict_cumulative_risk,
)
from mobilise_d.expansion import compute_expansion_from_frames  # noqa: E402
from mobilise_d.observations import (  # noqa: E402
    FEATURES,
    FEATURE_MAP,
    ds_pcd_config,
    wearable_config,
)

OUT_ROOT = PROJECT_ROOT.parents[1] / "output" / "ds_pcd_mobilise"
FIXED = OUT_ROOT / "20261114-fixed"
_CONTRACT_CANDIDATES = (
    OUT_ROOT / "stage8" / "20261114-contract-v2" / "models_contract",
    OUT_ROOT / "stage8" / "20261114-contract-v1" / "models_contract",
)
DEFAULT_CONTRACT_DIR = next(
    (item for item in _CONTRACT_CANDIDATES if item.exists()), None
)
STAGE5 = OUT_ROOT / "stage5"
STAGE7 = OUT_ROOT / "stage7"
TOLERANCE = 1e-9


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 7 纵向服务演示")
    parser.add_argument("--person", default="1049")
    parser.add_argument("--landmark", default="2023-04-22")
    parser.add_argument("--out", default=str(STAGE7))
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_CONTRACT_DIR) if DEFAULT_CONTRACT_DIR is not None else None,
        help="阶段 8 带输入契约的模型目录；省略时优先使用阶段 8 导出，否则源模型",
    )
    return parser


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _wearable_observations(person: str, landmark: str) -> tuple[list[DailyObservation], dict[str, dict[str, Any]]]:
    daily = pd.read_csv(FIXED / "data" / "daily_gait.csv", dtype={"participantid": str})
    precision = pd.read_csv(STAGE5 / "se_causal" / "main_precision.csv", dtype={"participantid": str})
    daily = daily.loc[
        daily["participantid"].eq(person)
        & daily["has_core_gait"].astype(str).str.lower().eq("true")
    ].copy()
    daily["day"] = pd.to_datetime(daily["day"]).dt.normalize()
    precision["day"] = pd.to_datetime(precision["day"]).dt.normalize()
    daily = daily.loc[daily["day"] <= pd.Timestamp(landmark)].merge(
        precision, on=["participantid", "visit_number", "day"], how="left",
        validate="one_to_one",
    ).sort_values("day")
    observations = []
    metadata = {}
    for row in daily.itertuples(index=False):
        features = {name: float(getattr(row, column)) for name, column in FEATURE_MAP.items()}
        feature_se = {}
        for name in FEATURES:
            column = getattr(row, f"se_{name}", None)
            if column is not None and pd.notna(column) and float(column) > 0:
                feature_se[name] = float(column)
        precision_payload = {
            "methods": {name: getattr(row, f"method_{name}", None) for name in FEATURES},
            "source": "stage5_causal_precision",
        }
        observations.append(DailyObservation.from_dict({
            "person_id": person,
            "observed_at": pd.Timestamp(row.day).date().isoformat(),
            "features": features,
            "feature_se": feature_se,
            "precision": precision_payload,
            "quality": {
                "valid_frames": 0, "valid_steps": 0,
                "walking_duration_s": 0.0, "mean_confidence": 0.0,
            },
            "source": "mobilise-d-cvs",
            "timezone": "UTC",
        }))
        metadata[pd.Timestamp(row.day).date().isoformat()] = {
            "visit_number": str(row.visit_number),
            "wear_h": float(row.wear_h_waking) if pd.notna(row.wear_h_waking) else None,
            "n_bouts": int(row.n_bouts) if pd.notna(getattr(row, "n_bouts", None)) else None,
        }
    return observations, metadata


def _make_wearable_profile(
    directory: Path, person: str, observations: list[DailyObservation],
    baseline_dates: list[str], ds_config: DsPcdConfig,
) -> tuple[ProfileStore, Any, Any]:
    store = ProfileStore(directory, person)
    for observation in observations:
        store.append_observation(observation)
    rms_config = wearable_config()
    rms_baseline = build_baseline(observations, rms_config)
    store.save_baseline(rms_baseline)
    records = [
        DayRecord(
            date=item.observation_date,
            features=dict(item.features),
            feature_se=dict(item.feature_se),
            valid=True,
        )
        for item in observations
    ]
    baseline_dates_tuple = tuple(baseline_dates)
    ds_baseline = fit_baseline(
        records, person, ds_config, baseline_dates=baseline_dates_tuple,
    )
    store.save_ds_pcd_baseline(ds_baseline)
    return store, rms_baseline, ds_baseline


def _service_payload(result) -> dict[str, Any]:
    return {
        "status": result.status,
        "distance": result.distance,
        "deviation_index": result.deviation_index,
        "persistent_deviation_days": result.persistent_deviation_days,
        "risk_7d": result.risk_7d,
        "risk_30d": result.risk_30d,
        "as_of": result.as_of,
        "baseline_end_date": result.baseline_end_date,
        "research_covariates": dict(result.research_covariates),
        "research_covariate_status": dict(result.research_covariate_status),
        "valid_day_counts": dict(result.valid_day_counts),
        "provenance": dict(result.provenance),
        "ds_pcd": result.ds_pcd,
    }


def _old_video_profile(directory: Path) -> dict[str, Any]:
    """旧版 11 特征视频画像：验证旧基线读取与未校准概率为空。"""
    features = {
        "gait_speed_proxy": 0.31, "cadence_spm": 92.0, "step_time_mean_s": 0.65,
        "step_time_cv": 0.08, "stance_width_norm": 0.74, "gait_asymmetry": 0.05,
        "trunk_tilt_mean_deg": 4.2, "trunk_sway_rms_deg": 1.5,
        "aspect_ratio_mean": 0.42, "aspect_ratio_cv": 0.07, "emos_proxy": 0.28,
    }
    if directory.exists():
        shutil.rmtree(directory)
    start = date(2026, 10, 1)
    observations = []
    for offset in range(15):
        payload = dict(features)
        payload["cadence_spm"] = 92.0 + offset * 0.05
        observations.append(DailyObservation.from_dict({
            "person_id": "elder_001",
            "observed_at": (start + __import__("datetime").timedelta(days=offset)).isoformat(),
            "features": payload,
            "quality": {
                "valid_frames": 300, "valid_steps": 40,
                "walking_duration_s": 30.0, "mean_confidence": 0.9,
            },
            "source": "legacy-video-demo",
            "timezone": "UTC",
        }))
    store = ProfileStore(directory, "elder_001")
    for item in observations:
        store.append_observation(item)
    config = LongitudinalConfig()
    baseline = build_baseline(observations, config)
    store.save_baseline(baseline)
    service = LongitudinalRiskService(store, config, modality="video",
                                      population="community older adults",
                                      feature_version="video-pose-v1")
    result = service.evaluate("elder_001", observations[-1], persist=False)
    return {
        "baseline_version": baseline.version,
        "baseline_valid_days": baseline.valid_days,
        "status": result.status,
        "distance": result.distance,
        "risk_7d": result.risk_7d,
        "risk_30d": result.risk_30d,
        "research_covariate_status": dict(result.research_covariate_status),
        "provenance": dict(result.provenance),
    }


def expansion_statistics() -> dict[str, Any]:
    landmarks = pd.read_csv(FIXED / "data" / "landmarks.csv", dtype={"participantid": str})
    windows = pd.read_csv(FIXED / "data" / "landmark_windows.csv", dtype={"participantid": str})
    summary = compute_expansion_from_frames(landmarks, windows)
    summary = dict(summary)
    summary["definition"] = (
        "S14 = 至少 14 个既往有效日（当前主分析）；"
        "S7 = 至少 7 个既往有效日（预设探索入组）"
    )
    summary["new_positive_people_ids"] = sorted(
        summary["positive_people_sets"]["from_new_people"]
    )
    return summary

def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    person = str(args.person)
    landmark = str(args.landmark)
    scores = pd.read_csv(FIXED / "stage2" / "scores_session_se.csv", dtype={"participantid": str})
    expected = scores.loc[
        scores["participantid"].eq(person) & scores["landmark_day"].astype(str).eq(landmark)
    ]
    if expected.empty:
        raise ValueError(f"阶段 6 评分表中没有 {person}/{landmark}（session_se 可评日）")
    expected = expected.iloc[0]
    predictions = pd.read_csv(FIXED / "predictions.csv", dtype={"participantid": str})
    expected_prediction = predictions.loc[
        predictions["variant"].eq("fixed_session") & predictions["horizon"].eq(30)
        & predictions["method"].eq("B_rms_cusum")
        & predictions["participantid"].eq(person)
        & predictions["landmark_day"].astype(str).eq(landmark)
    ]
    if expected_prediction.empty:
        raise ValueError("阶段 6 折外预测中缺少对应行")
    expected_prediction = expected_prediction.iloc[0]
    folds = pd.read_csv(FIXED / "splits.csv", dtype={"participantid": str})
    fold = int(folds.loc[folds["participantid"].eq(person), "fold"].iloc[0])
    baseline_rows = pd.read_csv(FIXED / "data" / "baseline_days.csv", dtype={"participantid": str})
    baseline_dates = sorted(
        baseline_rows.loc[
            baseline_rows["participantid"].eq(person)
            & baseline_rows["landmark_day"].astype(str).eq(landmark),
            "baseline_day",
        ].tolist()
    )
    run_config = json.loads((FIXED / "stage2" / "replay_summary.json").read_text(encoding="utf-8"))
    ds_config = DsPcdConfig.from_dict(run_config["modes"]["session_se"]["config"])
    observations, metadata = _wearable_observations(person, landmark)
    profile_dir = out / "demo_profile"
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
    store, rms_baseline, ds_baseline = _make_wearable_profile(
        profile_dir, person, observations, baseline_dates, ds_config,
    )
    service = LongitudinalRiskService(
        store, wearable_config(), ds_pcd_config=ds_config,
        ds_pcd_alarm_threshold=5.2, modality="wearable",
        population="65+ Parkinson (Mobilise-D CVS)",
        feature_version="mobilise-d-daily-v1",
        score_version="scoring-v2-causal-precision",
    )
    current = {item.observation_date: item for item in store.load_observations()}[landmark]
    result = service.evaluate(person, current, as_of=landmark, persist=False)
    service_payload = _service_payload(result)
    ds = result.ds_pcd or {}
    comparison = {
        "distance_diff": abs(float(result.distance) - float(expected["rms_distance"])),
        "cusum_diff": abs(float(ds.get("cusum")) - float(expected["cusum"])),
        "T_diff": abs(float(ds.get("T")) - float(expected["T"])),
        "q_diff": abs(float(ds.get("q")) - float(expected["q"])),
        "status_match": str(ds.get("status")) == str(expected["ds_status"]),
        "baseline_end_match": str(ds.get("baseline_end")) == str(expected["ds_baseline_end"]),
        "precision_mode_match": str(ds.get("precision_mode")) == str(expected["precision_mode"]),
        "n_valid_days_match": int(result.valid_day_counts["n_valid_days"]) == int(expected["n_valid_days"]),
        "n_valid_days_before_match": int(result.valid_day_counts["n_valid_days_before_landmark"])
        == int(expected["n_valid_days_before_landmark"]),
    }
    if args.model_dir:
        model_path = (
            Path(args.model_dir) / "fixed_session" / "30d"
            / f"B_rms_cusum_fold{fold}.json"
        )
    else:
        model_path = (
            FIXED / "fold_models" / "fixed_session" / "30d"
            / f"B_rms_cusum_fold{fold}.json"
        )
    artifact = CoxModelArtifact.from_dict(json.loads(model_path.read_text(encoding="utf-8")))
    research = research_prediction(result, artifact)
    research["fold"] = fold
    research["model_file"] = str(model_path)
    research["stage6_risk"] = float(expected_prediction["risk"])
    research["risk_diff"] = (
        abs(float(research["risk"]) - float(expected_prediction["risk"]))
        if research.get("risk") is not None else None
    )
    clinical = {
        "service_risk_7d": result.risk_7d,
        "service_risk_30d": result.risk_30d,
        "gate_error": None,
    }
    try:
        predict_cumulative_risk(
            {"rms_distance": result.research_covariates["rms_distance"],
             "cusum": result.research_covariates["cusum"]},
            artifact,
        )
    except ValueError as exc:
        clinical["gate_error"] = str(exc)
    # 无 DS 基线的同一前缀
    missing_dir = out / "demo_profile_no_ds"
    if missing_dir.exists():
        shutil.rmtree(missing_dir)
    missing_store = ProfileStore(missing_dir, person)
    for item in observations:
        missing_store.append_observation(item)
    missing_store.save_baseline(rms_baseline)
    missing_service = LongitudinalRiskService(
        missing_store, wearable_config(), ds_pcd_config=ds_config,
        modality="wearable", population="65+ Parkinson (Mobilise-D CVS)",
        feature_version="mobilise-d-daily-v1",
        score_version="scoring-v2-causal-precision",
    )
    missing_current = {item.observation_date: item for item in missing_store.load_observations()}[landmark]
    missing_result = missing_service.evaluate(
        person, missing_current, as_of=landmark, persist=False,
    )
    missing_research = research_prediction(missing_result, artifact)
    # 配置/来源不匹配：用视频 DS 配置读取穿戴画像
    mismatch_service = LongitudinalRiskService(
        store, wearable_config(), ds_pcd_config=DsPcdConfig(),
        modality="wearable", population="65+ Parkinson (Mobilise-D CVS)",
        feature_version="mobilise-d-daily-v1",
        score_version="scoring-v2-causal-precision",
    )
    mismatch_result = mismatch_service.evaluate(
        person, current, as_of=landmark, persist=False,
    )
    old = _old_video_profile(out / "old_profile")
    expansion = expansion_statistics()
    payload = {
        "person": person,
        "landmark": landmark,
        "fold": fold,
        "baseline_dates": baseline_dates,
        "service": service_payload,
        "expected_stage6": {
            "rms_distance": float(expected["rms_distance"]),
            "cusum": float(expected["cusum"]),
            "T": float(expected["T"]),
            "q": float(expected["q"]),
            "ds_status": str(expected["ds_status"]),
            "ds_baseline_end": str(expected["ds_baseline_end"]),
            "precision_mode": str(expected["precision_mode"]),
            "n_valid_days": int(expected["n_valid_days"]),
            "n_valid_days_before_landmark": int(expected["n_valid_days_before_landmark"]),
        },
        "comparison": comparison,
        "research": research,
        "clinical_probability": clinical,
        "missing_ds_baseline": {
            "service_status": missing_result.status,
            "ds_status": (missing_result.ds_pcd or {}).get("status"),
            "distance": missing_result.distance,
            "cusum": missing_result.research_covariates.get("cusum"),
            "covariate_status": dict(missing_result.research_covariate_status),
            "research": missing_research,
        },
        "config_mismatch": {
            "service_status": mismatch_result.status,
            "ds_status": (mismatch_result.ds_pcd or {}).get("status"),
            "message": (mismatch_result.ds_pcd or {}).get("message"),
            "cusum": mismatch_result.research_covariates.get("cusum"),
            "covariate_status": dict(mismatch_result.research_covariate_status),
        },
        "old_profile": old,
        "expansion": expansion,
        "metadata": {
            "observation_days": len(observations),
            "first_observation": observations[0].observation_date,
            "last_observation": observations[-1].observation_date,
            "wear_metadata_sample": dict(list(metadata.items())[:3]),
        },
        "tolerance": TOLERANCE,
    }
    _write_json(out / "demo_output.json", payload)
    mapping = {
        "wearable": {
            "person_id": "DailyObservation.person_id",
            "day": "DailyObservation.observed_at / local_date（UTC）",
            "walking_speed_m_s": "daily_gait.ws_30_avg_d（≥30 秒步行段，m/s）",
            "cadence_spm": "daily_gait.cadence_all_avg_d（steps/min）",
            "stride_duration_s": "daily_gait.strdur_all_avg_d（s，跨步周期）",
            "feature_se": "stage5/se_causal/main_precision.csv 的 se_*（同日冻结）",
            "quality": "佩戴/步态筛选由阶段 1 has_core_gait 与阶段 5 精度覆盖；服务使用零阈值配置",
            "source": "mobilise-d-cvs",
            "modality": "wearable",
            "precision_mode": "session_se",
        },
        "video": {
            "features": "longitudinal.pose_features.summarize_pose_sequence 的 11 项特征",
            "feature_se": "pose_session_precision（--with-uncertainty）",
            "quality": "有效帧数、步数、步行时长、平均置信度",
            "modality": "video",
        },
        "research_covariates": {
            "rms_distance": "服务 distance（同日 RMS）",
            "cusum": "服务 DS-PCD C（同日累计）",
            "distance": "旧模型既有字段，保留",
        },
        "model_input_contract": {
            "A_rms": ["rms_distance"],
            "B_rms_cusum": ["rms_distance", "cusum"],
            "calibrated": False,
            "research_only": True,
        },
    }
    _write_json(out / "input_field_mapping.json", mapping)
    print(json.dumps({
        "person": person, "landmark": landmark, "fold": fold,
        "service_status": service_payload["status"],
        "distance": service_payload["distance"],
        "cusum": ds.get("cusum"),
        "T": ds.get("T"),
        "comparison": comparison,
        "research": {k: research[k] for k in ("status", "risk", "calibrated", "research_only", "fold", "risk_diff")},
        "clinical": clinical,
        "missing_ds": payload["missing_ds_baseline"],
        "config_mismatch": payload["config_mismatch"],
        "old_profile": {k: old[k] for k in ("baseline_version", "status", "distance", "risk_7d", "risk_30d")},
        "expansion": {k: expansion[k] for k in ("S14", "S7", "overlapping_people", "new_people", "new_positive_people", "new_positive_landmarks")},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
