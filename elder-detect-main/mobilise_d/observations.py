"""把阶段 1 冻结队列适配为 DS-PCD 的逐日记录与 RMS 基线。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from longitudinal.baseline import PersonalBaseline, _feature_stats, config_fingerprint
from longitudinal.config import LongitudinalConfig
from longitudinal.ds_pcd import DayRecord, DsPcdConfig
from longitudinal.service import trend_7d_calendar


FEATURES: tuple[str, ...] = ("walking_speed_m_s", "cadence_spm", "stride_duration_s")
DIRECTIONS: dict[str, float] = {
    "walking_speed_m_s": -1.0,
    "cadence_spm": -1.0,
    "stride_duration_s": 1.0,
}
FEATURE_MAP: dict[str, str] = {
    "walking_speed_m_s": "ws_30_avg_d",
    "cadence_spm": "cadence_all_avg_d",
    "stride_duration_s": "strdur_all_avg_d",
}
# 发布方逐日值的精度：步速 0.01 m/s、步频 1 steps/min、跨步周期 0.01 s。
TAU_FLOOR = {"walking_speed_m_s": 0.005, "cadence_spm": 0.5, "stride_duration_s": 0.005}
SE_FLOOR = {"walking_speed_m_s": 1e-4, "cadence_spm": 1e-2, "stride_duration_s": 1e-4}


@dataclass(frozen=True)
class Stage1Cohort:
    participants: pd.DataFrame
    daily: pd.DataFrame
    landmarks: pd.DataFrame
    baseline_days: pd.DataFrame
    windows: pd.DataFrame

    @property
    def person_ids(self) -> list[str]:
        return sorted(self.participants["participantid"].unique())


def load_stage1(bundle: Path) -> Stage1Cohort:
    participants = pd.read_csv(bundle / "participants.csv", dtype={"participantid": str})
    daily = pd.read_csv(bundle / "daily_gait.csv", dtype={"participantid": str})
    daily["day"] = pd.to_datetime(daily["day"]).dt.normalize()
    daily["has_core_gait"] = daily["has_core_gait"].astype(str).str.lower().eq("true")
    landmarks = pd.read_csv(bundle / "landmarks.csv", dtype={"participantid": str})
    landmarks["landmark_day"] = pd.to_datetime(landmarks["landmark_day"]).dt.normalize()
    landmarks["baseline_eligible"] = (
        landmarks["baseline_eligible"].astype(str).str.lower().eq("true")
    )
    baseline_days = pd.read_csv(bundle / "baseline_days.csv", dtype={"participantid": str})
    baseline_days["baseline_day"] = pd.to_datetime(baseline_days["baseline_day"]).dt.normalize()
    baseline_days["landmark_day"] = pd.to_datetime(baseline_days["landmark_day"]).dt.normalize()
    windows = pd.read_csv(bundle / "landmark_windows.csv", dtype={"participantid": str})
    windows["landmark_day"] = pd.to_datetime(windows["landmark_day"]).dt.normalize()
    windows["event_bool"] = windows["event"].map({"True": True, "False": False})
    return Stage1Cohort(
        participants=participants, daily=daily, landmarks=landmarks,
        baseline_days=baseline_days, windows=windows,
    )


def wearable_config() -> LongitudinalConfig:
    """穿戴设备适配器的质量门控：有效性由阶段 1 的步态筛选保证。"""
    return LongitudinalConfig(
        core_features=FEATURES,
        min_valid_frames=0,
        min_valid_steps=0,
        min_walking_duration_s=0.0,
        min_mean_confidence=0.0,
        min_core_feature_coverage=1.0,
    )


def ds_pcd_config(
    *,
    precision_mode: str = "session_se",
    null_samples: int = 20000,
    quantile_points: int = 1000,
    k_ref: float = 0.5,
    gap_reset_days: int = 7,
    bootstrap_samples: int = 200,
    min_bouts: int = 5,
) -> DsPcdConfig:
    return DsPcdConfig(
        features=FEATURES,
        directions=DIRECTIONS,
        precision_mode=precision_mode,
        null_samples=null_samples,
        quantile_points=quantile_points,
        k_ref=k_ref,
        gap_reset_days=gap_reset_days,
        tau_floor=TAU_FLOOR,
        se_floor=SE_FLOOR,
        bootstrap_samples=bootstrap_samples,
        min_bouts=min_bouts,
    )


def build_records(
    cohort: Stage1Cohort,
    precision: pd.DataFrame | None,
    *,
    precision_mode: str,
    person_ids: Iterable[str] | None = None,
) -> dict[str, list[DayRecord]]:
    """按人构造逐日记录；只有阶段 1 的有效步态日进入记录。"""
    daily = cohort.daily.loc[cohort.daily["has_core_gait"]].copy()
    if person_ids is not None:
        daily = daily.loc[daily["participantid"].isin(set(person_ids))]
    if precision is not None and len(precision):
        precision = precision.copy()
        precision["day"] = pd.to_datetime(precision["day"]).dt.normalize()
        daily = daily.merge(
            precision, on=["participantid", "visit_number", "day"], how="left",
            validate="one_to_one",
        )
    records: dict[str, list[DayRecord]] = {}
    for row in daily.itertuples(index=False):
        features = {
            name: float(getattr(row, column)) for name, column in FEATURE_MAP.items()
        }
        feature_se: dict[str, float] = {}
        if precision_mode == "session_se":
            for name in FEATURES:
                column = getattr(row, f"se_{name}", None)
                if column is not None and pd.notna(column) and float(column) > 0:
                    feature_se[name] = float(column)
        records.setdefault(str(row.participantid), []).append(DayRecord(
            date=pd.Timestamp(row.day).date().isoformat(),
            features=features,
            feature_se=feature_se,
            valid=True,
            note=f"visit={row.visit_number}",
        ))
    for items in records.values():
        items.sort(key=lambda item: item.date)
    return records


def fit_rms_baseline(
    person_id: str,
    records_by_date: Mapping[str, DayRecord],
    baseline_dates: Iterable[str],
    config: LongitudinalConfig,
) -> PersonalBaseline:
    """用与 DS-PCD 相同的基线日建立旧 RMS 规则的中位数/MAD 基线。"""
    dates = tuple(baseline_dates)
    missing = [day for day in dates if day not in records_by_date]
    if missing:
        raise ValueError(f"RMS 基线日缺少观测：{missing[:5]}")
    features = {
        name: _feature_stats([records_by_date[day].features[name] for day in dates])
        for name in config.core_features
    }
    return PersonalBaseline(
        person_id=str(person_id),
        created_at="mobilise-stage2",
        valid_days=len(dates),
        valid_dates=dates,
        features=features,
        config_fingerprint=config_fingerprint(config),
        updated_at="mobilise-stage2",
        config_snapshot=asdict(config),
        slow_reference={name: stats.median for name, stats in features.items()},
    )


def distance_history(
    records: list[DayRecord],
    baseline: PersonalBaseline,
    baseline_end: str,
) -> list[tuple[str, float]]:
    from longitudinal.deviation import compute_deviation

    history = []
    for record in records:
        if record.date <= baseline_end:
            continue
        history.append((record.date, compute_deviation(record.features, baseline).distance))
    return history


def trend_and_persistence(
    history: list[tuple[str, float]],
    day: str,
    *,
    threshold: float = 3.0,
) -> tuple[float, dict[str, Any], int]:
    """返回该预测日的 7 日趋势、窗口信息与持续偏离天数。"""
    upto = [(item_day, value) for item_day, value in history if item_day <= day]
    trend, window = trend_7d_calendar(upto, day)
    persistent = 0
    for _, value in reversed(upto):
        if value < threshold:
            break
        persistent += 1
    return trend, window, persistent
