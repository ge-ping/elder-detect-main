"""阶段 12：真实与半合成输入共用同一 RMS/DS-PCD 评分入口。

* DS-PCD 一律调用 ``longitudinal.ds_pcd``（不复制公式）。
* RMS 一律调用 ``longitudinal.deviation.compute_deviation``，基线由
  ``mobilise_d.observations.fit_rms_baseline`` 在同一批基线日上建立。
* 真实观测与虚拟序列使用同一个函数签名，返回值带 ``source`` 与版本指纹。
* 未达基线资格、预测日缺特征或精度不足都保留明确状态，不以不明回退充数。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from longitudinal.baseline import PersonalBaseline
from longitudinal.config import LongitudinalConfig
from longitudinal.deviation import compute_deviation
from longitudinal.ds_pcd import (
    BaselineUnavailable,
    DayRecord,
    DsPcdBaseline,
    DsPcdConfig,
    config_fingerprint,
    fit_baseline,
    score_timeline,
)
from .observations import fit_rms_baseline
from .round2 import R2_DIRECTIONS, R2_FEATURES


# 发布值的日表精度：步速 0.01 m/s、步长 1 cm；短窗变异为无量纲。
TAU_FLOOR: dict[str, float] = {
    "gait_speed_mean": 0.005,
    "stride_length_mean_cm": 0.5,
    "stride_duration_bout_cv": 0.002,
}
SE_FLOOR: dict[str, float] = {
    "gait_speed_mean": 1e-4,
    "stride_length_mean_cm": 1e-3,
    "stride_duration_bout_cv": 1e-5,
}


def r2_ds_pcd_config(
    *,
    precision_mode: str = "session_se",
    features: Sequence[str] = R2_FEATURES,
    directions: Mapping[str, float] = R2_DIRECTIONS,
    null_samples: int = 20000,
    quantile_points: int = 1000,
    k_ref: float = 0.5,
    gap_reset_days: int = 7,
    shrink_lambda: float = 0.5,
    tau_floor: Mapping[str, float] | None = None,
    se_floor: Mapping[str, float] | None = None,
) -> DsPcdConfig:
    features = tuple(features)
    tau = dict(tau_floor or TAU_FLOOR)
    se = dict(se_floor or SE_FLOOR)
    return DsPcdConfig(
        features=features,
        directions={name: float(directions[name]) for name in features},
        precision_mode=precision_mode,
        null_samples=null_samples,
        quantile_points=quantile_points,
        k_ref=k_ref,
        gap_reset_days=gap_reset_days,
        shrink_lambda=shrink_lambda,
        tau_floor={name: tau[name] for name in features if name in tau},
        se_floor={name: se[name] for name in features if name in se},
    )


def scoring_fingerprint(config: DsPcdConfig) -> str:
    payload = {
        "ds_pcd": config.to_dict(),
        "config_fingerprint": config_fingerprint(config),
        "features": list(config.features),
        "directions": dict(config.directions),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def rms_config(
    features: Sequence[str] = R2_FEATURES, *, min_baseline_days: int = 14
) -> LongitudinalConfig:
    return LongitudinalConfig(
        core_features=tuple(features),
        min_valid_frames=0,
        min_valid_steps=0,
        min_walking_duration_s=0.0,
        min_mean_confidence=0.0,
        min_core_feature_coverage=1.0,
        min_baseline_days=min_baseline_days,
    )


@dataclass(frozen=True)
class PersonSeries:
    """一个人（真实或虚拟）按日排列的特征、标准误与来源。"""

    person_id: str
    days: tuple[str, ...]
    features: dict[str, tuple[float, ...]]
    feature_se: dict[str, tuple[float, ...]]
    source: str
    meta: dict[str, Any]

    def records(self, indices: Iterable[int] | None = None) -> list[DayRecord]:
        chosen = list(indices) if indices is not None else list(range(len(self.days)))
        out = []
        for index in chosen:
            values = {
                name: float(self.features[name][index]) for name in self.features
            }
            se = {
                name: float(self.feature_se[name][index]) for name in self.feature_se
            }
            out.append(DayRecord(date=self.days[index], features=values, feature_se=se))
        return out


def series_from_frame(
    frame: pd.DataFrame,
    person_id: str,
    *,
    features: Sequence[str] = R2_FEATURES,
    se_columns: Mapping[str, str] | None = None,
    source: str = "observed",
    meta: Mapping[str, Any] | None = None,
) -> PersonSeries:
    """把逐日 DataFrame 转成评分序列；缺值保留 NaN，由评分层处理。"""
    frame = frame.sort_values("day", kind="stable")
    days = tuple(pd.to_datetime(frame["day"]).dt.strftime("%Y-%m-%d"))
    se_columns = dict(se_columns or {
        "gait_speed_mean": "se_gait_speed_mean",
        "stride_length_mean_cm": "se_stride_length_mean_cm",
        "stride_duration_bout_cv": "se_stride_duration_bout_cv",
    })
    feature_values: dict[str, tuple[float, ...]] = {}
    se_values: dict[str, tuple[float, ...]] = {}
    for name in features:
        feature_values[name] = tuple(
            pd.to_numeric(frame[name], errors="coerce").astype(float)
        )
        column = se_columns.get(name)
        se_values[name] = (
            tuple(pd.to_numeric(frame[column], errors="coerce").astype(float))
            if column and column in frame.columns
            else tuple(float("nan") for _ in range(len(frame)))
        )
    return PersonSeries(
        person_id=str(person_id), days=days, features=feature_values,
        feature_se=se_values, source=source, meta=dict(meta or {}),
    )


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def score_person(
    series: PersonSeries,
    config: DsPcdConfig,
    *,
    baseline_days: int = 14,
    snapshot_day: str | None = None,
    alarm_threshold: float | None = None,
    rms_threshold: float | None = None,
    baseline_dates: Sequence[str] | None = None,
) -> dict[str, Any]:
    """对一个人评分：DS-PCD 快照、逐日前缀与 RMS 距离。

    给出 ``baseline_dates`` 时强制使用冻结的基线日；缺失即记基线不可用，
    不悄悄把监测日当成基线。
    """
    records = series.records()
    if not records:
        return {"person_id": series.person_id, "status": "no_records",
                "source": series.source}
    try:
        baseline_obj = fit_baseline(
            records, series.person_id, config,
            baseline_dates=tuple(baseline_dates) if baseline_dates else None,
        )
        timeline = score_timeline(
            records, series.person_id, config, baseline=baseline_obj,
            alarm_threshold=alarm_threshold,
        )
    except BaselineUnavailable as error:
        return {
            "person_id": series.person_id,
            "status": f"baseline_unavailable:{error.reason}",
            "reason": str(error),
            "source": series.source,
        }
    except Exception as error:  # noqa: BLE001 - 保留失败状态
        return {
            "person_id": series.person_id,
            "status": f"scoring_failed:{type(error).__name__}",
            "reason": str(error),
            "source": series.source,
        }
    rows = timeline.rows
    scored = [row for row in rows if row.status == "scored"]
    snapshot = None
    if snapshot_day is not None:
        for row in scored:
            if row.date == snapshot_day:
                snapshot = row
                break
    elif scored:
        snapshot = scored[-1]

    # RMS：同一批基线日
    baseline_dates = list(timeline.baseline.baseline_dates)
    rms_values: list[tuple[str, float]] = []
    baseline = None
    try:
        mapping = {record.date: record for record in records}
        baseline = fit_rms_baseline(
            series.person_id, mapping, baseline_dates, rms_config(config.features)
        )
        for row in rows:
            record_day = row.date
            if record_day <= timeline.baseline.baseline_end:
                continue
            if record_day not in mapping:
                continue
            values = mapping[record_day].features
            usable = {
                name: value for name, value in values.items() if _finite(value)
            }
            if len(usable) < 2:
                continue
            rms_values.append((record_day, compute_deviation(usable, baseline).distance))
    except Exception as error:  # noqa: BLE001
        baseline = None
        rms_values = []
        rms_note = f"{type(error).__name__}: {error}"
    else:
        rms_note = None

    rms_by_day = dict(rms_values)
    rms_snapshot = rms_by_day.get(snapshot_day) if snapshot_day else (
        rms_values[-1][1] if rms_values else None
    )
    first_rms_alarm = None
    if rms_threshold is not None:
        for day, value in rms_values:
            if value >= rms_threshold:
                first_rms_alarm = day
                break
    return {
        "person_id": series.person_id,
        "source": series.source,
        **series.meta,
        "status": "scored" if snapshot is not None else "snapshot_unavailable",
        "baseline_end": timeline.baseline.baseline_end,
        "baseline_days_used": len(timeline.baseline.baseline_dates),
        "monitoring_days": len(rows),
        "scored_days": len(scored),
        "snapshot_date": snapshot.date if snapshot else None,
        "T": snapshot.statistic_t if snapshot else None,
        "q": snapshot.q if snapshot else None,
        "C": snapshot.cusum if snapshot else None,
        "omnibus_D": snapshot.omnibus_d if snapshot else None,
        "mask": ",".join(snapshot.mask) if snapshot else None,
        "excluded_missing_features": timeline.summary["excluded_missing_features"],
        "excluded_missing_se": timeline.summary["excluded_missing_se"],
        "cusum_resets": timeline.summary["cusum_resets"],
        "max_C": max((row.cusum or 0.0) for row in scored) if scored else None,
        "max_q": max((row.q or 0.0) for row in scored) if scored else None,
        "rms_snapshot": rms_snapshot,
        "rms_max": max(value for _, value in rms_values) if rms_values else None,
        "rms_first_alarm_day": first_rms_alarm,
        "rms_note": rms_note,
        "daily": [
            {
                "day": row.date,
                "q": row.q,
                "C": row.cusum,
                "T": row.statistic_t,
                "status": row.status,
                "omnibus_D": row.omnibus_d,
                "omnibus_p": row.omnibus_p,
                "rms": rms_by_day.get(row.date),
                "alarm": (
                    None if alarm_threshold is None or row.cusum is None
                    else bool(row.cusum >= alarm_threshold)
                ),
            }
            for row in rows
        ],
    }


def level_and_change(
    series: PersonSeries,
    baseline_indices: Sequence[int],
    monitoring_indices: Sequence[int],
    *,
    features: Sequence[str] = R2_FEATURES,
) -> dict[str, float]:
    """绝对水平（监测块均值）与朴素变化（块均值 − 基线均值）。"""
    out: dict[str, float] = {}
    for name in features:
        values = np.asarray(series.features[name], dtype=float)
        baseline = values[list(baseline_indices)]
        monitoring = values[list(monitoring_indices)]
        baseline = baseline[np.isfinite(baseline)]
        monitoring = monitoring[np.isfinite(monitoring)]
        level = float(monitoring.mean()) if len(monitoring) else float("nan")
        base = float(baseline.mean()) if len(baseline) else float("nan")
        out[f"level_{name}"] = level
        out[f"change_{name}"] = level - base if math.isfinite(level) and math.isfinite(base) else float("nan")
    return out


def calibrate_alarm_thresholds(
    stable_rows: Sequence[Mapping[str, Any]],
    *,
    target_rate: float = 0.10,
) -> dict[str, Any]:
    """给 RMS 与 DS-PCD 同等机会，各标定“30 天内至少一次误报”=10%。

    阈值取稳定序列上“至少一次越界”比例为目标的经验分位点；若无法达到目标，
    保留可达到的最近水平并记录原因。
    """
    result: dict[str, Any] = {"target_rate": target_rate, "methods": {}}
    for method, column in (("rms", "rms_max"), ("ds_pcd", "max_C")):
        values = np.asarray(
            [float(row[column]) for row in stable_rows if row.get(column) is not None],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        entry: dict[str, Any] = {"usable_sequences": int(len(values))}
        if len(values) < 50:
            entry.update({"threshold": None, "reason": "too_few_stable_sequences"})
            result["methods"][method] = entry
            continue
        threshold = float(np.quantile(values, 1.0 - target_rate))
        achieved = float((values >= threshold).mean())
        # 分位点可能有并列值，用严格大于再校正一次
        if achieved > target_rate + 0.005:
            above = values[values > threshold]
            if len(above):
                threshold = float(above.min())
                achieved = float((values >= threshold).mean())
        entry.update({
            "threshold": threshold,
            "achieved_rate": achieved,
            "quantile_median": float(np.median(values)),
            "quantile_p90": float(np.quantile(values, 0.9)),
            "max": float(values.max()),
        })
        if achieved > target_rate + 0.01:
            entry["reason"] = "discrete_ties_prevent_lower_rate"
        result["methods"][method] = entry
    return result


def first_alarm_day(
    daily: Sequence[Mapping[str, Any]], *, method: str, threshold: float | None
) -> str | None:
    """按标定阈值返回首次报警日；无阈值或无报警返回 None。"""
    if threshold is None:
        return None
    key = "rms" if method == "rms" else "C"
    for row in daily:
        value = row.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric) and numeric >= threshold:
            return str(row.get("day"))
    return None
