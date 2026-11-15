"""读取每日步态聚合表，并按发布方字段语义生成可用性与质量标记。

``Aggregated DMO Data.zip`` 的 ``daily_agg_filtered.csv`` 是发布方已过滤的
每日聚合表；``daily_agg_all.csv`` 保留被发布方过滤规则剔除的日期。
``daily_weartime.csv`` 给出同人同日的佩戴时长。三张表都以
``participantid`` + ``visit_number`` + ``measurement_date`` 为键。
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import pandas as pd

from .sources import nested_frames


# 发布方逐日聚合列。单位来源：Data Dictionary 的 wb/lwb 行
# （duration=sec、cadence=Steps/min、speed=m/s、stride length=m，
# stride duration=sec），逐日列的合计与方差单位由同一行语义推出。
# 例外：逐日 ``strlen_*_avg_d`` 实测为厘米（ws_30 ≈ strlen/100/strdur，
# 中位比值 0.97），与步行段级的米不同；该差异在 unit_consistency_checks
# 中记录，不做未声明的单位换算。
FIELD_SPECS: tuple[dict, ...] = (
    {"column": "wb_all_sum_d", "unit": "count", "role": "activity",
     "description": "当日步行段数量", "evidence": "daily wb count (dictionary wb rows)"},
    {"column": "walkdur_all_sum_d", "unit": "min", "role": "activity",
     "description": "当日累计步行时间", "evidence": "arithmetic check vs steps/cadence"},
    {"column": "wbsteps_all_sum_d", "unit": "count", "role": "activity",
     "description": "当日累计步数", "evidence": "daily step count"},
    {"column": "wbdur_all_avg_d", "unit": "s", "role": "activity",
     "description": "步行段时长均值", "evidence": "dictionary wb duration = sec"},
    {"column": "wbdur_all_p90_d", "unit": "s", "role": "activity",
     "description": "步行段时长 P90", "evidence": "dictionary wb duration = sec"},
    {"column": "wbdur_all_var_d", "unit": "s^2", "role": "activity",
     "description": "步行段时长方差", "evidence": "squared duration (dictionary wb duration = sec)"},
    {"column": "cadence_all_avg_d", "unit": "steps/min", "role": "feature",
     "description": "当日平均步频", "evidence": "dictionary averagecadence = Steps/min; 120/strdur check"},
    {"column": "strdur_all_avg_d", "unit": "s", "role": "feature",
     "description": "当日平均跨步周期（步态周期）", "evidence": "dictionary averagestrideduration = sec; 120/cadence check"},
    {"column": "cadence_all_var_d", "unit": "(steps/min)^2", "role": "activity",
     "description": "步频方差", "evidence": "variance of cadence"},
    {"column": "strdur_all_var_d", "unit": "s^2", "role": "activity",
     "description": "跨步周期方差", "evidence": "variance of stride duration"},
    {"column": "ws_1030_avg_d", "unit": "m/s", "role": "activity",
     "description": "10–30 秒步行段的平均步速", "evidence": "dictionary averagestridespeed = m/s"},
    {"column": "strlen_1030_avg_d", "unit": "cm", "role": "activity",
     "description": "10–30 秒步行段的平均步长（逐日列为厘米）",
     "evidence": "daily column measured in cm: ws ≈ strlen/100/strdur"},
    {"column": "wb_10_sum_d", "unit": "count", "role": "activity",
     "description": "≥10 秒步行段数量", "evidence": "release column name"},
    {"column": "ws_10_p90_d", "unit": "m/s", "role": "activity",
     "description": "≥10 秒步行段步速 P90", "evidence": "dictionary averagestridespeed = m/s"},
    {"column": "wb_30_sum_d", "unit": "count", "role": "activity",
     "description": "≥30 秒步行段数量（步速主特征的来源段）", "evidence": "release column name"},
    {"column": "ws_30_avg_d", "unit": "m/s", "role": "feature",
     "description": "≥30 秒步行段的平均步速", "evidence": "dictionary averagestridespeed = m/s; ws≈strlen/strdur check"},
    {"column": "strlen_30_avg_d", "unit": "cm", "role": "activity",
     "description": "≥30 秒步行段的平均步长（逐日列为厘米）",
     "evidence": "daily column measured in cm: ws ≈ strlen/100/strdur"},
    {"column": "cadence_30_avg_d", "unit": "steps/min", "role": "activity",
     "description": "≥30 秒步行段平均步频", "evidence": "dictionary averagecadence = Steps/min"},
    {"column": "strdur_30_avg_d", "unit": "s", "role": "activity",
     "description": "≥30 秒步行段平均跨步周期", "evidence": "dictionary averagestrideduration = sec"},
    {"column": "ws_30_p90_d", "unit": "m/s", "role": "activity",
     "description": "≥30 秒步行段步速 P90", "evidence": "dictionary averagestridespeed = m/s"},
    {"column": "cadence_30_p90_d", "unit": "steps/min", "role": "activity",
     "description": "≥30 秒步行段步频 P90", "evidence": "dictionary averagecadence = Steps/min"},
    {"column": "ws_30_var_d", "unit": "(m/s)^2", "role": "activity",
     "description": "≥30 秒步行段步速方差", "evidence": "variance of walking speed"},
    {"column": "strlen_30_var_d", "unit": "m^2", "role": "activity",
     "description": "≥30 秒步行段步长方差", "evidence": "variance of stride length"},
    {"column": "wb_60_sum_d", "unit": "count", "role": "activity",
     "description": "≥60 秒步行段数量", "evidence": "release column name"},
    {"column": "total_worn_h_d", "unit": "h", "role": "quality",
     "description": "当日传感器佩戴小时数", "evidence": "release wear-time column; 24.0 = full day"},
    {"column": "total_worn_during_waking_h_d", "unit": "h", "role": "quality",
     "description": "当日清醒时段佩戴小时数", "evidence": "release wear-time column; 15.0 = waking window"},
)

FEATURE_COLUMNS: tuple[str, ...] = ("cadence_all_avg_d", "strdur_all_avg_d", "ws_30_avg_d")
WEAR_COLUMNS: tuple[str, ...] = ("total_worn_h_d", "total_worn_during_waking_h_d")


def field_mapping() -> dict:
    """字段映射：来源列名、单位、角色、证据。"""
    return {
        "source_table": "Aggregated DMO Data/T{1..5} Aggregated DMO Data.zip::cvs-T*-daily_agg_filtered.csv",
        "wear_time_table": "Aggregated DMO Data/T{1..5} Aggregated DMO Data.zip::cvs-T*-daily_weartime.csv",
        "join_keys": ["participantid", "visit_number", "measurement_date"],
        "fields": list(FIELD_SPECS),
        "features": list(FEATURE_COLUMNS),
        "quality_fields": list(WEAR_COLUMNS),
        "notes": [
            "Data Dictionary 只逐条定义步行段级 DMO（wb/lwb：duration=sec、cadence=Steps/min、"
            "speed=m/s、stride length=m、stride duration=sec），未逐条定义逐日列名；"
            "逐日列单位按同一语义推出，并用表内算术关系核验。",
            "ws_30_avg_d 只使用 ≥30 秒的步行段，缺少该长度步行段的日子为空值。",
            "逐日 strlen_*_avg_d 的单位是厘米；Data Dictionary 的米是步行段级定义。",
        ],
    }


def _read_frames(path: Path, suffix: str) -> Iterable[pd.DataFrame]:
    for _, payload in nested_frames(path, suffix):
        yield pd.read_csv(BytesIO(payload), dtype=str)


@dataclass(frozen=True)
class DailyData:
    """每日步态表与加载统计。"""

    frame: pd.DataFrame
    all_days: int
    filtered_days: int
    wear_rows: int
    notes: dict


def _prepare_wear(wear: pd.DataFrame) -> pd.DataFrame:
    wear = wear.rename(columns={
        "total_worn_h": "wear_h_total",
        "total_worn_during_waking_h": "wear_h_waking",
    }).copy()
    wear["visit_number"] = wear["visit_number"].astype(str).str.strip().str.lower()
    wear["participantid"] = wear["participantid"].astype(str).str.strip()
    wear["day"] = pd.to_datetime(wear["measurement_date"], errors="coerce").dt.normalize()
    wear["wear_h_total"] = pd.to_numeric(wear["wear_h_total"], errors="coerce")
    wear["wear_h_waking"] = pd.to_numeric(wear["wear_h_waking"], errors="coerce")
    return wear.drop(columns=["measurement_date"])


def _prepare_daily(raw: pd.DataFrame, wear: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["visit_number"] = frame["visit_number"].astype(str).str.strip().str.lower()
    frame["participantid"] = frame["participantid"].astype(str).str.strip()
    frame["day"] = pd.to_datetime(frame["measurement_date"], errors="coerce").dt.normalize()
    for spec in FIELD_SPECS:
        frame[spec["column"]] = pd.to_numeric(frame[spec["column"]], errors="coerce")
    frame = frame.loc[frame["day"].notna()].copy()
    frame = frame.merge(
        wear, on=["participantid", "visit_number", "day"], how="left", validate="one_to_one"
    )
    features = frame[list(FEATURE_COLUMNS)]
    frame["has_core_gait"] = features.notna().all(axis=1) & features.gt(0).all(axis=1)
    frame["has_walk_bout"] = frame["wb_all_sum_d"].fillna(0).gt(0)
    return frame.sort_values(
        ["participantid", "day", "visit_number"], kind="stable"
    ).reset_index(drop=True)


def load_daily(path: Path) -> DailyData:
    """读取发布方 filtered/all 逐日聚合与佩戴时长表。"""
    filtered = pd.concat(list(_read_frames(path, "daily_agg_filtered.csv")), ignore_index=True)
    all_days = pd.concat(list(_read_frames(path, "daily_agg_all.csv")), ignore_index=True)
    wear_raw = pd.concat(list(_read_frames(path, "daily_weartime.csv")), ignore_index=True)
    wear = _prepare_wear(wear_raw)
    frame = _prepare_daily(filtered, wear)

    all_days[["participantid", "day"]] = all_days[["participantid", "measurement_date"]].copy()
    all_days["day"] = pd.to_datetime(all_days["day"], errors="coerce").dt.normalize()
    all_repeated_person_days = int(all_days.duplicated(["participantid", "day"]).sum())
    duplicate_days = int(frame.duplicated(["participantid", "day"]).sum())
    spanning = frame.loc[frame.duplicated(["participantid", "day"], keep=False)]
    contradictory_duplicates = int(
        spanning.groupby(["participantid", "day"])["visit_number"].nunique().gt(1).sum()
    )
    notes = {
        "daily_rows_filtered": int(len(frame)),
        "daily_rows_all": int(len(all_days)),
        "daily_rows_excluded_by_release_filter": int(len(all_days) - len(filtered)),
        "daily_people_filtered": int(frame["participantid"].nunique()),
        "daily_people_all": int(all_days["participantid"].nunique()),
        "wear_rows": int(len(wear)),
        "duplicate_person_day_rows": duplicate_days,
        "duplicate_person_day_rows_in_all_table": all_repeated_person_days,
        "person_days_spanning_two_visits": contradictory_duplicates,
        "core_gait_days": int(frame["has_core_gait"].sum()),
        "core_gait_people": int(frame.loc[frame["has_core_gait"], "participantid"].nunique()),
        "core_gait_days_missing_ws30": int(frame["ws_30_avg_d"].isna().sum()),
        "wear_merge_missing": int(frame["wear_h_waking"].isna().sum()),
        "wear_vs_agg_max_abs_diff_waking_h": float(
            (frame["wear_h_waking"] - frame["total_worn_during_waking_h_d"]).abs().max()
        ),
    }
    return DailyData(
        frame=frame,
        all_days=int(len(all_days)),
        filtered_days=int(len(filtered)),
        wear_rows=int(len(wear)),
        notes=notes,
    )


def load_all_days(path: Path) -> DailyData:
    """读取未套用发布方过滤规则的逐日表，用于过滤规则敏感性说明。"""
    all_days = pd.concat(list(_read_frames(path, "daily_agg_all.csv")), ignore_index=True)
    wear_raw = pd.concat(list(_read_frames(path, "daily_weartime.csv")), ignore_index=True)
    frame = _prepare_daily(all_days, _prepare_wear(wear_raw))
    return DailyData(
        frame=frame,
        all_days=int(len(frame)),
        filtered_days=int(len(frame)),
        wear_rows=int(len(wear_raw)),
        notes={"daily_rows_all_days": int(len(frame))},
    )


def unit_consistency_checks(frame: pd.DataFrame) -> dict:
    """用表内算术关系核验单位推断，而不是把推断写入配置就当作事实。"""
    valid = frame.loc[frame["has_core_gait"]].copy()
    checks = {
        "rows_checked": int(len(valid)),
        "median_cadence_steps_per_min": float(valid["cadence_all_avg_d"].median()),
        "median_stride_duration_s": float(valid["strdur_all_avg_d"].median()),
        "median_walking_speed_m_per_s": float(valid["ws_30_avg_d"].median()),
        "median_stride_length_cm": float(valid["strlen_30_avg_d"].median()),
        "median_daily_steps": float(valid["wbsteps_all_sum_d"].median()),
        "median_walking_minutes": float(valid["walkdur_all_sum_d"].median()),
        "median_wear_waking_h": float(valid["wear_h_waking"].median()),
    }
    implied_cadence = 120.0 / valid["strdur_all_avg_d"]
    cadence_ratio = (valid["cadence_all_avg_d"] / implied_cadence).dropna()
    checks["cadence_vs_120_over_stride"] = {
        "median_ratio": float(cadence_ratio.median()),
        "iqr_ratio": [float(cadence_ratio.quantile(0.25)), float(cadence_ratio.quantile(0.75))],
        "agrees_within_20pct_share": float(((cadence_ratio - 1).abs() <= 0.2).mean()),
    }
    implied_minutes = valid["wbsteps_all_sum_d"] / valid["cadence_all_avg_d"]
    walk_ratio = (valid["walkdur_all_sum_d"] / implied_minutes).replace([float("inf")], pd.NA).dropna()
    checks["walking_minutes_vs_steps_over_cadence"] = {
        "median_ratio": float(walk_ratio.median()),
        "agrees_within_20pct_share": float(((walk_ratio - 1).abs() <= 0.2).mean()),
    }
    implied_speed = valid["strlen_30_avg_d"] / 100.0 / valid["strdur_30_avg_d"]
    speed_ratio = (valid["ws_30_avg_d"] / implied_speed).replace([float("inf")], pd.NA).dropna()
    checks["speed_vs_stride_length_cm_over_duration"] = {
        "median_ratio": float(speed_ratio.median()),
        "agrees_within_20pct_share": float(((speed_ratio - 1).abs() <= 0.2).mean()),
        "unit_note": "逐日 strlen_30_avg_d 按厘米解读后与 ws_30_avg_d 一致；字典的米是段级单位。",
    }
    checks["wear_within_waking_window_share"] = float(
        (valid["wear_h_waking"] <= valid["wear_h_total"]).mean()
    )
    return checks
