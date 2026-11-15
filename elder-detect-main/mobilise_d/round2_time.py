"""阶段 16–18：短窗预测的时间边界定义与合格键重建。

本模块只做三件事，都被 16→17→18 共用：

1. **同日跌倒排除。** 预测日 ``t`` 当天只要有已记录的跌倒，整个预测日就从
   训练与评价中剔除（即使次日还有一次跌倒）。原始跌倒日期不改写。
2. **三个时段的口径。** ①评分/消融只读截至 ``t``（含当天）的观测；
   ②``monitoring_window`` 为基线结束后的第 1–30 个自然日，其后 7 天只用于
   判定结局；③已知变化的检出窗从变化起始日 ``u`` 起算，H 个自然日对应
   ``[u, u+H)``，与前瞻跌倒窗分开命名。
3. **合格键与边界样例。** 从原始事件日期与日记覆盖重建 1/3/7 天标签，
   输出带排除原因的键表与边界样例，供阶段 17/18 直接读取。

真实与模拟标签、人数、预测时点分别统计；不涉及模型训练或性能计算。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

HORIZONS_DAYS: tuple[int, ...] = (1, 3, 7)
PRIMARY_HORIZON_DAYS: int = 7
ALARM_WINDOW_DAYS: int = 30
OUTCOME_TAIL_DAYS: int = 7
KEYS_VERSION: str = "round2-keys-v2-sameday-excluded"
SCORING_VERSION: str = "round2-scoring-v2-truncated"
THRESHOLD_VERSION: str = "round2-threshold-v2-monitoring30"

# 只用于核对旧复核数字，不参与任何计算。
REFERENCE_CHECK: dict[str, Any] = {
    "old_real_test_rows": 172,
    "old_real_test_people": 134,
    "reference_rows_after_same_day_removal": 171,
    "reference_people_after_same_day_removal": 133,
    "reference_positives_1_3_7": [0, 1, 3],
    "old_synthetic_test_same_day_rows": 117,
    "old_threshold_30d_false_alarm_ds_pcd": 0.1062,
    "old_threshold_30d_false_alarm_rms": 0.2029,
}


def window_definitions() -> dict[str, Any]:
    """冻结的三个时段口径与端点（阶段 17/18 引用此文件）。"""
    return {
        "keys_version": KEYS_VERSION,
        "scoring_version": SCORING_VERSION,
        "threshold_version": THRESHOLD_VERSION,
        "horizons_days": list(HORIZONS_DAYS),
        "primary_horizon_days": PRIMARY_HORIZON_DAYS,
        "observation_cutoff": {
            "rule": "所有主模型、消融与评分只读取日期 <= t 的观测（含 t 当天）",
            "t_definition": "该日观测结束后的评估；1 天表示下一自然日的风险",
            "forbidden": "t 之后的步态、精度、q/C 与事件真值不得进入 t 的输入",
            "check": "截断/追加/扰动 t 之后的步态后，t 当天输入与预测须一致",
        },
        "fall_window": {
            "rule": "结局为 (t, t+H] 内至少一次跌倒；t 当天已发生的跌倒不算未来事件",
            "same_day_exclusion": (
                "预测日 t 当天存在任何已记录跌倒时，整个预测日从训练与评价中剔除，"
                "即使 (t, t+H] 内还有一次跌倒"
            ),
            "same_day_evidence": "Clinical Data.zip / falls-reports.csv 的 falldate",
            "starts_after_t": True,
            "ends_inclusive": True,
        },
        "monitoring_window": {
            "start": "个人基线结束后的第 1 个自然日",
            "end": "第 30 个自然日（含）",
            "days": ALARM_WINDOW_DAYS,
            "definitions": {
                "false_alarm": "监测第 1–30 天内至少一次越过冻结阈值",
                "threshold_input": "RMS 取监测第 1–30 天内最大距离；DS-PCD 取同一窗口内最大 C",
            },
            "excluded": "第 31 天起（含 7 天结局尾部）的越界不计入误报，也不参与阈值标定",
        },
        "outcome_tail": {
            "start": "监测第 31 个自然日",
            "end": "第 37 个自然日（含）",
            "days": OUTCOME_TAIL_DAYS,
            "role": "只用于判定 (t, t+7] 结局，不进入报警阈值与误报统计",
        },
        "change_detection_window": {
            "onset": "潜在健康过程首次偏离基线均值超过 0.25|delta| 的自然日 u",
            "window": "H 个自然日对应 [u, u+H)，H 取 1/3/7",
            "naming": "change_detection_window，区别于 fall_window",
            "first_post_change_alarm": "变化后第一次报警计为检出",
            "pre_change_alarm": "变化前报警单独记录，不当作检出，也不取消后续检出资格",
            "censoring": "监测结束导致 [u, u+H) 越界时记录可观察天数与截尾数",
        },
    }


def dated_falls(clinical) -> pd.DataFrame:
    """原始包的逐次跌倒日期表（只保留有日期的记录）。"""
    reports = clinical.reports.copy()
    reports["participantid"] = reports["participantid"].astype(str)
    reports["visit_number"] = reports["visit_number"].astype(str).str.lower()
    reports["day"] = pd.to_datetime(reports["day"], errors="coerce").dt.normalize()
    reports["fall_day"] = reports["day"]
    out = reports.loc[reports["day"].notna(), [
        "participantid", "visit_number", "fallrptnum", "fall_day",
    ]].copy()
    return out.sort_values(["participantid", "fall_day"], kind="stable").reset_index(drop=True)


def diary_coverage(clinical) -> pd.DataFrame:
    """每个（人，访视）的日记覆盖截止日。"""
    diaries = clinical.diaries.copy()
    diaries["participantid"] = diaries["participantid"].astype(str)
    diaries["visit_number"] = diaries["visit_number"].astype(str).str.lower()
    diaries["diary_day"] = pd.to_datetime(diaries["day"], errors="coerce").dt.normalize()
    out = diaries[["participantid", "visit_number", "diary_day"]].dropna(subset=["diary_day"])
    return out.drop_duplicates(["participantid", "visit_number"]).reset_index(drop=True)


def rebuild_windows(
    candidates: pd.DataFrame,
    falls: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    horizons: Sequence[int] = HORIZONS_DAYS,
) -> pd.DataFrame:
    """从原始事件日期重建每个候选预测日的 1/3/7 天标签与覆盖状态。"""
    falls_by_person: dict[str, np.ndarray] = {}
    for person, group in falls.groupby("participantid", sort=False):
        falls_by_person[str(person)] = np.sort(
            group["fall_day"].to_numpy(dtype="datetime64[ns]")
        )
    coverage_lookup: dict[tuple[str, str], pd.Timestamp] = {}
    for row in coverage.itertuples(index=False):
        coverage_lookup[(str(row.participantid), str(row.visit_number))] = pd.Timestamp(row.diary_day)

    rows: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        person = str(row.participantid)
        landmark = pd.Timestamp(row.landmark_day).normalize()
        diary_visit = getattr(row, "diary_visit_number_x", None)
        if diary_visit is None or (isinstance(diary_visit, float) and math_isnan(diary_visit)):
            diary_visit = getattr(row, "diary_visit_number_y", None)
        diary_day = coverage_lookup.get((person, str(diary_visit).lower()))
        record: dict[str, Any] = {
            "participantid": person,
            "visit_number": str(row.visit_number).lower(),
            "landmark_day": landmark.date().isoformat(),
            "diary_visit_number": str(diary_visit).lower() if diary_visit is not None else None,
            "diary_day": diary_day.date().isoformat() if diary_day is not None else None,
            "n_dated_falls_total": int(len(falls_by_person.get(person, []))),
        }
        days = falls_by_person.get(person, np.array([], dtype="datetime64[ns]"))
        same_day = days[days == np.datetime64(landmark)]
        record["same_day_fall"] = bool(len(same_day))
        record["same_day_fall_dates"] = ",".join(
            pd.Timestamp(day).date().isoformat() for day in same_day
        ) or None
        for horizon in horizons:
            limit = (landmark + pd.Timedelta(days=horizon)).normalize()
            covered = bool(diary_day is not None and limit <= diary_day)
            in_window = days[(days > np.datetime64(landmark)) & (days <= np.datetime64(limit))]
            record[f"followup_ok_{horizon}d"] = covered
            record[f"y_{horizon}d"] = bool(len(in_window)) if covered else None
            record[f"first_event_{horizon}d"] = (
                pd.Timestamp(in_window.min()).date().isoformat() if len(in_window) else None
            )
            record[f"n_falls_in_window_{horizon}d"] = int(len(in_window)) if covered else None
            record[f"event_offset_days_{horizon}d"] = (
                int((pd.Timestamp(in_window.min()) - landmark).days) if len(in_window) else None
            )
            record[f"days_short_of_window_end_{horizon}d"] = (
                None if diary_day is None else max(0, int((limit - diary_day).days))
            )
        rows.append(record)
    return pd.DataFrame(rows)


def math_isnan(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def real_key_table(
    candidates: pd.DataFrame,
    windows: pd.DataFrame,
    scores: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    horizons: Sequence[int] = HORIZONS_DAYS,
) -> pd.DataFrame:
    """把候选、重建标签、可评分状态与人员折合成合格键表。"""
    scores = scores.copy()
    scores["participantid"] = scores["participantid"].astype(str)
    scores["landmark_day"] = pd.to_datetime(scores["landmark_day"]).dt.strftime("%Y-%m-%d")
    scored = scores.loc[scores["status"] == "scored", ["participantid", "landmark_day"]].copy()
    scored["precision_scorable"] = True
    status_map = dict(
        zip(
            zip(scores["participantid"], scores["landmark_day"]),
            scores["status"],
        )
    )
    frame = candidates.copy()
    frame["participantid"] = frame["participantid"].astype(str)
    frame["landmark_day"] = pd.to_datetime(frame["landmark_day"]).dt.strftime("%Y-%m-%d")
    frame["visit_number"] = frame["visit_number"].astype(str).str.lower()
    keep = ["participantid", "visit_number", "landmark_day", "diary_day", "diary_visit_number",
            "same_day_fall", "same_day_fall_dates", "n_dated_falls_total"]
    for horizon in horizons:
        keep.extend([f"followup_ok_{horizon}d", f"y_{horizon}d", f"first_event_{horizon}d",
                     f"n_falls_in_window_{horizon}d", f"event_offset_days_{horizon}d",
                     f"days_short_of_window_end_{horizon}d"])
    keep = [column for column in keep if column in windows.columns]
    frame = frame.merge(windows[keep], on=["participantid", "visit_number", "landmark_day"],
                        how="left", validate="one_to_one", suffixes=("", "_rebuilt"))
    frame["precision_status"] = [
        status_map.get((person, day), "not_scored")
        for person, day in zip(frame["participantid"], frame["landmark_day"])
    ]
    frame["precision_scorable"] = frame["precision_status"].eq("scored")
    fold_map = dict(zip(splits["participantid"].astype(str), splits["fold"].astype(int)))
    frame["fold"] = frame["participantid"].map(fold_map)

    reasons: list[str] = []
    for row in frame.itertuples(index=False):
        if not bool(row.r_branch_eligible):
            reasons.append("not_r_branch_candidate")
        elif not bool(row.baseline_eligible):
            reasons.append("baseline_not_ready")
        elif not bool(row.precision_scorable):
            reasons.append(f"precision:{row.precision_status}")
        elif bool(row.same_day_fall):
            reasons.append("same_day_fall")
        else:
            reasons.append("")
    frame["exclusion_reason"] = reasons
    frame["corrected_key"] = frame["exclusion_reason"].eq("")

    # 同窗口标签一致性（旧值只作对照，不参与更正）
    for horizon in horizons:
        old_column = f"y_{horizon}d_num"
        new_column = f"y_{horizon}d"
        if old_column in frame.columns:
            frame[f"y_old_{horizon}d"] = frame[old_column]
    return frame


def nested_and_boundary_audit(keys: pd.DataFrame, falls: pd.DataFrame) -> dict:
    """嵌套、同日与 t±边界归属的独立核验（全部从原始事件日重算）。"""
    usable = keys.loc[keys["corrected_key"]].copy()
    known = usable.dropna(subset=[f"y_{h}d" for h in HORIZONS_DAYS])
    violations = known.loc[
        (known["y_1d"].astype(float) > known["y_3d"].astype(float))
        | (known["y_3d"].astype(float) > known["y_7d"].astype(float))
    ]
    fall_lookup: dict[str, set[str]] = {}
    for person, group in falls.groupby("participantid", sort=False):
        fall_lookup[str(person)] = set(
            pd.to_datetime(group["fall_day"]).dt.strftime("%Y-%m-%d")
        )
    boundary_rows: list[dict[str, Any]] = []
    for row in keys.itertuples(index=False):
        landmark = pd.Timestamp(row.landmark_day)
        days = fall_lookup.get(str(row.participantid), set())
        flags = {}
        for offset in range(0, 9):
            day = (landmark + pd.Timedelta(days=offset)).date().isoformat()
            flags[offset] = day in days
        expected_1d = flags[1]
        expected_3d = any(flags[offset] for offset in (1, 2, 3))
        expected_7d = any(flags[offset] for offset in range(1, 8))
        boundary_rows.append({
            "participantid": row.participantid,
            "landmark_day": row.landmark_day,
            "exclusion_reason": row.exclusion_reason,
            "corrected_key": bool(row.corrected_key),
            **{f"fall_at_t_plus_{offset}": bool(flags[offset]) for offset in range(0, 9)},
            "expected_y_1d": expected_1d,
            "expected_y_3d": expected_3d,
            "expected_y_7d": expected_7d,
            "y_1d": None if pd.isna(getattr(row, "y_1d", None)) else bool(getattr(row, "y_1d")),
            "y_3d": None if pd.isna(getattr(row, "y_3d", None)) else bool(getattr(row, "y_3d")),
            "y_7d": None if pd.isna(getattr(row, "y_7d", None)) else bool(getattr(row, "y_7d")),
        })
    boundary = pd.DataFrame(boundary_rows)
    corrected = boundary.loc[boundary["corrected_key"]]
    label_violations = {}
    for horizon in HORIZONS_DAYS:
        expected = corrected[f"expected_y_{horizon}d"].astype(bool)
        observed = corrected[f"y_{horizon}d"].astype(bool)
        label_violations[f"{horizon}d"] = int((expected != observed).sum())
    same_day_and_next = boundary.loc[
        boundary["fall_at_t_plus_0"] & boundary["fall_at_t_plus_1"]
    ]
    return {
        "corrected_rows": int(len(usable)),
        "corrected_people": int(usable["participantid"].nunique()),
        "nested_violations": int(len(violations)),
        "boundary_label_violations": label_violations,
        "same_day_records_all_candidates": int(boundary["fall_at_t_plus_0"].sum()),
        "same_day_records_corrected": int(corrected["fall_at_t_plus_0"].sum()),
        "same_day_and_next_day_records": same_day_and_next[[
            "participantid", "landmark_day", "exclusion_reason"
        ]].to_dict("records"),
        "t_plus_8_only_records": int(
            boundary["fall_at_t_plus_8"].sum()
            - (boundary["fall_at_t_plus_8"] & boundary[[
                f"fall_at_t_plus_{offset}" for offset in range(1, 8)
            ]].any(axis=1)).sum()
        ),
        "boundary_table": boundary,
    }


def synthetic_same_day(scores: pd.DataFrame) -> pd.DataFrame:
    """模拟序列中事件恰在预测日的记录（须在训练与评价前剔除）。"""
    frame = scores.loc[scores["status"] == "scored"].copy()
    gap = pd.to_numeric(frame["event_gap_from_snapshot"], errors="coerce")
    frame["event_gap"] = gap
    excluded = frame.loc[gap == 0].copy()
    excluded["exclusion_reason"] = "same_day_event"
    return excluded[[
        "series_id", "fold", "mechanism", "noise_family", "trajectory",
        "event_day", "event_gap", "batch", "exclusion_reason",
    ] if "batch" in excluded.columns else [
        "series_id", "fold", "mechanism", "noise_family", "trajectory",
        "event_day", "event_gap", "exclusion_reason",
    ]]


def synthetic_key_counts(scores: pd.DataFrame, *, batch: str) -> dict:
    """模拟训练/测试的同日排除计数与既往事件说明。"""
    frame = scores.copy()
    gap = pd.to_numeric(frame["event_gap_from_snapshot"], errors="coerce")
    scored = frame["status"].eq("scored")
    same_day = scored & gap.eq(0)
    prior_event = scored & gap.lt(0)
    cells: dict[str, Any] = {}
    for (fold, mechanism, noise), group in frame.groupby(
        ["fold", "mechanism", "noise_family"], sort=True
    ):
        group_gap = pd.to_numeric(group["event_gap_from_snapshot"], errors="coerce")
        group_scored = group["status"].eq("scored")
        key = f"fold{int(fold)}|{mechanism}|{noise}"
        cells[key] = {
            "rows": int(len(group)),
            "scored": int(group_scored.sum()),
            "same_day_excluded": int((group_scored & group_gap.eq(0)).sum()),
            "prior_event_kept": int((group_scored & group_gap.lt(0)).sum()),
            "positives_1d": int((group_scored & group_gap.gt(0) & group_gap.le(1)).sum()),
            "positives_3d": int((group_scored & group_gap.gt(0) & group_gap.le(3)).sum()),
            "positives_7d": int((group_scored & group_gap.gt(0) & group_gap.le(7)).sum()),
        }
    return {
        "batch": batch,
        "rows": int(len(frame)),
        "scored": int(scored.sum()),
        "same_day_excluded": int(same_day.sum()),
        "prior_event_kept": int(prior_event.sum()),
        "cells": cells,
        "limitation": (
            "模拟事件过程只保存首次事件：预测日前的首次事件按既往史保留，"
            "不能表示同一人后续重复跌倒"
        ),
    }


def cohort_stage_table(
    keys: pd.DataFrame,
    participants: pd.DataFrame,
    *,
    horizons: Sequence[int] = HORIZONS_DAYS,
) -> pd.DataFrame:
    """真实分支逐阶段分母：候选→基线→可评分→同日排除→共同测试。"""
    pd_keys = keys.loc[keys["cohort"].eq("pd")].copy()
    stages = [
        ("older_pd_participants", None),
        ("candidate_landmarks", pd_keys),
        ("baseline_ready", pd_keys.loc[pd_keys["baseline_eligible"].astype(bool)]),
        ("precision_scorable", pd_keys.loc[
            pd_keys["baseline_eligible"].astype(bool) & pd_keys["precision_scorable"]
        ]),
        ("same_day_excluded", pd_keys.loc[
            pd_keys["baseline_eligible"].astype(bool) & pd_keys["precision_scorable"]
            & pd_keys["same_day_fall"]
        ]),
        ("common_test", pd_keys.loc[pd_keys["corrected_key"]]),
    ]
    rows: list[dict[str, Any]] = []
    for name, subset in stages:
        if subset is None:
            rows.append({
                "stage": name,
                "prediction_days": None,
                "people": int(len(participants)),
                **{f"positive_days_{h}d": None for h in horizons},
                **{f"positive_people_{h}d": None for h in horizons},
                **{f"fold{fold}_days": None for fold in (0, 1, 2)},
            })
            continue
        entry: dict[str, Any] = {
            "stage": name,
            "prediction_days": int(len(subset)),
            "people": int(subset["participantid"].nunique()),
        }
        for horizon in horizons:
            column = f"y_{horizon}d"
            positive = subset.loc[subset[column].eq(True)]
            entry[f"positive_days_{horizon}d"] = int(len(positive))
            entry[f"positive_people_{horizon}d"] = int(positive["participantid"].nunique())
        for fold in (0, 1, 2):
            entry[f"fold{fold}_days"] = int((subset["fold"] == fold).sum())
        rows.append(entry)
    return pd.DataFrame(rows)


def alarm_ledger(
    daily: pd.DataFrame,
    *,
    alarm_column: str,
    onset_offset: int | None = None,
    monitoring_days: int = ALARM_WINDOW_DAYS,
    change_windows: Sequence[int] = HORIZONS_DAYS,
    extended_days: int = ALARM_WINDOW_DAYS + OUTCOME_TAIL_DAYS,
) -> dict[str, Any]:
    """从逐日报警记录计算误报归属与变化后检出（阶段 17/18 共用）。

    ``daily`` 需含 ``monitoring_offset``（监测第 n 个自然日，1 起）与布尔报警列。
    误报只在监测第 1–30 天归属；变化检出以 ``onset_offset`` 为第一个机会，
    H 个自然日对应 ``[onset, onset+H-1]``，监测结束使其越界时记为截尾。
    """
    frame = daily.dropna(subset=[alarm_column]).copy()
    frame["monitoring_offset"] = pd.to_numeric(
        frame["monitoring_offset"], errors="coerce").astype("Int64")
    frame = frame.loc[frame["monitoring_offset"].notna()]
    alarms = frame.loc[frame[alarm_column].astype(bool)]
    within = alarms.loc[alarms["monitoring_offset"] <= monitoring_days]
    offsets = sorted({int(value) for value in within["monitoring_offset"]})
    record: dict[str, Any] = {
        "first_alarm_offset": offsets[0] if offsets else None,
        "false_alarm_within_monitoring": bool(offsets),
        "n_alarms_within_monitoring": int(len(offsets)),
        "alarms_after_monitoring": int(
            (alarms["monitoring_offset"] > monitoring_days).sum()
        ),
    }
    if onset_offset is None:
        return record
    onset = int(onset_offset)
    record["onset_offset"] = onset
    pre = [value for value in offsets if value < onset]
    post = [value for value in offsets if value >= onset]
    record["pre_change_alarm"] = bool(pre)
    record["first_pre_change_offset"] = pre[0] if pre else None
    record["first_post_change_offset"] = post[0] if post else None
    extended = sorted({
        int(value) for value in alarms["monitoring_offset"]
        if onset <= int(value) <= extended_days
    })
    record["first_post_change_extended_offset"] = extended[0] if extended else None
    for window in change_windows:
        last_observable = min(monitoring_days, onset + window - 1)
        observable = max(0, last_observable - onset + 1)
        record[f"observable_days_{window}d"] = int(observable)
        record[f"censored_{window}d"] = bool(onset + window - 1 > monitoring_days)
        hit = None
        for value in post:
            if value - onset <= window - 1:
                hit = value
                break
        record[f"detected_{window}d"] = bool(hit is not None)
        record[f"latency_days_{window}d"] = (
            int(hit - onset) if hit is not None else None
        )
        extended_hit = None
        for value in extended:
            if value - onset <= window - 1:
                extended_hit = value
                break
        record[f"detected_extended_{window}d"] = bool(extended_hit is not None)
    return record


def detection_ledger_from_frame(
    frame: pd.DataFrame,
    *,
    alarm_column: str,
    group_columns: Sequence[str] = ("series_id",),
    onset_column: str = "onset_offset",
    monitoring_days: int = ALARM_WINDOW_DAYS,
) -> pd.DataFrame:
    """按序列汇总报警台账（误报、变化前/后首次报警、1/3/7 天检出）。"""
    rows: list[dict[str, Any]] = []
    keys = list(group_columns)
    for key, group in frame.groupby(keys, sort=False, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        base = dict(zip(keys, values))
        onset = None
        if onset_column in group.columns:
            onset_values = pd.to_numeric(group[onset_column], errors="coerce").dropna()
            onset = int(onset_values.iloc[0]) if len(onset_values) else None
        record = alarm_ledger(
            group, alarm_column=alarm_column, onset_offset=onset,
            monitoring_days=monitoring_days,
        )
        rows.append({**base, **record})
    return pd.DataFrame(rows)


def scenario_examples() -> dict[str, pd.DataFrame]:
    """阶段 17 核验用的小样例：30/31 天越阈、变化前后报警、截尾窗口。"""
    def frame(rows: Sequence[tuple[int, bool]]) -> pd.DataFrame:
        return pd.DataFrame({
            "monitoring_offset": [item[0] for item in rows],
            "alarm": [item[1] for item in rows],
        })

    return {
        "alarm_day_30": frame([(29, False), (30, True)]),
        "alarm_day_31": frame([(30, False), (31, True)]),
        "pre_then_post": frame([(5, True), (10, False), (12, True), (14, False)]),
        "late_onset": frame([(5, True), (27, False), (28, True)]),
    }
