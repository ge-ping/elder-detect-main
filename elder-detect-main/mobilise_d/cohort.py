"""从每日步态与临床表构造可评估队列、候选预测时点与未来跌倒结局。

时间与身份约定（与实施计划一致）：

* 稳定键是 ``participantid`` + 观测日 + ``visit_number``（t1..t5）。
* 预测日 t 取每次有效步态采集最后一天；个人基线只使用 t 之前最早
  ``baseline_days`` 个合格日期。
* 结局时窗为 (t, t+H]；同日跌倒不计入。只有日记总数与逐次报告一致、
  且逐次报告日期都落在日记覆盖区间内，无事件记录才算阴性。
* 7/30 天窗口都必须被日记覆盖到 ``t+H``，否则记为随访不足而不是阴性。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from .clinical import ClinicalData
from .daily import FEATURE_COLUMNS, WEAR_COLUMNS


@dataclass(frozen=True)
class CohortConfig:
    """阶段 1 冻结的人群与结局口径。"""

    min_age: float = 65.0
    cohort_column: str = "cohort pd"
    baseline_days: int = 14
    alignment_days: int = 14
    diary_window_months: int = 6
    horizons: tuple[int, ...] = (7, 30)
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    wear_sensitivity_hours: tuple[float, ...] = (8.0, 10.0, 12.0)
    require_wear_hours: float | None = None

    def to_dict(self) -> dict:
        return {
            "min_age": self.min_age,
            "cohort_column": self.cohort_column,
            "baseline_days": self.baseline_days,
            "alignment_days": self.alignment_days,
            "diary_window_months": self.diary_window_months,
            "horizons": list(self.horizons),
            "feature_columns": list(self.feature_columns),
            "wear_sensitivity_hours": list(self.wear_sensitivity_hours),
            "require_wear_hours": self.require_wear_hours,
        }


@dataclass
class CohortResult:
    """阶段 1 的派生表与计数。"""

    participants: pd.DataFrame
    visits: pd.DataFrame
    daily: pd.DataFrame
    diaries: pd.DataFrame
    reports: pd.DataFrame
    gates: pd.DataFrame
    landmarks: pd.DataFrame
    baseline_days: pd.DataFrame
    windows: pd.DataFrame
    flow: pd.DataFrame
    counts: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)


GATE_COLUMNS: tuple[str, ...] = (
    "participantid", "visit_number", "previous_visit_number", "diary_day",
    "fallstotal", "fallsreports", "n_report_rows", "n_dated_reports",
    "n_report_rows_outside_window", "has_undated_report", "previous_visit_day",
    "covered_from", "landmark_day", "block_first_day", "n_valid_days_in_previous_visit",
    "block_first_offset_days", "landmark_offset_days", "has_previous_visit_date",
    "fall_reports_consistent", "previous_visit_has_gait", "monitoring_aligned",
    "landmark_in_diary_window", "candidate", "exclusion_reason",
)

LANDMARK_COLUMNS: tuple[str, ...] = (
    "participantid", "landmark_day", "visit_number", "visit_number_index",
    "diary_visit_number", "diary_day", "covered_from",
    "days_from_previous_visit_to_landmark", "n_valid_days_in_block",
    "block_first_day", "block_last_day", "block_max_gap_days",
    "block_gaps_over_7_days", "block_median_gap_days", "n_prior_valid_days",
    "baseline_eligible", "baseline_days_used", "baseline_first_day",
    "baseline_end_day", "baseline_span_days", "baseline_max_gap_days",
    "days_from_baseline_end_to_landmark", "monitoring_start_day",
    "n_monitored_days_in_block", "landmark_cadence", "landmark_stride_duration_s",
    "landmark_walking_speed_m_s", "landmark_wear_waking_h",
    "block_min_wear_waking_h", "block_days_wear_lt_10h",
)

BASELINE_COLUMNS: tuple[str, ...] = (
    "participantid", "landmark_day", "baseline_index", "baseline_day",
)

WINDOW_COLUMNS: tuple[str, ...] = (
    "participantid", "visit_number", "diary_visit_number", "landmark_day",
    "covered_from", "diary_day", "coverage_evidence", "horizon_days", "status",
    "window_end", "diary_covers_until", "days_short_of_window_end", "event",
    "n_reports_in_window", "first_event_day", "baseline_eligible",
)


def diary_window(
    previous_visit_day: pd.Timestamp, diary_day: pd.Timestamp, months: int
) -> pd.Timestamp:
    """日记覆盖区间下界：上次访视日与访视日前 N 个月的较晚者。"""
    return max(previous_visit_day, (diary_day - pd.DateOffset(months=months)).normalize())


def gap_summary(days: np.ndarray) -> dict:
    """相邻观测日的间隔统计，供间断采样规则使用。"""
    if len(days) < 2:
        return {"max_gap_days": 0, "gaps_over_7_days": 0, "median_gap_days": 0.0}
    gaps = np.diff(np.sort(days)).astype("timedelta64[D]").astype(float)
    return {
        "max_gap_days": int(gaps.max()),
        "gaps_over_7_days": int((gaps > 7).sum()),
        "median_gap_days": float(np.median(gaps)),
    }


def _visit_days(visits: pd.DataFrame) -> dict[tuple[str, str], pd.Timestamp]:
    lookup: dict[tuple[str, str], pd.Timestamp] = {}
    for person, visit, day in zip(visits["participantid"], visits["visit_number"], visits["day"]):
        if pd.isna(day):
            continue
        key = (str(person), str(visit))
        if key in lookup and lookup[key] != day:
            raise ValueError(f"访视日期不唯一：{key}")
        lookup[key] = day
    return lookup


def _gait_days(daily: pd.DataFrame) -> tuple[dict, dict]:
    by_person_visit: dict[tuple[str, str], np.ndarray] = {}
    valid = daily.loc[daily["has_core_gait"], ["participantid", "visit_number", "day"]]
    for (person, visit), group in valid.groupby(["participantid", "visit_number"], sort=False):
        by_person_visit[(str(person), str(visit))] = np.sort(
            group["day"].to_numpy(dtype="datetime64[ns]")
        )
    by_person: dict[str, np.ndarray] = {}
    for person, group in valid.groupby("participantid", sort=False):
        by_person[str(person)] = np.sort(group["day"].to_numpy(dtype="datetime64[ns]"))
    return by_person_visit, by_person


def _report_days(reports: pd.DataFrame) -> dict[tuple[str, str], list[pd.Timestamp]]:
    lookup: dict[tuple[str, str], list[pd.Timestamp]] = {}
    for (person, visit), group in reports.groupby(["participantid", "visit_number"], sort=False):
        lookup[(str(person), str(visit))] = list(group["day"])
    return lookup


def _window_rows(
    landmark: pd.Timestamp,
    diary_day: pd.Timestamp,
    horizons: tuple[int, ...],
    report_days: list[pd.Timestamp],
    baseline_eligible: bool,
) -> list[dict]:
    dated = [day.normalize() for day in report_days if pd.notna(day)]
    rows = []
    for horizon in horizons:
        limit = (landmark + timedelta(days=horizon)).normalize()
        covered = limit <= diary_day
        in_window = [day for day in dated if landmark < day <= limit]
        rows.append({
            "horizon_days": horizon,
            "status": "eligible" if covered else "insufficient_followup",
            "window_end": limit.date().isoformat(),
            "diary_covers_until": diary_day.date().isoformat(),
            "days_short_of_window_end": max(0, (limit - diary_day).days),
            "event": bool(in_window) if covered else None,
            "n_reports_in_window": len(in_window),
            "first_event_day": min(in_window).date().isoformat() if in_window else None,
            "baseline_eligible": bool(baseline_eligible),
        })
    return rows


def _person_summary(daily: pd.DataFrame) -> pd.DataFrame:
    valid = daily.loc[daily["has_core_gait"]]
    summary = valid.groupby("participantid", sort=False).agg(
        n_valid_days=("day", "nunique"),
        n_visits_with_gait=("visit_number", "nunique"),
        first_valid_day=("day", "min"),
        last_valid_day=("day", "max"),
    ).reset_index()
    summary["first_valid_day"] = summary["first_valid_day"].dt.date.astype(str)
    summary["last_valid_day"] = summary["last_valid_day"].dt.date.astype(str)
    return summary


def build_cohort(
    clinical: ClinicalData,
    daily: pd.DataFrame,
    config: CohortConfig,
) -> CohortResult:
    """构造候选预测时点、结局窗口与筛选计数。

    ``daily`` 需包含 ``has_core_gait`` 列；``config.require_wear_hours`` 非空时
    再按当日清醒佩戴时长收缩有效日集合（用于佩戴质控敏感性分析）。
    """
    features = list(config.feature_columns)
    frame = daily.copy()
    frame["participantid"] = frame["participantid"].astype(str)
    frame["visit_number"] = frame["visit_number"].astype(str).str.lower()
    frame["has_core_gait"] = (
        frame[features].notna().all(axis=1) & frame[features].gt(0).all(axis=1)
    )
    if config.require_wear_hours is not None:
        frame = frame.loc[
            frame["has_core_gait"] & frame[WEAR_COLUMNS[1]].ge(config.require_wear_hours)
        ].copy()

    participants = clinical.participants.copy()
    participants["participantid"] = participants["participantid"].astype(str)
    participants["is_pd"] = participants[config.cohort_column].fillna("F").eq("T")
    participants["older_pd"] = participants["is_pd"] & participants["age"].ge(config.min_age)
    participants = participants.loc[participants["older_pd"]].copy()
    target = participants["participantid"].tolist()
    target_set = set(target)

    frame = frame.loc[frame["participantid"].isin(target_set)].copy()
    visits = clinical.visits.copy()
    visits["participantid"] = visits["participantid"].astype(str)
    visits["visit_number"] = visits["visit_number"].astype(str).str.lower()
    visits = visits.loc[visits["participantid"].isin(target_set)].copy()
    reports = clinical.reports.copy()
    reports["participantid"] = reports["participantid"].astype(str)
    reports["visit_number"] = reports["visit_number"].astype(str).str.lower()
    reports = reports.loc[reports["participantid"].isin(target_set)].copy()
    diaries = clinical.diaries.copy()
    diaries["participantid"] = diaries["participantid"].astype(str)
    diaries["visit_number"] = diaries["visit_number"].astype(str).str.lower()
    diaries = diaries.loc[diaries["participantid"].isin(target_set)].copy()

    visit_lookup = _visit_days(visits)
    by_person_visit, by_person = _gait_days(frame)
    reports_lookup = _report_days(reports)

    participants = participants.merge(_person_summary(frame), on="participantid", how="left")
    report_counts = (
        reports.loc[reports["day"].notna()]
        .groupby("participantid")["day"].nunique()
        .rename("n_dated_fall_reports")
        .reset_index()
    )
    participants = participants.merge(report_counts, on="participantid", how="left")
    participants["n_valid_days"] = participants["n_valid_days"].fillna(0).astype(int)
    participants["n_visits_with_gait"] = participants["n_visits_with_gait"].fillna(0).astype(int)
    participants["n_dated_fall_reports"] = participants["n_dated_fall_reports"].fillna(0).astype(int)

    gate_rows: list[dict] = []
    landmark_rows: list[dict] = []
    baseline_rows: list[dict] = []
    window_rows: list[dict] = []

    for row in diaries.itertuples(index=False):
        person = str(row.participantid)
        visit = str(row.visit_number).lower()
        diary_day = row.day
        digit = visit[1:]
        previous = "t" + str(int(digit) - 1) if digit.isdigit() else None
        previous_day = visit_lookup.get((person, previous)) if previous else None

        report_list = reports_lookup.get((person, visit), [])
        dated_reports = [day.normalize() for day in report_list if pd.notna(day)]
        has_undated = len(dated_reports) != len(report_list)
        has_window = pd.notna(previous_day) and pd.notna(diary_day) and previous_day < diary_day
        covered_from = (
            diary_window(previous_day, diary_day, config.diary_window_months)
            if has_window else None
        )
        reports_outside = (
            sum(1 for day in dated_reports if not (covered_from <= day <= diary_day))
            if covered_from is not None else None
        )
        total_agrees = pd.notna(row.fallstotal) and int(row.fallstotal) == len(report_list)
        reports_consistent = bool(
            has_window and total_agrees and not has_undated and reports_outside == 0
        )

        block = by_person_visit.get((person, previous)) if previous else None
        block_first = pd.Timestamp(block[0]).normalize() if block is not None and len(block) else None
        block_last = pd.Timestamp(block[-1]).normalize() if block is not None and len(block) else None
        block_offset = int((block_first - previous_day).days) if block_first is not None and has_window else None
        aligned = bool(block_offset is not None and abs(block_offset) <= config.alignment_days)
        in_window = bool(
            covered_from is not None and block_last is not None
            and covered_from <= block_last < diary_day
        )

        gates = {
            "has_previous_visit_date": bool(has_window),
            "fall_reports_consistent": reports_consistent,
            "previous_visit_has_gait": bool(block is not None and len(block)),
            "monitoring_aligned": aligned,
            "landmark_in_diary_window": in_window,
        }
        candidate = all(gates.values())
        if not gates["has_previous_visit_date"]:
            reason = "missing_or_invalid_previous_visit"
        elif not gates["fall_reports_consistent"]:
            reason = "fall_reports_not_usable"
        elif not gates["previous_visit_has_gait"]:
            reason = "no_valid_gait_in_previous_visit"
        elif not gates["monitoring_aligned"]:
            reason = "monitoring_not_aligned_with_previous_visit"
        elif not gates["landmark_in_diary_window"]:
            reason = "landmark_outside_diary_window"
        else:
            reason = None

        gate_rows.append({
            "participantid": person,
            "visit_number": visit,
            "previous_visit_number": previous,
            "diary_day": diary_day.date().isoformat() if pd.notna(diary_day) else None,
            "fallstotal": None if pd.isna(row.fallstotal) else int(row.fallstotal),
            "fallsreports": None if pd.isna(row.fallsreports) else float(row.fallsreports),
            "n_report_rows": len(report_list),
            "n_dated_reports": len(dated_reports),
            "n_report_rows_outside_window": reports_outside,
            "has_undated_report": bool(has_undated),
            "previous_visit_day": previous_day.date().isoformat() if pd.notna(previous_day) else None,
            "covered_from": covered_from.date().isoformat() if covered_from is not None else None,
            "landmark_day": block_last.date().isoformat() if block_last is not None else None,
            "block_first_day": block_first.date().isoformat() if block_first is not None else None,
            "n_valid_days_in_previous_visit": int(len(block)) if block is not None else 0,
            "block_first_offset_days": block_offset,
            "landmark_offset_days": (
                int((block_last - previous_day).days)
                if block_last is not None and has_window else None
            ),
            **gates,
            "candidate": bool(candidate),
            "exclusion_reason": reason,
        })
        if not candidate:
            continue

        landmark = block_last
        person_days = by_person[person]
        prior = person_days[person_days < landmark.to_datetime64()]
        n_prior = int(len(prior))
        baseline_arr = prior[: config.baseline_days]
        baseline_eligible = n_prior >= config.baseline_days
        baseline_start = pd.Timestamp(baseline_arr[0]).normalize() if len(baseline_arr) else None
        baseline_end = pd.Timestamp(baseline_arr[-1]).normalize() if len(baseline_arr) else None
        block_gaps = gap_summary(block)
        baseline_gaps = gap_summary(baseline_arr)
        monitoring_start = (
            baseline_end + timedelta(days=1) if baseline_eligible else previous_day
        )
        monitored_days = [
            pd.Timestamp(day).normalize() for day in block
            if pd.Timestamp(day).normalize() >= monitoring_start
        ]
        landmark_row = frame.loc[
            (frame["participantid"] == person)
            & (frame["visit_number"] == previous)
            & (frame["day"] == landmark)
        ].iloc[0]
        block_frame = frame.loc[
            (frame["participantid"] == person)
            & (frame["visit_number"] == previous)
            & frame["day"].isin([pd.Timestamp(day).normalize() for day in block])
        ]
        landmark_rows.append({
            "participantid": person,
            "landmark_day": landmark.date().isoformat(),
            "visit_number": previous,
            "visit_number_index": int(previous[1:]),
            "diary_visit_number": visit,
            "diary_day": diary_day.date().isoformat(),
            "covered_from": covered_from.date().isoformat(),
            "days_from_previous_visit_to_landmark": int((landmark - previous_day).days),
            "n_valid_days_in_block": int(len(block)),
            "block_first_day": pd.Timestamp(block[0]).normalize().date().isoformat(),
            "block_last_day": landmark.date().isoformat(),
            "block_max_gap_days": block_gaps["max_gap_days"],
            "block_gaps_over_7_days": block_gaps["gaps_over_7_days"],
            "block_median_gap_days": block_gaps["median_gap_days"],
            "n_prior_valid_days": n_prior,
            "baseline_eligible": bool(baseline_eligible),
            "baseline_days_used": int(len(baseline_arr)),
            "baseline_first_day": baseline_start.date().isoformat() if baseline_start is not None else None,
            "baseline_end_day": baseline_end.date().isoformat() if baseline_end is not None else None,
            "baseline_span_days": (
                int((baseline_end - baseline_start).days) if baseline_start is not None else None
            ),
            "baseline_max_gap_days": baseline_gaps["max_gap_days"],
            "days_from_baseline_end_to_landmark": (
                int((landmark - baseline_end).days) if baseline_end is not None else None
            ),
            "monitoring_start_day": monitoring_start.date().isoformat(),
            "n_monitored_days_in_block": len(monitored_days),
            "landmark_cadence": float(landmark_row["cadence_all_avg_d"]),
            "landmark_stride_duration_s": float(landmark_row["strdur_all_avg_d"]),
            "landmark_walking_speed_m_s": float(landmark_row["ws_30_avg_d"]),
            "landmark_wear_waking_h": (
                None if pd.isna(landmark_row["wear_h_waking"]) else float(landmark_row["wear_h_waking"])
            ),
            "block_min_wear_waking_h": (
                None if block_frame["wear_h_waking"].isna().all()
                else float(block_frame["wear_h_waking"].min())
            ),
            "block_days_wear_lt_10h": int((block_frame["wear_h_waking"] < 10.0).sum()),
        })
        for index, baseline_day in enumerate(baseline_arr, start=1):
            baseline_rows.append({
                "participantid": person,
                "landmark_day": landmark.date().isoformat(),
                "baseline_index": index,
                "baseline_day": pd.Timestamp(baseline_day).normalize().date().isoformat(),
            })
        for item in _window_rows(
            landmark, diary_day, tuple(config.horizons),
            report_list, baseline_eligible,
        ):
            window_rows.append({
                "participantid": person,
                "visit_number": previous,
                "diary_visit_number": visit,
                "landmark_day": landmark.date().isoformat(),
                "covered_from": covered_from.date().isoformat(),
                "diary_day": diary_day.date().isoformat(),
                "coverage_evidence": (
                    f"falls diary {visit} dated {diary_day.date().isoformat()}; "
                    f"window starts {covered_from.date().isoformat()}; "
                    f"dated fall reports agree ({len(dated_reports)})"
                ),
                **item,
            })

    gates_frame = pd.DataFrame(gate_rows, columns=list(GATE_COLUMNS))
    landmarks = pd.DataFrame(landmark_rows, columns=list(LANDMARK_COLUMNS))
    baselines = pd.DataFrame(baseline_rows, columns=list(BASELINE_COLUMNS))
    windows = pd.DataFrame(window_rows, columns=list(WINDOW_COLUMNS))

    for table in (gates_frame, landmarks, baselines, windows):
        if not table.empty:
            table["participantid"] = table["participantid"].astype(str)
            sort_keys = [
                column for column in ("participantid", "landmark_day", "horizon_days", "visit_number")
                if column in table.columns
            ]
            table.sort_values(sort_keys, kind="stable", inplace=True)
            table.reset_index(drop=True, inplace=True)

    participants.attrs["release_participants"] = int(len(clinical.participants))
    participants.attrs["pd_participants"] = int(
        clinical.participants["cohort pd"].fillna("F").eq("T").sum()
    )
    participants.attrs["older_pd_with_valid_gait"] = int(frame["participantid"].nunique())
    counts = _counts(
        participants=participants,
        diaries=diaries,
        gates=gates_frame,
        landmarks=landmarks,
        windows=windows,
        reports=reports,
        frame=frame,
        all_participants=clinical.participants,
        config=config,
    )
    return CohortResult(
        participants=participants,
        visits=visits,
        daily=frame,
        diaries=diaries,
        reports=reports,
        gates=gates_frame,
        landmarks=landmarks,
        baseline_days=baselines,
        windows=windows,
        flow=_flow(participants, diaries, gates_frame, windows, config),
        counts=counts,
        notes={
            "participants_in_release": int(len(clinical.participants)),
            "pd_participants": int(clinical.participants["cohort pd"].fillna("F").eq("T").sum()),
            "diary_rows_outside_target_population": int(
                len(clinical.diaries) - len(diaries)
            ),
            "diary_rows_of_target_without_any_valid_gait_day": int(
                sum(1 for person in diaries["participantid"]
                    if person not in by_person)
            ),
            "visits_with_duplicate_person_visit": int(
                visits.duplicated(["participantid", "visit_number"]).sum()
            ),
            "diaries_with_duplicate_person_visit": int(
                diaries.duplicated(["participantid", "visit_number"]).sum()
            ),
        },
    )


def _counts(
    *,
    participants: pd.DataFrame,
    diaries: pd.DataFrame,
    gates: pd.DataFrame,
    landmarks: pd.DataFrame,
    windows: pd.DataFrame,
    reports: pd.DataFrame,
    frame: pd.DataFrame,
    all_participants: pd.DataFrame,
    config: CohortConfig,
) -> dict:
    counts = {
        "release_participants": int(len(all_participants)),
        "pd_participants": int(all_participants["cohort pd"].fillna("F").eq("T").sum()),
        "older_pd_participants": int(len(participants)),
        "older_pd_participants_with_valid_gait": int(frame["participantid"].nunique()),
        "daily_rows_in_target_population": int(len(frame)),
        "older_pd_diary_rows": int(len(diaries)),
        "dated_fall_reports_in_target_population": int(reports["day"].notna().sum()),
        "diary_rows_considered": int(len(gates)),
        "candidate_landmarks": int(len(landmarks)),
        "candidate_people": int(landmarks["participantid"].nunique()) if len(landmarks) else 0,
        "baseline_ready_landmarks": int(landmarks["baseline_eligible"].sum()) if len(landmarks) else 0,
        "baseline_ready_people": int(
            landmarks.loc[landmarks["baseline_eligible"], "participantid"].nunique()
        ) if len(landmarks) else 0,
        "horizons": {},
    }
    for horizon in config.horizons:
        part = windows.loc[(windows["horizon_days"] == horizon) & (windows["status"] == "eligible")]
        positive = part.loc[part["event"] == True]  # noqa: E712
        ready = part.loc[part["baseline_eligible"]]
        ready_positive = ready.loc[ready["event"] == True]  # noqa: E712
        counts["horizons"][f"{horizon}d"] = {
            "eligible_landmarks": int(len(part)),
            "eligible_people": int(part["participantid"].nunique()) if len(part) else 0,
            "positive_landmarks": int(len(positive)),
            "positive_people": int(positive["participantid"].nunique()) if len(positive) else 0,
            "insufficient_followup_landmarks": int(
                ((windows["horizon_days"] == horizon) & (windows["status"] == "insufficient_followup")).sum()
            ),
            "baseline_ready_eligible_landmarks": int(len(ready)),
            "baseline_ready_eligible_people": int(ready["participantid"].nunique()) if len(ready) else 0,
            "baseline_ready_positive_landmarks": int(len(ready_positive)),
            "baseline_ready_positive_people": (
                int(ready_positive["participantid"].nunique()) if len(ready_positive) else 0
            ),
        }
    return counts


def _flow(
    participants: pd.DataFrame,
    diaries: pd.DataFrame,
    gates: pd.DataFrame,
    windows: pd.DataFrame,
    config: CohortConfig,
) -> pd.DataFrame:
    """逐步筛选计数；顺序与上次核验口径一致，便于逐项对照。"""
    rows = []

    def add(step: str, unit: str, rows_in: int, rows_out: int, detail: str) -> None:
        rows.append({
            "step": step,
            "unit": unit,
            "rows_in": int(rows_in),
            "rows_out": int(rows_out),
            "removed": int(rows_in - rows_out),
            "detail": detail,
        })

    release = int(participants.attrs.get("release_participants", len(participants)))
    add("release_participants", "person", release, int(participants.attrs.get("pd_participants", release)),
        "Cohort Site 表中的受试者")
    add("older_pd", "person", int(participants.attrs.get("pd_participants", release)), len(participants),
        f"cohort pd == T 且基线年龄 >= {config.min_age:g} 岁")
    add("older_pd_with_valid_gait", "person", len(participants),
        int(participants.attrs.get("older_pd_with_valid_gait", int(participants["n_valid_days"].gt(0).sum()))),
        "至少有一个步频、跨步周期、步速齐全的有效观测日")
    add("older_pd_diary_rows", "diary row", len(diaries), len(gates),
        "目标人群在 falls-diaries 中的访视记录")
    if gates.empty:
        return pd.DataFrame(rows)

    steps = [
        ("has_previous_visit_date", "能对上上一次访视日期且早于日记日期"),
        ("fall_reports_consistent", "日记半年总数与逐次报告条数一致，报告日期都在覆盖区间"),
        ("previous_visit_has_gait", "上一次访视至少有一个有效步态日"),
        ("monitoring_aligned", f"采集首日与上次访视相差不超过 {config.alignment_days} 天"),
        ("landmark_in_diary_window", "采集最后一天落在日记覆盖的半年区间内"),
    ]
    for index, (column, detail) in enumerate(steps):
        subset = gates
        for prior_column, _ in steps[:index]:
            subset = subset.loc[subset[prior_column]]
        add(column, "diary row", len(subset), int(subset[column].sum()), detail)
    candidates = int(gates["candidate"].sum())
    add("candidate_landmarks", "landmark", candidates, candidates,
        "全部结局与覆盖条件通过，形成候选预测时点")
    if not windows.empty:
        for horizon in config.horizons:
            sub = windows["horizon_days"] == horizon
            covered = int((sub & windows["status"].eq("eligible")).sum())
            add(f"{horizon}d_window_covered", "landmark × horizon", int(sub.sum()), covered,
                f"日记覆盖到 t+{horizon} 天；未覆盖记为随访不足")
    return pd.DataFrame(rows)
