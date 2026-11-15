"""阶段 1 队列构造的单元测试（合成数据，不依赖网络与原始包）。"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mobilise_d.clinical import ClinicalData  # noqa: E402
from mobilise_d.cohort import (  # noqa: E402
    CohortConfig,
    _window_rows,
    build_cohort,
    diary_window,
    gap_summary,
)


def _day(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _daily(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["participantid"] = frame["participantid"].astype(str)
    frame["visit_number"] = frame["visit_number"].astype(str)
    frame["day"] = pd.to_datetime(frame["day"]).dt.normalize()
    for column, value in (
        ("cadence_all_avg_d", 90.0),
        ("strdur_all_avg_d", 1.3),
        ("ws_30_avg_d", 0.9),
        ("wb_all_sum_d", 200.0),
        ("total_worn_h_d", 24.0),
        ("total_worn_during_waking_h_d", 15.0),
        ("wear_h_total", 24.0),
        ("wear_h_waking", 15.0),
    ):
        frame[column] = frame.get(column, value)
    frame["has_core_gait"] = True
    return frame


def _clinical(
    participants: list[dict],
    visits: list[dict],
    diaries: list[dict],
    reports: list[dict],
) -> ClinicalData:
    participants_frame = pd.DataFrame(participants)
    participants_frame["participantid"] = participants_frame["participantid"].astype(str)
    visits_frame = pd.DataFrame(visits)
    visits_frame["participantid"] = visits_frame["participantid"].astype(str)
    visits_frame["visit_number"] = visits_frame["visit_number"].astype(str)
    visits_frame["day"] = pd.to_datetime(visits_frame["day"]).dt.normalize()
    diaries_frame = pd.DataFrame(diaries)
    diaries_frame["participantid"] = diaries_frame["participantid"].astype(str)
    diaries_frame["visit_number"] = diaries_frame["visit_number"].astype(str)
    diaries_frame["day"] = pd.to_datetime(diaries_frame["day"]).dt.normalize()
    reports_frame = pd.DataFrame(reports, columns=["participantid", "visit_number", "day"])
    reports_frame["participantid"] = reports_frame["participantid"].astype(str)
    reports_frame["visit_number"] = reports_frame["visit_number"].astype(str)
    reports_frame["day"] = pd.to_datetime(reports_frame["day"]).dt.normalize()
    return ClinicalData(
        participants=participants_frame,
        visits=visits_frame,
        diaries=diaries_frame,
        reports=reports_frame,
    )


def test_diary_window_uses_later_of_previous_visit_and_six_months():
    assert diary_window(_day("2024-01-01"), _day("2024-09-01"), 6) == _day("2024-03-01")
    assert diary_window(_day("2024-08-01"), _day("2024-09-01"), 6) == _day("2024-08-01")


def test_gap_summary_counts_long_gaps():
    days = np.array(["2024-01-01", "2024-01-02", "2024-01-10"], dtype="datetime64[ns]")
    summary = gap_summary(days)
    assert summary["max_gap_days"] == 8
    assert summary["gaps_over_7_days"] == 1
    assert summary["median_gap_days"] == 4.5
    assert gap_summary(days[:1]) == {"max_gap_days": 0, "gaps_over_7_days": 0, "median_gap_days": 0.0}


def test_window_excludes_same_day_fall_and_requires_coverage():
    landmark = _day("2024-01-10")
    reports = [_day("2024-01-10"), _day("2024-01-17"), _day("2024-02-09")]
    rows = _window_rows(landmark, _day("2024-02-05"), (7, 30), reports, True)
    seven, thirty = rows
    assert seven["status"] == "eligible"
    assert seven["event"] is True
    assert seven["n_reports_in_window"] == 1  # 同日跌倒不计入；第 7 天计入
    assert seven["first_event_day"] == "2024-01-17"
    assert thirty["status"] == "insufficient_followup"
    assert thirty["event"] is None
    assert thirty["days_short_of_window_end"] == 4


def test_window_full_coverage_includes_last_day():
    rows = _window_rows(
        _day("2024-01-10"), _day("2024-01-17"), (7,), [_day("2024-01-17")], True
    )
    assert rows[0]["status"] == "eligible"
    assert rows[0]["event"] is True


def test_build_cohort_marks_candidate_and_baseline_days():
    participants = [{"participantid": "1", "cohort pd": "T", "age": 72.0}]
    visits = [
        {"participantid": "1", "visit_number": "t1", "day": "2024-01-01", "pstatus": "Attended"},
        {"participantid": "1", "visit_number": "t2", "day": "2024-06-20", "pstatus": "Attended"},
    ]
    gait_days = [f"2024-01-{day:02d}" for day in range(2, 12)]  # 10 天
    gait_rows = [
        {"participantid": "1", "visit_number": "t1", "day": day} for day in gait_days
    ]
    gait_rows += [
        {"participantid": "1", "visit_number": "t2", "day": day}
        for day in ["2024-06-14", "2024-06-15", "2024-06-16", "2024-06-17"]
    ]
    diaries = [
        {"participantid": "1", "visit_number": "t2", "day": "2024-06-20", "fallstotal": 0,
         "fallsreports": 0.0}
    ]
    clinical = _clinical(participants, visits, diaries, [])
    result = build_cohort(clinical, _daily(gait_rows), CohortConfig())
    assert result.counts["candidate_landmarks"] == 1
    landmark = result.landmarks.iloc[0]
    assert landmark["landmark_day"] == "2024-01-11"
    assert landmark["n_prior_valid_days"] == 9
    assert not landmark["baseline_eligible"]
    assert landmark["baseline_days_used"] == 9
    assert landmark["days_from_previous_visit_to_landmark"] == 10
    assert result.windows["status"].eq("eligible").all()
    assert len(result.baseline_days) == 9
    assert result.baseline_days["baseline_day"].iloc[0] == "2024-01-02"
    assert result.flow.set_index("step").loc["candidate_landmarks", "rows_out"] == 1


def test_build_cohort_uses_only_earliest_baseline_days_and_tracks_gaps():
    participants = [{"participantid": "1", "cohort pd": "T", "age": 70.0}]
    visits = [
        {"participantid": "1", "visit_number": "t1", "day": "2024-01-01"},
        {"participantid": "1", "visit_number": "t2", "day": "2024-06-20"},
        {"participantid": "1", "visit_number": "t3", "day": "2024-12-10"},
    ]
    gait_rows = [
        {"participantid": "1", "visit_number": "t1", "day": f"2024-01-{day:02d}"}
        for day in range(2, 12)
    ] + [
        {"participantid": "1", "visit_number": "t2", "day": day}
        for day in ["2024-06-21", "2024-06-22", "2024-06-23", "2024-06-24", "2024-06-25"]
    ]
    diaries = [{"participantid": "1", "visit_number": "t3", "day": "2024-12-10",
                "fallstotal": 0, "fallsreports": 0.0}]
    result = build_cohort(_clinical(participants, visits, diaries, []), _daily(gait_rows), CohortConfig())
    landmark = result.landmarks.iloc[0]
    assert landmark["landmark_day"] == "2024-06-25"
    assert landmark["n_prior_valid_days"] == 14
    assert landmark["baseline_eligible"]
    assert landmark["baseline_days_used"] == 14
    assert landmark["baseline_first_day"] == "2024-01-02"
    assert landmark["baseline_end_day"] == "2024-06-24"
    assert landmark["baseline_max_gap_days"] > 30
    assert landmark["days_from_baseline_end_to_landmark"] == 1
    assert landmark["monitoring_start_day"] == "2024-06-25"
    assert landmark["n_monitored_days_in_block"] == 1
    assert len(result.baseline_days) == 14
    assert result.baseline_days["baseline_day"].is_monotonic_increasing


def test_build_cohort_records_exclusion_reasons():
    participants = [{"participantid": "1", "cohort pd": "T", "age": 70.0},
                    {"participantid": "2", "cohort pd": "F", "age": 70.0}]
    visits = [
        {"participantid": "1", "visit_number": "t1", "day": "2024-01-01"},
        {"participantid": "1", "visit_number": "t2", "day": "2024-06-20"},
        {"participantid": "1", "visit_number": "t3", "day": "2024-12-10"},
        {"participantid": "2", "visit_number": "t1", "day": "2024-01-01"},
    ]
    gait = [
        {"participantid": "1", "visit_number": "t1", "day": "2024-01-05"},
        {"participantid": "1", "visit_number": "t2", "day": "2024-06-15"},
    ] + [
        {"participantid": "1", "visit_number": "t3", "day": f"2024-12-{day:02d}"}
        for day in range(1, 4)
    ]
    diaries = [
        # 日记总数与报告数不一致
        {"participantid": "1", "visit_number": "t2", "day": "2024-06-20", "fallstotal": 2,
         "fallsreports": 1.0},
        # 采集结束日早于日记覆盖下界（访视间隔超过半年）
        {"participantid": "1", "visit_number": "t3", "day": "2024-12-10", "fallstotal": 0,
         "fallsreports": 0.0},
        # 非帕金森队列，不进入目标人群
        {"participantid": "2", "visit_number": "t1", "day": "2024-06-01", "fallstotal": 0,
         "fallsreports": 0.0},
    ]
    reports = [{"participantid": "1", "visit_number": "t2", "day": "2024-06-18"}]
    result = build_cohort(_clinical(participants, visits, diaries, reports), _daily(gait), CohortConfig())
    reasons = result.gates.set_index("visit_number")["exclusion_reason"].to_dict()
    assert reasons["t2"] == "fall_reports_not_usable"
    assert reasons["t3"] == "landmark_outside_diary_window"
    assert result.counts["candidate_landmarks"] == 0
    assert set(result.gates["participantid"]) == {"1"}


def test_build_cohort_excludes_misaligned_monitoring():
    participants = [{"participantid": "1", "cohort pd": "T", "age": 70.0}]
    visits = [
        {"participantid": "1", "visit_number": "t1", "day": "2024-01-01"},
        {"participantid": "1", "visit_number": "t2", "day": "2024-06-20"},
    ]
    gait = [{"participantid": "1", "visit_number": "t1", "day": "2024-01-20"}]
    diaries = [{"participantid": "1", "visit_number": "t2", "day": "2024-06-20",
                "fallstotal": 0, "fallsreports": 0.0}]
    result = build_cohort(_clinical(participants, visits, diaries, []), _daily(gait), CohortConfig())
    row = result.gates.iloc[0]
    assert row["exclusion_reason"] == "monitoring_not_aligned_with_previous_visit"
    assert row["block_first_offset_days"] == 19


def test_build_cohort_wear_scenario_and_age_gate():
    participants = [{"participantid": "1", "cohort pd": "T", "age": 64.0},
                    {"participantid": "2", "cohort pd": "T", "age": 70.0}]
    visits = [
        {"participantid": "1", "visit_number": "t1", "day": "2024-01-01"},
        {"participantid": "1", "visit_number": "t2", "day": "2024-06-20"},
        {"participantid": "2", "visit_number": "t1", "day": "2024-01-01"},
        {"participantid": "2", "visit_number": "t2", "day": "2024-06-20"},
    ]
    gait = []
    for person in ("1", "2"):
        gait.append({"participantid": person, "visit_number": "t1", "day": "2024-01-03"})
    gait.append({"participantid": "2", "visit_number": "t1", "day": "2024-01-04",
                 "total_worn_during_waking_h_d": 6.0, "wear_h_waking": 6.0})
    diaries = [
        {"participantid": "1", "visit_number": "t2", "day": "2024-06-20", "fallstotal": 0,
         "fallsreports": 0.0},
        {"participantid": "2", "visit_number": "t2", "day": "2024-06-20", "fallstotal": 0,
         "fallsreports": 0.0},
    ]
    frame = _daily(gait)
    result = build_cohort(_clinical(participants, visits, diaries, []), frame, CohortConfig())
    assert set(result.landmarks["participantid"]) == {"2"}
    scenario = build_cohort(
        _clinical(participants, visits, diaries, []),
        frame,
        CohortConfig(require_wear_hours=10.0),
    )
    assert len(scenario.landmarks) == 0
    assert len(scenario.windows) == 0


def test_landmarks_and_windows_are_sorted_and_keys_unique():
    participants = [{"participantid": str(person), "cohort pd": "T", "age": 70.0}
                    for person in (2, 1)]
    visits = []
    gait = []
    diaries = []
    for person in (2, 1):
        visits.append({"participantid": str(person), "visit_number": "t1", "day": "2024-01-01"})
        visits.append({"participantid": str(person), "visit_number": "t2", "day": "2024-06-20"})
        gait.append({"participantid": str(person), "visit_number": "t1", "day": "2024-01-04"})
        gait.append({"participantid": str(person), "visit_number": "t1", "day": "2024-01-05"})
        diaries.append({"participantid": str(person), "visit_number": "t2", "day": "2024-06-20",
                        "fallstotal": 0, "fallsreports": 0.0})
    result = build_cohort(_clinical(participants, visits, diaries, []), _daily(gait), CohortConfig())
    assert result.landmarks["participantid"].tolist() == ["1", "2"]
    assert not result.windows.duplicated(["participantid", "landmark_day", "horizon_days"]).any()
    assert result.windows["status"].eq("eligible").all()
