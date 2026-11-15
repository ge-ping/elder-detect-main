"""阶段 2 穿戴设备适配器与批量评分函数测试（合成表，不读原始包）。"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mobilise_d.observations import (  # noqa: E402
    Stage1Cohort,
    build_records,
    ds_pcd_config,
    fit_rms_baseline,
    distance_history,
    trend_and_persistence,
    wearable_config,
)
from scripts.replay_mobilise_stage2 import _score_person  # noqa: E402

START = date(2024, 3, 1)


def make_cohort() -> tuple[Stage1Cohort, pd.DataFrame]:
    days = []
    rng = np.random.default_rng(5)
    for offset in range(15):
        days.append({
            "participantid": "1",
            "visit_number": "t1" if offset < 14 else "t2",
            "day": pd.Timestamp(START + timedelta(days=offset)),
            "has_core_gait": True,
            "ws_30_avg_d": 0.9 + 0.01 * rng.normal(),
            "cadence_all_avg_d": 92.0 + 0.5 * rng.normal(),
            "strdur_all_avg_d": 1.18 + 0.005 * rng.normal(),
        })
    daily = pd.DataFrame(days)
    participants = pd.DataFrame([{
        "participantid": "1", "age": 70.0, "n_valid_days": 15,
    }])
    landmarks = pd.DataFrame([{
        "participantid": "1",
        "landmark_day": pd.Timestamp(START + timedelta(days=14)),
        "visit_number": "t2", "diary_visit_number": "t3",
        "n_prior_valid_days": 14, "baseline_eligible": True,
    }])
    baseline_days = pd.DataFrame([
        {
            "participantid": "1",
            "landmark_day": pd.Timestamp(START + timedelta(days=14)),
            "baseline_index": index + 1,
            "baseline_day": pd.Timestamp(START + timedelta(days=index)),
        }
        for index in range(14)
    ])
    windows = pd.DataFrame([
        {
            "participantid": "1", "landmark_day": pd.Timestamp(START + timedelta(days=14)),
            "horizon_days": horizon, "status": "eligible", "event": False,
        }
        for horizon in (7, 30)
    ])
    precision = pd.DataFrame([
        {
            "participantid": "1", "visit_number": row["visit_number"],
            "day": row["day"],
            "se_walking_speed_m_s": 0.01,
            "se_cadence_spm": 0.2,
            "se_stride_duration_s": 0.002,
        }
        for row in days
    ])
    return Stage1Cohort(
        participants=participants, daily=daily, landmarks=landmarks,
        baseline_days=baseline_days, windows=windows,
    ), precision


class Stage2AdapterTest(unittest.TestCase):
    def test_records_and_rms_baseline_use_stage1_dates(self):
        cohort, precision = make_cohort()
        records = build_records(cohort, precision, precision_mode="session_se")
        self.assertEqual(len(records["1"]), 15)
        baseline = fit_rms_baseline(
            "1", {record.date: record for record in records["1"]},
            [row.baseline_day.date().isoformat() for row in cohort.baseline_days.itertuples()],
            wearable_config(),
        )
        self.assertEqual(baseline.valid_days, 14)
        self.assertEqual(baseline.valid_dates[0], "2024-03-01")
        history = distance_history(records["1"], baseline, baseline.valid_dates[-1])
        self.assertEqual(len(history), 1)
        trend, window, persistent = trend_and_persistence(history, "2024-03-15")
        # 基线结束日是 03-14，监测只有预测日当天，因此窗口只有 1 个点
        self.assertEqual(window["points"], 1)
        self.assertEqual(trend, 0.0)
        self.assertGreaterEqual(persistent, 0)

    def test_score_person_outputs_rms_and_ds_pcd(self):
        cohort, precision = make_cohort()
        config = ds_pcd_config(precision_mode="session_se", null_samples=1000,
                               quantile_points=200)
        records = build_records(cohort, precision, precision_mode="session_se")
        baseline_dates = tuple(
            pd.Timestamp(value).date().isoformat()
            for value in cohort.baseline_days["baseline_day"]
        )
        rows = _score_person(
            "1", cohort.landmarks, records["1"], baseline_dates,
            wearable_config(), config, None, "session_se",
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["precision_mode"], "session_se")
        self.assertIsNotNone(row["rms_distance"])
        self.assertEqual(row["ds_status"], "scored")
        self.assertIsNotNone(row["T"])
        self.assertIsNotNone(row["cusum"])
        self.assertEqual(row["ds_baseline_end"], "2024-03-14")
        self.assertEqual(row["n_valid_days"], 15)
        self.assertEqual(row["n_valid_days_before_landmark"], 14)
        self.assertTrue(row["n_valid_days_includes_landmark"])

    def test_missing_se_makes_session_mode_unscorable_but_keeps_rms(self):
        cohort, precision = make_cohort()
        precision = precision.copy()
        precision.loc[0, "se_walking_speed_m_s"] = np.nan
        config = ds_pcd_config(precision_mode="session_se", null_samples=1000,
                               quantile_points=200)
        records = build_records(cohort, precision, precision_mode="session_se")
        baseline_dates = tuple(
            pd.Timestamp(value).date().isoformat()
            for value in cohort.baseline_days["baseline_day"]
        )
        rows = _score_person(
            "1", cohort.landmarks, records["1"], baseline_dates,
            wearable_config(), config, None, "session_se",
        )
        self.assertIsNotNone(rows[0]["rms_distance"])
        self.assertEqual(rows[0]["ds_status"], "baseline_dates_unavailable")


if __name__ == "__main__":
    unittest.main()
