"""阶段 9：S7 扩样集合统计与旧字段语义测试。"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mobilise_d.expansion import (  # noqa: E402
    analyze_expansion, compute_expansion_from_frames,
)


def frames():
    """S14 4 人/4 个预测日；S7 5 人/9 个预测日。

    A: S14 阴性，S7 额外阳性 -> 原有人员新进入阳性集合
    B: 仅 S7，阳性 -> 新增受试者中的阳性
    C: S14 阳性，S7 额外阳性 -> 已阳性人员的重复事件
    D: S14/S7 均阴性
    E: S14 阳性，S7 无额外事件
    """
    landmarks = pd.DataFrame([
        ("A", "2024-01-01", True, 20),
        ("A", "2024-01-02", False, 10),
        ("B", "2024-01-03", False, 10),
        ("C", "2024-01-04", True, 20),
        ("C", "2024-01-05", False, 10),
        ("D", "2024-01-06", True, 20),
        ("D", "2024-01-07", False, 10),
        ("E", "2024-01-08", True, 20),
        ("E", "2024-01-09", False, 10),
    ], columns=["participantid", "landmark_day", "baseline_eligible", "n_prior_valid_days"])
    events = {
        ("A", "2024-01-01"): False,
        ("A", "2024-01-02"): True,
        ("B", "2024-01-03"): True,
        ("C", "2024-01-04"): True,
        ("C", "2024-01-05"): True,
        ("D", "2024-01-06"): False,
        ("D", "2024-01-07"): False,
        ("E", "2024-01-08"): True,
        ("E", "2024-01-09"): False,
    }
    windows = pd.DataFrame([
        {"participantid": p, "landmark_day": d, "horizon_days": 30, "event": e}
        for (p, d), e in events.items()
    ])
    return landmarks, windows


class ExpansionTest(unittest.TestCase):
    def test_minimal_case_counts_and_categories(self):
        landmarks, windows = frames()
        result = analyze_expansion(landmarks, windows)
        summary = result["summary"]
        self.assertEqual(summary["S14"], {
            "landmarks": 4, "people": 4,
            "positive_landmarks": 2, "positive_people": 2,
        })
        self.assertEqual(summary["S7"], {
            "landmarks": 9, "people": 5,
            "positive_landmarks": 5, "positive_people": 4,
        })
        self.assertEqual(summary["new_people"], 1)
        self.assertEqual(summary["s7_new_positive_people_total"], 2)
        self.assertEqual(summary["s7_new_positive_people_from_new_people"], 1)
        self.assertEqual(summary["s7_existing_people_newly_positive"], 1)
        self.assertEqual(summary["s7_extra_positive_landmarks_total"], 3)
        self.assertEqual(summary["s7_extra_positive_landmarks_from_new_people"], 1)
        self.assertEqual(
            summary["s7_extra_positive_landmarks_from_existing_people_newly_positive"], 1
        )
        self.assertEqual(
            summary["s7_extra_positive_landmarks_from_already_positive_people"], 1
        )
        self.assertTrue(all(summary["identities"].values()))
        extra = result["extra_positive_landmarks"]
        self.assertEqual(
            extra.set_index("participantid")["extra_positive_category"].to_dict(),
            {
                "A": "from_existing_people_newly_positive",
                "B": "from_new_people",
                "C": "from_already_positive_people",
            },
        )

    def test_historical_fields_keep_old_meaning(self):
        landmarks, windows = frames()
        summary = compute_expansion_from_frames(landmarks, windows)
        # 旧字段只统计新增受试者中的阳性，不是全部新增阳性。
        self.assertEqual(summary["new_positive_people"], 1)
        self.assertEqual(summary["new_positive_landmarks"], 1)
        self.assertEqual(summary["s7_new_positive_people_total"], 2)
        self.assertEqual(summary["s7_extra_positive_landmarks_total"], 3)

    def test_duplicate_landmark_key_rejected(self):
        landmarks, windows = frames()
        duplicated = pd.concat([landmarks, landmarks.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "重复"):
            analyze_expansion(duplicated, windows)

    def test_missing_outcome_rejected(self):
        landmarks, windows = frames()
        with self.assertRaisesRegex(ValueError, "缺少 30 天结局"):
            analyze_expansion(landmarks, windows.iloc[:-1])

    def test_horizon_filter_uses_30_day_window(self):
        landmarks, windows = frames()
        seven = windows.assign(horizon_days=7)
        with self.assertRaisesRegex(ValueError, "缺少 30 天结局"):
            analyze_expansion(landmarks, seven)


if __name__ == "__main__":
    unittest.main()
