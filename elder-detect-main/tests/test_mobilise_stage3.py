"""阶段 3 分析工具（折分配、AUROC/Brier、校准斜率）的单元测试。"""
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

from scripts.analyse_mobilise_stage3 import (  # noqa: E402
    auroc,
    assign_folds,
    brier,
    calibration_slope,
)


def synthetic_frame(people: int = 30, event_people: int = 6) -> pd.DataFrame:
    rows = []
    for person in range(people):
        has_event = person < event_people
        for landmark in range(2):
            rows.append({
                "participantid": f"p{person:03d}",
                "landmark_day": pd.Timestamp(date(2024, 1, 1) + timedelta(days=person)),
                "event_bool_30d": has_event and landmark == 0,
                "person_has_event_30d": has_event,
            })
    return pd.DataFrame(rows)


class Stage3ToolsTest(unittest.TestCase):
    def test_auroc_matches_pairwise_with_ties(self):
        labels = np.array([1, 1, 0, 0, 0, 1])
        scores = np.array([0.9, 0.5, 0.5, 0.2, 0.8, 0.5])
        positives = scores[labels == 1]
        negatives = scores[labels == 0]
        expected = float(
            ((positives[:, None] > negatives[None, :]).sum()
             + 0.5 * (positives[:, None] == negatives[None, :]).sum())
            / (len(positives) * len(negatives))
        )
        self.assertAlmostEqual(auroc(labels, scores), expected)
        self.assertIsNone(auroc(np.ones(4, dtype=int), np.arange(4.0)))
        self.assertAlmostEqual(auroc(labels, labels.astype(float)), 1.0)

    def test_brier_and_calibration_slope(self):
        labels = np.array([0, 1, 0, 1])
        probabilities = np.array([0.1, 0.9, 0.2, 0.8])
        self.assertAlmostEqual(
            brier(labels, probabilities), np.mean((labels - probabilities) ** 2)
        )
        rng = np.random.default_rng(7)
        latent = rng.normal(0, 1.5, 4000)
        probabilities = 1 / (1 + np.exp(-latent))
        labels = rng.binomial(1, probabilities)
        slope = calibration_slope(labels, probabilities)
        self.assertIsNotNone(slope)
        self.assertAlmostEqual(slope, 1.0, delta=0.25)
        self.assertIsNone(calibration_slope(np.array([0, 1, 0, 1]), np.array([0.2, 0.8, 0.3, 0.7])))

    def test_assign_folds_is_person_grouped_balanced_and_reproducible(self):
        frame = synthetic_frame()
        first = assign_folds(frame, 3, 20261113)
        second = assign_folds(frame, 3, 20261113)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 30)
        self.assertEqual(first["fold"].nunique(), 3)
        self.assertEqual(first.groupby("fold")["has_event"].sum().min(), 2)
        mapped = frame.merge(first[["participantid", "fold"]], on="participantid")
        self.assertEqual(mapped.groupby("participantid")["fold"].nunique().max(), 1)

    def test_assign_folds_refuses_fold_without_events(self):
        frame = synthetic_frame(people=6, event_people=1)
        with self.assertRaisesRegex(ValueError, "阳性"):
            assign_folds(frame, 3, 1)


if __name__ == "__main__":
    unittest.main()
