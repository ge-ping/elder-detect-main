"""会话测量标准误估计的单元测试。"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.uncertainty import (  # noqa: E402
    bootstrap_mean_se,
    pose_session_precision,
)


class BootstrapTest(unittest.TestCase):
    def test_iid_bootstrap_matches_analytic_se(self):
        rng = np.random.default_rng(4)
        values = rng.normal(10.0, 2.0, 200)
        result = bootstrap_mean_se(values, samples=400, seed_material="iid")
        self.assertEqual(result["method"], "iid_bootstrap")
        self.assertAlmostEqual(
            result["se"], float(values.std(ddof=1) / np.sqrt(len(values))), delta=0.15
        )

    def test_block_bootstrap_is_larger_under_autocorrelation(self):
        rng = np.random.default_rng(5)
        size = 400
        innov = rng.normal(0, 1, size)
        values = np.empty(size)
        values[0] = innov[0]
        for index in range(1, size):
            values[index] = 0.8 * values[index - 1] + innov[index]
        result = bootstrap_mean_se(values, samples=300, seed_material="ar1")
        self.assertEqual(result["method"], "block2_bootstrap")
        # 块长 2 只保留滞后 1 相关，不等同于完整的长期方差，
        # 因此只要求高于独立同分布标准误，并与块长 2 的矩估计接近。
        self.assertGreater(result["se"], result["diagnostic"]["analytic_se"] * 1.2)
        n = len(values)
        centered = values - values.mean()
        g0 = float((centered ** 2).mean())
        g1 = float((centered[:-1] * centered[1:]).mean())
        expected = float(np.sqrt((2 * g0 + 2 * g1) / (2 * n)))
        self.assertAlmostEqual(result["se"], expected, delta=0.25 * expected)

    def test_too_few_values_is_unavailable(self):
        result = bootstrap_mean_se([1.0, 2.0], min_count=5)
        self.assertIsNone(result["se"])
        self.assertEqual(result["method"], "unavailable")
        self.assertEqual(result["reason"], "too_few_observations")


def synthetic_walk(step_interval: float = 2.0 / 3.0, duration: float = 30.0, fps: float = 25.0):
    timestamps = np.arange(0.0, duration, 1.0 / fps)
    points = np.zeros((len(timestamps), 33, 2))
    phase = 2 * np.pi * timestamps / step_interval
    points[:, 11] = (-0.2, -1.0)
    points[:, 12] = (0.2, -1.0)
    points[:, 23] = (-0.1, -0.5)
    points[:, 24] = (0.1, -0.5)
    points[:, 27] = np.stack(
        (-0.15 + 0.05 * np.sin(phase), 0.02 * np.sin(phase)), axis=1
    )
    points[:, 28] = np.stack(
        (0.15 - 0.05 * np.sin(phase), -0.02 * np.sin(phase)), axis=1
    )
    points[:, 0] = (0.0, -1.1)
    confidence = np.ones((len(timestamps), 33))
    return points, timestamps, confidence


class PosePrecisionTest(unittest.TestCase):
    def test_time_features_get_positive_se(self):
        points, timestamps, confidence = synthetic_walk()
        result = pose_session_precision(
            points, timestamps, confidence, samples=100, seed_material="walk",
        )
        self.assertGreater(result.n_steps, 30)
        self.assertIn("cadence_spm", result.feature_se)
        self.assertIn("step_time_mean_s", result.feature_se)
        self.assertGreater(result.feature_se["cadence_spm"], 0.0)
        self.assertLess(result.feature_se["cadence_spm"], 5.0)
        repeated = pose_session_precision(
            points, timestamps, confidence, samples=100, seed_material="walk",
        )
        self.assertEqual(result.to_dict(), repeated.to_dict())

    def test_short_sequence_reports_reasons(self):
        points, timestamps, confidence = synthetic_walk(duration=2.0)
        result = pose_session_precision(points, timestamps, confidence, samples=50)
        self.assertLess(result.n_steps, 8)
        self.assertTrue(any("insufficient_step_events" in reason for reason in result.reasons))
        self.assertEqual(result.feature_se, {})
