from datetime import date, timedelta
import math
import unittest

from longitudinal.baseline import (
    BaselineFeatureStats,
    PersonalBaseline,
    build_baseline,
)
from longitudinal.config import LongitudinalConfig
from longitudinal.deviation import (
    DeviationResult,
    compute_deviation,
    count_persistent_deviation,
)
from longitudinal.schema import DailyObservation


def observation(day: int, features: dict[str, float]) -> DailyObservation:
    observed = date(2026, 10, 1) + timedelta(days=day)
    return DailyObservation.from_dict({
        "person_id": "elder_001",
        "observed_at": observed.isoformat(),
        "features": features,
        "quality": {
            "valid_frames": 200,
            "valid_steps": 30,
            "walking_duration_s": 20,
            "mean_confidence": 0.8,
        },
    })


class PersonalBaselineTest(unittest.TestCase):
    def test_requires_fourteen_valid_days(self):
        config = LongitudinalConfig(
            min_baseline_days=14,
            core_features=("cadence_spm",),
        )
        observations = [observation(i, {"cadence_spm": 80 + i}) for i in range(13)]

        with self.assertRaisesRegex(ValueError, "至少需要 14 个有效观测日"):
            build_baseline(observations, config)

    def test_builds_median_and_mad_scale_at_fourteen_days(self):
        config = LongitudinalConfig(core_features=("cadence_spm",))
        observations = [observation(i, {"cadence_spm": i + 1}) for i in range(14)]

        baseline = build_baseline(observations, config)

        stats = baseline.features["cadence_spm"]
        self.assertEqual(baseline.valid_days, 14)
        self.assertAlmostEqual(stats.median, 7.5)
        self.assertAlmostEqual(stats.scale, 1.4826 * 3.5)
        self.assertEqual(stats.scale_method, "mad")
        self.assertTrue(stats.active)

    def test_uses_iqr_when_mad_is_zero_and_disables_constant_feature(self):
        config = LongitudinalConfig(core_features=("variable", "constant"))
        values = [1.0] * 8 + [2.0] * 6
        observations = [
            observation(i, {"variable": value, "constant": 5.0})
            for i, value in enumerate(values)
        ]

        baseline = build_baseline(observations, config)

        self.assertEqual(baseline.features["variable"].scale_method, "iqr")
        self.assertGreater(baseline.features["variable"].scale, 0.0)
        self.assertFalse(baseline.features["constant"].active)
        self.assertEqual(baseline.features["constant"].scale_method, "constant")

    def test_round_trip_preserves_configuration_fingerprint(self):
        config = LongitudinalConfig(core_features=("cadence_spm",))
        observations = [observation(i, {"cadence_spm": 80 + i}) for i in range(14)]
        baseline = build_baseline(observations, config)

        restored = PersonalBaseline.from_dict(baseline.to_dict())

        self.assertEqual(restored.config_fingerprint, baseline.config_fingerprint)
        self.assertEqual(restored.features["cadence_spm"].count, 14)
        self.assertEqual(restored.config_snapshot["min_baseline_days"], 14)
        self.assertEqual(restored.slow_reference["cadence_spm"], baseline.features["cadence_spm"].median)

    def test_initial_anchor_uses_first_fourteen_valid_days_only(self):
        config = LongitudinalConfig(core_features=("cadence_spm",))
        observations = [observation(i, {"cadence_spm": 80 + i}) for i in range(14)]
        observations.append(observation(14, {"cadence_spm": 500.0}))

        baseline = build_baseline(observations, config)

        self.assertEqual(baseline.valid_days, 14)
        self.assertEqual(baseline.valid_dates[-1], "2026-10-14")
        self.assertAlmostEqual(baseline.features["cadence_spm"].median, 86.5)


class DeviationTest(unittest.TestCase):
    def setUp(self):
        self.baseline = PersonalBaseline(
            person_id="elder_001",
            created_at="2026-11-01T00:00:00",
            valid_days=14,
            valid_dates=tuple(f"2026-10-{day:02d}" for day in range(1, 15)),
            features={
                "a": BaselineFeatureStats(0.0, 1.0, "mad", 14, True),
                "b": BaselineFeatureStats(0.0, 1.0, "mad", 14, True),
                "constant": BaselineFeatureStats(2.0, 0.0, "constant", 14, False),
            },
            config_fingerprint="abc",
        )

    def test_clips_each_z_score_before_rms_distance(self):
        result = compute_deviation({"a": 100.0, "b": 0.0}, self.baseline)

        self.assertEqual(result.feature_deviations["a"], 100.0)
        self.assertAlmostEqual(result.distance, math.sqrt(12.5))
        self.assertAlmostEqual(result.deviation_index, math.sqrt(12.5) / 5.0)
        self.assertEqual(result.valid_feature_count, 2)

    def test_persistent_count_only_uses_consecutive_tail(self):
        distances = [1.0, 3.1, 3.2, 3.3]
        results = [
            DeviationResult(value, min(value / 5, 1), {}, 1)
            for value in distances
        ]

        self.assertEqual(count_persistent_deviation(results, threshold=3.0), 3)


if __name__ == "__main__":
    unittest.main()
