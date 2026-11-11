from datetime import date, timedelta
import tempfile
import unittest

from longitudinal.baseline import build_baseline
from longitudinal.config import LongitudinalConfig
from longitudinal.schema import DailyObservation
from longitudinal.service import LongitudinalRiskService
from longitudinal.storage import ProfileStore
from longitudinal.survival import CoxModelArtifact


def observation(
    day: int,
    cadence: float,
    *,
    confidence: float = 0.8,
) -> DailyObservation:
    observed = date(2026, 10, 1) + timedelta(days=day)
    return DailyObservation.from_dict({
        "person_id": "elder_001",
        "observed_at": observed.isoformat(),
        "features": {"cadence_spm": cadence},
        "quality": {
            "valid_frames": 200,
            "valid_steps": 30,
            "walking_duration_s": 20,
            "mean_confidence": confidence,
        },
    })


def model(horizon: int, baseline_survival: float) -> CoxModelArtifact:
    return CoxModelArtifact.from_dict({
        "schema_version": 2,
        "model_type": "landmark_cox",
        "bundle_id": "service-test-bundle",
        "horizon_days": horizon,
        "feature_names": ["distance", "persistent_deviation_days"],
        "coefficients": [0.0, 0.0],
        "feature_centers": [0.0, 0.0],
        "baseline_survival": baseline_survival,
        "calibrated": True,
        "trained_at": "2026-11-11T00:00:00+00:00",
        "evidence_ids": ["mehdizadeh_2021"],
    })


class LongitudinalRiskServiceTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = ProfileStore(self.directory.name, "elder_001")
        self.config = LongitudinalConfig(core_features=("cadence_spm",))

    def seed_baseline(self):
        items = [observation(i, 80.0 + i) for i in range(14)]
        for item in items:
            self.store.append_observation(item)
        baseline = build_baseline(items, self.config)
        self.store.save_baseline(baseline)
        return baseline

    def test_insufficient_quality_returns_reasons_without_probability(self):
        service = LongitudinalRiskService(self.store, self.config)

        result = service.evaluate("elder_001", observation(20, 90.0, confidence=0.2))

        self.assertEqual(result.status, "insufficient_quality")
        self.assertIsNone(result.risk_7d)
        self.assertTrue(any("置信度" in reason for reason in result.quality_reasons))

    def test_collecting_reports_valid_days_before_baseline(self):
        for item in [observation(i, 80.0 + i) for i in range(5)]:
            self.store.append_observation(item)
        service = LongitudinalRiskService(self.store, self.config)

        result = service.evaluate("elder_001", observation(6, 86.0))

        self.assertEqual(result.status, "collecting")
        self.assertEqual(result.baseline_valid_days, 6)
        self.assertIsNone(result.deviation_index)

    def test_baseline_ready_returns_deviation_without_probability(self):
        self.seed_baseline()
        service = LongitudinalRiskService(self.store, self.config)

        result = service.evaluate("elder_001", observation(14, 86.5))

        self.assertEqual(result.status, "baseline_ready")
        self.assertAlmostEqual(result.deviation_index, 0.0)
        self.assertIsNone(result.risk_7d)
        self.assertIsNone(result.risk_30d)

    def test_three_consecutive_large_deviations_require_calibration(self):
        self.seed_baseline()
        abnormal = [observation(day, 140.0) for day in (14, 15, 16)]
        for item in abnormal:
            self.store.append_observation(item)
        service = LongitudinalRiskService(self.store, self.config)

        result = service.evaluate("elder_001", abnormal[-1])

        self.assertEqual(result.status, "calibration_required")
        self.assertEqual(result.persistent_deviation_days, 3)
        self.assertIn("未经跌倒概率校准", result.message)

    def test_calibrated_bundle_outputs_seven_and_thirty_day_probability(self):
        self.seed_baseline()
        service = LongitudinalRiskService(
            self.store,
            self.config,
            models={7: model(7, 0.8), 30: model(30, 0.6)},
        )

        result = service.evaluate("elder_001", observation(14, 86.5))

        self.assertEqual(result.status, "calibrated")
        self.assertAlmostEqual(result.risk_7d, 0.2)
        self.assertAlmostEqual(result.risk_30d, 0.4)

    def test_configuration_mismatch_is_rejected(self):
        self.seed_baseline()
        different = LongitudinalConfig(
            core_features=("cadence_spm",), min_valid_steps=25
        )
        service = LongitudinalRiskService(self.store, different)

        with self.assertRaisesRegex(ValueError, "配置不一致"):
            service.evaluate("elder_001", observation(14, 86.5))


if __name__ == "__main__":
    unittest.main()
