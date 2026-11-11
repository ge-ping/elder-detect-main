import math
import unittest

from longitudinal.config import LongitudinalConfig
from longitudinal.quality import assess_quality
from longitudinal.schema import DailyObservation


class DailyObservationTest(unittest.TestCase):
    def test_from_dict_normalizes_date_and_round_trips(self):
        data = {
            "person_id": "elder_001",
            "observed_at": "2026-11-01T08:30:00",
            "features": {"cadence_spm": 92.5, "step_time_cv": 0.08},
            "quality": {
                "valid_frames": 200,
                "valid_steps": 24,
                "walking_duration_s": 15.0,
                "mean_confidence": 0.82,
            },
            "psychological": {"fes_i": 24.0},
            "source": "daily.json",
        }

        observation = DailyObservation.from_dict(data)

        self.assertEqual(observation.observation_date, "2026-11-01")
        self.assertEqual(observation.to_dict()["features"]["cadence_spm"], 92.5)
        self.assertEqual(observation.to_dict()["psychological"]["fes_i"], 24.0)

    def test_from_dict_rejects_non_finite_feature(self):
        with self.assertRaisesRegex(ValueError, "有限数值"):
            DailyObservation.from_dict({
                "person_id": "elder_001",
                "observed_at": "2026-11-01",
                "features": {"cadence_spm": math.nan},
                "quality": {},
            })

    def test_quality_metrics_reject_non_finite_or_out_of_range_values(self):
        for quality in (
            {"valid_frames": 150, "valid_steps": 20, "walking_duration_s": math.nan,
             "mean_confidence": 0.8},
            {"valid_frames": 150, "valid_steps": 20, "walking_duration_s": 10,
             "mean_confidence": math.inf},
            {"valid_frames": 150, "valid_steps": 20, "walking_duration_s": 10,
             "mean_confidence": 1.2},
            {"valid_frames": True, "valid_steps": 20, "walking_duration_s": 10,
             "mean_confidence": 0.8},
        ):
            with self.subTest(quality=quality):
                with self.assertRaises(ValueError):
                    DailyObservation.from_dict({
                        "person_id": "elder_001",
                        "observed_at": "2026-11-01",
                        "features": {"cadence_spm": 90.0},
                        "quality": quality,
                    })

    def test_mixed_timezone_inputs_are_normalized_but_keep_local_day(self):
        local = DailyObservation.from_dict({
            "person_id": "elder_001",
            "observed_at": "2026-11-01T00:30:00+08:00",
            "features": {},
            "quality": {},
        })
        date_only = DailyObservation.from_dict({
            "person_id": "elder_001",
            "observed_at": "2026-11-02",
            "features": {},
            "quality": {},
        })

        ordered = sorted([date_only, local], key=lambda item: item.observed_at)

        self.assertEqual(ordered[0].observation_date, "2026-11-01")
        self.assertEqual(local.observed_at.utcoffset().total_seconds(), 0)
        self.assertEqual(local.timezone, "Asia/Shanghai")


class QualityAssessmentTest(unittest.TestCase):
    def test_default_thresholds_accept_complete_observation(self):
        config = LongitudinalConfig(core_features=("cadence_spm", "step_time_cv"))
        observation = DailyObservation.from_dict({
            "person_id": "elder_001",
            "observed_at": "2026-11-01",
            "features": {"cadence_spm": 92.5, "step_time_cv": 0.08},
            "quality": {
                "valid_frames": 150,
                "valid_steps": 20,
                "walking_duration_s": 10.0,
                "mean_confidence": 0.5,
            },
        })

        result = assess_quality(observation, config)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.reasons, ())

    def test_quality_returns_all_chinese_rejection_reasons(self):
        config = LongitudinalConfig(
            core_features=("cadence_spm", "step_time_cv", "trunk_sway_rms_deg")
        )
        observation = DailyObservation.from_dict({
            "person_id": "elder_001",
            "observed_at": "2026-11-01",
            "features": {"cadence_spm": 92.5},
            "quality": {
                "valid_frames": 100,
                "valid_steps": 10,
                "walking_duration_s": 5.0,
                "mean_confidence": 0.4,
            },
        })

        result = assess_quality(observation, config)

        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.reasons), 5)
        self.assertTrue(any("有效帧数" in reason for reason in result.reasons))
        self.assertTrue(any("核心特征覆盖率" in reason for reason in result.reasons))


if __name__ == "__main__":
    unittest.main()
