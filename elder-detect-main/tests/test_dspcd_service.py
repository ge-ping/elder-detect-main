"""DS-PCD 接入纵向服务后的时间截断与一致性测试。"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.baseline import build_baseline  # noqa: E402
from longitudinal.config import LongitudinalConfig  # noqa: E402
from longitudinal.deviation import compute_deviation  # noqa: E402
from longitudinal.ds_pcd import (  # noqa: E402
    DayRecord,
    DsPcdConfig,
    fit_baseline,
)
from longitudinal.schema import DailyObservation  # noqa: E402
from longitudinal.service import LongitudinalRiskService  # noqa: E402
from longitudinal.storage import ProfileStore  # noqa: E402


FEATURES = ("walking_speed_m_s", "cadence_spm", "stride_duration_s")
DIRECTIONS = {"walking_speed_m_s": -1.0, "cadence_spm": -1.0, "stride_duration_s": 1.0}
BASE = {"walking_speed_m_s": 0.9, "cadence_spm": 92.0, "stride_duration_s": 1.18}
START = date(2024, 1, 1)


def wearable_config(**overrides) -> LongitudinalConfig:
    payload = dict(
        core_features=FEATURES,
        min_valid_frames=0,
        min_valid_steps=0,
        min_walking_duration_s=0.0,
        min_mean_confidence=0.0,
        min_core_feature_coverage=1.0,
    )
    payload.update(overrides)
    return LongitudinalConfig(**payload)


def observation(
    day: int,
    features: dict[str, float],
    *,
    se: dict[str, float] | None = None,
    person: str = "p1",
) -> DailyObservation:
    payload = {
        "person_id": person,
        "observed_at": (START + timedelta(days=day)).isoformat(),
        "features": {name: float(features[name]) for name in FEATURES},
        "quality": {
            "valid_frames": 0,
            "valid_steps": 0,
            "walking_duration_s": 0.0,
            "mean_confidence": 0.0,
        },
        "source": "unit-test",
    }
    if se is not None:
        payload["feature_se"] = {name: float(se[name]) for name in FEATURES}
    return DailyObservation.from_dict(payload)


def baseline_values(count: int = 14, seed: int = 3) -> dict[int, dict[str, float]]:
    rng = np.random.default_rng(seed)
    values = {}
    for day in range(count):
        values[day] = {
            "walking_speed_m_s": BASE["walking_speed_m_s"] + 0.01 * rng.normal(),
            "cadence_spm": BASE["cadence_spm"] + 0.5 * rng.normal(),
            "stride_duration_s": BASE["stride_duration_s"] + 0.005 * rng.normal(),
        }
    return values


def records_for(values: dict[int, dict[str, float]], se: dict[str, float] | None = None):
    return [
        DayRecord(
            date=(START + timedelta(days=day)).isoformat(),
            features=features,
            feature_se=se or {},
        )
        for day, features in sorted(values.items())
    ]


class DsPcdServiceTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = ProfileStore(self.directory.name, "p1")
        self.config = wearable_config()
        self.values = baseline_values()
        for day, features in self.values.items():
            self.store.append_observation(observation(day, features))
        self.rms_baseline = build_baseline(self.store.load_observations(), self.config)
        self.store.save_baseline(self.rms_baseline)
        self.ds_config = DsPcdConfig(
            features=FEATURES,
            directions=DIRECTIONS,
            precision_mode="none",
            null_samples=2000,
            quantile_points=200,
            outlier_z=100.0,
        )
        self.ds_baseline = fit_baseline(
            records_for(self.values), "p1", self.ds_config
        )
        self.store.save_ds_pcd_baseline(self.ds_baseline)

    def test_same_day_outputs_rms_and_ds_pcd(self):
        service = LongitudinalRiskService(
            self.store, self.config, ds_pcd_config=self.ds_config
        )
        current = observation(14, {
            "walking_speed_m_s": 0.82, "cadence_spm": 90.0, "stride_duration_s": 1.22,
        })
        result = service.evaluate("p1", current, persist=False)
        self.assertEqual(result.status, "baseline_ready")
        expected = compute_deviation(current.features, self.rms_baseline)
        self.assertAlmostEqual(result.distance, expected.distance)
        self.assertIsNotNone(result.ds_pcd)
        self.assertEqual(result.ds_pcd["status"], "scored")
        self.assertEqual(result.ds_pcd["detector"], "ds_pcd")
        self.assertAlmostEqual(
            result.ds_pcd["cusum"],
            max(0.0, result.ds_pcd["q"] - self.ds_config.k_ref),
        )
        self.assertEqual(result.as_of, current.observation_date)
        self.assertEqual(result.baseline_source, "stored")

    def test_appending_later_anomalies_does_not_change_earlier_result(self):
        service = LongitudinalRiskService(
            self.store, self.config, ds_pcd_config=self.ds_config
        )
        current = observation(14, {
            "walking_speed_m_s": 0.84, "cadence_spm": 90.5, "stride_duration_s": 1.2,
        })
        before = service.evaluate("p1", current, persist=False).to_dict()
        for day in (18, 19, 20):
            self.store.append_observation(observation(day, {
                "walking_speed_m_s": 0.4, "cadence_spm": 120.0, "stride_duration_s": 0.7,
            }))
        after = service.evaluate("p1", current, persist=False).to_dict()
        self.assertEqual(before, after)

    def test_day_by_day_profile_matches_isolated_replay(self):
        service = LongitudinalRiskService(
            self.store, self.config, ds_pcd_config=self.ds_config
        )
        current = observation(14, {
            "walking_speed_m_s": 0.86, "cadence_spm": 91.0, "stride_duration_s": 1.19,
        })
        full = service.evaluate("p1", current, persist=False).to_dict()

        with tempfile.TemporaryDirectory() as directory:
            isolated = ProfileStore(directory, "p1")
            for day in range(15):
                isolated.append_observation(observation(day, self.values.get(day, {
                    "walking_speed_m_s": 0.86, "cadence_spm": 91.0, "stride_duration_s": 1.19,
                })))
            isolated.save_baseline(self.rms_baseline)
            isolated.save_ds_pcd_baseline(self.ds_baseline)
            replay = LongitudinalRiskService(
                isolated, self.config, ds_pcd_config=self.ds_config
            ).evaluate("p1", current, persist=False).to_dict()
        for key in ("distance", "deviation_index", "ds_pcd", "trend_7d_window"):
            self.assertEqual(full[key], replay[key])

    def test_strict_baseline_order_rejects_prediction_on_or_before_baseline_end(self):
        service = LongitudinalRiskService(
            self.store, self.config, ds_pcd_config=self.ds_config
        )
        # 基线建立当天：默认严格，允许旧行为时显式标记
        inside = observation(13, self.values[13])
        with self.assertRaisesRegex(ValueError, "预测日"):
            service.evaluate("p1", inside, as_of=inside.observation_date)
        lenient = service.evaluate(
            "p1", inside, as_of=inside.observation_date,
            strict_baseline_order=False, persist=False,
        )
        self.assertTrue(lenient.baseline_contains_prediction)
        self.assertEqual(lenient.baseline_source, "stored_contains_prediction")

    def test_sparse_timeline_with_gap_matches_isolated_replay(self):
        service = LongitudinalRiskService(
            self.store, self.config, ds_pcd_config=self.ds_config
        )
        sparse_days = {
            14: {"walking_speed_m_s": 0.88, "cadence_spm": 91.0, "stride_duration_s": 1.19},
            15: {"walking_speed_m_s": 0.86, "cadence_spm": 90.0, "stride_duration_s": 1.2},
            23: {"walking_speed_m_s": 0.80, "cadence_spm": 88.0, "stride_duration_s": 1.24},
            24: {"walking_speed_m_s": 0.78, "cadence_spm": 87.0, "stride_duration_s": 1.25},
        }
        for day, features in sparse_days.items():
            self.store.append_observation(observation(day, features))
        target = observation(24, sparse_days[24])
        full = service.evaluate("p1", target, persist=False).to_dict()
        self.assertGreaterEqual(full["ds_pcd"]["cusum_resets_to_as_of"], 1)
        with tempfile.TemporaryDirectory() as directory:
            isolated = ProfileStore(directory, "p1")
            for day in range(14):
                isolated.append_observation(observation(day, self.values[day]))
            for day, features in sparse_days.items():
                isolated.append_observation(observation(day, features))
            isolated.save_baseline(self.rms_baseline)
            isolated.save_ds_pcd_baseline(self.ds_baseline)
            replay = LongitudinalRiskService(
                isolated, self.config, ds_pcd_config=self.ds_config
            ).evaluate("p1", target, persist=False).to_dict()
        for key in ("distance", "ds_pcd", "trend_7d_window"):
            self.assertEqual(full[key], replay[key])

    def test_missing_ds_baseline_is_reported_without_breaking_rms(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(directory, "p1")
            for day, features in self.values.items():
                store.append_observation(observation(day, features))
            store.save_baseline(self.rms_baseline)
            result = LongitudinalRiskService(
                store, self.config, ds_pcd_config=self.ds_config
            ).evaluate("p1", observation(14, BASE), persist=False)
        self.assertEqual(result.status, "baseline_ready")
        self.assertEqual(result.ds_pcd["status"], "no_baseline")

    def test_trend_7d_uses_calendar_days(self):
        service = LongitudinalRiskService(
            self.store, self.config, ds_pcd_config=self.ds_config
        )
        early = observation(14, {
            "walking_speed_m_s": 0.84, "cadence_spm": 90.0, "stride_duration_s": 1.2,
        })
        late = observation(20, {
            "walking_speed_m_s": 0.80, "cadence_spm": 89.0, "stride_duration_s": 1.24,
        })
        self.store.append_observation(early)
        first = service.evaluate("p1", early, persist=False)
        second = service.evaluate("p1", late, persist=False)
        window = second.trend_7d_window
        self.assertEqual(window["points"], 2)
        self.assertEqual(window["span_days"], 6)
        expected = (second.distance - first.distance) / 6
        self.assertAlmostEqual(second.to_dict()["trend_7d_window"]["span_days"], 6)
        # 与旧口径（按观测条数）不同：两天相隔 6 天时不应除以 1
        self.assertNotAlmostEqual(expected, second.distance - first.distance)

    def test_session_se_mode_excludes_days_without_se(self):
        se = {"walking_speed_m_s": 0.01, "cadence_spm": 0.2, "stride_duration_s": 0.002}
        config = DsPcdConfig(
            features=FEATURES, directions=DIRECTIONS, precision_mode="session_se",
            null_samples=2000, quantile_points=200, outlier_z=100.0,
        )
        baseline = fit_baseline(records_for(self.values, se), "p1", config)
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(directory, "p1")
            for day, features in self.values.items():
                store.append_observation(observation(day, features, se=se))
            store.save_baseline(self.rms_baseline)
            store.save_ds_pcd_baseline(baseline)
            service = LongitudinalRiskService(store, self.config, ds_pcd_config=config)
            without = service.evaluate(
                "p1", observation(14, BASE), persist=False,
            )
            with_se = service.evaluate(
                "p1", observation(14, BASE, se=se), persist=False,
            )
        self.assertEqual(without.ds_pcd["status"], "excluded_missing_se")
        self.assertEqual(with_se.ds_pcd["status"], "scored")


if __name__ == "__main__":
    unittest.main()
