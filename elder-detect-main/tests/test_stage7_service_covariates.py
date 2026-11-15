"""阶段 7：服务研究协变量、来源信息与研究接口的单元测试。"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.baseline import build_baseline  # noqa: E402
from longitudinal.config import LongitudinalConfig  # noqa: E402
from longitudinal.ds_pcd import (  # noqa: E402
    DayRecord, DsPcdConfig, config_fingerprint as ds_pcd_fingerprint, fit_baseline,
)
from longitudinal.model_contract import build_input_contract  # noqa: E402
from longitudinal.research import research_prediction  # noqa: E402
from longitudinal.schema import DailyObservation  # noqa: E402
from longitudinal.service import LongitudinalRiskService  # noqa: E402
from longitudinal.storage import ProfileStore  # noqa: E402
from longitudinal.survival import (  # noqa: E402
    CoxModelArtifact,
    predict_cumulative_risk,
    predict_cumulative_risk_research,
)


FEATURES = ("walking_speed_m_s", "cadence_spm", "stride_duration_s")
DIRECTIONS = {"walking_speed_m_s": -1.0, "cadence_spm": -1.0, "stride_duration_s": 1.0}
START = date(2024, 1, 1)


def wearable_config() -> LongitudinalConfig:
    return LongitudinalConfig(
        core_features=FEATURES,
        min_valid_frames=0, min_valid_steps=0, min_walking_duration_s=0.0,
        min_mean_confidence=0.0, min_core_feature_coverage=1.0,
    )


def observation(day: int, features: dict[str, float], *, se: dict[str, float] | None = None):
    payload = {
        "person_id": "p1",
        "observed_at": (START + timedelta(days=day)).isoformat(),
        "features": features,
        "quality": {"valid_frames": 0, "valid_steps": 0,
                    "walking_duration_s": 0.0, "mean_confidence": 0.0},
        "source": "mobilise-d-cvs",
        "timezone": "UTC",
    }
    if se is not None:
        payload["feature_se"] = se
    return DailyObservation.from_dict(payload)


def baseline_values() -> dict[int, dict[str, float]]:
    rng = np.random.default_rng(4)
    return {
        day: {
            "walking_speed_m_s": 0.9 + 0.01 * rng.normal(),
            "cadence_spm": 92.0 + 0.5 * rng.normal(),
            "stride_duration_s": 1.18 + 0.005 * rng.normal(),
        }
        for day in range(15)
    }


def contract(features: list[str], ds_config: DsPcdConfig | None = None) -> dict:
    feature_specs = {
        "rms_distance": {
            "definition": "root_mean_square_distance_from_personal_baseline",
            "source": "service.distance", "requires_ds_pcd": False,
        },
    }
    ds = {"required": False}
    if "cusum" in features:
        assert ds_config is not None
        feature_specs["cusum"] = {
            "definition": "ds_pcd_cusum_directional_accumulation",
            "source": "service.ds_pcd.cusum", "requires_ds_pcd": True,
        }
        ds = {
            "required": True,
            "precision_mode": ds_config.precision_mode,
            "config_fingerprint": ds_pcd_fingerprint(ds_config),
            "features": list(ds_config.features),
            "directions": dict(ds_config.directions),
            "k_ref": ds_config.k_ref,
            "gap_reset_days": ds_config.gap_reset_days,
        }
    return build_input_contract(
        modality="wearable", feature_version="mobilise-d-daily-v1",
        features={name: feature_specs[name] for name in features}, ds_pcd=ds,
    )


def model(features: list[str], calibrated: bool,
          input_contract: dict | None = None) -> CoxModelArtifact:
    payload = {
        "schema_version": 2, "model_type": "landmark_cox",
        "bundle_id": "stage7-test", "horizon_days": 30,
        "feature_names": features,
        "coefficients": [0.1 for _ in features],
        "feature_centers": [0.0 for _ in features],
        "baseline_survival": 0.9, "calibrated": calibrated,
        "trained_at": "2024-01-01T00:00:00+00:00", "evidence_ids": ["test"],
    }
    if input_contract is not None:
        payload["input_contract"] = input_contract
    return CoxModelArtifact.from_dict(payload)


class Stage7ServiceTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = ProfileStore(self.directory.name, "p1")
        self.config = wearable_config()
        self.values = baseline_values()
        self.se = {"walking_speed_m_s": 0.01, "cadence_spm": 0.2, "stride_duration_s": 0.002}
        for day, features in self.values.items():
            self.store.append_observation(observation(day, features, se=self.se))
        self.rms = build_baseline(self.store.load_observations(), self.config)
        self.store.save_baseline(self.rms)
        self.ds_config = DsPcdConfig(
            features=FEATURES, directions=DIRECTIONS, precision_mode="session_se",
            null_samples=1000, quantile_points=200, outlier_z=100.0,
        )
        self.ds_baseline = fit_baseline(
            [DayRecord(date=(START + timedelta(days=d)).isoformat(), features=f,
                       feature_se=self.se)
             for d, f in self.values.items()],
            "p1", self.ds_config, baseline_dates=tuple(
                (START + timedelta(days=d)).isoformat() for d in range(14)
            ),
        )
        self.store.save_ds_pcd_baseline(self.ds_baseline)

    def test_service_exposes_research_covariates_and_provenance(self):
        service = LongitudinalRiskService(
            self.store, self.config, ds_pcd_config=self.ds_config,
            modality="wearable", population="65+ Parkinson",
            feature_version="mobilise-d-daily-v1",
        )
        current = observation(14, {
            "walking_speed_m_s": 0.84, "cadence_spm": 90.0, "stride_duration_s": 1.2,
        }, se=self.se)
        result = service.evaluate("p1", current, persist=False)
        self.assertEqual(result.status, "baseline_ready")
        self.assertIsNotNone(result.distance)
        self.assertAlmostEqual(result.research_covariates["rms_distance"], result.distance)
        self.assertAlmostEqual(result.research_covariates["distance"], result.distance)
        self.assertIsNotNone(result.research_covariates["cusum"])
        self.assertEqual(result.research_covariate_status["cusum"], "available")
        self.assertEqual(result.provenance["modality"], "wearable")
        self.assertEqual(result.provenance["population"], "65+ Parkinson")
        self.assertEqual(result.provenance["feature_version"], "mobilise-d-daily-v1")
        self.assertEqual(result.provenance["precision_mode"], "session_se")
        self.assertEqual(result.valid_day_counts["n_valid_days"], 15)
        self.assertEqual(result.valid_day_counts["n_valid_days_before_landmark"], 14)
        self.assertIsNone(result.risk_7d)
        self.assertIsNone(result.risk_30d)

    def test_research_prediction_uses_service_covariates(self):
        service = LongitudinalRiskService(
            self.store, self.config, ds_pcd_config=self.ds_config,
            modality="wearable", feature_version="mobilise-d-daily-v1",
        )
        current = observation(14, {
            "walking_speed_m_s": 0.84, "cadence_spm": 90.0, "stride_duration_s": 1.2,
        }, se=self.se)
        result = service.evaluate("p1", current, persist=False)
        artifact = model(
            ["rms_distance", "cusum"], calibrated=False,
            input_contract=contract(["rms_distance", "cusum"], self.ds_config),
        )
        prediction = research_prediction(result, artifact)
        self.assertEqual(prediction["status"], "ok")
        self.assertFalse(prediction["calibrated"])
        self.assertTrue(prediction["research_only"])
        expected = predict_cumulative_risk_research(
            {"rms_distance": result.research_covariates["rms_distance"],
             "cusum": result.research_covariates["cusum"]},
            artifact,
        )
        self.assertAlmostEqual(prediction["risk"], expected)
        with self.assertRaisesRegex(ValueError, "尚未校准"):
            predict_cumulative_risk(
                {"rms_distance": 1.0, "cusum": 0.2}, artifact,
            )

    def test_missing_ds_baseline_reports_cusum_status_without_breaking_rms(self):
        with __import__("tempfile").TemporaryDirectory() as directory:
            store = ProfileStore(directory, "p1")
            for day, features in self.values.items():
                store.append_observation(observation(day, features, se=self.se))
            store.save_baseline(self.rms)
            service = LongitudinalRiskService(
                store, self.config, ds_pcd_config=self.ds_config,
                modality="wearable", feature_version="mobilise-d-daily-v1",
            )
            current = observation(14, {
                "walking_speed_m_s": 0.84, "cadence_spm": 90.0, "stride_duration_s": 1.2,
            }, se=self.se)
            result = service.evaluate("p1", current, persist=False)
        self.assertIsNotNone(result.distance)
        self.assertIsNone(result.research_covariates["cusum"])
        self.assertTrue(result.research_covariate_status["cusum"].startswith("missing:"))
        prediction = research_prediction(
            result,
            model(
                ["rms_distance", "cusum"], False,
                input_contract=contract(["rms_distance", "cusum"], self.ds_config),
            ),
        )
        self.assertEqual(prediction["status"], "missing_covariates")
        self.assertEqual(prediction["missing"], ["cusum"])

    def test_model_requiring_missing_cusum_keeps_rms_and_reports_incomplete(self):
        with __import__("tempfile").TemporaryDirectory() as directory:
            store = ProfileStore(directory, "p1")
            for day, features in self.values.items():
                store.append_observation(observation(day, features, se=self.se))
            store.save_baseline(self.rms)
            service = LongitudinalRiskService(
                store, self.config,
                models={7: model(["rms_distance", "cusum"], True),
                        30: model(["rms_distance", "cusum"], True)},
                ds_pcd_config=self.ds_config, modality="wearable",
            )
            current = observation(14, {
                "walking_speed_m_s": 0.84, "cadence_spm": 90.0, "stride_duration_s": 1.2,
            }, se=self.se)
            result = service.evaluate("p1", current, persist=False)
        self.assertEqual(result.status, "model_inputs_incomplete")
        self.assertIsNotNone(result.distance)
        self.assertIsNone(result.risk_30d)

    def test_old_distance_only_model_still_works(self):
        service = LongitudinalRiskService(
            self.store, self.config,
            models={7: model(["distance", "persistent_deviation_days"], True),
                    30: model(["distance", "persistent_deviation_days"], True)},
        )
        current = observation(14, {
            "walking_speed_m_s": 0.84, "cadence_spm": 90.0, "stride_duration_s": 1.2,
        }, se=self.se)
        result = service.evaluate("p1", current, persist=False)
        self.assertEqual(result.status, "calibrated")
        self.assertIsNotNone(result.risk_30d)
        self.assertAlmostEqual(result.research_covariates["rms_distance"], result.distance)
        self.assertIsNone(result.research_covariates["cusum"])


if __name__ == "__main__":
    unittest.main()
