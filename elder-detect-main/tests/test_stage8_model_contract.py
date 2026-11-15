"""阶段 8：模型输入契约、研究接口拒绝规则与服务入口测试。"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.baseline import (  # noqa: E402
    build_baseline, rms_distance_config_fingerprint,
)
from longitudinal.config import LongitudinalConfig  # noqa: E402
from longitudinal.ds_pcd import (  # noqa: E402
    DayRecord, DsPcdConfig, config_fingerprint as ds_pcd_fingerprint,
    fit_baseline, scoring_config_fingerprint as ds_pcd_scoring_fingerprint,
)
from longitudinal.model_contract import (  # noqa: E402
    build_input_contract, validate_input_contract,
)
from longitudinal.research import research_prediction  # noqa: E402
from longitudinal.schema import DailyObservation  # noqa: E402
from longitudinal.service import LongitudinalRiskService  # noqa: E402
from longitudinal.storage import ProfileStore  # noqa: E402
from longitudinal.survival import (  # noqa: E402
    CoxModelArtifact, predict_cumulative_risk_research,
)

FEATURES = ("walking_speed_m_s", "cadence_spm", "stride_duration_s")
DIRECTIONS = {"walking_speed_m_s": -1.0, "cadence_spm": -1.0, "stride_duration_s": 1.0}
START = date(2024, 1, 1)
PROVENANCE = {
    "modality": "wearable",
    "feature_version": "mobilise-d-daily-v1",
    "precision_mode": "session_se",
    "ds_pcd_config_fingerprint": "ds-fp",
}
COVARIATES = {"rms_distance": 1.25, "cusum": 3.5}


def make_contract(*, modality: str = "wearable",
                  feature_version: str = "mobilise-d-daily-v1",
                  precision: str = "session_se",
                  fingerprint: str = "ds-fp") -> dict:
    return build_input_contract(
        modality=modality, feature_version=feature_version,
        features={
            "rms_distance": {
                "definition": "root_mean_square_distance_from_personal_baseline",
                "source": "service.distance", "requires_ds_pcd": False,
            },
            "cusum": {
                "definition": "ds_pcd_cusum_directional_accumulation",
                "source": "service.ds_pcd.cusum", "requires_ds_pcd": True,
            },
        },
        ds_pcd={
            "required": True, "precision_mode": precision,
            "config_fingerprint": fingerprint,
        },
    )


def make_model(contract: dict | None, *, calibrated: bool = False,
               horizon: int = 30) -> CoxModelArtifact:
    payload = {
        "schema_version": 2, "model_type": "landmark_cox",
        "bundle_id": "stage8-test", "horizon_days": horizon,
        "feature_names": ["rms_distance", "cusum"],
        "coefficients": [0.1, 0.2], "feature_centers": [0.0, 0.0],
        "baseline_survival": 0.9, "calibrated": calibrated,
        "trained_at": "2024-01-01T00:00:00+00:00", "evidence_ids": ["test"],
    }
    if contract is not None:
        payload["input_contract"] = contract
    return CoxModelArtifact.from_dict(payload)


def result(*, provenance: dict | None = None,
           covariates: dict | None = None,
           ds_pcd: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        research_covariates=dict(COVARIATES if covariates is None else covariates),
        provenance=dict(PROVENANCE if provenance is None else provenance),
        ds_pcd=({} if ds_pcd is None else ds_pcd),
        research_covariate_status={"rms_distance": "available", "cusum": "available"},
    )


class Stage8ContractTest(unittest.TestCase):
    def test_serialization_roundtrip_and_match(self):
        contract = make_contract()
        model = make_model(contract)
        restored = CoxModelArtifact.from_dict(model.to_dict())
        self.assertEqual(restored.input_contract, contract)
        self.assertEqual(restored.to_dict(), model.to_dict())
        prediction = research_prediction(result(), restored)
        expected = predict_cumulative_risk_research(COVARIATES, restored)
        self.assertEqual(prediction["status"], "ok")
        self.assertAlmostEqual(prediction["risk"], expected)
        self.assertFalse(prediction["calibrated"])
        self.assertTrue(prediction["research_only"])

    def test_contract_missing_is_explicit(self):
        prediction = research_prediction(result(), make_model(None))
        self.assertEqual(prediction["status"], "contract_missing")
        self.assertIsNone(prediction["risk"])

    def test_modality_mismatch(self):
        prediction = research_prediction(
            result(provenance={**PROVENANCE, "modality": "video"}),
            make_model(make_contract()),
        )
        self.assertEqual(prediction["status"], "modality_mismatch")
        self.assertIsNone(prediction["risk"])

    def test_feature_version_mismatch(self):
        prediction = research_prediction(
            result(provenance={**PROVENANCE, "feature_version": "video-pose-v1"}),
            make_model(make_contract()),
        )
        self.assertEqual(prediction["status"], "feature_version_mismatch")
        self.assertIsNone(prediction["risk"])

    def test_precision_and_ds_config_mismatch(self):
        prediction = research_prediction(
            result(provenance={**PROVENANCE, "precision_mode": "none"}),
            make_model(make_contract()),
        )
        self.assertEqual(prediction["status"], "precision_mode_mismatch")
        self.assertIsNone(prediction["risk"])
        prediction = research_prediction(
            result(provenance={**PROVENANCE, "ds_pcd_config_fingerprint": "other"}),
            make_model(make_contract()),
        )
        self.assertEqual(prediction["status"], "ds_pcd_config_mismatch")
        self.assertIsNone(prediction["risk"])

    def test_missing_and_invalid_covariates(self):
        prediction = research_prediction(
            result(covariates={"rms_distance": 1.0, "cusum": None}),
            make_model(make_contract()),
        )
        self.assertEqual(prediction["status"], "missing_covariates")
        self.assertEqual(prediction["missing"], ["cusum"])
        self.assertIsNone(prediction["risk"])
        prediction = research_prediction(
            result(covariates={"rms_distance": 1.0, "cusum": float("nan")}),
            make_model(make_contract()),
        )
        self.assertEqual(prediction["status"], "invalid_covariates")
        self.assertIsNone(prediction["risk"])

    def test_missing_provenance_and_invalid_contract(self):
        prediction = research_prediction(result(provenance={}), make_model(make_contract()))
        self.assertEqual(prediction["status"], "missing_provenance")
        self.assertIsNone(prediction["risk"])
        bad = make_contract()
        bad["schema_version"] = 99
        prediction = research_prediction(result(), make_model(bad))
        self.assertEqual(prediction["status"], "contract_invalid")
        self.assertIsNone(prediction["risk"])
        bad = make_contract()
        del bad["features"]["cusum"]
        prediction = research_prediction(result(), make_model(bad))
        self.assertEqual(prediction["status"], "contract_invalid")
        self.assertIsNone(prediction["risk"])

    def test_contract_validation_rejects_unknown_version(self):
        with self.assertRaises(ValueError):
            validate_input_contract({"schema_version": 2})

    def test_population_is_informational_not_a_hard_gate(self):
        contract = make_contract()
        contract["population"] = "model population"
        prediction = research_prediction(
            result(provenance={**PROVENANCE, "population": "different display text"}),
            make_model(contract),
        )
        self.assertEqual(prediction["status"], "ok")
        self.assertEqual(prediction["matched"]["population"], "different display text")

    def test_scoped_rms_fingerprint_matches_full_fingerprint_mismatch(self):
        contract = make_contract()
        contract["rms_distance_config_fingerprint"] = "rms-scoped"
        provenance = {**PROVENANCE, "rms_distance_config_fingerprint": "rms-scoped",
                      "rms_config_fingerprint": "different-full"}
        self.assertEqual(
            research_prediction(result(provenance=provenance), make_model(contract))["status"],
            "ok",
        )
        provenance["rms_distance_config_fingerprint"] = "other"
        prediction = research_prediction(result(provenance=provenance), make_model(contract))
        self.assertEqual(prediction["status"], "rms_config_mismatch")
        self.assertIsNone(prediction["risk"])

    def test_scoped_ds_fingerprint_and_legacy_full_fingerprint(self):
        contract = make_contract()
        contract["ds_pcd"].pop("config_fingerprint", None)
        contract["ds_pcd"]["scoring_config_fingerprint"] = "ds-scoped"
        provenance = {**PROVENANCE, "ds_pcd_scoring_config_fingerprint": "ds-scoped",
                      "ds_pcd_config_fingerprint": "different-full"}
        self.assertEqual(
            research_prediction(result(provenance=provenance), make_model(contract))["status"],
            "ok",
        )
        provenance["ds_pcd_scoring_config_fingerprint"] = "other"
        prediction = research_prediction(result(provenance=provenance), make_model(contract))
        self.assertEqual(prediction["status"], "ds_pcd_config_mismatch")
        self.assertIsNone(prediction["risk"])
        legacy = make_contract()
        legacy["ds_pcd"].pop("config_fingerprint", None)
        legacy["ds_pcd"]["config_fingerprint"] = "legacy-full"
        provenance = {**PROVENANCE, "ds_pcd_config_fingerprint": "legacy-full"}
        self.assertEqual(
            research_prediction(result(provenance=provenance), make_model(legacy))["status"],
            "ok",
        )

    def test_fingerprint_helpers_only_cover_relevant_fields(self):
        base = LongitudinalConfig()
        same = replace(base, persistent_days=99, persistent_distance_threshold=9.0,
                       timezone="UTC")
        changed = replace(base, min_valid_steps=99)
        self.assertEqual(
            rms_distance_config_fingerprint(base),
            rms_distance_config_fingerprint(same),
        )
        self.assertNotEqual(
            rms_distance_config_fingerprint(base),
            rms_distance_config_fingerprint(changed),
        )
        ds_base = DsPcdConfig()
        ds_same = replace(ds_base, bootstrap_samples=500, min_bouts=10)
        ds_changed = replace(ds_base, k_ref=1.0)
        self.assertEqual(
            ds_pcd_scoring_fingerprint(ds_base),
            ds_pcd_scoring_fingerprint(ds_same),
        )
        self.assertNotEqual(
            ds_pcd_scoring_fingerprint(ds_base),
            ds_pcd_scoring_fingerprint(ds_changed),
        )


class Stage8ServiceTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = ProfileStore(self.directory.name, "p1")
        self.config = LongitudinalConfig(
            core_features=FEATURES,
            min_valid_frames=0, min_valid_steps=0, min_walking_duration_s=0.0,
            min_mean_confidence=0.0, min_core_feature_coverage=1.0,
        )
        self.se = {"walking_speed_m_s": 0.01, "cadence_spm": 0.2, "stride_duration_s": 0.002}
        values = {}
        rng = np.random.default_rng(7)
        for day in range(15):
            values[day] = {
                "walking_speed_m_s": 0.9 + 0.01 * rng.normal(),
                "cadence_spm": 92.0 + 0.5 * rng.normal(),
                "stride_duration_s": 1.18 + 0.005 * rng.normal(),
            }
            self.store.append_observation(self._observation(day, values[day]))
        self.values = values
        self.store.save_baseline(build_baseline(self.store.load_observations(), self.config))
        self.ds_config = DsPcdConfig(
            features=FEATURES, directions=DIRECTIONS, precision_mode="session_se",
            null_samples=300, quantile_points=100, outlier_z=100.0,
        )
        self.store.save_ds_pcd_baseline(fit_baseline(
            [DayRecord(date=self._date(day), features=features, feature_se=self.se)
             for day, features in values.items()],
            "p1", self.ds_config,
            baseline_dates=tuple(self._date(day) for day in range(14)),
        ))
        self.contract = build_input_contract(
            modality="wearable", feature_version="mobilise-d-daily-v1",
            features={
                "rms_distance": {
                    "definition": "root_mean_square_distance_from_personal_baseline",
                    "source": "service.distance", "requires_ds_pcd": False,
                },
                "cusum": {
                    "definition": "ds_pcd_cusum_directional_accumulation",
                    "source": "service.ds_pcd.cusum", "requires_ds_pcd": True,
                },
            },
            ds_pcd={
                "required": True, "precision_mode": "session_se",
                "config_fingerprint": ds_pcd_fingerprint(self.ds_config),
            },
        )

    @staticmethod
    def _date(day: int) -> str:
        return (START + timedelta(days=day)).isoformat()

    def _observation(self, day: int, features: dict) -> DailyObservation:
        return DailyObservation.from_dict({
            "person_id": "p1", "observed_at": self._date(day),
            "features": features, "feature_se": self.se,
            "quality": {"valid_frames": 0, "valid_steps": 0,
                        "walking_duration_s": 0.0, "mean_confidence": 0.0},
            "source": "mobilise-d-cvs", "timezone": "UTC",
        })

    def _service(self, models):
        return LongitudinalRiskService(
            self.store, self.config, models=models,
            ds_pcd_config=self.ds_config, modality="wearable",
            feature_version="mobilise-d-daily-v1",
        )

    def test_service_uses_declared_contract(self):
        service = self._service({
            7: make_model(self.contract, calibrated=True, horizon=7),
            30: make_model(self.contract, calibrated=True, horizon=30),
        })
        current = self._observation(14, {
            "walking_speed_m_s": 0.84, "cadence_spm": 90.0, "stride_duration_s": 1.2,
        })
        result = service.evaluate("p1", current, persist=False)
        self.assertEqual(result.status, "calibrated")
        self.assertIsNotNone(result.risk_7d)
        self.assertIsNotNone(result.risk_30d)
        self.assertEqual(result.model_input_status["30"]["status"], "ok")
        self.assertEqual(result.provenance["score_version"], "legacy-rms-v1")

    def test_service_rejects_contract_mismatch(self):
        bad = dict(self.contract)
        bad["feature_version"] = "video-pose-v1"
        service = self._service({
            7: make_model(self.contract, calibrated=True, horizon=7),
            30: make_model(bad, calibrated=True, horizon=30),
        })
        current = self._observation(14, {
            "walking_speed_m_s": 0.84, "cadence_spm": 90.0, "stride_duration_s": 1.2,
        })
        result = service.evaluate("p1", current, persist=False)
        self.assertEqual(result.status, "model_input_mismatch")
        self.assertIsNone(result.risk_7d)
        self.assertIsNone(result.risk_30d)
        self.assertEqual(result.model_input_status["30"]["status"], "feature_version_mismatch")

    def test_service_keeps_legacy_model_without_contract(self):
        service = self._service({
            7: make_model(None, calibrated=True, horizon=7),
            30: make_model(None, calibrated=True, horizon=30),
        })
        current = self._observation(14, {
            "walking_speed_m_s": 0.84, "cadence_spm": 90.0, "stride_duration_s": 1.2,
        })
        result = service.evaluate("p1", current, persist=False)
        self.assertEqual(result.status, "calibrated")
        self.assertEqual(result.model_input_status["30"]["status"], "legacy_no_contract")
        self.assertIsNotNone(result.risk_30d)


if __name__ == "__main__":
    unittest.main()
