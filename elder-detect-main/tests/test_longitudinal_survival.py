import json
import math
from pathlib import Path
import tempfile
import unittest

from longitudinal.survival import (
    CoxModelArtifact,
    load_model_bundle,
    predict_cumulative_risk,
)


def artifact_data(
    horizon: int = 7,
    calibrated: bool = True,
    bundle_id: str = "bundle-001",
):
    return {
        "schema_version": 2,
        "model_type": "landmark_cox",
        "bundle_id": bundle_id,
        "horizon_days": horizon,
        "feature_names": ["distance", "trend_7d"],
        "coefficients": [math.log(2.0), 0.0],
        "feature_centers": [1.0, 0.0],
        "baseline_survival": 0.8 if horizon == 7 else 0.6,
        "calibrated": calibrated,
        "trained_at": "2026-11-11T00:00:00+00:00",
        "evidence_ids": ["mehdizadeh_2021", "cox_1972", "landmarking_2007"],
        "training_summary": {"persons": 100, "events": 20},
    }


class CoxPredictionTest(unittest.TestCase):
    def test_predicts_cumulative_risk_from_cox_formula(self):
        artifact = CoxModelArtifact.from_dict(artifact_data())

        risk = predict_cumulative_risk(
            {"distance": 2.0, "trend_7d": 3.0}, artifact
        )

        self.assertAlmostEqual(risk, 0.36)

    def test_uncalibrated_artifact_cannot_output_probability(self):
        artifact = CoxModelArtifact.from_dict(artifact_data(calibrated=False))

        with self.assertRaisesRegex(ValueError, "尚未校准"):
            predict_cumulative_risk({"distance": 1.0, "trend_7d": 0.0}, artifact)

    def test_string_false_cannot_bypass_calibration_gate(self):
        data = artifact_data()
        data["calibrated"] = "false"

        with self.assertRaisesRegex(ValueError, "calibrated"):
            CoxModelArtifact.from_dict(data)

    def test_missing_feature_is_rejected(self):
        artifact = CoxModelArtifact.from_dict(artifact_data())

        with self.assertRaisesRegex(ValueError, "缺少模型特征"):
            predict_cumulative_risk({"distance": 1.0}, artifact)

    def test_invalid_horizon_and_survival_are_rejected(self):
        bad_horizon = artifact_data(horizon=24)
        with self.assertRaisesRegex(ValueError, "7天或30天"):
            CoxModelArtifact.from_dict(bad_horizon)
        bad_survival = artifact_data()
        bad_survival["baseline_survival"] = 1.2
        with self.assertRaisesRegex(ValueError, "baseline_survival"):
            CoxModelArtifact.from_dict(bad_survival)

    def test_model_bundle_requires_paired_seven_and_thirty_day_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seven = root / "risk_7d.json"
            thirty = root / "risk_30d.json"
            seven.write_text(json.dumps(artifact_data(7)), encoding="utf-8")
            thirty.write_text(json.dumps(artifact_data(30)), encoding="utf-8")

            bundle = load_model_bundle(seven, thirty)

            self.assertEqual(set(bundle), {7, 30})
            self.assertEqual(bundle[30].horizon_days, 30)

    def test_model_bundle_allows_contract_ids_differing_by_horizon(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seven = root / "risk_7d.json"
            thirty = root / "risk_30d.json"
            data7 = artifact_data(7)
            data30 = artifact_data(30)
            data7["input_contract"] = {"schema_version": 1, "contract_id": "same-contract-7d"}
            data30["input_contract"] = {"schema_version": 1, "contract_id": "same-contract-30d"}
            seven.write_text(json.dumps(data7), encoding="utf-8")
            thirty.write_text(json.dumps(data30), encoding="utf-8")
            bundle = load_model_bundle(seven, thirty)
            self.assertEqual(set(bundle), {7, 30})

    def test_model_bundle_rejects_different_training_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seven = root / "risk_7d.json"
            thirty = root / "risk_30d.json"
            seven.write_text(json.dumps(artifact_data(7, bundle_id="a")), encoding="utf-8")
            thirty.write_text(json.dumps(artifact_data(30, bundle_id="b")), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "同一训练批次"):
                load_model_bundle(seven, thirty)


if __name__ == "__main__":
    unittest.main()
