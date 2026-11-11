import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.train_landmark_cox import _step_survival_at, _validation_ok


class LandmarkTrainingContractTest(unittest.TestCase):
    def test_validation_requires_real_boolean_and_finite_metrics(self):
        invalid_documents = [
            {"independent_validation": "false", "horizons": {"7": {}, "30": {}}},
            {"independent_validation": True, "horizons": {"7": None, "30": None}},
            {"independent_validation": True, "horizons": {
                "7": {"auroc": math.nan, "calibration_slope": 1.0},
                "30": {"auroc": 0.7, "calibration_slope": 1.0},
            }},
        ]
        for document in invalid_documents:
            with self.subTest(document=document), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "validation.json"
                path.write_text(json.dumps(document), encoding="utf-8")
                ok, _ = _validation_ok(str(path))
                self.assertFalse(ok)

    def test_validation_accepts_both_horizons_with_finite_metrics(self):
        document = {
            "independent_validation": True,
            "validation_id": "external-cohort-001",
            "horizons": {
                "7": {"auroc": 0.78, "calibration_slope": 0.96},
                "30": {"auroc": 0.70, "calibration_slope": 0.91},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            ok, loaded = _validation_ok(str(path))
        self.assertTrue(ok)
        self.assertEqual(loaded["validation_id"], "external-cohort-001")

    def test_baseline_survival_uses_step_value_not_linear_interpolation(self):
        times = np.array([1.0, 5.0, 10.0])
        survival = np.array([0.95, 0.80, 0.60])

        self.assertEqual(_step_survival_at(times, survival, 7.0), 0.80)
        self.assertEqual(_step_survival_at(times, survival, 0.5), 1.0)


if __name__ == "__main__":
    unittest.main()
