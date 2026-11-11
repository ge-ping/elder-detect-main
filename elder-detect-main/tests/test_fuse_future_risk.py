import json
from pathlib import Path
import tempfile
import unittest

from scripts.fuse_risk import attach_future_risk, load_future_risk


class FutureRiskAttachmentTest(unittest.TestCase):
    def test_loads_valid_uncalibrated_result_without_inventing_probability(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest_risk.json"
            path.write_text(json.dumps({
                "person_id": "elder_001",
                "status": "calibration_required",
                "risk_7d": None,
                "risk_30d": None,
                "deviation_index": 0.6,
            }), encoding="utf-8")

            result = load_future_risk(path)

            self.assertEqual(result["person_id"], "elder_001")
            self.assertIsNone(result["risk_7d"])

    def test_uncalibrated_result_with_probability_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({
                "person_id": "elder_001",
                "status": "baseline_ready",
                "risk_7d": 0.2,
                "risk_30d": None,
            }), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "未校准"):
                load_future_risk(path)

    def test_calibrated_probability_must_be_json_number_not_string_or_boolean(self):
        for value in ("0.2", True):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.json"
                path.write_text(json.dumps({
                    "person_id": "elder_001",
                    "status": "calibrated",
                    "risk_7d": value,
                    "risk_30d": 0.3,
                }), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "JSON 数值"):
                    load_future_risk(path)

    def test_no_path_keeps_existing_summary_shape_unchanged(self):
        payload = {"aggregate": {"n_videos": 1}, "per_video": []}

        result = attach_future_risk(payload, None)

        self.assertEqual(result, payload)
        self.assertNotIn("future_risk", result)


if __name__ == "__main__":
    unittest.main()
