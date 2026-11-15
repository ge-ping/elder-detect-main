"""DS-PCD 命令行接入测试：采集、建基线、按预测日输出两套指标。"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import date, timedelta
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.config import DEFAULT_CORE_FEATURES  # noqa: E402
from scripts.build_personal_baseline import main as build_main  # noqa: E402
from scripts.collect_longitudinal import main as collect_main  # noqa: E402
from scripts.predict_future_fall import main as predict_main  # noqa: E402


DS_FEATURES = ("gait_speed_proxy", "cadence_spm", "step_time_mean_s")


class DsPcdCliTest(unittest.TestCase):
    def test_collect_build_and_predict_both_detectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = root / "profiles"
            start = date(2026, 10, 1)
            for offset in range(15):
                features = {
                    name: float(index + 1) + offset * 0.01
                    for index, name in enumerate(DEFAULT_CORE_FEATURES)
                }
                payload = {
                    "features": features,
                    "quality": {
                        "valid_frames": 300,
                        "valid_steps": 40,
                        "walking_duration_s": 30.0,
                        "mean_confidence": 0.9,
                    },
                    "feature_se": {
                        "gait_speed_proxy": 0.01,
                        "cadence_spm": 0.2,
                        "step_time_mean_s": 0.002,
                    },
                }
                source = root / f"day_{offset:02d}.json"
                source.write_text(json.dumps(payload), encoding="utf-8")
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(collect_main([
                        "--input", str(source),
                        "--person-id", "elder_001",
                        "--observed-at", (start + timedelta(days=offset)).isoformat(),
                        "--profile-dir", str(profiles),
                    ]), 0)

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(build_main([
                    "--person-id", "elder_001",
                    "--profile-dir", str(profiles),
                    "--detector", "both",
                    "--precision-mode", "session_se",
                    "--null-samples", "2000",
                    "--quantile-points", "200",
                ]), 0)
            profile = profiles / "elder_001"
            self.assertTrue((profile / "baseline.json").exists())
            self.assertTrue((profile / "ds_pcd_baseline.json").exists())

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(predict_main([
                    "--person-id", "elder_001",
                    "--profile-dir", str(profiles),
                    "--detector", "both",
                    "--as-of", (start + timedelta(days=14)).isoformat(),
                ]), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["status"], "baseline_ready")
            self.assertIsNotNone(result["distance"])
            self.assertEqual(result["ds_pcd"]["status"], "scored")
            self.assertEqual(result["ds_pcd"]["precision_mode"], "session_se")
            self.assertIn("T", result["ds_pcd"])
            self.assertIn("cusum", result["ds_pcd"])

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(predict_main([
                    "--person-id", "elder_001",
                    "--profile-dir", str(profiles),
                    "--detector", "rms",
                    "--as-of", (start + timedelta(days=14)).isoformat(),
                ]), 0)
            rms_only = json.loads(output.getvalue())
            self.assertIsNone(rms_only["ds_pcd"])
            self.assertAlmostEqual(rms_only["distance"], result["distance"])


if __name__ == "__main__":
    unittest.main()
