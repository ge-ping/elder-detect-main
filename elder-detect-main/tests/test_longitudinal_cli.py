from contextlib import redirect_stdout
from datetime import date, timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from longitudinal.config import DEFAULT_CORE_FEATURES
from scripts.build_personal_baseline import main as build_baseline_main
from scripts.collect_longitudinal import _load_npz, main as collect_main
from scripts.predict_future_fall import main as predict_main


class LongitudinalCliTest(unittest.TestCase):
    def test_feature_json_to_baseline_and_uncalibrated_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = root / "profiles"
            start = date(2026, 10, 1)
            for offset in range(14):
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
                }
                source = root / f"day_{offset:02d}.json"
                source.write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
                with redirect_stdout(io.StringIO()):
                    code = collect_main([
                        "--input", str(source),
                        "--person-id", "elder_001",
                        "--observed-at", (start + timedelta(days=offset)).isoformat(),
                        "--profile-dir", str(profiles),
                    ])
                self.assertEqual(code, 0)

            with redirect_stdout(io.StringIO()):
                self.assertEqual(build_baseline_main([
                    "--person-id", "elder_001",
                    "--profile-dir", str(profiles),
                ]), 0)

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(predict_main([
                    "--person-id", "elder_001",
                    "--profile-dir", str(profiles),
                ]), 0)

            result = json.loads(output.getvalue())
            self.assertEqual(result["status"], "baseline_ready")
            self.assertIsNone(result["risk_7d"])
            self.assertIsNone(result["risk_30d"])
            self.assertTrue(
                (profiles / "elder_001" / "baseline.json").exists()
            )
            self.assertTrue(
                (profiles / "elder_001" / "latest_risk.json").exists()
            )

    def test_collect_rejects_json_without_features(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.json"
            source.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "features"):
                collect_main([
                    "--input", str(source),
                    "--person-id", "elder_001",
                    "--observed-at", "2026-11-01",
                    "--profile-dir", str(Path(directory) / "profiles"),
                ])

    def test_npz_requires_landmarks_timestamps_and_visibility(self):
        landmarks = np.zeros((4, 33, 3), dtype=np.float64)
        timestamps = np.arange(4, dtype=np.float64) / 25.0
        visibility = np.ones((4, 33), dtype=np.float64)
        complete = {
            "landmarks": landmarks,
            "timestamps": timestamps,
            "visibility": visibility,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for missing in complete:
                with self.subTest(missing=missing):
                    source = root / f"missing_{missing}.npz"
                    np.savez(
                        source,
                        **{
                            name: value
                            for name, value in complete.items()
                            if name != missing
                        },
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "landmarks.*timestamps.*visibility",
                    ):
                        _load_npz(source, min_confidence=0.5)

    def test_requirements_pin_numpy_and_include_flask(self):
        requirements = (
            Path(__file__).resolve().parents[1] / "requirements.txt"
        ).read_text(encoding="utf-8").splitlines()
        effective_dependencies = {
            line.split("#", 1)[0].strip()
            for line in requirements
            if line.strip() and not line.lstrip().startswith("#")
        }

        self.assertIn("numpy>=1.24,<2", requirements)
        self.assertIn("Flask>=3.0,<4", requirements)
        self.assertIn("pillow>=9.0.0", effective_dependencies)
        self.assertIn("seaborn>=0.12.0", effective_dependencies)

    def test_npz_millisecond_timestamps_are_converted_to_seconds(self):
        captured: dict[str, np.ndarray] = {}

        class Summary:
            @staticmethod
            def to_dict():
                return {"features": {}, "quality": {}}

        def capture_summary(
            landmarks,
            timestamps,
            confidence,
            *,
            min_confidence,
        ):
            captured["timestamps"] = timestamps.copy()
            return Summary()

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "milliseconds.npz"
            np.savez(
                source,
                landmarks=np.zeros((3, 33, 3), dtype=np.float64),
                timestamps=np.array([0.0, 40.0, 80.0]),
                visibility=np.ones((3, 33), dtype=np.float64),
            )
            with patch(
                "scripts.collect_longitudinal.summarize_pose_sequence",
                side_effect=capture_summary,
            ):
                _load_npz(source, min_confidence=0.5)

        np.testing.assert_allclose(
            captured["timestamps"],
            np.array([0.0, 0.04, 0.08]),
        )


if __name__ == "__main__":
    unittest.main()
