import io
from pathlib import Path
import tempfile
import unittest


class FakeAnalyzer:
    def __init__(self):
        self.received_path = None
        self.call_count = 0

    def analyze(self, video_path: Path) -> dict:
        self.call_count += 1
        self.received_path = Path(video_path)
        self.path_existed_during_analysis = self.received_path.exists()
        return {
            "mode": "real_observation",
            "features": {
                "cadence_spm": 91.2,
                "step_time_cv": 0.08,
                "trunk_sway_rms_deg": 1.4,
                "emos_proxy": 0.27,
            },
            "quality": {
                "valid_frames": 420,
                "valid_steps": 36,
                "walking_duration_s": 28.0,
                "mean_confidence": 0.88,
            },
        }


class WebDemoTest(unittest.TestCase):
    def setUp(self):
        from web_demo.app import create_app

        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.app = create_app({
            "TESTING": True,
            "UPLOAD_ROOT": self.temp_dir.name,
        })
        self.client = self.app.test_client()

    def test_index_contains_required_dashboard_regions(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("长者步态画像与未来跌倒预警", html)
        self.assertIn('id="upload-form"', html)
        self.assertIn('id="profile-panel"', html)
        self.assertIn('id="baseline-chart"', html)
        self.assertIn('id="deviation-chart"', html)
        self.assertIn("示例数据", html)

    def test_profile_api_exposes_sample_baseline_without_probability(self):
        response = self.client.get("/api/profile")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["data_mode"], "sample")
        self.assertEqual(payload["baseline"]["valid_days"], 14)
        self.assertEqual(payload["baseline"]["required_days"], 14)
        self.assertGreaterEqual(len(payload["history"]), 21)
        for key in (
            "cadence_spm", "step_time_cv", "trunk_sway_rms_deg",
            "emos_proxy", "deviation_index",
        ):
            self.assertIn(key, payload["metrics"])
        self.assertIsNone(payload["risk"]["risk_7d"])
        self.assertIsNone(payload["risk"]["risk_30d"])
        self.assertIn("等待真实纵向标签校准", payload["risk"]["message"])
        self.assertTrue(all(
            {"date", "value", "baseline_median", "baseline_low",
             "baseline_high", "distance"}.issubset(item)
            for item in payload["history"]
        ))

    def test_health_endpoint(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_analyze_rejects_missing_file(self):
        response = self.client.post("/api/analyze", data={})

        self.assertEqual(response.status_code, 400)
        self.assertIn("视频", response.get_json()["error"])

    def test_analyze_rejects_unsupported_extension(self):
        response = self.client.post(
            "/api/analyze",
            data={"video": (io.BytesIO(b"not-a-video"), "walk.txt")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("格式", response.get_json()["error"])

    def test_missing_model_returns_explicit_sample_fallback(self):
        response = self.client.post(
            "/api/analyze",
            data={
                "video": (io.BytesIO(b"demo-video"), "walk.mp4"),
                "person_name": "李奶奶",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["processing"]["mode"], "sample_fallback")
        self.assertEqual(
            payload["processing"]["warning"],
            "当前未安装姿态模型，页面展示示例画像，不代表上传视频分析结果",
        )
        self.assertIsNone(payload["profile"]["risk"]["risk_7d"])
        self.assertEqual(payload["upload"]["filename"], "walk.mp4")
        self.assertEqual(payload["profile"]["person"]["name"], "李奶奶")

    def test_injected_analyzer_returns_real_observation_and_cleans_temp_file(self):
        from web_demo.app import create_app

        analyzer = FakeAnalyzer()
        app = create_app({
            "TESTING": True,
            "UPLOAD_ROOT": self.temp_dir.name,
        }, analyzer=analyzer)
        response = app.test_client().post(
            "/api/analyze",
            data={
                "video": (io.BytesIO(b"demo-video"), "safe walk.mp4"),
                "person_name": "王爷爷",
                "person_id": "elder_009",
                "age": "78",
                "observed_at": "2026-11-11",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(analyzer.path_existed_during_analysis)
        self.assertFalse(analyzer.received_path.exists())
        self.assertEqual(payload["processing"]["mode"], "real_observation")
        self.assertIsNone(payload["processing"]["warning"])
        self.assertAlmostEqual(payload["profile"]["metrics"]["cadence_spm"], 91.2)
        self.assertEqual(payload["profile"]["person"]["id"], "elder_009")
        self.assertIsNone(payload["profile"]["risk"]["risk_30d"])
        self.assertEqual(payload["observation"]["observed_at"], "2026-11-11")

    def test_invalid_observed_at_is_rejected_before_analyzer_runs(self):
        from web_demo.app import create_app

        analyzer = FakeAnalyzer()
        app = create_app({
            "TESTING": True,
            "UPLOAD_ROOT": self.temp_dir.name,
        }, analyzer=analyzer)
        response = app.test_client().post(
            "/api/analyze",
            data={
                "video": (io.BytesIO(b"demo-video"), "walk.mp4"),
                "observed_at": "2026-99-99",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("观测日期", response.get_json()["error"])
        self.assertEqual(analyzer.call_count, 0)


if __name__ == "__main__":
    unittest.main()
