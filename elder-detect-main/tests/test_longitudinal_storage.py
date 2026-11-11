import json
from pathlib import Path
import tempfile
import unittest

from longitudinal.schema import DailyObservation
from longitudinal.storage import ProfileStore


def make_observation(day: int, cadence: float = 90.0) -> DailyObservation:
    return DailyObservation.from_dict({
        "person_id": "elder_001",
        "observed_at": f"2026-11-{day:02d}T08:00:00",
        "features": {"cadence_spm": cadence},
        "quality": {
            "valid_frames": 200,
            "valid_steps": 30,
            "walking_duration_s": 20,
            "mean_confidence": 0.8,
        },
    })


class ProfileStoreTest(unittest.TestCase):
    def test_append_creates_profile_and_loads_in_date_order(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(directory, "elder_001")
            store.append_observation(make_observation(2))
            store.append_observation(make_observation(1))

            observations = store.load_observations()

            self.assertEqual([item.observation_date for item in observations], [
                "2026-11-01", "2026-11-02"
            ])
            self.assertTrue(store.observations_path.exists())

    def test_duplicate_date_is_rejected_unless_replace_is_true(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(directory, "elder_001")
            store.append_observation(make_observation(1, 90.0))

            with self.assertRaisesRegex(ValueError, "重复观测日期"):
                store.append_observation(make_observation(1, 80.0))

            store.append_observation(make_observation(1, 80.0), replace=True)
            self.assertEqual(store.load_observations()[0].features["cadence_spm"], 80.0)

    def test_json_documents_are_saved_atomically_without_temp_residue(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(directory, "elder_001")
            store.save_baseline({"person_id": "elder_001", "version": 1})
            store.save_latest_result({"status": "baseline_ready"})

            self.assertEqual(store.load_baseline()["version"], 1)
            self.assertEqual(
                json.loads(store.latest_risk_path.read_text(encoding="utf-8"))["status"],
                "baseline_ready",
            )
            self.assertEqual(list(Path(store.profile_dir).glob("*.tmp")), [])

    def test_person_id_cannot_escape_profile_root(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "person_id"):
                ProfileStore(directory, "../outside")

    def test_loaded_observation_must_match_store_person(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(directory, "elder_001")
            store.profile_dir.mkdir(parents=True)
            bad = make_observation(1).to_dict()
            bad["person_id"] = "elder_002"
            store.observations_path.write_text(
                json.dumps(bad, ensure_ascii=False) + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "person_id"):
                store.load_observations()


if __name__ == "__main__":
    unittest.main()
