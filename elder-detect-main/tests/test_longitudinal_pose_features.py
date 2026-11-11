import math
import unittest

import numpy as np

from longitudinal.pose_features import summarize_pose_sequence


MP = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_ankle": 27,
    "right_ankle": 28,
}


def periodic_walk(seconds: float = 10.0, fps: int = 25):
    timestamps = np.arange(0.0, seconds, 1.0 / fps)
    landmarks = np.zeros((len(timestamps), 33, 2), dtype=np.float64)
    landmarks[:, MP["nose"], :] = [0.50, 0.15]
    landmarks[:, MP["left_shoulder"], :] = [0.45, 0.30]
    landmarks[:, MP["right_shoulder"], :] = [0.55, 0.30]
    landmarks[:, MP["left_hip"], :] = [0.47, 0.55]
    landmarks[:, MP["right_hip"], :] = [0.53, 0.55]
    phase = 2.0 * np.pi * timestamps
    landmarks[:, MP["left_ankle"], 0] = 0.46 + 0.02 * np.sin(phase)
    landmarks[:, MP["right_ankle"], 0] = 0.54 - 0.02 * np.sin(phase)
    landmarks[:, MP["left_ankle"], 1] = 0.88 + 0.025 * np.sin(phase)
    landmarks[:, MP["right_ankle"], 1] = 0.88 - 0.025 * np.sin(phase)
    confidence = np.full((len(timestamps), 33), 0.9, dtype=np.float64)
    return landmarks, timestamps, confidence


class PoseFeatureSummaryTest(unittest.TestCase):
    def test_periodic_walk_produces_expected_temporal_and_posture_features(self):
        landmarks, timestamps, confidence = periodic_walk()

        summary = summarize_pose_sequence(
            landmarks,
            timestamps,
            confidence,
            joint_indices=MP,
            min_confidence=0.5,
        )

        self.assertGreaterEqual(summary.quality["valid_steps"], 18)
        self.assertAlmostEqual(summary.features["cadence_spm"], 120.0, delta=8.0)
        self.assertAlmostEqual(summary.features["step_time_mean_s"], 0.5, delta=0.08)
        self.assertLess(summary.features["step_time_cv"], 0.12)
        self.assertLess(summary.features["gait_asymmetry"], 0.12)
        self.assertAlmostEqual(summary.features["trunk_tilt_mean_deg"], 0.0, delta=1.0)
        self.assertLess(summary.features["trunk_sway_rms_deg"], 1.0)
        self.assertGreater(summary.features["stance_width_norm"], 0.0)
        self.assertTrue(all(math.isfinite(value) for value in summary.features.values()))

    def test_low_confidence_frames_are_excluded_from_quality(self):
        landmarks, timestamps, confidence = periodic_walk(seconds=4.0)
        confidence[:25, :] = 0.1

        summary = summarize_pose_sequence(
            landmarks,
            timestamps,
            confidence,
            joint_indices=MP,
            min_confidence=0.5,
        )

        self.assertEqual(summary.quality["valid_frames"], len(timestamps) - 25)
        self.assertAlmostEqual(summary.quality["mean_confidence"], 0.9)

    def test_invalid_shapes_raise_clear_error(self):
        with self.assertRaisesRegex(ValueError, "landmarks"):
            summarize_pose_sequence(
                np.zeros((10, 33)),
                np.arange(10),
                np.ones((10, 33)),
                joint_indices=MP,
            )

    def test_whole_body_vertical_motion_is_not_misclassified_as_steps(self):
        landmarks, timestamps, confidence = periodic_walk(seconds=4.0)
        landmarks[:, MP["left_ankle"], 1] = 0.88
        landmarks[:, MP["right_ankle"], 1] = 0.88
        translation = 0.03 * np.sin(2.0 * np.pi * timestamps)
        for index in MP.values():
            landmarks[:, index, 1] += translation

        summary = summarize_pose_sequence(
            landmarks, timestamps, confidence, joint_indices=MP
        )

        self.assertEqual(summary.quality["valid_steps"], 0)
        self.assertNotIn("gait_asymmetry", summary.features)


if __name__ == "__main__":
    unittest.main()
