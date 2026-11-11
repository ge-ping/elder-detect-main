"""从单目人体关键点序列生成每日步态和姿态摘要。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_MEDIAPIPE_JOINTS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_ankle": 27,
    "right_ankle": 28,
}


@dataclass(frozen=True)
class PoseSequenceSummary:
    features: dict[str, float]
    quality: dict[str, float | int]

    def to_dict(self) -> dict[str, Any]:
        return {"features": dict(self.features), "quality": dict(self.quality)}


def _step_events(
    signal: np.ndarray,
    timestamps: np.ndarray,
    valid: np.ndarray,
    min_interval_s: float = 0.35,
) -> list[int]:
    candidates: list[int] = []
    for index in range(1, len(signal) - 1):
        if not (valid[index - 1] and valid[index] and valid[index + 1]):
            continue
        if signal[index] < signal[index - 1] and signal[index] <= signal[index + 1]:
            prominence = max(signal[index - 1], signal[index + 1]) - signal[index]
            if prominence > 1e-5:
                candidates.append(index)
    selected: list[int] = []
    for index in candidates:
        if not selected or timestamps[index] - timestamps[selected[-1]] >= min_interval_s:
            selected.append(index)
        elif signal[index] < signal[selected[-1]]:
            selected[-1] = index
    return selected


def _coefficient_of_variation(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    mean = float(np.mean(values))
    return float(np.std(values) / mean) if abs(mean) > 1e-12 else 0.0


def summarize_pose_sequence(
    landmarks: np.ndarray,
    timestamps_s: np.ndarray,
    confidence: np.ndarray,
    *,
    joint_indices: dict[str, int] | None = None,
    min_confidence: float = 0.5,
) -> PoseSequenceSummary:
    """汇总一段步行关键点序列。

    空间结果均为同一摄像机条件下的归一化代理量。函数不把单目坐标
    解释为实验室级米制步速或真实动态稳定裕度。
    """
    points = np.asarray(landmarks, dtype=np.float64)
    timestamps = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    scores = np.asarray(confidence, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] not in (2, 3):
        raise ValueError("landmarks 必须是 (T,J,2) 或 (T,J,3) 数组")
    if len(points) != len(timestamps):
        raise ValueError("landmarks 与 timestamps 长度必须一致")
    if scores.ndim == 1:
        scores = np.repeat(scores[:, None], points.shape[1], axis=1)
    if scores.shape[:2] != points.shape[:2]:
        raise ValueError("confidence 必须是 (T,J) 或 (T,) 数组")
    if len(points) < 3 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("时间戳必须严格递增且至少包含3帧")

    joints = dict(DEFAULT_MEDIAPIPE_JOINTS if joint_indices is None else joint_indices)
    required_names = tuple(DEFAULT_MEDIAPIPE_JOINTS)
    missing = [name for name in required_names if name not in joints]
    if missing:
        raise ValueError(f"joint_indices 缺少关节：{', '.join(missing)}")
    indices = np.asarray([joints[name] for name in required_names], dtype=int)
    if indices.min() < 0 or indices.max() >= points.shape[1]:
        raise ValueError("joint_indices 超出 landmarks 的关节范围")

    xy = points[:, :, :2]
    required_scores = scores[:, indices]
    required_points = xy[:, indices, :]
    valid = (
        np.all(np.isfinite(required_points), axis=(1, 2))
        & np.all(np.isfinite(required_scores), axis=1)
        & np.all(required_scores >= min_confidence, axis=1)
    )

    ls = xy[:, joints["left_shoulder"]]
    rs = xy[:, joints["right_shoulder"]]
    lh = xy[:, joints["left_hip"]]
    rh = xy[:, joints["right_hip"]]
    la = xy[:, joints["left_ankle"]]
    ra = xy[:, joints["right_ankle"]]
    nose = xy[:, joints["nose"]]

    shoulder_mid = (ls + rs) / 2.0
    hip_mid = (lh + rh) / 2.0
    ankle_mid = (la + ra) / 2.0
    shoulder_width = np.linalg.norm(ls - rs, axis=1)
    safe_width = np.where(shoulder_width > 1e-6, shoulder_width, np.nan)

    body_height_proxy = np.linalg.norm(shoulder_mid - ankle_mid, axis=1)
    safe_height = np.where(body_height_proxy > 1e-6, body_height_proxy, np.nan)
    left_ankle_relative = (la[:, 1] - hip_mid[:, 1]) / safe_height
    right_ankle_relative = (ra[:, 1] - hip_mid[:, 1]) / safe_height
    left_events = _step_events(left_ankle_relative, timestamps, valid)
    right_events = _step_events(right_ankle_relative, timestamps, valid)
    combined_events = sorted(left_events + right_events)
    event_times = timestamps[combined_events]
    step_times = np.diff(event_times)

    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) >= 2:
        duration = float(timestamps[valid_indices[-1]] - timestamps[valid_indices[0]])
    else:
        duration = 0.0
    cadence = len(combined_events) / duration * 60.0 if duration > 0 else 0.0

    left_periods = np.diff(timestamps[left_events])
    right_periods = np.diff(timestamps[right_events])
    if len(left_periods) and len(right_periods):
        left_mean = float(np.mean(left_periods))
        right_mean = float(np.mean(right_periods))
        asymmetry = abs(left_mean - right_mean) / max((left_mean + right_mean) / 2, 1e-12)
    else:
        asymmetry = 0.0

    trunk_dx = shoulder_mid[:, 0] - hip_mid[:, 0]
    trunk_dy = shoulder_mid[:, 1] - hip_mid[:, 1]
    trunk_tilt = np.degrees(np.arctan2(np.abs(trunk_dx), np.abs(trunk_dy)))
    stance_width = np.abs(la[:, 0] - ra[:, 0]) / safe_width

    box_points = np.stack((nose, ls, rs, lh, rh, la, ra), axis=1)
    box_width = np.ptp(box_points[:, :, 0], axis=1)
    box_height = np.ptp(box_points[:, :, 1], axis=1)
    aspect_ratio = np.divide(
        box_width,
        box_height,
        out=np.zeros_like(box_width),
        where=box_height > 1e-6,
    )

    com_proxy_x = 0.4 * shoulder_mid[:, 0] + 0.6 * hip_mid[:, 0]
    support_half_width = np.abs(la[:, 0] - ra[:, 0]) / 2.0
    emos_proxy = (
        support_half_width - np.abs(com_proxy_x - ankle_mid[:, 0])
    ) / safe_width

    hip_velocity = np.linalg.norm(np.diff(hip_mid, axis=0), axis=1) / np.diff(timestamps)
    velocity_width = (safe_width[:-1] + safe_width[1:]) / 2.0
    gait_speed_proxy = np.divide(
        hip_velocity,
        velocity_width,
        out=np.zeros_like(hip_velocity),
        where=np.isfinite(velocity_width),
    )
    velocity_valid = valid[:-1] & valid[1:]

    def valid_mean(values: np.ndarray, mask: np.ndarray = valid) -> float:
        selected = values[mask & np.isfinite(values)]
        return float(np.mean(selected)) if len(selected) else 0.0

    tilt_values = trunk_tilt[valid & np.isfinite(trunk_tilt)]
    tilt_mean = float(np.mean(tilt_values)) if len(tilt_values) else 0.0
    tilt_rms = (
        float(np.sqrt(np.mean((tilt_values - tilt_mean) ** 2)))
        if len(tilt_values) else 0.0
    )
    features: dict[str, float] = {
        "gait_speed_proxy": valid_mean(gait_speed_proxy, velocity_valid),
        "stance_width_norm": valid_mean(stance_width),
        "trunk_tilt_mean_deg": tilt_mean,
        "trunk_sway_rms_deg": tilt_rms,
        "aspect_ratio_mean": valid_mean(aspect_ratio),
        "aspect_ratio_cv": _coefficient_of_variation(
            aspect_ratio[valid & np.isfinite(aspect_ratio)]
        ),
        "emos_proxy": valid_mean(emos_proxy),
    }
    if len(step_times):
        features.update({
            "cadence_spm": float(cadence),
            "step_time_mean_s": float(np.mean(step_times)),
            "step_time_cv": _coefficient_of_variation(step_times),
        })
    if len(left_periods) and len(right_periods):
        features["gait_asymmetry"] = float(asymmetry)
    quality = {
        "valid_frames": int(np.count_nonzero(valid)),
        "valid_steps": int(len(combined_events)),
        "walking_duration_s": duration,
        "mean_confidence": (
            float(np.mean(required_scores[valid])) if np.any(valid) else 0.0
        ),
    }
    return PoseSequenceSummary(features=features, quality=quality)
