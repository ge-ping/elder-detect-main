"""会话测量标准误：步行段 bootstrap 与姿态序列精度估计。

这里的标准误只描述“同一会话内由有限步数/帧数造成的估计不确定性”，
不包含传感器、机位、尺度或关键点系统偏差。调用方必须把该限制随结果
一起保存，不能把它当作完整的测量误差。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .ds_pcd import stable_seed
from .pose_features import DEFAULT_MEDIAPIPE_JOINTS, _step_events


POSE_FEATURES: tuple[str, ...] = (
    "gait_speed_proxy", "cadence_spm", "step_time_mean_s",
)


@dataclass(frozen=True)
class PrecisionEstimate:
    """一次会话的精度估计与回退原因。"""

    feature_se: dict[str, float] = field(default_factory=dict)
    methods: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    n_steps: int = 0
    n_frames: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_se": dict(self.feature_se),
            "methods": dict(self.methods),
            "diagnostics": {key: dict(value) for key, value in self.diagnostics.items()},
            "reasons": list(self.reasons),
            "n_steps": self.n_steps,
            "n_frames": self.n_frames,
        }


def _lag1(values: np.ndarray) -> float | None:
    if len(values) < 6:
        return None
    left, right = values[:-1], values[1:]
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def bootstrap_mean_se(
    values: Sequence[float],
    *,
    samples: int = 200,
    min_count: int = 5,
    block_length: int | None = None,
    seed_material: str = "bootstrap-v1",
    statistic: Callable[[np.ndarray], float] | None = None,
) -> dict[str, Any]:
    """估计统计量的抽样标准误；块长 2 时保留相邻观测相关性。"""
    array = np.asarray([float(value) for value in values if math.isfinite(float(value))],
                       dtype=np.float64)
    n = len(array)
    compute = statistic or (lambda sample: float(sample.mean()))
    diagnostic: dict[str, Any] = {
        "n": n,
        "mean": float(compute(array)) if n else None,
        "sd": float(array.std(ddof=1)) if n > 1 else None,
        "lag1": _lag1(array),
        "analytic_se": (
            float(array.std(ddof=1) / math.sqrt(n)) if n > 1 else None
        ),
    }
    if n < min_count:
        return {
            "se": None, "method": "unavailable", "block_length": None,
            "diagnostic": diagnostic, "reason": "too_few_observations",
        }
    lag1 = diagnostic["lag1"]
    if block_length is None:
        block_length = 2 if lag1 is not None and lag1 >= 0.1 else 1
    block_length = max(1, min(int(block_length), n))
    rng = np.random.default_rng(stable_seed(seed_material))
    if block_length == 1:
        means = np.array([
            compute(array[rng.integers(0, n, n)]) for _ in range(samples)
        ])
        method = "iid_bootstrap"
    else:
        n_blocks = int(math.ceil(n / block_length))
        starts_high = max(1, n - block_length + 1)
        means = np.empty(samples, dtype=np.float64)
        for index in range(samples):
            starts = rng.integers(0, starts_high, n_blocks)
            blocks = [array[start:start + block_length] for start in starts]
            sample = np.concatenate(blocks)[:n]
            means[index] = compute(sample)
        method = f"block{block_length}_bootstrap"
    se = float(means.std(ddof=1))
    diagnostic["bootstrap_means_sd"] = se
    if not math.isfinite(se) or se <= 0:
        analytic = diagnostic["analytic_se"]
        if analytic is not None and math.isfinite(analytic) and analytic > 0:
            return {
                "se": float(analytic), "method": "analytic",
                "block_length": block_length, "diagnostic": diagnostic,
                "reason": "bootstrap_degenerate",
            }
        return {
            "se": None, "method": "unavailable", "block_length": block_length,
            "diagnostic": diagnostic, "reason": "degenerate_variance",
        }
    return {
        "se": se, "method": method, "block_length": block_length,
        "diagnostic": diagnostic, "reason": None,
    }


def _prepare_sequence(
    landmarks: np.ndarray,
    timestamps_s: np.ndarray,
    confidence: np.ndarray,
    *,
    min_confidence: float,
    joint_indices: Mapping[str, int] | None,
) -> dict[str, Any]:
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
    indices = np.asarray([joints[name] for name in DEFAULT_MEDIAPIPE_JOINTS], dtype=int)
    xy = points[:, :, :2]
    required = scores[:, indices]
    required_points = xy[:, indices, :]
    valid = (
        np.all(np.isfinite(required_points), axis=(1, 2))
        & np.all(np.isfinite(required), axis=1)
        & np.all(required >= min_confidence, axis=1)
    )
    left_shoulder = xy[:, joints["left_shoulder"]]
    right_shoulder = xy[:, joints["right_shoulder"]]
    left_hip = xy[:, joints["left_hip"]]
    right_hip = xy[:, joints["right_hip"]]
    left_ankle = xy[:, joints["left_ankle"]]
    right_ankle = xy[:, joints["right_ankle"]]
    hip_mid = (left_hip + right_hip) / 2.0
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
    safe_width = np.where(shoulder_width > 1e-6, shoulder_width, np.nan)
    body_height = np.linalg.norm(
        (left_shoulder + right_shoulder) / 2.0 - (left_ankle + right_ankle) / 2.0,
        axis=1,
    )
    safe_height = np.where(body_height > 1e-6, body_height, np.nan)
    left_relative = (left_ankle[:, 1] - hip_mid[:, 1]) / safe_height
    right_relative = (right_ankle[:, 1] - hip_mid[:, 1]) / safe_height
    left_events = _step_events(left_relative, timestamps, valid)
    right_events = _step_events(right_relative, timestamps, valid)
    combined = sorted(left_events + right_events)
    hip_velocity = np.linalg.norm(np.diff(hip_mid, axis=0), axis=1) / np.diff(timestamps)
    velocity_width = (safe_width[:-1] + safe_width[1:]) / 2.0
    speed_values = np.divide(
        hip_velocity, velocity_width,
        out=np.full_like(hip_velocity, np.nan),
        where=np.isfinite(velocity_width),
    )
    speed_valid = valid[:-1] & valid[1:] & np.isfinite(speed_values)
    return {
        "timestamps": timestamps,
        "valid": valid,
        "events": combined,
        "event_times": timestamps[combined] if combined else np.array([]),
        "speed_values": speed_values[speed_valid],
        "frame_intervals": np.diff(timestamps),
        "valid_frames": int(np.count_nonzero(valid)),
    }


def pose_session_precision(
    landmarks: np.ndarray,
    timestamps_s: np.ndarray,
    confidence: np.ndarray,
    *,
    features: Sequence[str] = POSE_FEATURES,
    samples: int = 200,
    min_confidence: float = 0.5,
    joint_indices: Mapping[str, int] | None = None,
    min_steps: int = 8,
    seed_material: str = "pose-session-v1",
) -> PrecisionEstimate:
    """用步事件与帧速度序列估计会话特征的测量标准误。

    * ``cadence_spm``、``step_time_mean_s``：对步间隔做移动块 bootstrap；
    * ``gait_speed_proxy``：对逐帧速度值做移动块 bootstrap，块长取中位步
      间隔对应的帧数，重采样的是已经算好的速度值，不拼接坐标，避免制造
      虚假的速度跳变。
    """
    prepared = _prepare_sequence(
        landmarks, timestamps_s, confidence,
        min_confidence=min_confidence, joint_indices=joint_indices,
    )
    intervals = np.diff(prepared["event_times"])
    n_steps = len(prepared["events"])
    reasons: list[str] = []
    if prepared["valid_frames"] < 20:
        reasons.append("insufficient_valid_frames")
    if n_steps < min_steps or len(intervals) < 2:
        reasons.append("insufficient_step_events")
    feature_se: dict[str, float] = {}
    methods: dict[str, str] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    material = "|".join((
        seed_material, str(prepared["valid_frames"]), str(n_steps),
        f"{prepared['timestamps'][-1]:.3f}",
    ))
    if len(intervals) >= 2 and n_steps >= min_steps:
        step = bootstrap_mean_se(
            intervals, samples=samples, seed_material=material + "|cadence",
            statistic=lambda sample: 60.0 / float(sample.mean()),
        )
        diagnostics["cadence_spm"] = {**step["diagnostic"], "reason": step["reason"]}
        methods["cadence_spm"] = step["method"]
        if step["se"] is not None:
            feature_se["cadence_spm"] = float(step["se"])
        step_time = bootstrap_mean_se(
            intervals, samples=samples, seed_material=material + "|step_time",
        )
        diagnostics["step_time_mean_s"] = {
            **step_time["diagnostic"], "reason": step_time["reason"],
        }
        methods["step_time_mean_s"] = step_time["method"]
        if step_time["se"] is not None:
            feature_se["step_time_mean_s"] = float(step_time["se"])
    if len(prepared["speed_values"]) >= 20:
        median_interval = float(np.median(intervals)) if len(intervals) else 1.0
        median_dt = float(np.median(prepared["frame_intervals"]))
        block = max(1, int(round(median_interval / median_dt))) if median_dt > 0 else 1
        speed = bootstrap_mean_se(
            prepared["speed_values"], samples=samples, block_length=block,
            seed_material=material + "|speed",
        )
        diagnostics["gait_speed_proxy"] = {
            **speed["diagnostic"], "reason": speed["reason"],
        }
        methods["gait_speed_proxy"] = speed["method"]
        if speed["se"] is not None:
            feature_se["gait_speed_proxy"] = float(speed["se"])
    for name in features:
        if name not in feature_se:
            methods.setdefault(name, "unavailable")
            if prepared["valid_frames"] < 20:
                reason = "insufficient_valid_frames"
            elif n_steps < min_steps:
                reason = "insufficient_step_events"
            else:
                reason = str(
                    diagnostics.get(name, {}).get("reason") or "not_estimated"
                )
            feature_se.pop(name, None)
            reasons.append(f"{name}:{reason}")
    return PrecisionEstimate(
        feature_se=feature_se,
        methods=methods,
        diagnostics=diagnostics,
        reasons=tuple(dict.fromkeys(reasons)),
        n_steps=n_steps,
        n_frames=int(len(prepared["timestamps"])),
    )
