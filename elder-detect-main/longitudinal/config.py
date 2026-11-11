"""纵向预警的集中配置。"""
from __future__ import annotations

from dataclasses import dataclass


DEFAULT_CORE_FEATURES = (
    "gait_speed_proxy",
    "cadence_spm",
    "step_time_mean_s",
    "step_time_cv",
    "stance_width_norm",
    "gait_asymmetry",
    "trunk_tilt_mean_deg",
    "trunk_sway_rms_deg",
    "aspect_ratio_mean",
    "aspect_ratio_cv",
    "emos_proxy",
)


@dataclass(frozen=True)
class LongitudinalConfig:
    """个人基线与质量门控配置。"""

    min_baseline_days: int = 14
    min_valid_frames: int = 150
    min_valid_steps: int = 20
    min_walking_duration_s: float = 10.0
    min_mean_confidence: float = 0.5
    min_core_feature_coverage: float = 0.8
    persistent_distance_threshold: float = 3.0
    persistent_days: int = 3
    timezone: str = "Asia/Shanghai"
    core_features: tuple[str, ...] = DEFAULT_CORE_FEATURES
