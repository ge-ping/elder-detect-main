"""每日观测质量门控。"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .config import LongitudinalConfig
from .schema import DailyObservation


@dataclass(frozen=True)
class QualityAssessment:
    """质量判断及全部拒绝原因。"""

    is_valid: bool
    reasons: tuple[str, ...]
    core_feature_coverage: float


def assess_quality(
    observation: DailyObservation,
    config: LongitudinalConfig,
) -> QualityAssessment:
    """检查观测是否可以进入个人基线和偏离计算。"""
    reasons: list[str] = []
    quality = observation.quality
    if quality.valid_frames < 0 or quality.valid_steps < 0:
        reasons.append("有效帧数和有效步数必须是非负整数")
    if not math.isfinite(quality.walking_duration_s) or quality.walking_duration_s < 0:
        reasons.append("有效步行时长必须是非负有限数值")
    if not math.isfinite(quality.mean_confidence) or not 0.0 <= quality.mean_confidence <= 1.0:
        reasons.append("平均姿态置信度必须位于0到1之间")
    if quality.valid_frames < config.min_valid_frames:
        reasons.append(
            f"有效帧数不足：{quality.valid_frames} < {config.min_valid_frames}"
        )
    if quality.valid_steps < config.min_valid_steps:
        reasons.append(
            f"有效步数不足：{quality.valid_steps} < {config.min_valid_steps}"
        )
    if quality.walking_duration_s < config.min_walking_duration_s:
        reasons.append(
            "有效步行时长不足："
            f"{quality.walking_duration_s:g}秒 < {config.min_walking_duration_s:g}秒"
        )
    if quality.mean_confidence < config.min_mean_confidence:
        reasons.append(
            "平均姿态置信度不足："
            f"{quality.mean_confidence:g} < {config.min_mean_confidence:g}"
        )

    total = len(config.core_features)
    present = sum(name in observation.features for name in config.core_features)
    coverage = present / total if total else 1.0
    if coverage < config.min_core_feature_coverage:
        reasons.append(
            "核心特征覆盖率不足："
            f"{coverage:.1%} < {config.min_core_feature_coverage:.1%}"
        )

    return QualityAssessment(
        is_valid=not reasons,
        reasons=tuple(reasons),
        core_feature_coverage=coverage,
    )
