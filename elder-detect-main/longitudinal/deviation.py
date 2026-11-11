"""每日观测相对个人基线的稳健偏离。"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .baseline import PersonalBaseline


@dataclass(frozen=True)
class DeviationResult:
    distance: float
    deviation_index: float
    feature_deviations: dict[str, float]
    valid_feature_count: int


def compute_deviation(
    features: dict[str, float],
    baseline: PersonalBaseline,
) -> DeviationResult:
    """计算稳健 z 分数和截断后的多维 RMS 距离。"""
    deviations: dict[str, float] = {}
    clipped: list[float] = []
    for name, stats in baseline.features.items():
        if not stats.active or name not in features:
            continue
        z_score = (float(features[name]) - stats.median) / stats.scale
        if not math.isfinite(z_score):
            continue
        deviations[name] = z_score
        clipped.append(max(-5.0, min(5.0, z_score)))
    if not clipped:
        raise ValueError("当前观测没有可用于偏离计算的有效特征")
    distance = math.sqrt(sum(value * value for value in clipped) / len(clipped))
    return DeviationResult(
        distance=distance,
        deviation_index=min(distance / 5.0, 1.0),
        feature_deviations=deviations,
        valid_feature_count=len(clipped),
    )


def count_persistent_deviation(
    results: list[DeviationResult],
    threshold: float = 3.0,
) -> int:
    """统计序列末尾连续超过统计阈值的有效观测日。"""
    count = 0
    for result in reversed(results):
        if result.distance < threshold:
            break
        count += 1
    return count
