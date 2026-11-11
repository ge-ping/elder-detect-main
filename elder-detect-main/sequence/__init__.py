"""
sequence — 时序窗口构建模块

将 T 帧 PoseResult + 标量特征 → (N, 35, 29) LSTM 输入张量。

提供:
    - WindowConfig: 窗口/特征维度配置（无硬编码）
    - FrameFeatureBuilder: 单帧 → K 维向量拼接
    - SlidingWindowBuilder: T 帧 → (N, window_size, K) 滑动窗口

使用示例:
    >>> from sequence import WindowConfig, SlidingWindowBuilder
    >>> config = WindowConfig(window_size=35, stride=1)
    >>> builder = SlidingWindowBuilder(config)
    >>> X = builder.build(results, {
    ...     "gait_speed": speeds,
    ...     "trunk_angle": angles,
    ...     "aspect_ratio": ratios,
    ... })
    >>> X.shape  # (N, 35, 29)
"""

from .config import WindowConfig
from .feature_vector import FrameFeatureBuilder
from .window import SlidingWindowBuilder

__all__ = [
    "WindowConfig",
    "FrameFeatureBuilder",
    "SlidingWindowBuilder",
]

__version__ = "0.1.0"
