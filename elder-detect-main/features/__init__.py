"""
features — 时序特征提取模块

从 PoseResult 序列提取论文5四大判别特征，输出 FeatureResult → numpy → LSTM。

提供:
    - FeaturePipeline: 特征提取管道（串联多个提取器）
    - TorsoAngleExtractor:   2.2.1 躯干角度（逐帧）
    - GaitSpeedExtractor:     2.2.2 步行速度（窗口级）
    - SkeletonFeatureExtractor: 2.2.4 骨架点特征（逐帧）
    - FeatureResult:          特征输出 dataclass → .to_array()

使用示例:
    >>> from pose import PoseExtractor, load_pose_config
    >>> from features import (
    ...     FeaturePipeline, TorsoAngleExtractor,
    ...     GaitSpeedExtractor, SkeletonFeatureExtractor,
    ... )
    >>> pipeline = FeaturePipeline([
    ...     TorsoAngleExtractor(),
    ...     SkeletonFeatureExtractor(),
    ...     GaitSpeedExtractor(window_size=30),
    ... ])
    >>> # → pipeline.process_frame(result) 或 pipeline.process_window(buffer)

对齐论文:
    - 论文5: 四大判别特征（躯干角 + 步速 + 宽高比 + 骨骼时序）→ 注意力GRU
    - 论文4: 躯干角 + 头髋比 → 轻量 LSTM
    - 论文2: RP 归一化 → LSTM/GRU
"""

from .base import FeatureExtractor
from .types import FeatureResult
from .torso import TorsoAngleExtractor
from .gait import GaitSpeedExtractor
from .skeleton import SkeletonFeatureExtractor
from .pipeline import FeaturePipeline

__all__ = [
    # 管道
    "FeaturePipeline",
    # 基类
    "FeatureExtractor",
    # 提取器
    "TorsoAngleExtractor",
    "GaitSpeedExtractor",
    "SkeletonFeatureExtractor",
    # 数据类型
    "FeatureResult",
]

__version__ = "0.1.0"
