"""
pose — MediaPipe Pose 姿态提取模块

提供：
- PoseExtractor: 核心提取器，封装 MediaPipe Tasks API
- Keypoint / PoseResult: 结构化数据模型（x,y ∈ [0,1] MediaPipe 原生归一化）
- BodyPart: 33个关键点枚举 + 预定义分组
- load_pose_config: 从 YAML 配置加载

使用示例:
    >>> from pose import PoseExtractor, PoseExtractorConfig, load_pose_config
    >>> config = load_pose_config("config/pose_config.yaml")
    >>> with PoseExtractor(config) as extractor:
    ...     for result in extractor.process_video("test.mp4"):
    ...         print(result.array.shape)
"""

from .keypoint import (
    Keypoint,
    PoseResult,
    BodyPart,
    CORE_JOINTS_13,
    MINIMAL_JOINTS_3,
    ALL_JOINTS_33,
    JOINT_SET_MAP,
)

from .extractor import (
    PoseExtractor,
    PoseExtractorConfig,
)

from .config_loader import (
    load_pose_config,
    create_config_from_dict,
)

__all__ = [
    # 核心类
    "PoseExtractor",
    "PoseExtractorConfig",
    # 数据模型
    "Keypoint",
    "PoseResult",
    "BodyPart",
    # 关键点分组
    "CORE_JOINTS_13",
    "MINIMAL_JOINTS_3",
    "ALL_JOINTS_33",
    "JOINT_SET_MAP",
    # 配置加载
    "load_pose_config",
    "create_config_from_dict",
]

__version__ = "0.2.0"
