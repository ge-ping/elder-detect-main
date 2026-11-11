"""
配置加载器

从 YAML 配置文件构建 PoseExtractorConfig。
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Optional

from .extractor import PoseExtractorConfig
from .keypoint import BodyPart, JOINT_SET_MAP


# 项目根目录（pose/config_loader.py → pose/..）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_pose_config(config_path: str) -> PoseExtractorConfig:
    """从 YAML 配置文件加载姿态提取配置

    Args:
        config_path: YAML 配置文件路径（相对项目根或绝对路径）

    Returns:
        PoseExtractorConfig 实例

    Example:
        >>> config = load_pose_config("config/pose_config.yaml")
        >>> extractor = PoseExtractor(config)

    配置文件格式见 config/pose_config.yaml
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Config file is empty: {path}")

    pose_cfg = raw.get("pose")
    if pose_cfg is None:
        raise ValueError(f"Config missing 'pose' section: {path}")

    return _parse_pose_section(pose_cfg)


def create_config_from_dict(pose_cfg: dict[str, Any]) -> PoseExtractorConfig:
    """从字典直接创建配置（无需 YAML 文件）

    Args:
        pose_cfg: 配置字典（对应 YAML 中 'pose' section）

    Returns:
        PoseExtractorConfig 实例
    """
    return _parse_pose_section(pose_cfg)


def _parse_pose_section(pose_cfg: dict[str, Any]) -> PoseExtractorConfig:
    """解析 YAML 'pose' section → PoseExtractorConfig"""
    model_name = pose_cfg.get("model_name", "mediapipe_pose")
    model_version = pose_cfg.get("model_version", "lite")

    # 模型路径: model/{model_name}/pose_landmarker_{model_version}.task
    model_dir = _PROJECT_ROOT / "model" / model_name
    model_file = f"pose_landmarker_{model_version}.task"
    model_path = str(model_dir / model_file)

    # 解析关键点集
    joint_set_name = pose_cfg.get("selected_joints")  # null / "core_13" / "minimal_3" / "all_33"
    selected_joints: Optional[list[BodyPart]] = None
    if isinstance(joint_set_name, str) and joint_set_name in JOINT_SET_MAP:
        selected_joints = JOINT_SET_MAP[joint_set_name]
    elif isinstance(joint_set_name, list):
        # 直接指定了 BodyPart 名称列表
        selected_joints = [BodyPart[name] for name in joint_set_name]

    # 尺寸
    resize = pose_cfg.get("resize_width"), pose_cfg.get("resize_height")
    if resize[0] is None or resize[1] is None:
        resize = None

    return PoseExtractorConfig(
        model_path=model_path,
        min_detection_confidence=float(pose_cfg.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(pose_cfg.get("min_tracking_confidence", 0.5)),
        min_pose_presence_confidence=float(pose_cfg.get("min_pose_presence_confidence", 0.6)),
        output_world_landmarks=bool(pose_cfg.get("output_world_landmarks", True)),
        selected_joints=selected_joints,
        frame_skip=int(pose_cfg.get("frame_skip", 1)),
        use_gpu=bool(pose_cfg.get("use_gpu", True)),
    )
