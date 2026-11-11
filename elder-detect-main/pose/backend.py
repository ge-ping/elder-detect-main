"""
姿态检测后端抽象 + 工厂（可插拔双后端）

统一接口：detect_frame(frame: np.ndarray (H,W,3 BGR), timestamp_ms: int)
           -> Optional[PoseResult]

- "mediapipe"：委托现有 PoseExtractor（零改动，仅在此包装）
- "yolo"     ：YoloPoseBackend（ultralytics，2D 输出，world_landmarks=None）

下游 demo / 训练脚本只依赖 PoseResult 这一标准输出类型，
因此换后端对它们完全透明。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from pose.keypoint import PoseResult

# 默认 MediaPipe 模型路径（与现有 demo 保持一致）
DEFAULT_MEDIAPIPE_TASK = "model/mediapipe_pose/pose_landmarker_lite.task"


class PoseBackend(ABC):
    """姿态检测后端抽象基类。

    子类只需实现 detect_frame：输入一帧 BGR 图像 + 时间戳，
    返回标准 PoseResult（无人则返回 None）。
    """

    @abstractmethod
    def detect_frame(self, frame: np.ndarray, timestamp_ms: int) -> Optional[PoseResult]:
        """检测单帧姿态。

        Args:
            frame: BGR 图像 (H, W, 3)
            timestamp_ms: 帧时间戳（毫秒，单调递增）

        Returns:
            PoseResult（含 landmarks）；无人时返回 None
        """
        raise NotImplementedError

    def __enter__(self) -> "PoseBackend":
        return self

    def __exit__(self, *exc) -> bool:
        return False


def create_pose_backend(name: str = "mediapipe", **cfg) -> PoseBackend:
    """工厂：按名字创建姿态后端。

    Args:
        name: "mediapipe" | "yolo"
        cfg:  后端专属配置（透传给具体实现）
              - mediapipe: model_path / output_world_landmarks / use_gpu / min_pose_presence_confidence
              - yolo:      model / device / conf / iou

    Returns:
        PoseBackend 实例
    """
    name = (name or "mediapipe").lower()
    if name == "mediapipe":
        from pose.extractor import PoseExtractor, PoseExtractorConfig
        config = PoseExtractorConfig(
            model_path=cfg.get("model_path", DEFAULT_MEDIAPIPE_TASK),
            output_world_landmarks=cfg.get("output_world_landmarks", True),
            use_gpu=cfg.get("use_gpu", True),
            min_pose_presence_confidence=cfg.get("min_pose_presence_confidence", 0.6),
            min_detection_confidence=cfg.get("min_detection_confidence", 0.5),
            min_tracking_confidence=cfg.get("min_tracking_confidence", 0.5),
        )
        return PoseExtractor(config)

    if name == "yolo":
        from pose.yolo_backend import YoloPoseBackend
        return YoloPoseBackend(
            model=cfg.get("model", "yolov8n-pose.pt"),
            device=cfg.get("device", "cuda"),
            conf=cfg.get("conf", 0.5),
            iou=cfg.get("iou", 0.7),
        )

    raise ValueError(f"Unknown pose backend: {name!r} (expected 'mediapipe' or 'yolo')")
