"""
YOLO 姿态后端（ultralytics YOLOv8/v11-pose）

- 输出 2D 关键点（x, y 像素 + conf），**无 3D 世界坐标**
- 转为标准 PoseResult：landmarks 填 13 核心关节 (x,y 归一化 [0,1])，
  visibility=conf，world_landmarks=None
- 实现 PoseBackend 接口，与 MediaPipe 后端输出类型一致 → 下游透明

独立性：本文件为全新模块，不 import / 不修改 pose/extractor.py 与
pose/keypoint.py。仅依赖 ultralytics（环境已装 8.3.0）。
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from pose.backend import PoseBackend
from pose.coco_joints import COCO_CORE_INDICES, COCO_TO_BODYPART
from pose.keypoint import BodyPart, Keypoint, PoseResult


class YoloPoseBackend(PoseBackend):
    """基于 ultralytics 的 2D 姿态估计后端。"""

    def __init__(self,
                 model: str = "yolov8n-pose.pt",
                 device: str = "cuda",
                 conf: float = 0.5,
                 iou: float = 0.7) -> None:
        """
        Args:
            model: YOLO pose 权重（默认 yolov8n-pose.pt，首次自动下载）
            device: "cuda" / "cpu" / "0"
            conf:  关键点置信度阈值
            iou:   NMS IoU 阈值
        """
        from ultralytics import YOLO

        self.model_name = model
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device

        self.model = YOLO(model)
        # 尝试放到目标设备；失败则回退 CPU（不影响功能，仅降速）
        try:
            self.model.to(device)
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] YOLO to({device}) failed, fallback cpu: {e}")
            self.model.to("cpu")
            self.device = "cpu"

    # ==================================================================
    # 主接口
    # ==================================================================

    def detect_frame(self, frame: np.ndarray, timestamp_ms: int) -> Optional[PoseResult]:
        """检测单帧。

        Returns:
            PoseResult（13 核心关节 2D，world_landmarks=None）；无人返回 None
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.model(
            rgb, verbose=False, conf=self.conf, iou=self.iou, device=self.device
        )
        res = results[0]
        kpts = res.keypoints
        if kpts is None or len(kpts.data) == 0:
            return None

        # kpts.data: (N_person, 17, 3) -> 转 numpy（兼容 torch / numpy 两种返回）
        raw = kpts.data
        if hasattr(raw, "cpu"):           # torch.Tensor -> numpy
            raw = raw.cpu().numpy()
        data = np.asarray(raw, dtype=np.float32)  # (N, 17, 3)

        # 无人 / 空检测（形状可能为 (0,17,3) 或 (1,0,3)）-> 返回 None
        if data.shape[0] == 0 or data.shape[1] == 0:
            return None

        # 选平均置信度最高的一人
        best = int(np.argmax(data[:, :, 2].mean(axis=1)))
        person = data[best]  # (17, 3) => x, y, conf

        # 全部低于阈值视为无人（与 MediaPipe 语义一致）
        if float(person[:, 2].max()) < self.conf:
            return None

        landmarks: list[Keypoint] = []
        for idx in COCO_CORE_INDICES:
            cx, cy, cconf = float(person[idx, 0]), float(person[idx, 1]), float(person[idx, 2])
            bp = COCO_TO_BODYPART[idx]
            if bp is None:
                continue
            # 归一化到 [0,1]（与 MediaPipe landmarks 约定一致：
            # 下游用 x*frame_width / y*frame_height 还原像素）
            landmarks.append(Keypoint(
                x=cx / w, y=cy / h, z=0.0,
                visibility=cconf, name=bp,
            ))

        if not landmarks:
            return None

        return PoseResult(
            landmarks=landmarks,
            world_landmarks=None,  # 2D 后端无 3D
            timestamp_ms=timestamp_ms,
            frame_width=w,
            frame_height=h,
        )

    def __repr__(self) -> str:
        return f"YoloPoseBackend(model={self.model_name}, device={self.device})"
