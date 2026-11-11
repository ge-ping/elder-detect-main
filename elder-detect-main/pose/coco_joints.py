"""
YOLO / COCO 姿态关键点映射表（与 MediaPipe 33 点完全解耦）

YOLOv8/v11-pose 输出 COCO-17 关键点（2D 像素坐标 + 每点 conf），
没有 MediaPipe 的 3D 世界坐标，也没有脚跟/脚尖（29/30/31/32）。

本模块只做「COCO 索引 → 现有 BodyPart 枚举」的映射，
**不修改** pose/keypoint.py 的 BodyPart 枚举（保持 MediaPipe 兼容）。

实际后端只用 13 个核心关节（丢弃 eyes/ears，跌倒图不需要面部细节），
与 LSTM/ST-GCN 的 CORE_JOINTS_13 一一对应。
"""
from __future__ import annotations

from typing import Optional

from pose.keypoint import BodyPart


# ---- COCO-17 关键点名称（索引即 ultralytics 输出列序）----
COCO_JOINT_NAMES: list[str] = [
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
]


# ---- COCO 索引 → BodyPart（None = 丢弃，如 eyes/ears）----
COCO_TO_BODYPART: dict[int, Optional[BodyPart]] = {
    0:  BodyPart.NOSE,
    1:  None,  # left_eye
    2:  None,  # right_eye
    3:  None,  # left_ear
    4:  None,  # right_ear
    5:  BodyPart.LEFT_SHOULDER,
    6:  BodyPart.RIGHT_SHOULDER,
    7:  BodyPart.LEFT_ELBOW,
    8:  BodyPart.RIGHT_ELBOW,
    9:  BodyPart.LEFT_WRIST,
    10: BodyPart.RIGHT_WRIST,
    11: BodyPart.LEFT_HIP,
    12: BodyPart.RIGHT_HIP,
    13: BodyPart.LEFT_KNEE,
    14: BodyPart.RIGHT_KNEE,
    15: BodyPart.LEFT_ANKLE,
    16: BodyPart.RIGHT_ANKLE,
}


# ---- 后端实际使用的 13 核心关节（COCO 索引，按升序）----
COCO_CORE_INDICES: list[int] = [
    i for i, bp in sorted(COCO_TO_BODYPART.items()) if bp is not None
]


# ---- 绘制用骨架边（按 BodyPart 对，跳过了 eyes/ears）----
# 与 ST-GCN 的 FALL_EDGES 同款「跌倒图」拓扑，便于直观对比。
YOLO_CORE_EDGES_BODYPART: list[tuple[BodyPart, BodyPart]] = [
    (BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER),    # 肩线
    (BodyPart.LEFT_SHOULDER, BodyPart.LEFT_ELBOW),
    (BodyPart.LEFT_ELBOW, BodyPart.LEFT_WRIST),
    (BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW),
    (BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST),
    (BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP),
    (BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_HIP),
    (BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP),              # 髋线
    (BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE),
    (BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE),
    (BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE),
    (BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE),
    (BodyPart.NOSE, BodyPart.LEFT_SHOULDER),
    (BodyPart.NOSE, BodyPart.RIGHT_SHOULDER),
]
