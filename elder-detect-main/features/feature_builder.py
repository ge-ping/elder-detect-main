"""
单帧 29 维特征拼接 (FrameFeatureBuilder)

复用现有 TorsoAngleExtractor + SkeletonFeatureExtractor，
组装: [gait_speed, trunk_angle, aspect_ratio, 13关节×2(x,y)]
"""
from __future__ import annotations

import numpy as np

from pose.keypoint import BodyPart, CORE_JOINTS_13
from .torso import TorsoAngleExtractor
from .skeleton import SkeletonFeatureExtractor


class FrameFeatureBuilder:
    """单帧 → 29 维特征向量

    复用:
        - TorsoAngleExtractor   → trunk_angle (三平面 arctan2 + df 择优)
        - SkeletonFeatureExtractor → aspect_ratio (W/H)
        - landmarks[CORE_JOINTS_13] → skeleton_26 (x,y)

    不拼接: z, visibility, symmetry, cadence
    """

    def __init__(self, use_angles: bool = False) -> None:
        self._torso = TorsoAngleExtractor()
        self._skeleton = SkeletonFeatureExtractor()
        self.use_angles = use_angles

    # ==================================================================
    # 主接口
    # ==================================================================

    def build(self, result, gait_speed: float) -> np.ndarray:
        """单帧 → (29,) float32

        Args:
            result: PoseResult（需 world_landmarks + landmarks）
            gait_speed: 逐帧步速 (m/s), 来自 GaitSpeedExtractor.compute_per_frame()

        Returns:
            (29,) ndarray float32
        """
        # 标量特征
        trunk_angle = self._torso.extract_frame(result)["torso_angle_deg"]
        aspect_ratio = self._skeleton.extract_frame(result)["aspect_ratio"]

        scalars = np.array([gait_speed, trunk_angle, aspect_ratio], dtype=np.float32)

        # 骨架 26 维
        skeleton = self._extract_skeleton(result)

        vec = np.concatenate([scalars, skeleton])

        # 关节角度 7 维（可选）
        if self.use_angles:
            from .joint_angle import extract_joint_angles
            angles = extract_joint_angles(result)
            vec = np.concatenate([vec, angles])

        return vec

    # ==================================================================
    # 内部
    # ==================================================================

    @staticmethod
    def _extract_skeleton(result) -> np.ndarray:
        """13 核心关节 × (x, y) → (26,) float32"""
        coords: list[float] = []
        for joint in CORE_JOINTS_13:
            try:
                kp = result.get(joint)
                coords.extend([float(kp.x), float(kp.y)])
            except KeyError:
                coords.extend([0.0, 0.0])
        return np.array(coords, dtype=np.float32)

    @property
    def dim(self) -> int:
        return 29
