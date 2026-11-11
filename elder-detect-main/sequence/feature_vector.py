"""
单帧特征拼接 (FrameFeatureBuilder)

将单帧 PoseResult + 标量特征拼接为 K 维向量（默认 29 维）。
"""
from __future__ import annotations

import numpy as np

from .config import WindowConfig


class FrameFeatureBuilder:
    """单帧特征拼接器

    输入: PoseResult + 标量特征 dict
    输出: (K,) ndarray — 论文5 默认 K=29

    K 维构成顺序:
        [scalars...,  joint_0_x, joint_0_y, joint_1_x, joint_1_y, ...]

    标量顺序由 config 开关控制:
        gait_speed (if enabled) → trunk_angle (if enabled) → aspect_ratio (if enabled)
    """

    def __init__(self, config: WindowConfig) -> None:
        self.config = config

    # ==================================================================
    # 主接口
    # ==================================================================

    def build(
        self,
        result,
        scalars: dict[str, float],
    ) -> np.ndarray:
        """单帧 → K 维特征向量

        Args:
            result: PoseResult（需含 landmarks）
            scalars: {"gait_speed": float, "trunk_angle": float, "aspect_ratio": float}
                     按 config 开关选择性提供，缺的自动填 0.0

        Returns:
            (K,) float32 ndarray
        """
        parts: list[np.ndarray] = []

        # ---- 标量特征 ----
        if self.config.use_gait_speed:
            parts.append(np.array([scalars.get("gait_speed", 0.0)], dtype=np.float32))
        if self.config.use_trunk_angle:
            parts.append(np.array([scalars.get("trunk_angle", 0.0)], dtype=np.float32))
        if self.config.use_aspect_ratio:
            parts.append(np.array([scalars.get("aspect_ratio", 0.0)], dtype=np.float32))

        # ---- 骨架坐标 ----
        skeleton = self._extract_skeleton(result)
        parts.append(skeleton)

        return np.concatenate(parts, dtype=np.float32)

    # ==================================================================
    # 内部
    # ==================================================================

    def _extract_skeleton(self, result) -> np.ndarray:
        """提取骨架 (2*N_joints,) 坐标向量

        内部保留 visibility 访问路径，当前 29 维不拼接 visibility，
        但不丢弃 —— 后续加 x,y,vis 只需改本方法 + config，无需重构上游。
        """
        coords: list[float] = []
        for joint in self.config.joint_set:
            try:
                kp = result.get(joint)
            except KeyError:
                coords.extend([0.0, 0.0])
                continue
            # kp.visibility  ← 保留访问路径，当前暂不拼入输出
            coords.extend([float(kp.x), float(kp.y)])
        return np.array(coords, dtype=np.float32)

    @property
    def dim(self) -> int:
        """单帧特征维度 K"""
        return self.config.feature_dim()
