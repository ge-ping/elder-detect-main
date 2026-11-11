"""
features/intent.py — 运动意图投影器（轨道C）

基于 pose 髋中心提取 2D 速度矢量，投射未来 INTENT_HORIZON 帧的预测区域 Zone(t)。
单目沿光轴失效的兜底：|v|₂D 小但 world_z 持续减小 → 判定沿光轴接近，
预测区域沿画面中心方向拉长（救回"正对镜头走来"的失效场景）。

设计见 docs/环境交互风险预测_设计.md §5 通道2。

用到的关键点：BodyPart.LEFT_HIP(23) / RIGHT_HIP(24)
  - landmarks（归一化2D [0,1]）→ 速度矢量
  - world_landmarks.z（米，相对髋中心）→ 深度代理兜底
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from .base import FeatureExtractor
from pose.keypoint import BodyPart
from scripts.build_config import (
    INTENT_HORIZON,
    INTENT_VEL_WINDOW,
    INTENT_ZONE_LONG_AXIS,
    INTENT_ZONE_SHORT_AXIS,
    INTENT_OPTICAL_VEL_THRESH,
    INTENT_OPTICAL_Z_THRESH,
)


class MotionIntentProjector(FeatureExtractor):
    """运动意图投影器

    逐帧输出 2D 速度矢量 + 预测区域 Zone(t) 椭圆参数，供 IntentGate 与
    环境掩码做重叠度计算。

    接口风格对齐 ``GaitSpeedExtractor``：``compute_per_frame`` + 内部历史缓存 + ``reset``。
    同时实现 ``extract_frame`` 以兼容 ``FeaturePipeline``。

    Args:
        vel_window: 速度矢量滑窗帧数 k（v = (p(t) − p(t−k)) / k）
        horizon: 预测未来帧数（p_pred = p(t) + v × horizon）
        zone_long_axis: 预测椭圆长轴（归一化坐标）
        zone_short_axis: 预测椭圆短轴（横向容差，归一化坐标）
        optical_vel_thresh: |v|₂D 低于此 → 疑似沿光轴运动
        optical_z_thresh: world_z 帧间变化小于此(米) → 沿光轴接近（z 持续减小）
    """

    def __init__(
        self,
        vel_window: int = INTENT_VEL_WINDOW,
        horizon: int = INTENT_HORIZON,
        zone_long_axis: float = INTENT_ZONE_LONG_AXIS,
        zone_short_axis: float = INTENT_ZONE_SHORT_AXIS,
        optical_vel_thresh: float = INTENT_OPTICAL_VEL_THRESH,
        optical_z_thresh: float = INTENT_OPTICAL_Z_THRESH,
    ) -> None:
        super().__init__(name="MotionIntent")
        self.vel_window = max(vel_window, 1)
        self.horizon = horizon
        self.zone_long_axis = zone_long_axis
        self.zone_short_axis = zone_short_axis
        self.optical_vel_thresh = optical_vel_thresh
        self.optical_z_thresh = optical_z_thresh
        # 缓存：(hip_center_2d, world_z, timestamp_ms)，maxlen = vel_window + 1
        self._history: deque[tuple[np.ndarray, float, int]] = deque(
            maxlen=self.vel_window + 1
        )

    # ==================================================================
    # FeatureExtractor 接口
    # ==================================================================

    def extract_frame(self, result) -> dict[str, float]:
        """FeaturePipeline 兼容入口，等价于 compute_per_frame。"""
        return self.compute_per_frame(result)

    def compute_per_frame(self, result) -> dict[str, float]:
        """逐帧计算 2D 速度矢量 + 预测区域 Zone(t)。

        Args:
            result: PoseResult（需 landmarks 含 LEFT_HIP/RIGHT_HIP；
                     world_landmarks 可选，缺失则跳过沿光轴兜底）

        Returns:
            dict 含：
              vel_x, vel_y     : 每帧位移（归一化坐标/帧）
              vel_mag          : 速度幅值
              zone_cx, zone_cy : 预测椭圆中心（归一化坐标，可能超出 [0,1]）
              zone_long_axis   : 椭圆长轴（归一化）
              zone_short_axis  : 椭圆短轴（归一化）
              zone_angle       : 椭圆长轴方向角（弧度）
              optical_axis_approach: 沿光轴接近标志（1.0/0.0）
        """
        p = self._hip_center_2d(result)    # 髋中心2D (2,) 归一化坐标
        foot_y = self._foot_y(result)      # 脚跟平均y（地面接触位，预测区域纵向锚点）
        z = self._hip_world_z(result)      # 髋世界z（米，缺失为 nan）
        self._history.append((p, foot_y, z, result.timestamp_ms))

        # 滑窗未满 → 速度 0
        if len(self._history) <= self.vel_window:
            v_per_frame = np.zeros(2)
            vel_mag = 0.0
            optical = False
        else:
            p_prev, _, z_prev, _ = self._history[0]
            dv = p - p_prev                    # (2,) 归一化位移
            v_per_frame = dv / self.vel_window  # 归一化坐标/帧
            vel_mag = float(np.linalg.norm(v_per_frame))
            optical = self._is_optical_axis_approach(vel_mag, z, z_prev)

        # 预测区域中心：纵向锚定脚下（与环境掩码地面ROI对齐），
        # 横向用速度外推——老人走到哪里脚踩到哪里，环境交互点在脚下而非髋部
        if optical:
            # 沿光轴接近：原地脚下，沿画面中心方向拉长
            center = np.array([p[0], foot_y])
            dir_to_center = np.array([0.5, 0.5]) - center
            norm = float(np.linalg.norm(dir_to_center))
            angle = float(math.atan2(dir_to_center[1], dir_to_center[0])) if norm > 1e-6 else 0.0
        else:
            # 正常：横向落点 = 髋x + 速度x × horizon，纵向 = 脚下
            center = np.array([p[0] + v_per_frame[0] * self.horizon, foot_y])
            angle = float(math.atan2(v_per_frame[1], v_per_frame[0]))

        return {
            "vel_x": float(v_per_frame[0]),
            "vel_y": float(v_per_frame[1]),
            "vel_mag": vel_mag,
            "zone_cx": float(center[0]),
            "zone_cy": float(center[1]),
            "zone_long_axis": self.zone_long_axis,
            "zone_short_axis": self.zone_short_axis,
            "zone_angle": angle,
            "optical_axis_approach": 1.0 if optical else 0.0,
        }

    # ==================================================================
    # 内部
    # ==================================================================

    @staticmethod
    def _hip_center_2d(result) -> np.ndarray:
        """髋中心 2D 归一化坐标 = (LEFT_HIP + RIGHT_HIP) / 2。"""
        lh = result.get(BodyPart.LEFT_HIP)
        rh = result.get(BodyPart.RIGHT_HIP)
        return np.array([(lh.x + rh.x) / 2, (lh.y + rh.y) / 2], dtype=np.float64)

    @staticmethod
    def _hip_world_z(result) -> float:
        """髋中心世界 z（米，相对髋中心）。world_landmarks 缺失 → nan。"""
        try:
            lh = result.get_world(BodyPart.LEFT_HIP)
            rh = result.get_world(BodyPart.RIGHT_HIP)
            return float((lh.z + rh.z) / 2)
        except (ValueError, KeyError):
            return float("nan")

    def _is_optical_axis_approach(self, vel_mag: float, z_now: float, z_prev: float) -> bool:
        """沿光轴接近判定：|v|₂D 小 且 world_z 持续减小。

        z 是 MediaPipe 伪深度（相对髋中心，单位米）。老人正对镜头走来时，
        z 会持续减小（靠近相机）。虽是伪深度，但相对变化有信号，
        能救回单目沿光轴失效的半数场景。
        """
        if vel_mag >= self.optical_vel_thresh:
            return False
        if math.isnan(z_now) or math.isnan(z_prev):
            return False
        # z_delta < 0 表示 z 减小（靠近相机）；小于阈值（负）即显著接近
        z_delta = z_now - z_prev
        return z_delta < self.optical_z_thresh

    @staticmethod
    def _foot_y(result) -> float:
        """脚跟平均y（归一化），fallback 脚踝。代表地面接触纵向位置。

        预测区域纵向锚定脚下，与环境掩码的地面 ROI 对齐——
        老人走到哪里脚踩到哪里，环境交互点在脚下而非髋部。
        无任何脚部关键点时 fallback 髋y下方 0.3（粗略脚下位）。
        """
        ys = []
        for part in (BodyPart.LEFT_HEEL, BodyPart.RIGHT_HEEL,
                     BodyPart.LEFT_ANKLE, BodyPart.RIGHT_ANKLE):
            try:
                kp = result.get(part)
            except KeyError:
                continue
            if kp.visibility >= 0.5:
                ys.append(kp.y)
        if ys:
            return float(sum(ys) / len(ys))
        # 无脚部关键点 → fallback 髋y下方
        try:
            hip_y = (result.get(BodyPart.LEFT_HIP).y + result.get(BodyPart.RIGHT_HIP).y) / 2
            return float(min(hip_y + 0.3, 1.0))
        except KeyError:
            return 0.8

    # ==================================================================
    # 生命周期
    # ==================================================================

    def reset(self) -> None:
        """重置内部历史缓存（切换视频时调用）。"""
        self._history.clear()
