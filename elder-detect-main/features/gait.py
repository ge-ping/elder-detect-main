"""
2.2.2 人体步行速度提取器 (Gait Speed Extractor)

论文5 (Lau et al., 2022): 步速 — 脚部中心在 3D 世界坐标中的位移速度，
是区分 Normal/Unstable/Fall 三分类的核心判别特征。

计算流程 (论文 §2.2.2):
    1. 取窗口首帧: M_t = (LEFT_ANKLE + RIGHT_ANKLE) / 2  (world_landmarks, 米)
    2. 取窗口尾帧: M_{t+N} = (LEFT_ANKLE + RIGHT_ANKLE) / 2
    3. D_f = |M_{t+N} - M_t|  (3D 欧氏距离)
    4. Δt = (timestamp_ms[N-1] - timestamp_ms[0]) / 1000  (秒)
    5. V_f = D_f / Δt  (m/s)

两种接口:
    - extract_window(results)  窗口级（论文原文，供 FeaturePipeline 用）
    - compute_per_frame(result) 逐帧（新增，供 build_dataset 用）
      内部维护 foot_center 历史缓存，V = ||foot(t) - foot(t-12)|| / Δt
      不足 12 帧 → 0.0
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .base import FeatureExtractor
from pose.keypoint import BodyPart


class GaitSpeedExtractor(FeatureExtractor):
    """步行速度提取器

    输入:
        PoseResult.world_landmarks (list[Keypoint], 单位: 米)
        需要 BodyPart: LEFT_ANKLE, RIGHT_ANKLE

    输出:
        {"walking_speed_ms": float}  — 窗口内脚部中心移动速度 (m/s)
        {"cadence": float}           — 暂未实现，固定返回 0.0

    论文参考:
        - 论文5 §2.2.2 Human Gait Speed: V_f = D_f / Δt
        - Lau et al., Int. J. Technology 13(6) 1173-1182 (2022)
    """

    # 默认窗口帧数（论文用 12 帧）
    DEFAULT_WINDOW_SIZE = 12

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
        """
        Args:
            window_size: 滑动窗口帧数，默认 12（论文值）
        """
        super().__init__(name="GaitSpeed")
        self.window_size = window_size
        self._history: deque[tuple[np.ndarray, int]] = deque(maxlen=window_size + 1)

    # ==================================================================
    # 逐帧特征（仅缓存）
    # ==================================================================

    def extract_frame(self, result) -> dict[str, float]:
        """逐帧调用不计算速度，仅供 FeaturePipeline 统一接口兼容。

        速度计算在 extract_window() 中进行。

        Returns:
            {"walking_speed_ms": 0.0}
        """
        return {"walking_speed_ms": 0.0}

    # ==================================================================
    # 窗口级特征
    # ==================================================================

    def extract_window(self, results: list) -> dict[str, float]:
        """从滑动窗口计算步行速度

        Args:
            results: 连续帧 PoseResult 列表
                     （需 world_landmarks 非 None，含 LEFT/RIGHT_ANKLE）

        Returns:
            {
                "walking_speed_ms": float,  # 脚中心移动速度 (m/s)
                "cadence": float,           # 暂为 0.0
            }

        公式:
            M_t = (LEFT_ANKLE + RIGHT_ANKLE) / 2   (米)
            D_f = |M_{last} - M_{first}|            (米, 3D欧氏)
            V_f = D_f / Δt                          (m/s, Δt 从时间戳差得出)
        """
        if len(results) < 2:
            return {"walking_speed_ms": 0.0, "cadence": 0.0}

        # ---- 1. 首尾帧脚部中心 ----
        first = results[0]
        last = results[-1]

        m_start = self._foot_center(first)   # (3,) ndarray, 米
        m_end = self._foot_center(last)

        # ---- 2. 3D 位移距离 (米) ----
        d_f = float(np.linalg.norm(m_end - m_start))

        # ---- 3. 时间跨度 (秒) ----
        dt = (last.timestamp_ms - first.timestamp_ms) / 1000.0
        if dt <= 0.0:
            return {"walking_speed_ms": 0.0, "cadence": 0.0}

        # ---- 4. 速度 (m/s) ----
        v_f = d_f / dt

        # cadence 暂未实现（需步态周期检测）
        return {"walking_speed_ms": v_f, "cadence": 0.0}

    # ==================================================================
    # 内部方法
    # ==================================================================

    @staticmethod
    def _foot_center(result) -> np.ndarray:
        """计算左右脚踝中点（world_landmarks，米）

        Args:
            result: PoseResult（需 world_landmarks 非 None）

        Returns:
            (3,) ndarray — 脚部中心 3D 坐标 (x, y, z) 米
        """
        la = result.get_world(BodyPart.LEFT_ANKLE)
        ra = result.get_world(BodyPart.RIGHT_ANKLE)
        return np.array([
            (la.x + ra.x) / 2,
            (la.y + ra.y) / 2,
            (la.z + ra.z) / 2,
        ], dtype=np.float64)

    # ==================================================================
    # 逐帧步速（新增 — 供 build_dataset 用）
    # ==================================================================

    def compute_per_frame(self, result) -> float:
        """逐帧步速

        维护内部 foot_center 历史缓存 (maxlen = window_size + 1)。
        不足 window_size 帧历史 → 返回 0.0。
        足 window_size 帧 → V = ||foot(t) - foot(t-window_size)|| / Δt

        Args:
            result: PoseResult（需 world_landmarks 非 None）

        Returns:
            步行速度 (m/s), 不足窗口时返回 0.0
        """
        center = self._foot_center(result)
        ts = result.timestamp_ms
        self._history.append((center, ts))

        if len(self._history) <= self.window_size:
            return 0.0

        prev_center, prev_ts = self._history[0]  # t - window_size
        dist = float(np.linalg.norm(center - prev_center))
        dt = (ts - prev_ts) / 1000.0
        if dt <= 0.0:
            return 0.0
        return dist / dt

    # ==================================================================
    # 生命周期
    # ==================================================================

    def reset(self) -> None:
        """重置内部历史缓存"""
        self._history.clear()
