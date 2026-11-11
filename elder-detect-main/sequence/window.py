"""
滑动窗口构建器 (SlidingWindowBuilder)

将 T 帧特征序列 → (N, window_size, K) 时序样本 → LSTM 输入。
"""
from __future__ import annotations

import numpy as np

from .config import WindowConfig
from .feature_vector import FrameFeatureBuilder


class SlidingWindowBuilder:
    """滑动窗口构建器

    输入: T 帧 PoseResult 列表 + 逐帧标量特征
    输出: (N, window_size, K) ndarray

    使用:
        >>> config = WindowConfig()
        >>> builder = SlidingWindowBuilder(config)
        >>> X = builder.build(results, {
        ...     "gait_speed": [v0, v1, ...],
        ...     "trunk_angle": [a0, a1, ...],
        ...     "aspect_ratio": [r0, r1, ...],
        ... })
        >>> X.shape  # (N, 35, 29)
    """

    def __init__(self, config: WindowConfig) -> None:
        self.config = config
        self._frame_builder = FrameFeatureBuilder(config)

    # ==================================================================
    # 主接口
    # ==================================================================

    def build(
        self,
        results: list,
        scalars_per_frame: dict[str, list[float]],
    ) -> np.ndarray:
        """构建时序窗口样本

        Args:
            results: T 帧 PoseResult 列表
            scalars_per_frame: 逐帧标量特征
                {"gait_speed": [T个], "trunk_angle": [T个], "aspect_ratio": [T个]}
                每个 list 长度应与 results 相同

        Returns:
            (N, window_size, K) float32 ndarray
            N = max(0, (T - window_size) // stride + 1)
        """
        T = len(results)
        window_size = self.config.window_size
        stride = self.config.stride

        # ---- 逐帧拼接为 (T, K) ----
        frames = self._build_all_frames(results, scalars_per_frame)

        # ---- 滑动窗口 ----
        if T < window_size:
            return np.empty((0, window_size, self._frame_builder.dim), dtype=np.float32)

        windows: list[np.ndarray] = []
        for start in range(0, T - window_size + 1, stride):
            windows.append(frames[start : start + window_size])

        if not windows:
            return np.empty((0, window_size, self._frame_builder.dim), dtype=np.float32)

        return np.stack(windows, axis=0).astype(np.float32)

    # ==================================================================
    # 辅助
    # ==================================================================

    def _build_all_frames(
        self,
        results: list,
        scalars_per_frame: dict[str, list[float]],
    ) -> np.ndarray:
        """所有帧 → (T, K) ndarray"""
        T = len(results)
        K = self._frame_builder.dim
        frames = np.empty((T, K), dtype=np.float32)

        for t in range(T):
            # 每帧的标量 dict
            sf: dict[str, float] = {}
            for key, seq in scalars_per_frame.items():
                if t < len(seq):
                    sf[key] = seq[t]
                else:
                    sf[key] = 0.0
            frames[t] = self._frame_builder.build(results[t], sf)

        return frames

    # ==================================================================
    # 直接从特征数组构建（供 build_dataset 用）
    # ==================================================================

    def build_from_array(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
        risk: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """从预构建的 (T, K) 特征数组生成滑动窗口

        Args:
            features: (T, K) float32
            labels:   (T,) int64 or None — 逐帧离散标签
            risk:     (T,) float32 or None — 逐帧连续 risk（回归用）

        Returns:
            X:       (N, window_size, K) float32
            y_cls:   (N,) int64 or None — 窗口最后一帧离散标签
            y_risk:  (N,) float32 or None — 窗口最后一帧连续 risk（回归目标）

        ★ 原行为不变：只传 labels 时 y_risk=None，调用方仍按 (X, y_cls) 用。
        """
        T = features.shape[0]
        window_size = self.config.window_size
        stride = self.config.stride

        if T < window_size:
            K = features.shape[1]
            empty_X = np.empty((0, window_size, K), dtype=np.float32)
            empty_cls = np.empty((0,), dtype=np.int64) if labels is not None else None
            empty_risk = np.empty((0,), dtype=np.float32) if risk is not None else None
            return empty_X, empty_cls, empty_risk

        windows_X: list[np.ndarray] = []
        windows_cls: list[int] = []
        windows_risk: list[float] = []

        for start in range(0, T - window_size + 1, stride):
            windows_X.append(features[start : start + window_size])
            if labels is not None:
                windows_cls.append(int(labels[start + window_size - 1]))  # 最后一帧标签
            if risk is not None:
                windows_risk.append(float(risk[start + window_size - 1]))  # 最后一帧 risk

        X = np.stack(windows_X, axis=0).astype(np.float32)
        y_cls = np.array(windows_cls, dtype=np.int64) if labels is not None else None
        y_risk = np.array(windows_risk, dtype=np.float32) if risk is not None else None
        return X, y_cls, y_risk

    def sample_count(self, total_frames: int) -> int:
        """给定总帧数，返回可切样本数"""
        w = self.config.window_size
        s = self.config.stride
        if total_frames < w:
            return 0
        return (total_frames - w) // s + 1
