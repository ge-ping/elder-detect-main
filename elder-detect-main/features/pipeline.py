"""
特征提取管道 (Feature Pipeline)

串联多个 FeatureExtractor，统一调用、合并输出 FeatureResult。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import FeatureExtractor
from .types import FeatureResult


class FeaturePipeline:
    """特征提取管道

    管理多个 FeatureExtractor，统一调用接口，合并输出为 FeatureResult。

    使用方式:
        >>> pipeline = FeaturePipeline([
        ...     TorsoAngleExtractor(),
        ...     SkeletonFeatureExtractor(),
        ...     GaitSpeedExtractor(window_size=30),
        ... ])
        >>> for result in extractor.process_video("walk.mp4"):
        ...     feat = pipeline.process_frame(result)
        ...     arr = feat.to_array()  # (5,) → LSTM

    窗口模式（含步速）:
        >>> buffer = []
        >>> for result in extractor.process_video("walk.mp4"):
        ...     buffer.append(result)
        ...     if len(buffer) >= 30:
        ...         feat = pipeline.process_window(buffer)
        ...         buffer.pop(0)
    """

    def __init__(self, extractors: list[FeatureExtractor]) -> None:
        """
        Args:
            extractors: FeatureExtractor 列表（顺序无关，各自独立计算）
        """
        self.extractors = extractors

    # ========== 逐帧处理 ==========

    def process_frame(self, result) -> FeatureResult:
        """逐帧处理：调用所有 extractor.extract_frame()

        注意: 不包含窗口级特征（walking_speed_ms/cadence 保持 0）。

        Args:
            result: PoseResult 单帧

        Returns:
            FeatureResult（逐帧特征已填充，窗口特征为 0）
        """
        features: dict[str, float] = {}
        for ext in self.extractors:
            try:
                frame_feats = ext.extract_frame(result)
                features.update(frame_feats)
            except NotImplementedError:
                # extract_frame 未被覆写，跳过
                pass

        return FeatureResult(
            torso_angle_deg=features.get("torso_angle_deg", 0.0),
            aspect_ratio=features.get("aspect_ratio", 0.0),
            symmetry=features.get("symmetry", 0.0),
            walking_speed_ms=features.get("walking_speed_ms", 0.0),
            cadence=features.get("cadence", 0.0),
            timestamp_ms=result.timestamp_ms,
        )

    # ========== 窗口处理 ==========

    def process_window(self, results: list) -> FeatureResult:
        """窗口处理：逐帧特征（取最后一帧）+ 窗口级特征合并

        Args:
            results: 连续帧 PoseResult 列表

        Returns:
            FeatureResult（所有特征已填充）
        """
        # 1. 逐帧特征：以窗口最后一帧为准
        frame_feat = self.process_frame(results[-1])

        # 2. 窗口级特征：调用支持窗口的 extractor
        window_feats: dict[str, float] = {}
        for ext in self.extractors:
            try:
                wf = ext.extract_window(results)
                window_feats.update(wf)
            except NotImplementedError:
                pass

        # 3. 合并：窗口特征覆盖逐帧的 0 值
        if window_feats:
            frame_feat.walking_speed_ms = window_feats.get(
                "walking_speed_ms", frame_feat.walking_speed_ms
            )
            frame_feat.cadence = window_feats.get(
                "cadence", frame_feat.cadence
            )

        return frame_feat

    # ========== 批量处理 ==========

    def process_sequence(
        self,
        results: list,
        window_size: int = 30,
    ) -> np.ndarray:
        """处理完整序列，输出 (T, K) 特征矩阵 → 直接送入 LSTM

        Args:
            results: 完整视频帧的 PoseResult 列表
            window_size: 滑动窗口大小（帧数）

        Returns:
            np.ndarray shape (T, K)，K = FeatureResult.feature_dim()
        """
        buffer: list = []
        features: list[np.ndarray] = []

        for result in results:
            buffer.append(result)
            if len(buffer) >= window_size:
                feat = self.process_window(buffer)
                features.append(feat.to_array())
                buffer.pop(0)

        if not features:
            return np.empty((0, FeatureResult.feature_dim()), dtype=np.float32)

        return np.stack(features, axis=0)

    # ========== 生命周期 ==========

    def reset(self) -> None:
        """重置所有 extractor 的内部状态"""
        for ext in self.extractors:
            ext.reset()

    def __repr__(self) -> str:
        names = [e.name for e in self.extractors]
        return f"FeaturePipeline({names})"
