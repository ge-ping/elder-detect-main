"""
特征提取器抽象基类

所有特征提取器（逐帧/窗口级）统一实现此接口，
由 FeaturePipeline 统一调度。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

# 延迟导入避免循环依赖
# PoseResult 在 extract_frame 参数中由调用方传入


class FeatureExtractor(ABC):
    """特征提取器抽象基类

    子类必须实现 extract_frame()。
    窗口级特征（如步速）额外覆写 extract_window()。

    Attributes:
        name: 提取器名称（用于调试和日志）
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    # ========== 逐帧特征（必须实现） ==========

    @abstractmethod
    def extract_frame(self, result) -> dict[str, float]:
        """从单帧 PoseResult 提取逐帧特征

        Args:
            result: PoseResult 对象（来自 pose 模块）

        Returns:
            dict[str, float]: 特征名 → 值
            例如 {"torso_angle_deg": 12.5, "aspect_ratio": 0.42, "symmetry": 0.95}
        """
        ...

    # ========== 窗口级特征（按需覆写） ==========

    def extract_window(self, results: list) -> dict[str, float]:
        """从滑动窗口提取窗口级特征

        默认 raise NotImplementedError。
        子类（如 GaitSpeedExtractor）覆写此方法。

        Args:
            results: 连续帧的 PoseResult 列表（长度 = window_size）

        Returns:
            dict[str, float]: 特征名 → 值
            例如 {"walking_speed_ms": 1.23, "cadence": 1.8}
        """
        raise NotImplementedError(
            f"{self.name} does not support window-level features"
        )

    # ========== 生命周期 ==========

    def reset(self) -> None:
        """重置内部状态

        切换视频/序列时调用，清空滑动窗口缓存等。
        默认无操作，子类（如 GaitSpeedExtractor）按需覆写。
        """
        pass

    def __repr__(self) -> str:
        return f"{self.name}()"
