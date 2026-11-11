"""
特征数据类型定义

FeatureResult: 单帧/窗口的特征集合，可直接转为 numpy 送入 LSTM。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class FeatureResult:
    """单帧或滑动窗口的特征集合

    各字段可能在不同帧类型中为 0（如逐帧特征不含步速时 walking_speed_ms=0）。

    Attributes:
        torso_angle_deg: 2.2.1 躯干偏离垂直轴角度（度）
        aspect_ratio: 2.2.4 人体外接矩形宽高比
        symmetry: 2.2.4 左右关键点对称性 (0~1, 1=完全对称)
        walking_speed_ms: 2.2.2 髋部中心移动速度 (m/s)
        cadence: 步频 (steps/s)，衍生自步速
        timestamp_ms: 帧时间戳
    """
    torso_angle_deg: float = 0.0
    aspect_ratio: float = 0.0
    symmetry: float = 0.0
    walking_speed_ms: float = 0.0
    cadence: float = 0.0
    timestamp_ms: int = 0

    # ---- 特征名称列表（固定顺序，to_array 以此为准） ----

    _FEATURE_NAMES: tuple[str, ...] = (
        "torso_angle_deg",
        "aspect_ratio",
        "symmetry",
        "walking_speed_ms",
        "cadence",
    )

    # ---- numpy 转换 ----

    def to_array(self) -> np.ndarray:
        """转为 (K,) float32 numpy 数组，直接送入 LSTM

        Returns:
            shape (5,) ndarray，dtype float32
            顺序: [torso_angle_deg, aspect_ratio, symmetry, walking_speed_ms, cadence]
        """
        return np.array([
            self.torso_angle_deg,
            self.aspect_ratio,
            self.symmetry,
            self.walking_speed_ms,
            self.cadence,
        ], dtype=np.float32)

    def to_dict(self) -> dict[str, float]:
        """转为字典"""
        return {
            "torso_angle_deg": self.torso_angle_deg,
            "aspect_ratio": self.aspect_ratio,
            "symmetry": self.symmetry,
            "walking_speed_ms": self.walking_speed_ms,
            "cadence": self.cadence,
        }

    @classmethod
    def feature_names(cls) -> tuple[str, ...]:
        """特征名称列表（固定顺序）"""
        return cls._FEATURE_NAMES

    @classmethod
    def feature_dim(cls) -> int:
        """特征向量维度"""
        return len(cls._FEATURE_NAMES)

    def __repr__(self) -> str:
        return (f"FeatureResult("
                f"torso={self.torso_angle_deg:.1f}°, "
                f"aspect={self.aspect_ratio:.3f}, "
                f"sym={self.symmetry:.3f}, "
                f"speed={self.walking_speed_ms:.3f}m/s, "
                f"cadence={self.cadence:.2f})")
