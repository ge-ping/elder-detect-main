"""
时序窗口配置 (WindowConfig)

所有参数可调，无硬编码。改配置即可适配论文2/4/5/8。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pose.keypoint import CORE_JOINTS_13, BodyPart


@dataclass
class WindowConfig:
    """时序窗口配置

    Attributes:
        window_size: 滑动窗口帧数（论文5 = 35）
        stride: 窗口步长（1 = 稠密采样）
        pad_mode: 边界处理 — "discard" 丢弃 | "zero_pad" 补零（暂未实现）
        joint_set: 骨架关键点列表
        use_gait_speed: 是否拼接步速
        use_trunk_angle: 是否拼接躯干角度
        use_aspect_ratio: 是否拼接宽高比
    """

    # 窗口参数
    window_size: int = 35
    stride: int = 1
    pad_mode: str = "discard"

    # 骨架关键点集
    joint_set: list[BodyPart] = field(default_factory=lambda: list(CORE_JOINTS_13))

    # 标量特征开关
    use_gait_speed: bool = True
    use_trunk_angle: bool = True
    use_aspect_ratio: bool = True

    def __post_init__(self):
        if self.window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {self.window_size}")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")

    # ---- 派生属性 ----

    def scalar_dim(self) -> int:
        """标量特征维度"""
        d = 0
        if self.use_gait_speed:   d += 1
        if self.use_trunk_angle:  d += 1
        if self.use_aspect_ratio: d += 1
        return d

    def skeleton_dim(self) -> int:
        """骨架坐标维度 = 关节数 × 2 (x, y)"""
        return len(self.joint_set) * 2

    def feature_dim(self) -> int:
        """单帧特征总维度 K"""
        return self.scalar_dim() + self.skeleton_dim()

    @property
    def joint_names(self) -> list[str]:
        """关节名称列表（按顺序）"""
        return [j.name for j in self.joint_set]
