"""
2.2.1 躯干角度提取器 (Torso Angle Extractor)

论文5 (Lau et al., 2022): 3D躯干倾角 — 人体躯干偏离垂直轴的角度，
是三分类（Normal/Unstable/Fall）的核心判别特征。

计算流程:
    1. world_landmarks 分离出两套副本:
       - 归一化 [-1,1] (clone_with_3d_normalized) → 三轴角度
       - 原始米坐标 → 肩髋距离 d_f 判断朝向
    2. 在 x-y、y-z、x-z 三平面分别以 arctan2 差值法求角度
    3. 根据 d_f 阈值 0.15m 选择最优角度:
       d_f < 0.15m → 正面 → min(θ1,θ2,θ3)
       d_f ≥ 0.15m → 侧身 → max(θ1,θ2,θ3)

逐帧特征：每帧独立计算，不依赖前后帧。
"""
from __future__ import annotations

import numpy as np

from pose.keypoint import BodyPart
from .base import FeatureExtractor


class TorsoAngleExtractor(FeatureExtractor):
    """躯干角度提取器

    输入:
        PoseResult（需 world_landmarks 非 None）

    输出:
        {"torso_angle_deg": float}  — 躯干倾角（度），0°=直立

    三平面定义:
        x-y: 前后俯仰  V1=(0,1)竖直
        y-z: 左右侧倾  V1=(0,1)竖直
        x-z: 水平扭转  V1=(0,1)=z+正前方

    论文参考:
        - 论文5 §2.2.1 Trunk Angle
        - Lau et al., Int. J. Technology 13(6) 1173-1182 (2022)
    """

    # df 阈值（米）：肩髋 x 轴距离，区分正面/侧身
    DF_THRESHOLD_M = 0.15

    # V1 在各平面的基准向量（均为 (0, 1)，即各平面第二轴正向）
    _V1 = np.array([0.0, 1.0], dtype=np.float64)

    def __init__(self) -> None:
        super().__init__(name="TorsoAngle")

    # ==================================================================
    # 逐帧特征
    # ==================================================================

    def extract_frame(self, result) -> dict[str, float]:
        """计算单帧躯干倾角

        Args:
            result: PoseResult（需 world_landmarks 非 None，
                    且包含 LEFT/RIGHT_SHOULDER + LEFT/RIGHT_HIP）

        Returns:
            {"torso_angle_deg": float}
        """
        # ---- 1. 输入校验 ----
        if result.world_landmarks is None:
            raise ValueError(
                "world_landmarks is None. "
                "Set output_world_landmarks=True in PoseExtractorConfig."
            )

        # ---- 2. 两套副本 ----
        # 副本A: 归一化 [-1, 1] → 角度计算
        norm = result.clone_with_3d_normalized()

        # 副本B: 原始米坐标 → df 距离判断
        # （直接用 result.world_landmarks）

        # ---- 3. 计算肩髋中点 ----
        # 归一化副本（角度用）
        m1_norm = self._midpoint_3d(norm, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER)
        m2_norm = self._midpoint_3d(norm, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)

        # 原始米坐标（df 用 — 只需 x 分量）
        m1_raw_x = self._midpoint_x_raw(result, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER)
        m2_raw_x = self._midpoint_x_raw(result, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)

        # ---- 4. 躯干向量 ----
        torso = m1_norm - m2_norm  # ndarray (3,), 归一化坐标系

        # ---- 5. 三平面角度 ----
        theta1 = self._plane_angle(0, 1, torso)   # x-y: 前后俯仰
        theta2 = self._plane_angle(2, 1, torso)   # y-z: 左右侧倾
        theta3 = self._plane_angle(0, 2, torso)   # x-z: 水平扭转

        # ---- 6. df 判断 + 最优选择 ----
        df = abs(m1_raw_x - m2_raw_x)  # 米

        if df < self.DF_THRESHOLD_M:
            angle_deg = min(theta1, theta2, theta3)
        else:
            angle_deg = max(theta1, theta2, theta3)

        return {"torso_angle_deg": float(angle_deg)}

    # ==================================================================
    # 内部方法
    # ==================================================================

    @staticmethod
    def _midpoint_3d(norm_result, part_a: BodyPart, part_b: BodyPart) -> np.ndarray:
        """归一化副本中取两关键点 3D 中点 → (3,) ndarray"""
        a = norm_result.get_world(part_a)
        b = norm_result.get_world(part_b)
        return np.array([
            (a.x + b.x) / 2,
            (a.y + b.y) / 2,
            (a.z + b.z) / 2,
        ], dtype=np.float64)

    @staticmethod
    def _midpoint_x_raw(raw_result, part_a: BodyPart, part_b: BodyPart) -> float:
        """原始米坐标中取两关键点 x 分量中点 → float（米）"""
        a = raw_result.get_world(part_a)
        b = raw_result.get_world(part_b)
        return (a.x + b.x) / 2.0

    @classmethod
    def _plane_angle(cls, idx_a: int, idx_b: int, torso: np.ndarray) -> float:
        """计算指定平面上的躯干倾斜角

        Args:
            idx_a: 平面第一轴在 torso 向量中的索引 (0=x, 1=y, 2=z)
            idx_b: 平面第二轴在 torso 向量中的索引 (0=x, 1=y, 2=z)
            torso: 躯干向量 (3,) 归一化坐标

        Returns:
            角度（度），0~180

        公式 (论文5):
            radians = |arctan2(V2.b, V2.a) - arctan2(V1.b, V1.a)|
            angle = radians * 180 / π
            其中 V1 = (0, 1)，V2 = (torso[idx_a], torso[idx_b])
        """
        v2 = np.array([torso[idx_a], torso[idx_b]], dtype=np.float64)
        mag = np.linalg.norm(v2)
        if mag < 1e-9:
            return 0.0  # 躯干在此平面无投影

        rad = abs(
            np.arctan2(v2[1], v2[0]) - np.arctan2(cls._V1[1], cls._V1[0])
        )
        deg = float(np.degrees(rad))
        return min(deg, 180.0)  # 钳制上界
