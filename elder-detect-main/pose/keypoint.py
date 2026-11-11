"""
MediaPipe Pose 关键点数据模型

33个人体关键点定义、Keypoint/PoseResult 数据类。
对齐论文5（注意力GRU三分类）的13核心关节筛选方案。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class BodyPart(Enum):
    """MediaPipe Pose 33个人体关键点枚举（0-indexed）"""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# ---- 预定义关键点分组 ----

# 论文5：13个核心躯干+四肢关节（去除面部/手部细节）
CORE_JOINTS_13: list[BodyPart] = [
    BodyPart.NOSE,
    BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER,
    BodyPart.LEFT_ELBOW, BodyPart.RIGHT_ELBOW,
    BodyPart.LEFT_WRIST, BodyPart.RIGHT_WRIST,
    BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP,
    BodyPart.LEFT_KNEE, BodyPart.RIGHT_KNEE,
    BodyPart.LEFT_ANKLE, BodyPart.RIGHT_ANKLE,
]

# 论文4：仅3个关键点（鼻、左肩、左髋），适配低算力嵌入式
MINIMAL_JOINTS_3: list[BodyPart] = [
    BodyPart.NOSE,
    BodyPart.LEFT_SHOULDER,
    BodyPart.LEFT_HIP,
]

# 全部33个关键点
ALL_JOINTS_33: list[BodyPart] = list(BodyPart)

# 字符串 → 列表映射（供配置加载使用）
JOINT_SET_MAP: dict[str, Optional[list[BodyPart]]] = {
    "all_33": ALL_JOINTS_33,
    "core_13": CORE_JOINTS_13,
    "minimal_3": MINIMAL_JOINTS_3,
}


@dataclass
class Keypoint:
    """单个关键点

    Attributes:
        x: MediaPipe 归一化 x [0, 1]（相对于图像宽度）
        y: MediaPipe 归一化 y [0, 1]（相对于图像高度）
        z: 深度值，相对于髋部中心（MediaPipe 世界坐标系为米）
        visibility: 置信度 0~1
        name: 对应的 BodyPart 枚举
    """
    x: float
    y: float
    z: float
    visibility: float
    name: BodyPart

    @property
    def confidence(self) -> float:
        return self.visibility

    @property
    def is_valid(self) -> bool:
        """置信度是否通过最低阈值（论文1：60%）"""
        return self.visibility >= 0.6

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.visibility)


@dataclass
class PoseResult:
    """单帧姿态提取结果

    Attributes:
        landmarks: 筛选后的关键点列表（N = 33/13/3 取决于配置），x,y ∈ [0,1]
        world_landmarks: 世界坐标系3D关键点（可选，论文1的3D重心计算用），单位：米
        timestamp_ms: 帧时间戳（毫秒）
        frame_width: 原始帧宽（像素），用于 x * frame_width 转为像素坐标
        frame_height: 原始帧高（像素），用于 y * frame_height 转为像素坐标
    """
    landmarks: list[Keypoint]
    world_landmarks: Optional[list[Keypoint]] = None
    timestamp_ms: int = 0
    frame_width: int = 0
    frame_height: int = 0

    # ---- 便捷属性 ----

    @property
    def num_keypoints(self) -> int:
        """实际关键点数量"""
        return len(self.landmarks)

    @property
    def array(self) -> np.ndarray:
        """(N, 4) numpy 数组 [x, y, z, visibility]"""
        return np.array([[kp.x, kp.y, kp.z, kp.visibility] for kp in self.landmarks],
                        dtype=np.float32)

    @property
    def array_2d(self) -> np.ndarray:
        """(N, 2) 仅 2D 像素坐标 [x, y]，给论文2那种只需2D的场景"""
        return np.array([[kp.x, kp.y] for kp in self.landmarks], dtype=np.float32)

    @property
    def array_3d(self) -> np.ndarray:
        """(N, 3) 仅 3D 坐标 [x, y, z]"""
        return np.array([[kp.x, kp.y, kp.z] for kp in self.landmarks], dtype=np.float32)

    @property
    def confidence_array(self) -> np.ndarray:
        """(N,) 置信度数组"""
        return np.array([kp.visibility for kp in self.landmarks], dtype=np.float32)

    @property
    def mean_confidence(self) -> float:
        """平均置信度"""
        return float(np.mean(self.confidence_array))

    @property
    def valid_mask(self) -> np.ndarray:
        """置信度 >= 0.6 的布尔掩码"""
        return self.confidence_array >= 0.6

    # ---- 便捷方法 ----

    def get(self, part: BodyPart) -> Keypoint:
        """按部位名从 landmarks 中获取单个关键点"""
        for kp in self.landmarks:
            if kp.name == part:
                return kp
        raise KeyError(f"Keypoint {part.name} not in landmarks. "
                       f"Available: {[kp.name.name for kp in self.landmarks]}")

    def get_world(self, part: BodyPart) -> Keypoint:
        """按部位名从 world_landmarks 中获取单个 3D 世界坐标关键点

        Args:
            part: BodyPart 枚举值

        Returns:
            对应的 Keypoint（坐标单位为米）

        Raises:
            ValueError: world_landmarks 为 None
            KeyError: 关键点不在 world_landmarks 中
        """
        if self.world_landmarks is None:
            raise ValueError(
                "world_landmarks is None. "
                "Set output_world_landmarks=True in PoseExtractorConfig."
            )
        for kp in self.world_landmarks:
            if kp.name == part:
                return kp
        raise KeyError(f"Keypoint {part.name} not in world_landmarks. "
                       f"Available: {[kp.name.name for kp in self.world_landmarks]}")

    def to_dict(self) -> dict[str, list[float]]:
        """转为字典，key=BodyPart名称，value=[x, y, z, visibility]"""
        return {kp.name.name: [kp.x, kp.y, kp.z, kp.visibility]
                for kp in self.landmarks}

    def clone_with_normalized_coords(self) -> "PoseResult":
        """返回坐标副本（已为 MediaPipe 原生 [0, 1] 归一化）

        自 v0.2 起，landmarks.x/y 默认为 MediaPipe 原生 [0, 1] 归一化坐标，
        不再做反归一化。此方法返回一份深拷贝，坐标值不变。

        .. note::
           如需像素坐标，手动乘以 frame_width/frame_height。
           如需 3D 归一化到 [-1, 1]，使用 :meth:`clone_with_3d_normalized`。

        Returns:
            新的 PoseResult，坐标与原始一致
        """
        # 坐标已是 [0, 1] 归一化，直接深拷贝
        new_landmarks = [
            Keypoint(
                x=kp.x, y=kp.y, z=kp.z,
                visibility=kp.visibility, name=kp.name,
            )
            for kp in self.landmarks
        ]
        new_world = None
        if self.world_landmarks:
            new_world = [
                Keypoint(
                    x=kp.x, y=kp.y, z=kp.z,
                    visibility=kp.visibility, name=kp.name,
                )
                for kp in self.world_landmarks
            ]
        return PoseResult(
            landmarks=new_landmarks,
            world_landmarks=new_world,
            timestamp_ms=self.timestamp_ms,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
        )

    def clone_with_3d_normalized(self) -> "PoseResult":
        """对 world_landmarks 做 [-1, 1] 归一化，原点移到髋部中心

        对齐论文1：
        "Each axis key points are normalized, which range from -1 to 1,
        and the origin of the 3D space starts from the hip center (0, 0, 0)."

        **仅处理 world_landmarks**，landmarks（像素坐标）保持不变。

        处理步骤:
            1. 平移: 将左髋+右髋中点移至原点 (0, 0, 0)
            2. 缩放: 所有坐标除以 max(|x|, |y|, |z|)，使值域落入 [-1, 1]

        Returns:
            新的 PoseResult:
            - ``landmarks`` 不变（像素坐标）
            - ``world_landmarks`` 归一化到 [-1, 1]，原点在髋部中心

        Raises:
            ValueError: world_landmarks 为 None
            KeyError: LEFT_HIP 或 RIGHT_HIP 不在 world_landmarks 中
        """
        if self.world_landmarks is None:
            raise ValueError(
                "world_landmarks is None. "
                "Set output_world_landmarks=True in PoseExtractorConfig."
            )

        # 1. 计算髋部中心（世界坐标，单位：米）
        left_hip = self.get_world(BodyPart.LEFT_HIP)
        right_hip = self.get_world(BodyPart.RIGHT_HIP)
        hip_center = np.array([
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2,
            (left_hip.z + right_hip.z) / 2,
        ])

        # 2. 平移所有 world_landmarks，使髋部中心为原点
        centered = np.array([
            [kp.x - hip_center[0],
             kp.y - hip_center[1],
             kp.z - hip_center[2]]
            for kp in self.world_landmarks
        ])

        # 3. 统一缩放：各轴除以 max(|x|, |y|, |z|)，使所有值 ∈ [-1, 1]
        max_abs = float(np.max(np.abs(centered)))
        scale = max_abs if max_abs > 1e-9 else 1.0

        new_world = [
            Keypoint(
                x=float(centered[i, 0] / scale),
                y=float(centered[i, 1] / scale),
                z=float(centered[i, 2] / scale),
                visibility=kp.visibility,
                name=kp.name,
            )
            for i, kp in enumerate(self.world_landmarks)
        ]

        return PoseResult(
            landmarks=self.landmarks,  # 像素坐标不变
            world_landmarks=new_world,
            timestamp_ms=self.timestamp_ms,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
        )

    def __repr__(self) -> str:
        return (f"PoseResult(keypoints={self.num_keypoints}, "
                f"ts={self.timestamp_ms}ms, "
                f"confidence={self.mean_confidence:.3f}, "
                f"size={self.frame_width}x{self.frame_height})")
