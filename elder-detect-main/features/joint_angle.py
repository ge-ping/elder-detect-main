"""
关节角度特征 — 7 个关节点夹角

基于 world_landmarks (3D, 米) 计算，比 2D 更准确（不受视角影响）。
"""
from __future__ import annotations

import numpy as np

from pose.keypoint import BodyPart


def _angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """计算 ∠ABC，即 BA 和 BC 的夹角（度）

    a, b, c: (3,) ndarray — 3D 坐标
    返回: 角度 (0~180°)
    """
    ba = a - b
    bc = c - b
    dot = np.dot(ba, bc)
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm < 1e-9:
        return 0.0
    cos = np.clip(dot / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def extract_joint_angles(result) -> np.ndarray:
    """从单帧 PoseResult 提取 7 个关节角度 → (7,) float32

    角度列表:
        [0] L_ELBOW:    ∠LEFT_SHOULDER-LEFT_ELBOW-LEFT_WRIST
        [1] R_ELBOW:    ∠RIGHT_SHOULDER-RIGHT_ELBOW-RIGHT_WRIST
        [2] L_KNEE:     ∠LEFT_HIP-LEFT_KNEE-LEFT_ANKLE
        [3] R_KNEE:     ∠RIGHT_HIP-RIGHT_KNEE-RIGHT_ANKLE
        [4] L_HIP:      ∠LEFT_SHOULDER-LEFT_HIP-LEFT_KNEE
        [5] R_HIP:      ∠RIGHT_SHOULDER-RIGHT_HIP-RIGHT_KNEE
        [6] HEAD_TRUNK: ∠NOSE-SHOULDER_MID-HIP_MID
    """
    angles = np.zeros(7, dtype=np.float32)

    try:
        # 获取所有需要的 3D 坐标
        def p3d(part):
            kp = result.get_world(part)
            return np.array([kp.x, kp.y, kp.z], dtype=np.float64)

        ls = p3d(BodyPart.LEFT_SHOULDER)
        le = p3d(BodyPart.LEFT_ELBOW)
        lw = p3d(BodyPart.LEFT_WRIST)
        rs = p3d(BodyPart.RIGHT_SHOULDER)
        re = p3d(BodyPart.RIGHT_ELBOW)
        rw = p3d(BodyPart.RIGHT_WRIST)
        lh = p3d(BodyPart.LEFT_HIP)
        lk = p3d(BodyPart.LEFT_KNEE)
        la = p3d(BodyPart.LEFT_ANKLE)
        rh = p3d(BodyPart.RIGHT_HIP)
        rk = p3d(BodyPart.RIGHT_KNEE)
        ra = p3d(BodyPart.RIGHT_ANKLE)
        nose = p3d(BodyPart.NOSE)

        shoulder_mid = (ls + rs) / 2
        hip_mid = (lh + rh) / 2

        angles[0] = _angle_3d(ls, le, lw)   # L_ELBOW
        angles[1] = _angle_3d(rs, re, rw)   # R_ELBOW
        angles[2] = _angle_3d(lh, lk, la)   # L_KNEE
        angles[3] = _angle_3d(rh, rk, ra)   # R_KNEE
        angles[4] = _angle_3d(ls, lh, lk)   # L_HIP
        angles[5] = _angle_3d(rs, rh, rk)   # R_HIP
        angles[6] = _angle_3d(nose, shoulder_mid, hip_mid)  # HEAD_TRUNK
    except (KeyError, ValueError):
        pass  # 缺失关键点时返回全零

    return angles
