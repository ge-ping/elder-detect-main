"""
逐帧运动特征: velocity / acceleration

  v_t = p_t - p_{t-1}
  a_t = v_t - v_{t-1} = p_t - 2·p_{t-1} + p_{t-2}
"""
import numpy as np


def add_velocity(
    features: np.ndarray,
    skeleton_start: int = 3,
    num_joints: int = 13,
) -> np.ndarray:
    """追加逐关节速度 → (T, K + 26)"""
    T = features.shape[0]
    sk_end = skeleton_start + num_joints * 2
    skeleton = features[:, skeleton_start:sk_end]

    velocity = np.zeros_like(skeleton)
    velocity[1:] = skeleton[1:] - skeleton[:-1]

    return np.concatenate([features, velocity], axis=1).astype(np.float32)


def add_acceleration(
    features: np.ndarray,
    skeleton_start: int = 3,
    num_joints: int = 13,
) -> np.ndarray:
    """追加逐关节加速度 → (T, K + 26)

    a_t = p_t - 2·p_{t-1} + p_{t-2}, 首两帧补 0
    """
    T = features.shape[0]
    sk_end = skeleton_start + num_joints * 2
    skeleton = features[:, skeleton_start:sk_end]

    accel = np.zeros_like(skeleton)
    if T > 2:
        accel[2:] = skeleton[2:] - 2 * skeleton[1:-1] + skeleton[:-2]

    return np.concatenate([features, accel], axis=1).astype(np.float32)
