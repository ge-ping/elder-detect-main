"""
features/feature_builder_2d.py — YOLO 2D 29 维特征（FrameFeatureBuilder 的 2D 等价）

不依赖 PoseResult.world_landmarks（YOLO 无 3D），直接从 (T,13,2) 归一化坐标算。
特征布局与 FrameFeatureBuilder 完全一致(29维)，仅把 3D 派生量改成 2D 近似：
    [gait_speed_2d, trunk_angle_2d, aspect_ratio, 13关节×2(x,y)]
  - gait_speed_2d   : 踝中点相邻帧 2D 位移 / dt（归一化单位/秒）
  - trunk_angle_2d  : 肩-髋向量在图像平面(x-y)的俯仰角(度)  ← 仅保留 x-y 单平面（丢了 z 相关的侧倾/扭转）
  - aspect_ratio    : 骨架包围盒 W/H（2D）
  - skeleton_26     : 13 关节 (x,y)（2D）

dim 仍为 29，LSTM 模型 input_dim=29 可直接复用，无需改 RiskRegressor。
"""
from __future__ import annotations

import numpy as np

# COCO13 关节索引（与 model/stgcn/graph.py COCO13_JOINTS 顺序一致）
NOSE, LSH, RSH, LEL, REL, LWRI, RWRI, LHIP, RHIP, LKN, RKN, LANK, RANK = range(13)


def _extract_skeleton_2d(lm2d: np.ndarray) -> np.ndarray:
    """13 关节 × (x, y) → (26,) float32"""
    return np.asarray(lm2d, dtype=np.float32).reshape(-1)


def _aspect_ratio_2d(lm2d: np.ndarray) -> float:
    xs = lm2d[:, 0]
    ys = lm2d[:, 1]
    w = float(xs.max() - xs.min())
    h = float(ys.max() - ys.min())
    if h < 1e-6:
        return 1.0
    return w / h


def _trunk_angle_2d(lm2d: np.ndarray) -> float:
    """肩-髋向量在图像平面(x右 y下)的俯仰角(度)。"""
    sx = (lm2d[LSH, 0] + lm2d[RSH, 0]) / 2.0
    sy = (lm2d[LSH, 1] + lm2d[RSH, 1]) / 2.0
    hx = (lm2d[LHIP, 0] + lm2d[RHIP, 0]) / 2.0
    hy = (lm2d[LHIP, 1] + lm2d[RHIP, 1]) / 2.0
    dx = hx - sx
    dy = hy - sy
    return float(np.degrees(np.arctan2(dy, dx)))


def build_frame_features(lm2d: np.ndarray, gait_2d: float) -> np.ndarray:
    """单帧 2D 坐标 (13,2) + 2D 步速 → (29,) float32"""
    scalars = np.array(
        [float(gait_2d), _trunk_angle_2d(lm2d), _aspect_ratio_2d(lm2d)],
        dtype=np.float32,
    )
    skel = _extract_skeleton_2d(lm2d)
    return np.concatenate([scalars, skel]).astype(np.float32)


def compute_gait_speed_2d(lm2d_seq: np.ndarray, dt: float) -> np.ndarray:
    """逐帧 2D 步速：踝中点相邻帧位移 / dt；首帧为 0。

    Args:
        lm2d_seq: (T, 13, 2) 归一化坐标
        dt:       帧间隔(秒)，如 1/25
    Returns:
        (T,) float32
    """
    lm2d_seq = np.asarray(lm2d_seq, dtype=np.float32)
    T = lm2d_seq.shape[0]
    gait = np.zeros(T, dtype=np.float32)
    if T < 2:
        return gait
    ankle_mid = (lm2d_seq[:, LANK, :] + lm2d_seq[:, RANK, :]) / 2.0   # (T,2)
    disp = np.linalg.norm(np.diff(ankle_mid, axis=0), axis=1)        # (T-1,)
    gait[1:] = disp / dt
    return gait
