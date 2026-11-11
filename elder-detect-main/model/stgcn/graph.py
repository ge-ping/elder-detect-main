"""
model/stgcn/graph.py — 17 关节「跌倒图」定义 + ST-GCN 空间邻接矩阵

独立模块，不影响既有 LSTM 路径、feature_builder、pose/keypoint 等。

关节集 FALL_JOINTS_17 = 论文5 的 CORE_JOINTS_13（鼻/双肩肘腕/双髋膝踝）
                      + 双脚跟(29/30) + 双脚尖(31/32)
骨骼边 FALL_EDGES：肩线、双臂链、脊柱、髋线、双腿链(髋→膝→踝→脚跟→脚尖)、
                  鼻连双肩（支撑头部姿态）。

build_spatial_graph() 返回 ST-GCN 空间分区邻接 A ∈ (3, V, V)：
  0) 自环（自身）
  1) 向心（邻节点 → 本节点，聚合）
  2) 离心（本节点 → 邻节点，传播）
均已行归一化，保证梯度稳定。

坐标提取：
  skeleton_from_arrays()    — 从缓存 (T,33,3) 世界坐标直接算（训练用，最快）
  skeleton_from_poseresult()— 从 PoseResult.world_landmarks 算（推理用，与训练一致）
两者都走「髋中心归一化到 [-1,1]」(clone_with_3d_normalized 同款数学)，
保证 train/infer 零漂移。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from pose.keypoint import BodyPart


# ---- 17 关节「跌倒图」关节集（顺序即图节点索引 0..16）----
FALL_JOINTS_17: list[BodyPart] = [
    BodyPart.NOSE,           # 0
    BodyPart.LEFT_SHOULDER,  # 1
    BodyPart.RIGHT_SHOULDER, # 2
    BodyPart.LEFT_ELBOW,     # 3
    BodyPart.RIGHT_ELBOW,    # 4
    BodyPart.LEFT_WRIST,     # 5
    BodyPart.RIGHT_WRIST,    # 6
    BodyPart.LEFT_HIP,       # 7
    BodyPart.RIGHT_HIP,      # 8
    BodyPart.LEFT_KNEE,      # 9
    BodyPart.RIGHT_KNEE,     # 10
    BodyPart.LEFT_ANKLE,     # 11
    BodyPart.RIGHT_ANKLE,    # 12
    BodyPart.LEFT_HEEL,      # 13
    BodyPart.RIGHT_HEEL,     # 14
    BodyPart.LEFT_FOOT_INDEX,# 15
    BodyPart.RIGHT_FOOT_INDEX,#16
]
FALL_JOINT_SET_NAME = "fall17"
NUM_FALL_JOINTS = len(FALL_JOINTS_17)  # 17

# ---- 骨骼边（以 FALL_JOINTS_17 的索引表示）----
FALL_EDGES: list[tuple[int, int]] = [
    # 肩线
    (1, 2),
    # 左臂链：肩→肘→腕
    (1, 3), (3, 5),
    # 右臂链
    (2, 4), (4, 6),
    # 脊柱：双肩→双髋
    (1, 7), (2, 8),
    # 髋线
    (7, 8),
    # 左腿链：髋→膝→踝→脚跟，踝→脚尖
    (7, 9), (9, 11), (11, 13), (11, 15),
    # 右腿链
    (8, 10), (10, 12), (12, 14), (12, 16),
    # 头部：鼻连双肩
    (0, 1), (0, 2),
]


# ==================================================================
# COCO-13 关节集（YOLO-pose 2D，对应 coco_joints.COCO_CORE_INDICES）
# 顺序即图节点索引 0..12（比 FALL_JOINTS_17 少了脚跟/脚尖，肢体末端到踝）
# ==================================================================
COCO13_JOINTS: list[BodyPart] = [
    BodyPart.NOSE,            # 0
    BodyPart.LEFT_SHOULDER,  # 1
    BodyPart.RIGHT_SHOULDER, # 2
    BodyPart.LEFT_ELBOW,     # 3
    BodyPart.RIGHT_ELBOW,    # 4
    BodyPart.LEFT_WRIST,     # 5
    BodyPart.RIGHT_WRIST,    # 6
    BodyPart.LEFT_HIP,       # 7
    BodyPart.RIGHT_HIP,      # 8
    BodyPart.LEFT_KNEE,      # 9
    BodyPart.RIGHT_KNEE,     # 10
    BodyPart.LEFT_ANKLE,     # 11
    BodyPart.RIGHT_ANKLE,    # 12
]
COCO13_JOINT_SET_NAME = "coco13"
NUM_COCO13_JOINTS = len(COCO13_JOINTS)  # 13

# 骨骼边（COCO-13 索引）：肩线 / 双臂 / 脊柱+髋线 / 双腿 / 鼻连双肩
COCO13_EDGES: list[tuple[int, int]] = [
    (1, 2),                       # 肩线
    (1, 3), (3, 5),               # 左臂链
    (2, 4), (4, 6),               # 右臂链
    (1, 7), (2, 8), (7, 8),       # 脊柱 + 髋线
    (7, 9), (9, 11),              # 左腿链（髋→膝→踝）
    (8, 10), (10, 12),            # 右腿链
    (0, 1), (0, 2),               # 头部：鼻连双肩
]


def select_joint_set(num_joints: int, joint_set: Optional[str] = None):
    """按关节数或显式命名返回 (joints, edges, name)。

    用于 ST-GCN 在 fall17(17关节3D) 与 coco13(13关节2D) 间自动切换图结构，
    保证邻接矩阵 V 维与数据 V 维一致（einsum 不越界）。

    - joint_set 显式给定（"fall17"/"coco13"）优先；
    - 否则按 num_joints 推断（13→coco13，17→fall17）。
    """
    if joint_set == "coco13" or (joint_set is None and num_joints == 13):
        return COCO13_JOINTS, COCO13_EDGES, COCO13_JOINT_SET_NAME
    if joint_set == "fall17" or (joint_set is None and num_joints == 17):
        return FALL_JOINTS_17, FALL_EDGES, FALL_JOINT_SET_NAME
    raise ValueError(
        f"不支持的关节集: joint_set={joint_set!r}, num_joints={num_joints} "
        f"(仅支持 fall17=17 关节 / coco13=13 关节)")


# ==================================================================
# 邻接矩阵 / 空间分区
# ==================================================================

def _hop_distance(num_node: int, edges: list[tuple[int, int]], max_hop: int = 1) -> np.ndarray:
    """BFS 计算每对节点的跳数距离（max_hop 截止）。"""
    A = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    dist = np.full((num_node, num_node), np.inf, dtype=np.float32)
    for src in range(num_node):
        dist[src, src] = 0.0
        visited = {src}
        queue = [src]
        d = 0
        while queue and d < max_hop:
            nxt = []
            d += 1
            for u in queue:
                for v in range(num_node):
                    if A[u, v] > 0 and v not in visited:
                        visited.add(v)
                        dist[src, v] = d
                        nxt.append(v)
            queue = nxt
    return dist


def build_spatial_graph(joints: list = FALL_JOINTS_17,
                        edges: list[tuple[int, int]] = FALL_EDGES,
                        max_hop: int = 1) -> np.ndarray:
    """构建 ST-GCN 空间分区邻接 A ∈ (3, V, V)

    0) 自环（自身）
    1) 向心（邻节点 → 本节点，聚合）
    2) 离心（本节点 → 邻节点，传播）

    每个分区均按行归一化（聚合权重和=1），保证梯度稳定。
    """
    V = len(joints)
    # 无向邻接
    A = np.zeros((V, V), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # 行归一化（每个节点把权重均分给其邻居）
    D = A.sum(axis=1)
    Dn = np.where(D > 0, 1.0 / D, 0.0)
    A_norm = (A * Dn[:, None]).astype(np.float32)

    self_link = np.eye(V, dtype=np.float32)
    # 离心分区独立行归一化（节点→邻居的权重和=1，保证聚合稳定）
    A_out = A_norm.T.copy()
    Do = A_out.sum(axis=1)
    Do_n = np.where(Do > 0, 1.0 / Do, 0.0)
    A_out = (A_out * Do_n[:, None]).astype(np.float32)
    A_part = np.stack([
        self_link,     # 0: self
        A_norm,        # 1: inward (neighbor -> node)
        A_out,         # 2: outward (node -> neighbor)
    ]).astype(np.float32)
    return A_part


# ==================================================================
# 坐标提取：髋中心归一化到 [-1,1]
# ==================================================================

def _index_of(joint_set: list[BodyPart]) -> list[int]:
    return [p.value for p in joint_set]


def _fill_invalid(coords: np.ndarray, valid: np.ndarray,
                  edges: list[tuple[int, int]]) -> None:
    """把 visibility<0.6 的关节用其邻接有效关节坐标均值填充；全无效则置 0。"""
    V = coords.shape[0]
    # 邻接表
    adj = [[] for _ in range(V)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    for i in range(V):
        if valid[i]:
            continue
        neigh = [j for j in adj[i] if valid[j]]
        if neigh:
            coords[i] = coords[neigh].mean(axis=0)
        else:
            coords[i] = 0.0
        valid[i] = True  # 填充后视为有效，避免后续 NaN


def skeleton_from_arrays(world: np.ndarray, vis: np.ndarray,
                         joint_set: list[BodyPart] = FALL_JOINTS_17,
                         edges: list[tuple[int, int]] = FALL_EDGES) -> np.ndarray:
    """从缓存数组直接算每帧 17 关节 3D 归一化坐标。

    Args:
        world: (T, 33, 3) 世界坐标（米）
        vis:   (T, 33)    可见性
        joint_set / edges: 关节集与边

    Returns:
        (T, V, 3) float32，坐标经髋中心归一化到 ~[-1,1]
    """
    world = np.asarray(world, dtype=np.float32)
    vis = np.asarray(vis, dtype=np.float32)
    T = world.shape[0]
    # 髋中心（世界坐标）
    hip = (world[:, BodyPart.LEFT_HIP.value] + world[:, BodyPart.RIGHT_HIP.value]) / 2.0  # (T,3)
    centered = world - hip[:, None, :]                                      # (T,33,3)
    max_abs = np.max(np.abs(centered), axis=(1, 2), keepdims=True)
    max_abs = np.where(max_abs > 1e-9, max_abs, 1.0)
    norm = centered / max_abs                                              # (T,33,3) ∈ [-1,1]
    idx = _index_of(joint_set)
    coords = norm[:, idx, :].copy()                                        # (T,V,3)
    vism = vis[:, idx]                                                     # (T,V)
    valid = vism >= 0.6
    for t in range(T):
        _fill_invalid(coords[t], valid[t], edges)
    return coords.astype(np.float32)


def skeleton_from_poseresult(result,
                             joint_set: list[BodyPart] = FALL_JOINTS_17,
                             edges: list[tuple[int, int]] = FALL_EDGES) -> np.ndarray:
    """从 PoseResult（推理时用）取 17 关节 3D 归一化坐标。

    与 skeleton_from_arrays 同源数学（髋中心归一化），保证 train/infer 一致。
    """
    if result.world_landmarks is None:
        raise ValueError("PoseResult.world_landmarks is None，无法取 3D 坐标")
    norm = result.clone_with_3d_normalized()
    V = len(joint_set)
    coords = np.zeros((V, 3), dtype=np.float32)
    valid = np.zeros(V, dtype=bool)
    for i, part in enumerate(joint_set):
        kp = norm.get_world(part)
        coords[i] = [kp.x, kp.y, kp.z]
        valid[i] = kp.visibility >= 0.6
    _fill_invalid(coords, valid, edges)
    return coords.astype(np.float32)


# ==================================================================
# 2D 坐标提取（YOLO-pose，无 3D 世界坐标）
# 髋中心索引固定为 COCO13 的 7(左髋)/8(右髋)；与 skeleton_from_arrays_2d
# 同源数学，保证 train(缓存) / infer(PoseResult) 零漂移。
# ==================================================================

def _coco13_hip() -> tuple[int, int]:
    """COCO13 关节集中左右髋的节点索引。"""
    return 7, 8


def skeleton_from_arrays_2d(landmarks_2d: np.ndarray, vis: np.ndarray,
                            joints: list = COCO13_JOINTS,
                            edges: list[tuple[int, int]] = COCO13_EDGES) -> np.ndarray:
    """从 YOLO 2D 缓存直接算每帧 13 关节 2D 归一化坐标。

    Args:
        landmarks_2d: (T, V, 2) 归一化坐标 [0,1]（YOLO 输出即此）
        vis:           (T, V)    每关节置信度

    Returns:
        (T, V, 2) float32，坐标经髋中心归一化到 ~[-1,1]
    """
    lm = np.asarray(landmarks_2d, dtype=np.float32)
    vis = np.asarray(vis, dtype=np.float32)
    T = lm.shape[0]
    V = lm.shape[1]
    hi, hj = _coco13_hip()
    hip = (lm[:, hi, :] + lm[:, hj, :]) / 2.0          # (T,2)
    centered = lm - hip[:, None, :]                    # (T,V,2)
    max_abs = np.max(np.abs(centered), axis=(1, 2), keepdims=True)
    max_abs = np.where(max_abs > 1e-9, max_abs, 1.0)
    norm = centered / max_abs                          # (T,V,2) ∈ [-1,1]
    vism = vis[:, :V]
    valid = vism >= 0.6
    for t in range(T):
        _fill_invalid(norm[t], valid[t], edges)
    return norm.astype(np.float32)


def skeleton_from_poseresult_2d(result,
                                joints: list = COCO13_JOINTS,
                                edges: list[tuple[int, int]] = COCO13_EDGES) -> np.ndarray:
    """从 YOLO PoseResult（推理时用）取 13 关节 2D 归一化坐标。

    与 skeleton_from_arrays_2d 同源数学（髋中心归一化），保证 train/infer 一致。
    """
    if result.landmarks is None:
        raise ValueError("PoseResult.landmarks is None，无法取 2D 坐标")
    V = len(result.landmarks)
    coords = np.zeros((V, 2), dtype=np.float32)
    valid = np.zeros(V, dtype=bool)
    for i, kp in enumerate(result.landmarks):
        coords[i] = [kp.x, kp.y]
        valid[i] = kp.visibility >= 0.6
    _fill_invalid(coords, valid, edges)
    return coords.astype(np.float32)
