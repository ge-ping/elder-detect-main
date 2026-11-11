"""
2.2.4 骨架点特征提取器 (Skeleton Point Feature Extractor)

论文5: 人体宽高比 (§2.2.3) + 骨骼时序 — 从 2D landmarks 提取人体几何特征。

特征:
    ① 宽高比 (aspect_ratio) = bbox_w / bbox_h
       - R_f < 1: 竖矩形 → 直立/行走
       - R_f ≈ 1: 方形 → 坐姿
       - R_f ≥ 1: 横矩形 → 跌倒/仰卧 (论文原文阈值笔误, 此处纠正)
    ② 左右对称性 (symmetry) — 左右关键点到中轴距离的归一化偏差

逐帧特征：每帧独立计算。
"""
from __future__ import annotations

import numpy as np

from .base import FeatureExtractor
from pose.keypoint import BodyPart


# 左右配对关键点（用于对称性计算）
_SYMMETRY_PAIRS: list[tuple[BodyPart, BodyPart]] = [
    (BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER),
    (BodyPart.LEFT_ELBOW, BodyPart.RIGHT_ELBOW),
    (BodyPart.LEFT_WRIST, BodyPart.RIGHT_WRIST),
    (BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP),
    (BodyPart.LEFT_KNEE, BodyPart.RIGHT_KNEE),
    (BodyPart.LEFT_ANKLE, BodyPart.RIGHT_ANKLE),
]


class SkeletonFeatureExtractor(FeatureExtractor):
    """骨架点几何特征提取器

    从 2D landmarks（[0, 1] 归一化坐标）提取人体几何特征。

    输入:
        PoseResult.landmarks (list[Keypoint], x,y ∈ [0,1])
        至少需要 CORE_JOINTS_13

    输出:
        {
            "aspect_ratio": float,   # 人体外接矩形 W/H
            "symmetry": float,       # 左右对称性 [0, 1]
        }

    论文参考:
        - 论文5 §2.2.3 Width-Height Ratio: R_f = W/H, 跌倒时形状剧变
        - 论文4 §头髋高度比: 失衡预判
        - 论文2 §RP归一化: 髋中心相对归一化
    """

    def __init__(self, use_rp_normalize: bool = True) -> None:
        """
        Args:
            use_rp_normalize: True=先以髋中心为原点做 RP 归一化（论文2），
                             对称性计算更稳健；宽高比不受影响
        """
        super().__init__(name="SkeletonFeature")
        self.use_rp_normalize = use_rp_normalize

    # ==================================================================
    # 逐帧特征
    # ==================================================================

    def extract_frame(self, result) -> dict[str, float]:
        """从单帧提取骨架点几何特征

        Args:
            result: PoseResult（使用 landmarks, x,y ∈ [0,1]）

        Returns:
            {
                "aspect_ratio": float,  # 人体宽高比 W/H
                "symmetry": float,      # 左右对称性 [0, 1]
            }
        """
        # ---- 提取所有关键点 2D 坐标 ----
        points = self._extract_points(result)

        if len(points) < 3:
            return {"aspect_ratio": 0.0, "symmetry": 0.0}

        # ---- RP 归一化（可选） ----
        if self.use_rp_normalize:
            hip_center = self._hip_center_2d(result)
            if hip_center is not None:
                points = {name: (x - hip_center[0], y - hip_center[1])
                          for name, (x, y) in points.items()}

        # ---- 1. 宽高比 ----
        xs = [p[0] for p in points.values()]
        ys = [p[1] for p in points.values()]
        bbox_w = max(xs) - min(xs)
        bbox_h = max(ys) - min(ys)
        aspect_ratio = bbox_w / bbox_h if bbox_h > 1e-9 else 0.0

        # ---- 2. 左右对称性 ----
        symmetry = self._compute_symmetry(points)

        return {"aspect_ratio": float(aspect_ratio), "symmetry": float(symmetry)}

    # ==================================================================
    # 内部方法
    # ==================================================================

    @staticmethod
    def _extract_points(result) -> dict[BodyPart, tuple[float, float]]:
        """从 PoseResult 提取所有 landmarks → {BodyPart: (x, y)}"""
        return {kp.name: (kp.x, kp.y) for kp in result.landmarks}

    @staticmethod
    def _hip_center_2d(result) -> np.ndarray | None:
        """髋中心 2D 坐标（论文2 RP 归一化原点）"""
        try:
            lh = result.get(BodyPart.LEFT_HIP)
            rh = result.get(BodyPart.RIGHT_HIP)
            return np.array([(lh.x + rh.x) / 2, (lh.y + rh.y) / 2])
        except KeyError:
            return None

    @staticmethod
    def _compute_symmetry(
        points: dict[BodyPart, tuple[float, float]],
    ) -> float:
        """计算左右对称性 [0, 1]

        方法:
            1. 从 points 中取髋肩四点 x 坐标均值作为躯干中轴线
            2. 对每对左右关键点，计算各自到中轴的距离差
            3. symmetry = 1 - mean_diff / max(1e-6, mean_distance)

        注意: 中线从 points（可能已 RP 归一化）计算，保证与传入坐标同系。
        """
        # 躯干中轴线 x 坐标 — 从 points 计算（与传入坐标同系）
        try:
            ls_x = points[BodyPart.LEFT_SHOULDER][0]
            rs_x = points[BodyPart.RIGHT_SHOULDER][0]
            lh_x = points[BodyPart.LEFT_HIP][0]
            rh_x = points[BodyPart.RIGHT_HIP][0]
            midline_x = (ls_x + rs_x + lh_x + rh_x) / 4.0
        except KeyError:
            return 1.0

        diffs = []
        distances = []
        for left_part, right_part in _SYMMETRY_PAIRS:
            try:
                left_pt = points[left_part]
                right_pt = points[right_part]
            except KeyError:
                continue
            d_left = abs(left_pt[0] - midline_x)
            d_right = abs(right_pt[0] - midline_x)
            diffs.append(abs(d_left - d_right))
            distances.append(max(d_left, d_right))

        if not diffs:
            return 1.0

        mean_diff = np.mean(diffs)
        mean_dist = np.mean(distances)
        norm = max(mean_diff / max(mean_dist, 1e-9), 0.0)

        return float(1.0 - min(norm, 1.0))
