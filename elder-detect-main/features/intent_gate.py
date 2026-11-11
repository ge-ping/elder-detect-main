"""
features/intent_gate.py — 干涉系数门控（I 门控）

I(t) = 预测区域 Zone(t) 落在环境掩码 M(t) 内的像素占比 ∈ [0,1]。

用"占比"而非 IoU：预测区域远小于掩码时，IoU 会因分母（并集）大而爆跳，
占比（交集 / Zone 面积）更稳。无重叠 → I = 0。

设计见 docs/环境交互风险预测_设计.md §5 通道3。

输入：
  - zone_params: MotionIntentProjector.compute_per_frame 输出 dict
                 （含 zone_cx/cy/long_axis/short_axis/angle）
  - mask: EnvMaskBuilder.build_per_frame 输出 (mask_h, mask_w) 二值掩码
输出：
  - I(t) ∈ [0,1]
"""
from __future__ import annotations

import numpy as np

from scripts.build_config import ENV_MASK_H, ENV_MASK_W


class IntentGate:
    """干涉系数门控

    Args:
        mask_h, mask_w: 与 ``EnvMaskBuilder`` 一致的掩码尺寸（用于生成 Zone 椭圆掩码）
    """

    def __init__(self, mask_h: int = ENV_MASK_H, mask_w: int = ENV_MASK_W) -> None:
        self.mask_h = mask_h
        self.mask_w = mask_w

    def compute(self, zone_params: dict, mask: np.ndarray) -> float:
        """计算干涉系数 I(t)。

        Args:
            zone_params: ``MotionIntentProjector.compute_per_frame`` 的输出 dict
            mask: ``EnvMaskBuilder.build_per_frame`` 的输出 (mask_h, mask_w) 二值掩码

        Returns:
            I(t) ∈ [0,1]，预测区域落在掩码内的像素占比；无重叠或 Zone 退化 → 0.0
        """
        zone = self._zone_ellipse_mask(zone_params)
        zone_area = int(zone.sum())
        if zone_area == 0:
            return 0.0
        # 占比 = Zone 与 M 的交集 / Zone 面积
        overlap = int((zone & (mask > 0)).sum())
        return overlap / zone_area

    # ==================================================================
    # 内部
    # ==================================================================

    def _zone_ellipse_mask(self, zone_params: dict) -> np.ndarray:
        """根据椭圆参数生成 Zone 二值掩码。

        椭圆方程在旋转后的主轴坐标系下：xr²/a² + yr²/b² ≤ 1。
        """
        cx = zone_params["zone_cx"] * self.mask_w
        cy = zone_params["zone_cy"] * self.mask_h
        a = max(zone_params["zone_long_axis"] * self.mask_w, 1.0)   # 长轴像素
        b = max(zone_params["zone_short_axis"] * self.mask_w, 1.0)  # 短轴像素
        angle = zone_params["zone_angle"]

        yy, xx = np.ogrid[:self.mask_h, :self.mask_w]
        ca, sa = np.cos(angle), np.sin(angle)
        # 旋转坐标到椭圆主轴系
        xr = (xx - cx) * ca + (yy - cy) * sa
        yr = -(xx - cx) * sa + (yy - cy) * ca
        return (xr * xr) / (a * a) + (yr * yr) / (b * b) <= 1.0
