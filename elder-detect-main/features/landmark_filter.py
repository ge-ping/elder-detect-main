#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
features/landmark_filter.py — 关键点坐标时序平滑（OneEuroFilter）

目的：压低 MediaPipe 单帧坐标抖动（高频噪声），减少"坐下/蹲下/弯腰"等
日常良性动作被误判为中间风险。作用在**归一化骨架 (V,3)** 上，与训练侧
build_skeleton.py 用的 skeleton_from_arrays 输出同形状、同顺序 → 保证
训练/推理零漂移。

为什么用 OneEuro 而不是卡尔曼：
  - OneEuro 专为传感器/关键点抖动设计，静止强平滑、快速运动弱平滑（beta 自适应），
    不会让跌倒这种快速动作变迟钝。
  - 纯确定性逐帧变换（dt=1 用"帧"为单位），训练侧整段序列平滑与推理侧逐帧
    平滑数学完全一致 → 零漂移（这是接入的硬约束）。

时间单位约定：用"帧"做时间步（dt=1.0），训练侧整段 & 推理侧逐帧都用同一约定，
因此不依赖视频真实 fps，零漂移成立。

用法：
  from features.landmark_filter import LandmarkSmoother
  # 推理侧（在线、逐帧）
  sm = LandmarkSmoother(enabled=True, min_cutoff=1.0, beta=0.02, dcutoff=1.0,
                        n_landmarks=17, channels=3)
  for frame in frames:
      skel = skeleton_from_poseresult(pr, FALL_JOINTS_17)   # (17,3)
      skel = sm.update(skel)                                # 平滑后 (17,3)
      sbuf.append(skel)
  # 训练侧（离线、整段）
  sm = LandmarkSmoother(enabled=True, min_cutoff=1.0, beta=0.02, dcutoff=1.0)
  coords = sm.smooth_sequence(coords)   # (T,17,3) -> (T,17,3)
"""
from __future__ import annotations

import numpy as np


class OneEuroFilter:
    """单标量通道的 1-Euro 滤波器 (Caseras et al., 2017)。

    低通 + 自适应截止频率：静止时强平滑（min_cutoff 主导），
    快速变化时弱平滑（beta·|dx| 提高截止频率，避免迟钝）。
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, dcutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev = None
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, dt: float = 1.0):
        x = float(x)
        if self.x_prev is None:
            self.x_prev = x
            return x
        dt = max(dt, 1e-6)
        edx = (x - self.x_prev) / dt
        a_d = self._alpha(self.dcutoff, dt)
        dx_hat = a_d * edx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class LandmarkSmoother:
    """对 (V, C) 关键点坐标做 OneEuro 时序平滑，每关节每坐标独立一个滤波器。

    提供两种接口：
      - update(frame (V,C))：在线逐帧，维护内部状态（推理用）。
      - smooth_sequence(seq (T,V,C))：离线整段，内部 reset 后逐帧 update（训练用）。
    两者都用相同 dt=1/fps（统一 25fps 约定），数学完全一致 → 训练/推理零漂移。

    注意 OneEuro 的 cutoff 单位是 Hz，必须配合真实时间步。本项目统一约定
    "每帧 = 1/25 秒"（fps=25），故 cutoff=1.0 对应 ~0.16s 时间常数（约 4 帧平滑），
    强度合理；若用 dt=1（帧）则 cutoff=1 等于完全不过滤（已踩坑）。

    enabled=False 时两个接口都原样返回输入，确保下游零侵入、可一键关闭。
    """

    def __init__(self, enabled: bool = True, min_cutoff: float = 1.0,
                 beta: float = 0.0, dcutoff: float = 1.0,
                 n_landmarks: int = 17, channels: int = 3, fps: float = 25.0):
        self.enabled = enabled
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.n_landmarks = n_landmarks
        self.channels = channels
        self.fps = float(fps)
        if enabled:
            self._alloc()

    def _alloc(self):
        self._filters = [
            [OneEuroFilter(self.min_cutoff, self.beta, self.dcutoff)
             for _ in range(self.channels)]
            for _ in range(self.n_landmarks)
        ]

    def reset(self):
        if self.enabled:
            self._alloc()

    def update(self, frame_coords):
        """在线逐帧平滑一帧坐标。frame_coords: (V, C) 数组 → 平滑后 (V, C)。"""
        if not self.enabled:
            return frame_coords
        dt = 1.0 / self.fps
        arr = np.asarray(frame_coords, dtype=np.float64)
        out = np.empty_like(arr)
        for v in range(self.n_landmarks):
            for c in range(self.channels):
                out[v, c] = self._filters[v][c](arr[v, c], dt=dt)
        return out.astype(np.float32)

    def smooth_sequence(self, seq):
        """离线整段平滑。seq: (T, V, C) → 平滑后 (T, V, C)。

        自动按实际形状重建滤波器（兼容不同关节数/维度数）。
        """
        if not self.enabled:
            return seq
        arr = np.asarray(seq, dtype=np.float64)
        T, V, C = arr.shape
        if V != self.n_landmarks or C != self.channels:
            self.n_landmarks, self.channels = V, C
            self._alloc()
        out = np.empty_like(arr)
        self.reset()
        for t in range(T):
            out[t] = self.update(arr[t])
        return out.astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "min_cutoff": self.min_cutoff,
            "beta": self.beta,
            "dcutoff": self.dcutoff,
            "fps": self.fps,
        }
