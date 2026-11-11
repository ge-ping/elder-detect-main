#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demo_gate.py — 实时跌倒风险 + 环境交互门控（边推理边显示）

基于 demo.py 的流式框架 + 环境掩码 + I 门控。
每帧：MediaPipe 提 pose → gait+feature → deque 窗口 B 推理 → 环境掩码 + Zone + I + E_eff
→ fuse 阈值 → cv2.imshow 实时显示。不用等全帧，真·实时。

环境掩码三种模式：
  absdiff (推荐·零依赖)：一张"干净空房基准图"做帧差 → 新增障碍物。不用 MOG2 / 不用 VLM。
      用法：--bg-image 干净基准图.jpg  （基准图可用 scripts/capture_baseline.py 生成）
            --bg-image 不给时，自动扫描视频前段无人帧中值合成（兜底）。
  bg_diff (旧)：MOG2 背景差分 + VLM 静态基线。
  vlm_coarse (旧)：依赖 VLM 标签。

用法（极简，只给视频即可）:
  python demo/demo_gate.py "你的视频.avi"

模型 / 基线 / GPU 全部自动探测，无需手动指定；进阶参数见下方 --help。
默认「跑完即保存」：标注视频 + 逐帧数据 CSV + 运行元数据，自动存到
demo/runs/<视频名>_<时间戳>/（不覆盖旧结果）。
  python demo/demo_gate.py "你的视频.avi" --no-save   # 只看不存
  python demo/demo_gate.py "你的视频.avi" --out-dir ./myrun   # 存到指定目录
  python demo/demo_gate.py "你的视频.avi" --bg-image clean.jpg  # absdiff + 干净基准图
  python demo/demo_gate.py "你的视频.avi" --profile    # 打印各阶段耗时看瓶颈
  python demo/demo_gate.py "你的视频.avi" --model X --baseline Y   # 手动覆盖
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pose.extractor import PoseExtractor, PoseExtractorConfig  # noqa: E402
from pose.keypoint import CORE_JOINTS_13  # noqa: E402
from features.feature_builder import FrameFeatureBuilder  # noqa: E402
from features.gait import GaitSpeedExtractor  # noqa: E402
from features.intent import MotionIntentProjector  # noqa: E402
from features.env_mask import EnvMaskBuilder  # noqa: E402
from features.intent_gate import IntentGate  # noqa: E402
from model.lstm.risk_model import RiskRegressor  # noqa: E402
# ST-GCN（独立 B 模型，仅当 model_info.json 带 model_type=stgcn 时加载，不影响 LSTM 默认路径）
from model.stgcn.stgcn import load_stgcn  # noqa: E402
from model.stgcn.graph import FALL_JOINTS_17, skeleton_from_poseresult  # noqa: E402
from scripts.build_config import (  # noqa: E402
    ENV_MASK_POKE_PERSON, ENV_MASK_PERSON_MARGIN, ENV_MASK_USE_SPECULAR,
    ENV_MASK_SPEC_MIN_AREA, ENV_MASK_MOG2_VAR, ENV_MASK_MIN_AREA_RATIO,
    ENV_MASK_MAX_AREA_RATIO, ENV_MASK_CLOSE_KERNEL,
    ENV_MASK_ABS_THRESH, ENV_MASK_ABS_BLUR, ENV_MASK_ABS_USE_COLOR, ENV_MASK_GROUND_ROI,
)


class GateRunner:
    """流式实时处理：每帧 pose → B 推理 → 门控 → 显示。"""

    def __init__(self, model_path, window_size=35, input_dim=29, baseline=None,
                 mask_builder=None, force_e=0.5, min_detection=0.3, mask_h=480, mask_w=640,
                 resize_w=960, use_gpu=False, mask_kwargs=None):
        info_path = Path(model_path).parent / "model_info.json"
        info = {}
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            input_dim = info.get("input_dim", input_dim)
            window_size = info.get("window_size", window_size)
        # 模型类型分支：默认 lstm（不影响既有功能）；stgcn 独立路径
        model_type = info.get("model_type", "lstm")
        self.model_type = model_type
        self.window = window_size
        if model_type == "stgcn":
            self.model = load_stgcn(model_path)
            self.sbuf = deque(maxlen=window_size)   # (V,3) 关节缓冲
            self.joint_set = FALL_JOINTS_17
            self.fb = None
            self.gait = None
        else:
            self.model = RiskRegressor(input_dim=input_dim)
            self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            self.model.eval()
            self.fb = FrameFeatureBuilder()
            self.gait = GaitSpeedExtractor()
            self.buf = deque(maxlen=window_size)
        self.proj = MotionIntentProjector()
        mk = {"mask_h": mask_h, "mask_w": mask_w}
        if mask_kwargs:
            mk.update(mask_kwargs)
        if mask_builder is not None:
            self.mb = mask_builder
        elif baseline:
            self.mb = EnvMaskBuilder.from_baseline(baseline, **mk)
        else:
            self.mb = EnvMaskBuilder(mode="bg_diff", **mk)
        self.gate = IntentGate(mask_h=mask_h, mask_w=mask_w)
        self.E = force_e
        self.ext = PoseExtractor(PoseExtractorConfig(
            model_path=str(PROJECT_ROOT / "model/mediapipe_pose/pose_landmarker_lite.task"),
            use_gpu=use_gpu,
            min_detection_confidence=min_detection, min_tracking_confidence=min_detection,
            min_pose_presence_confidence=min_detection))
        self.ext.__enter__()
        self.H = mask_h
        self.W = mask_w
        self.resize_w = resize_w  # resize 给 MediaPipe 加速（pose 归一化坐标不受影响）
        self.profile = False      # --profile 时记录各阶段耗时

    def process_frame(self, frame, ts_ms):
        st = {}
        h, w = frame.shape[:2]
        # resize 给 MediaPipe（高分辨率 2560宽 CPU 慢，缩到 resize_w 加速；pose 是归一化坐标不受影响）
        if self.resize_w and w > self.resize_w:
            scale = self.resize_w / w
            small = cv2.resize(frame, (self.resize_w, int(h * scale)))
        else:
            small = frame
        t0 = time.perf_counter()
        pr = self.ext.detect_frame(small, ts_ms)
        st["pose_ms"] = (time.perf_counter() - t0) * 1000
        if pr is None:
            return {"risk": None, "has_person": False, "_stage_ms": st}
        t0 = time.perf_counter()
        if self.model_type == "stgcn":
            # ST-GCN 分支：每帧 17 关节 3D 归一化坐标 → (V,3) 缓冲 → 满窗 (1,3,T,V)
            skel = skeleton_from_poseresult(pr, self.joint_set)   # (V,3)
            self.sbuf.append(skel)
            risk = 0.0
            if len(self.sbuf) >= self.window:
                x = torch.from_numpy(np.stack(self.sbuf)).float()  # (T,V,3)
                x = x.permute(2, 0, 1).unsqueeze(0)               # (1,3,T,V)
                with torch.no_grad():
                    risk = float(self.model(x).item())
        else:
            # LSTM 分支（默认，行为不变）
            gs = self.gait.compute_per_frame(pr)
            vec = self.fb.build(pr, gs)
            self.buf.append(np.asarray(vec, dtype=np.float32))
            risk = 0.0
            if len(self.buf) >= self.window:
                x = torch.from_numpy(np.stack(self.buf)).float().unsqueeze(0)
                with torch.no_grad():
                    risk = float(self.model(x).item())
        st["b_ms"] = (time.perf_counter() - t0) * 1000
        zone = self.proj.compute_per_frame(pr)
        # 环境掩码在低分辨率下构建：bg_diff/HSV/形态学都在小图上跑，比原图快约一个数量级
        mw, mh = self.W, self.H
        t0 = time.perf_counter()
        if mw and mh and (small.shape[1] != mw or small.shape[0] != mh):
            mask_frame = cv2.resize(small, (mw, mh))
        else:
            mask_frame = small
        mask = self.mb.build_per_frame(pr, frame_bgr=mask_frame)
        st["mask_ms"] = (time.perf_counter() - t0) * 1000
        I_val = self.gate.compute(zone, mask)
        E_eff = self.E * I_val
        theta = max(0.45, 0.6 - 0.12 * E_eff)  # 简化 fuse 阈值（THR_ALARM_BASE - CTX_GAIN*ctx）
        alarm = risk >= theta
        return {"risk": risk, "E_eff": E_eff, "I": I_val, "theta": theta,
                "alarm": alarm, "mask": mask, "zone": zone, "pr": pr, "has_person": True,
                "_stage_ms": st if self.profile else {}}

    def draw(self, frame, info):
        h, w = frame.shape[:2]
        vis = frame.copy()
        if info.get("mask") is not None:
            m = info["mask"]
            # 掩码可能是低分辨率，绘制前对齐到画面尺寸
            if m.shape[:2] != vis.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), (vis.shape[1], vis.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            red = np.zeros_like(vis); red[m > 0] = [0, 0, 255]
            vis = cv2.addWeighted(vis, 0.7, red, 0.3, 0)
        z = info.get("zone")
        if z:
            cx = int(z["zone_cx"] * w); cy = int(z["zone_cy"] * h)
            a = max(int(z["zone_long_axis"] * w), 3)
            b = max(int(z["zone_short_axis"] * w), 3)
            cv2.ellipse(vis, (cx, cy), (a, b), np.degrees(z["zone_angle"]), 0, 360, (0, 255, 0), 2)
        pr = info.get("pr")
        if pr:
            for part in CORE_JOINTS_13:
                try:
                    kp = pr.get(part)
                    if kp.visibility >= 0.5:
                        cv2.circle(vis, (int(kp.x * w), int(kp.y * h)), 3, (0, 255, 255), -1)
                except KeyError:
                    pass
        risk = info.get("risk") or 0.0
        E_eff = info.get("E_eff", 0.0) or 0.0
        I_val = info.get("I", 0.0) or 0.0
        theta = info.get("theta", 0.6) or 0.6
        alm = info.get("alarm", False)
        b_color = (0, 0, 255) if risk >= 0.6 else ((0, 165, 255) if risk >= 0.3 else (0, 200, 0))
        status = "DANGER" if risk >= 0.6 else ("WARNING" if risk >= 0.3 else "SAFE")
        cv2.rectangle(vis, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.putText(vis, f"Status: {status}   Risk: {risk:.3f}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, b_color, 2)
        # 风险条（demo.py 风格）
        bx, by, bw, bh = 8, 34, 240, 16
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (80, 80, 80), -1)
        cv2.rectangle(vis, (bx, by), (bx + int(bw * min(risk, 1.0)), by + bh), b_color, -1)
        cv2.putText(vis, f"E_eff={E_eff:.2f}  I={I_val:.2f}  theta={theta:.2f}  alarm={'YES' if alm else 'no'}",
                    (8, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(vis, f"B={risk:.2f}", (w - 180, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, b_color, 3)
        return vis

    def close(self):
        self.ext.__exit__(None, None, None)


def _resolve_path(override, candidates, glob_patterns, label):
    """自动定位模型/基线：优先手动指定 → 已知候选 → 通配搜索。"""
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            return str(p)
        print(f"[WARN] 指定的{label}不存在，忽略: {p}")
    for c in candidates:
        p = PROJECT_ROOT / c
        if p.exists():
            return str(p)
    import glob
    for pat in glob_patterns:
        hits = sorted(glob.glob(str(PROJECT_ROOT / pat)))
        if hits:
            return hits[0]
    return None


# 已知默认位置（存在则直接用它，省去手动指定）
DEFAULT_MODEL_CANDIDATES = [
    "model/lstm/risk_v1/run_03/best_model.pt",
    "model/lstm/risk_v1/run_01/best_model.pt",
]
DEFAULT_BASELINE_CANDIDATES = [
    "env_baselines/Pre-VFall/CLFM1.json",
]


def _auto_baseline(ext, cap, args, mask_w, mask_h, max_scan=150, min_empty=5):
    """absdiff 无 --bg-image 时的兜底：扫描视频前段，用 pose 找『无人』帧 → 中值合成干净基准图。

    复用已构建的 pose 提取器 ext（避免再次加载 MediaPipe）。扫描后把 cap 复位到开头，
    供主循环重新顺序读取。返回 (mask_h, mask_w) BGR 基准图；无人帧不足则返回 None。
    """
    imgs = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n = 0
    while n < max_scan:
        ok, fr = cap.read()
        if not ok:
            break
        if fr.shape[1] > args.resize_width:
            s = args.resize_width / fr.shape[1]
            fr = cv2.resize(fr, (args.resize_width, int(fr.shape[0] * s)))
        try:
            pr = ext.detect_frame(fr, n * 33)
        except Exception:
            pr = None
        if pr is None:  # 无人 → 视为干净帧
            imgs.append(fr.astype(np.float32))
        n += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 复位，主循环从头读
    if len(imgs) < min_empty:
        return None
    med = np.median(np.stack(imgs, axis=0), axis=0).astype(np.uint8)
    med = cv2.resize(med, (mask_w, mask_h), interpolation=cv2.INTER_AREA)
    return med


def main():
    p = argparse.ArgumentParser(
        description="实时跌倒风险 + 环境交互门控（只给视频即可，模型/基线/GPU 自动探测）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python demo/demo_gate.py \"视频.avi\"                # 跑 + 自动保存\n"
               "  python demo/demo_gate.py \"视频.avi\" --no-save       # 只看不存\n"
               "  python demo/demo_gate.py \"视频.avi\" --out-dir ./r   # 存到指定目录\n"
               "  python demo/demo_gate.py \"视频.avi\" --profile")
    p.add_argument("video", help="输入视频文件（必填，其余全部自动探测）")
    p.add_argument("--model", default=None, help="手动指定 RiskRegressor best_model.pt（默认自动探测）")
    p.add_argument("--baseline", default=None, help="手动指定 VLM 静态基线 JSON（默认自动探测）")
    # 进阶参数（均有合理默认，一般无需改）
    p.add_argument("--force-e", type=float, default=0.5, help="环境交互强度 E（默认0.5）")
    p.add_argument("--min-detection", type=float, default=0.3, help="MediaPipe 检测阈值（默认0.3）")
    p.add_argument("--process-every", type=int, default=4, help="每N帧推理1次（默认4，越大越快）")
    p.add_argument("--resize-width", type=int, default=720, help="解码/推理缩放宽度（默认720）")
    p.add_argument("--gpu", action="store_true", help="强制用 GPU（默认有独显自动启用）")
    p.add_argument("--cpu", action="store_true", help="强制用 CPU（关 GPU）")
    p.add_argument("--no-save", action="store_true",
                   help="不保存（默认会保存标注视频+逐帧CSV+元数据到 demo/runs/<名>_<时间戳>/）")
    p.add_argument("--out-dir", default=None,
                   help="自定义保存目录（默认 demo/runs/<视频名>_<时间戳>）")
    p.add_argument("--profile", action="store_true", help="打印各阶段耗时（pose/B/掩码）定位瓶颈")
    # 掩码误报抑制（默认已收紧；仍误报可调这些）
    p.add_argument("--no-poke-person", action="store_true",
                   help="关闭『挖掉整个人』（默认开，避免 MOG2 把移动中的老人当危险）")
    p.add_argument("--mask-specular", action="store_true",
                   help="启用 HSV 反光(水渍)检测（默认关，误报最重：白墙/亮地/幕布都中标）")
    p.add_argument("--mask-var", type=int, default=ENV_MASK_MOG2_VAR,
                   help=f"MOG2 varThreshold（默认{ENV_MASK_MOG2_VAR}，越高噪声越少）")
    p.add_argument("--mask-min-area", type=float, default=ENV_MASK_MIN_AREA_RATIO,
                   help=f"前景连通域最小面积占比，小于丢弃（默认{ENV_MASK_MIN_AREA_RATIO}）")
    p.add_argument("--mask-close", type=int, default=ENV_MASK_CLOSE_KERNEL,
                   help=f"闭运算核大小，0=跳过（默认{ENV_MASK_CLOSE_KERNEL}）")
    p.add_argument("--mask-max-area", type=float, default=ENV_MASK_MAX_AREA_RATIO,
                   help=f"前景总占比超此值(光照突变/切场景)整帧清空，防整帧误报（默认{ENV_MASK_MAX_AREA_RATIO}）")
    # 帧差(absdiff)掩码：干净基准图做 absdiff，不用 MOG2、不用 VLM
    p.add_argument("--mode", choices=["bg_diff", "absdiff", "vlm_coarse"], default=None,
                   help="掩码模式：absdiff=干净基准图帧差(推荐,零依赖) | bg_diff=MOG2 | vlm_coarse(旧)。"
                        "默认：给了 --bg-image 用 absdiff，否则 bg_diff")
    p.add_argument("--bg-image", default=None,
                   help="absdiff 模式的干净基准图(路径)。部署时截一张地面干净无人的图。"
                        "可用 scripts/capture_baseline.py 从干净视频自动生成")
    p.add_argument("--abs-threshold", type=int, default=ENV_MASK_ABS_THRESH,
                   help=f"absdiff 二值化阈值（默认{ENV_MASK_ABS_THRESH}，越大越严格）")
    p.add_argument("--abs-blur", type=int, default=ENV_MASK_ABS_BLUR,
                   help=f"差分前高斯模糊核（默认{ENV_MASK_ABS_BLUR}，0=不模糊，须奇数）")
    p.add_argument("--no-abs-color", action="store_true",
                   help="absdiff 改用灰度差分（默认取三通道最大值，能抓彩色异物）")
    p.add_argument("--no-ground-roi", action="store_true",
                   help="absdiff 不过滤地面区域（默认只保留下方 1/3，匹配'地上多了一块东西'）")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERR] 打不开: {args.video}"); return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # —— 自动探测：模型 / 基线 / GPU ——
    model_path = _resolve_path(args.model, DEFAULT_MODEL_CANDIDATES,
                               "model/lstm/**/best_model.pt", "模型")
    baseline_path = _resolve_path(args.baseline, DEFAULT_BASELINE_CANDIDATES,
                                  "env_baselines/**/*.json", "基线")
    if not model_path:
        print("[ERR] 找不到 B 模型（LSTM 或 ST-GCN，也无 --model 指定）。默认 LSTM 在 model/lstm/，ST-GCN 用 --model 指定其 best_model.pt")
        return
    if args.cpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = True
    else:
        use_gpu = torch.cuda.is_available()  # 有独显自动启用，无则回落 CPU

    # 统一目标分辨率：保持宽高比，cap.set 不可靠，统一在 read 后 resize
    if args.resize_width and W and args.resize_width < W:
        target_w = args.resize_width
        target_h = int(round(H * target_w / W))
    else:
        target_w = W or 640
        target_h = H or 480
    aw, ah = target_w, target_h
    mask_w = min(aw, 480)                       # 掩码更低分辨率（~10x 加速），不影响 pose 精度
    mask_h = int(ah * mask_w / aw) if aw else 480
    if mode == "absdiff":
        base_name = f"absdiff(基准={args.bg_image or '自动标定'})"
    else:
        base_name = Path(baseline_path).name if baseline_path else "bg_diff(无基线)"
    print(f"=== 实时门控 demo ===\n  视频  : {args.video}\n  模型  : {model_path}"
          f"\n  掩码  : {base_name}\n  原始 {W}x{H} → 解码 {aw}x{ah} @ {fps:.1f}fps"
          f"\n  gpu={use_gpu}  process_every={args.process_every}  resize_w={args.resize_width}"
          f"  mask={mask_w}x{mask_h}")

    mask_kwargs = dict(
        poke_person=not args.no_poke_person,
        person_margin=ENV_MASK_PERSON_MARGIN,
        use_specular=args.mask_specular,
        spec_min_area=ENV_MASK_SPEC_MIN_AREA,
        mog2_var=args.mask_var,
        min_area_ratio=args.mask_min_area,
        max_area_ratio=args.mask_max_area,
        close_kernel=args.mask_close,
        # absdiff (v2)
        abs_threshold=args.abs_threshold,
        abs_blur=args.abs_blur,
        abs_use_color=not args.no_abs_color,
        ground_roi=not args.no_ground_roi,
    )

    # —— 掩码模式判定 ——
    mode = args.mode
    if mode is None:
        mode = "absdiff" if args.bg_image else "bg_diff"

    # —— 构建掩码构建器（absdiff 需干净基准图）——
    mask_builder = None
    if mode == "absdiff" and args.bg_image:
        mask_builder = EnvMaskBuilder.from_baseline_image(args.bg_image, **mask_kwargs)

    runner = GateRunner(model_path, baseline=baseline_path, mask_builder=mask_builder,
                        force_e=args.force_e,
                        min_detection=args.min_detection, mask_h=mask_h, mask_w=mask_w,
                        resize_w=args.resize_width, use_gpu=use_gpu, mask_kwargs=mask_kwargs)
    runner.profile = args.profile

    # absdiff 且无 --bg-image：复用 runner 的 pose 提取器扫描视频前段无人帧 → 中值合成干净基准
    if mode == "absdiff" and mask_builder is None:
        print("[INFO] absdiff 未给 --bg-image，尝试从视频自动标定干净基准（扫描无人帧→中值合成）…")
        bg = _auto_baseline(runner.ext, cap, args, mask_w, mask_h)
        if bg is None:
            print("[ERR] 自动标定失败：视频前段找不到足够'无人'帧。请用 scripts/capture_baseline.py "
                  "在干净空房视频上生成基准图，再用 --bg-image 指定。")
            return
        runner.mb = EnvMaskBuilder.from_baseline_image(bg, **mask_kwargs)
        print(f"[INFO] 自动标定成功（无人帧中值合成基准，{mask_w}x{mask_h}）")

    # —— 保存目录（默认开启）：标注视频 + 逐帧 CSV + 运行元数据 ——
    save_dir = None
    writer = None
    csv_f = None
    csv_w = None
    if not args.no_save:
        if args.out_dir:
            save_dir = Path(args.out_dir)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = PROJECT_ROOT / "demo" / "runs" / f"{Path(args.video).stem}_{ts}"
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(save_dir / "annotated.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (aw, ah))
        csv_f = open(save_dir / "frames.csv", "w", newline="", encoding="utf-8")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["frame", "ts_ms", "has_person", "risk", "I", "E_eff",
                        "theta", "alarm", "pose_ms", "b_ms", "mask_ms"])
        print(f"  保存 → {save_dir}")

    cv2.namedWindow("gate demo (q=quit)", cv2.WINDOW_NORMAL)
    print("  按 q 退出，Space 暂停")
    paused = False
    last_vis = np.zeros((ah, aw, 3), dtype=np.uint8)
    last_info = {"risk": 0.0, "E_eff": 0.0, "I": 0.0, "theta": 0.6, "alarm": False, "_stage_ms": {}}
    t = 0
    fps_ema = 0.0
    last_t = time.perf_counter()
    profile_acc = {"pose_ms": 0.0, "b_ms": 0.0, "mask_ms": 0.0, "n": 0}
    # 按视频帧率同步播放：25fps→40ms；处理更慢时不会更快
    play_delay_ms = max(1, int(round(1000.0 / fps))) if fps > 0 else 40
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok: break
            # 强制 resize 到统一工作分辨率：cap.set 不可靠，这里保证画幅/显示/保存一致
            if frame.shape[1] != target_w or frame.shape[0] != target_h:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if t % max(args.process_every, 1) == 0:
                last_info = runner.process_frame(frame, int(t * 1000 / fps))
                if args.profile and last_info.get("_stage_ms"):
                    sm = last_info["_stage_ms"]
                    for k in ("pose_ms", "b_ms", "mask_ms"):
                        profile_acc[k] += sm.get(k, 0.0)
                    profile_acc["n"] += 1
            vis = runner.draw(frame, last_info)
            if writer: writer.write(vis)
            # 逐帧记录（写入 CSV，便于事后分析 risk/I/theta 曲线与耗时）
            if csv_w is not None:
                sm = last_info.get("_stage_ms", {}) or {}
                risk_v = last_info.get("risk")
                csv_w.writerow([
                    t, int(t * 1000 / fps),
                    1 if last_info.get("has_person") else 0,
                    "" if risk_v is None else round(float(risk_v), 4),
                    round(float(last_info.get("I", 0.0) or 0.0), 4),
                    round(float(last_info.get("E_eff", 0.0) or 0.0), 4),
                    round(float(last_info.get("theta", 0.6) or 0.6), 4),
                    1 if last_info.get("alarm") else 0,
                    round(float(sm.get("pose_ms", 0.0) or 0.0), 2),
                    round(float(sm.get("b_ms", 0.0) or 0.0), 2),
                    round(float(sm.get("mask_ms", 0.0) or 0.0), 2),
                ])
            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            if dt > 0:
                inst = 1.0 / dt
                fps_ema = inst if fps_ema == 0 else fps_ema * 0.9 + inst * 0.1
            t += 1
        else:
            vis = last_vis
        if vis is None:
            vis = np.zeros((ah, aw, 3), dtype=np.uint8)
        last_vis = vis
        cv2.putText(vis, f"FPS={fps_ema:.1f}", (8, ah - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("gate demo (q=quit)", vis)
        if paused:
            # 暂停时停止帧率同步，等待空格或 q
            key = cv2.waitKey(0) & 0xFF
        else:
            key = cv2.waitKey(play_delay_ms) & 0xFF
        if key == ord("q"): break
        elif key == ord(" "): paused = not paused
    cap.release()
    if writer: writer.release()
    if csv_f: csv_f.close()
    runner.close()
    cv2.destroyAllWindows()
    print(f"  跑了 {t} 帧，平均显示 ~{fps_ema:.1f} FPS")
    if args.profile and profile_acc["n"]:
        n = profile_acc["n"]
        print(f"  [profile] 平均每重算帧: pose={profile_acc['pose_ms']/n:.1f}ms"
              f"  B={profile_acc['b_ms']/n:.1f}ms  mask={profile_acc['mask_ms']/n:.1f}ms")
    # 写运行元数据
    if save_dir is not None:
        alarm_frames = 0
        if csv_w is not None:
            # 重新统计报警帧数（CSV 已写，这里用轻量扫描）
            with open(save_dir / "frames.csv", "r", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                alarm_frames = sum(1 for r in rdr if r["alarm"] == "1")
        prof = None
        if args.profile and profile_acc["n"]:
            n = profile_acc["n"]
            prof = {k: round(profile_acc[k] / n, 2) for k in ("pose_ms", "b_ms", "mask_ms")}
        meta = {
            "video": str(Path(args.video).resolve()),
            "model": model_path,
            "baseline": baseline_path,
            "fps": fps,
            "input_res": [W, H],
            "decode_res": [aw, ah],
            "mask_res": [mask_w, mask_h],
            "total_frames": t,
            "avg_display_fps": round(fps_ema, 2),
            "alarm_frames": alarm_frames,
            "params": {"force_e": args.force_e, "min_detection": args.min_detection,
                       "process_every": args.process_every, "resize_width": args.resize_width,
                       "use_gpu": use_gpu},
            "profile_per_recompute_frame_ms": prof,
            "outputs": {"video": "annotated.mp4", "frames_csv": "frames.csv",
                        "run_meta": "run_meta.json"},
        }
        (save_dir / "run_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  产物 → {save_dir}  (视频 / 逐帧CSV / 元数据)")


if __name__ == "__main__":
    main()
