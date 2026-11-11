#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intent_gate_demo_video2.py — 视频渲染 demo（帧+掩码+Zone+risk曲线，导出可播放 mp4）

逐帧渲染：
  左 = 原始帧 + bg_diff 掩码红色叠加 + 预测 Zone 绿色椭圆 + 状态文字(B/E_eff/I/θ/alarm)
  右 = risk 曲线 panel（B/E_eff/I/θ_alarm 历史 + alarm 红点标记）
导出 mp4 可播放，直观看 risk 随帧变化。

默认用合成跌倒 B 曲线（前60%正常→爬升→跌倒=1.0），让 risk 变化可见。
也可 --b-risk 常量。

用法:
  python demo/intent_gate_demo_video2.py --video "D:/my_datasets/.../FallBackwardsS1.avi" --baseline env_baselines/Subject.1_Fall_backwards.json --force-e 0.5 --frames 150
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pose.extractor import PoseExtractor, PoseExtractorConfig  # noqa: E402
from features.intent import MotionIntentProjector  # noqa: E402
from features.env_mask import EnvMaskBuilder  # noqa: E402
from features.intent_gate import IntentGate  # noqa: E402
from scripts.fuse_risk import fuse  # noqa: E402

PANEL_W = 420


def synth_fall_risk(T: int) -> np.ndarray:
    """合成跌倒 B：前60% 0.15正常 → 爬升0.3→1.0 → 跌倒1.0。"""
    r = np.full(T, 0.15, dtype=np.float32)
    fs, fp = int(T * 0.6), int(T * 0.85)
    n = max(fp - fs, 1)
    r[fs:fp] = 0.3 + 0.7 * (1 - np.exp(-np.arange(n) / 4.0))
    r[fp:] = 1.0
    return r


def draw_curves(hist: dict, h: int) -> np.ndarray:
    """右侧 risk 曲线 panel（cv2 纯绘制，快）。"""
    panel = np.full((h, PANEL_W, 3), 24, dtype=np.uint8)
    x0, x1 = 45, PANEL_W - 15
    y0, y1 = 15, h - 25
    cv2.line(panel, (x0, y0), (x0, y1), (110, 110, 110), 1)
    cv2.line(panel, (x0, y1), (x1, y1), (110, 110, 110), 1)
    for v, txt in [(0.3, "0.3"), (0.6, "0.6"), (1.0, "1.0")]:
        y = int(y1 - v * (y1 - y0))
        cv2.line(panel, (x0, y), (x1, y), (60, 60, 60), 1)
        cv2.putText(panel, txt, (5, y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    n = len(hist["B"])
    if n > 1:
        for key, color in [("B", (120, 120, 255)), ("E_eff", (120, 255, 120)),
                           ("I", (180, 180, 60)), ("theta", (210, 210, 210))]:
            pts = [(x0 + int(i / (n - 1) * (x1 - x0)),
                    int(y1 - hist[key][i] * (y1 - y0))) for i in range(n)]
            for i in range(1, len(pts)):
                cv2.line(panel, pts[i - 1], pts[i], color, 2)
        for i, a in enumerate(hist["alarm"]):
            if a:
                x = x0 + int(i / max(n - 1, 1) * (x1 - x0))
                cv2.circle(panel, (x, y1 + 8), 2, (60, 60, 255), -1)
    legends = [("B", (120, 120, 255)), ("E_eff", (120, 255, 120)),
               ("I", (180, 180, 60)), ("theta", (210, 210, 210))]
    for i, (name, c) in enumerate(legends):
        cv2.line(panel, (x0 + 5 + i * 80, y0 + 5), (x0 + 25 + i * 80, y0 + 5), c, 2)
        cv2.putText(panel, name, (x0 + 28 + i * 80, y0 + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
    cv2.putText(panel, "red dot = alarm", (x0 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 255), 1)
    cv2.putText(panel, "risk curves (gate ON)", (x0 + 5, y0 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return panel


def main():
    p = argparse.ArgumentParser(description="视频渲染 demo（帧+掩码+risk曲线，导出 mp4）")
    p.add_argument("--video", required=True, help="avi 视频路径")
    p.add_argument("--baseline", default=None, help="VLM 静态基线 JSON")
    p.add_argument("--b-risk", type=float, default=None,
                   help="B 常量（提供则用，否则合成跌倒曲线看变化）")
    p.add_argument("--force-e", type=float, default=0.5)
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--out", default="demo/demo_video_risk.mp4")
    p.add_argument("--out-json", default="demo/demo_video2.json")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERR] 打不开: {args.video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"=== 视频渲染 demo ===\n  {args.video}\n  {W}x{H} @ {fps:.1f}fps")

    if args.baseline:
        mb = EnvMaskBuilder.from_baseline(args.baseline, mask_h=H, mask_w=W)
        print(f"  VLM基线: ground_top_y={mb.ground_top_y}, obstacles={len(mb.obstacles)}")
    else:
        mb = EnvMaskBuilder(mode="bg_diff", mask_h=H, mask_w=W)
    proj = MotionIntentProjector()
    gate = IntentGate(mask_h=H, mask_w=W)
    ext = PoseExtractor(PoseExtractorConfig(
        model_path=str(PROJECT_ROOT / "model/mediapipe_pose/pose_landmarker_lite.task"),
        use_gpu=False,
        min_detection_confidence=0.3, min_tracking_confidence=0.3,
        min_pose_presence_confidence=0.3))  # 降阈值适应高分辨率视频(Pre-VFall 2560宽)
    ext.__enter__()

    E = float(args.force_e)
    room = "demo"
    frames, zones, masks, I_arr = [], [], [], []
    t = 0
    while t < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        pr = ext.detect_frame(frame, int(t * 1000 / fps))
        if pr is None:
            zones.append(None); masks.append(None); I_arr.append(0.0)
        else:
            z = proj.compute_per_frame(pr)
            m = mb.build_per_frame(pr, frame_bgr=frame)
            I_arr.append(gate.compute(z, m))
            zones.append(z); masks.append(m)
        frames.append(frame)
        t += 1
    cap.release()
    ext.__exit__(None, None, None)
    T = len(frames)
    if T == 0:
        print("[ERR] 无帧")
        return

    I_arr = np.array(I_arr, dtype=np.float32)
    if args.b_risk is not None:
        B = np.full(T, float(args.b_risk), dtype=np.float32)
        b_src = f"const({args.b_risk})"
    else:
        B = synth_fall_risk(T)
        b_src = "synth_fall"
    _, _, alarm_off, _, _, _, _ = fuse(B, E, 0.0, False, room, I=None)
    _, _, alarm_on, _, ta_on, _, _ = fuse(B, E, 0.0, False, room, I=I_arr)
    n_off, n_on = int(alarm_off.sum()), int(alarm_on.sum())
    ta_on_a = np.full(T, float(np.mean(ta_on))) if np.ndim(ta_on) == 0 else np.asarray(ta_on, float)

    print(f"  B={b_src} E={E} P=0  T={T}")
    print(f"  I mean={I_arr.mean():.3f}  alarm 关={n_off} 开={n_on} (变化{n_on - n_off:+d})")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (W + PANEL_W, H))
    hist = {"B": [], "E_eff": [], "I": [], "theta": [], "alarm": []}
    for t in range(T):
        frame = frames[t]
        z = zones[t]; m = masks[t]
        I_val = float(I_arr[t])
        E_eff = E * I_val
        B_val = float(B[t])
        theta = float(ta_on_a[t])
        alm = bool(alarm_on[t])

        vis = frame.copy()
        if m is not None:
            red = np.zeros_like(vis); red[m > 0] = [0, 0, 255]
            vis = cv2.addWeighted(vis, 0.65, red, 0.35, 0)
        if z is not None:
            cx = int(z["zone_cx"] * W); cy = int(z["zone_cy"] * H)
            a = max(int(z["zone_long_axis"] * W), 3)
            b = max(int(z["zone_short_axis"] * W), 3)
            cv2.ellipse(vis, (cx, cy), (a, b), np.degrees(z["zone_angle"]), 0, 360, (0, 255, 0), 2)
        color_alm = (0, 0, 255) if alm else (0, 200, 0)
        cv2.rectangle(vis, (0, 0), (W, 100), (0, 0, 0), -1)
        cv2.putText(vis, f"frame {t}/{T}  B={B_val:.2f}  E_eff={E_eff:.2f}  I={I_val:.2f}",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"theta={theta:.2f}  alarm={'YES' if alm else 'no'}",
                    (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_alm, 2)
        cv2.putText(vis, "green=Zone(predict)  red=mask(hazard)",
                    (8, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        hist["B"].append(B_val); hist["E_eff"].append(E_eff)
        hist["I"].append(I_val); hist["theta"].append(theta)
        hist["alarm"].append(alm)
        panel = draw_curves(hist, H)
        canvas = np.zeros((H, W + PANEL_W, 3), dtype=np.uint8)
        canvas[:, :W] = vis
        canvas[:, W:] = panel
        out.write(canvas)
    out.release()
    print(f"\n  视频 → {args.out}")
    print(f"  左=画面+掩码+Zone  右=risk曲线(红点=alarm)  可直接播放")
    Path(args.out_json).write_text(json.dumps({
        "video": args.video, "T": T, "B": b_src, "E": E,
        "I_mean": float(I_arr.mean()),
        "alarm_off": n_off, "alarm_on": n_on, "delta": n_on - n_off,
        "out": args.out}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
