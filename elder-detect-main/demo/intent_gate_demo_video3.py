#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intent_gate_demo_video3.py — 视频渲染 demo（真实 B 模型 + 纯算法掩码 + I 门控）

在 v2 基础上把合成 B 换成真实 RiskRegressor 模型推理：
  逐帧 PoseExtractor 提 pose → GaitSpeed + FrameFeatureBuilder 算29维特征
  → 35帧滑窗 RiskRegressor 推理 → B(t)（真实姿态风险曲线）
  + bg_diff 掩码 + I 门控 + 融合 → 渲染 mp4（左画面+右risk曲线）

用法:
  python demo/intent_gate_demo_video3.py --video "D:/my_datasets/.../FallBackwardsS1.avi" --baseline env_baselines/Subject.1_Fall_backwards.json --run-dir model/lstm/risk_v1/run_03 --force-e 0.5 --frames 150
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
from pose.keypoint import BodyPart, CORE_JOINTS_13  # noqa: E402
from features.intent import MotionIntentProjector  # noqa: E402
from features.env_mask import EnvMaskBuilder  # noqa: E402
from features.intent_gate import IntentGate  # noqa: E402
from scripts.fuse_risk import fuse  # noqa: E402
from scripts.build_config import WINDOW_SIZE, STRIDE  # noqa: E402

PANEL_W = 420


def compute_B_realtime(poses, run_dir: Path):
    """真实 RiskRegressor 推理：逐帧特征 → 35帧滑窗 → B(t)。"""
    import torch
    from model.lstm.risk_model import RiskRegressor
    from features.feature_builder import FrameFeatureBuilder
    from features.gait import GaitSpeedExtractor
    from sequence import WindowConfig, SlidingWindowBuilder

    info = json.loads((run_dir / "model_info.json").read_text(encoding="utf-8"))
    input_dim = int(info.get("input_dim", 29))
    window = int(info.get("window_size", WINDOW_SIZE))
    model = RiskRegressor(input_dim=input_dim)
    model.load_state_dict(torch.load(str(run_dir / "best_model.pt"),
                                      map_location="cpu", weights_only=True))
    model.eval()

    gait = GaitSpeedExtractor()
    fb = FrameFeatureBuilder()
    feats = []
    for res in poses:
        if res is None:
            feats.append(np.zeros(input_dim, dtype=np.float32))
        else:
            gs = gait.compute_per_frame(res)
            feats.append(np.asarray(fb.build(res, gs), dtype=np.float32))
    X = np.array(feats, dtype=np.float32)
    T = len(X)
    if T < window:
        return np.zeros(T, dtype=np.float32), f"model(too few frames {T}<{window})"
    wb = SlidingWindowBuilder(WindowConfig(window_size=window, stride=STRIDE))
    wins = wb.build_from_array(X, np.zeros(T, dtype=np.int64))[0]
    with torch.no_grad():
        out = model(torch.from_numpy(wins).float()).numpy().reshape(-1)
    centers = np.arange(window // 2, T - window // 2, STRIDE)
    B = np.zeros(T, dtype=np.float32)
    m = min(len(out), len(centers))
    B[centers[:m]] = out[:m]
    return B, f"model({run_dir.name})"


SKELETON_CONN = [
    (BodyPart.NOSE, BodyPart.LEFT_SHOULDER), (BodyPart.NOSE, BodyPart.RIGHT_SHOULDER),
    (BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER),
    (BodyPart.LEFT_SHOULDER, BodyPart.LEFT_ELBOW), (BodyPart.LEFT_ELBOW, BodyPart.LEFT_WRIST),
    (BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW), (BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST),
    (BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP), (BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_HIP),
    (BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP),
    (BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE), (BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE),
    (BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE), (BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE),
]


def draw_skeleton(vis, pr, W, H):
    """画 13 核心骨骼点 + 骨架连线（黄线 + 红点）。"""
    if pr is None:
        return
    pts = {}
    for part in CORE_JOINTS_13:
        try:
            kp = pr.get(part)
        except KeyError:
            continue
        if kp.visibility < 0.5:
            continue
        pts[part] = (int(kp.x * W), int(kp.y * H))
    for a, b in SKELETON_CONN:
        if a in pts and b in pts:
            cv2.line(vis, pts[a], pts[b], (0, 255, 255), 2)
    for part, (x, y) in pts.items():
        cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)
        cv2.circle(vis, (x, y), 4, (255, 255, 255), 1)


def draw_coords(vis, pr, W, H):
    """左下角列 13 点归一化坐标 (x,y)。"""
    if pr is None:
        return
    y0 = H - 16 * 13 - 22
    cv2.rectangle(vis, (0, y0 - 6), (215, H), (0, 0, 0), -1)
    cv2.putText(vis, "13 joints (norm x,y):", (4, y0 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    for i, part in enumerate(CORE_JOINTS_13):
        try:
            kp = pr.get(part)
            txt = f"{part.name[:10]:10}:{kp.x:.2f},{kp.y:.2f}"
        except KeyError:
            txt = f"{part.name[:10]:10}:-"
        cv2.putText(vis, txt, (4, y0 + 14 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1)


def draw_curves(hist: dict, h: int) -> np.ndarray:
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
    for i, (name, c) in enumerate([("B", (120, 120, 255)), ("E_eff", (120, 255, 120)),
                                   ("I", (180, 180, 60)), ("theta", (210, 210, 210))]):
        cv2.line(panel, (x0 + 5 + i * 80, y0 + 5), (x0 + 25 + i * 80, y0 + 5), c, 2)
        cv2.putText(panel, name, (x0 + 28 + i * 80, y0 + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
    cv2.putText(panel, "red dot = alarm", (x0 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 255), 1)
    cv2.putText(panel, "risk curves (gate ON, real B model)", (x0 + 5, y0 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return panel


def main():
    p = argparse.ArgumentParser(description="视频渲染 demo（真实 B 模型 + I 门控）")
    p.add_argument("--video", required=True)
    p.add_argument("--baseline", default=None)
    p.add_argument("--run-dir", default="model/lstm/risk_v1/run_03", help="RiskRegressor 目录")
    p.add_argument("--force-e", type=float, default=0.5)
    p.add_argument("--frames", type=int, default=0, help="最多跑多少帧（0=全视频，参考 demo.py）")
    p.add_argument("--out", default="demo/demo_video_realB.mp4")
    p.add_argument("--out-json", default="demo/demo_video3.json")
    p.add_argument("--min-detection", type=float, default=0.3,
                   help="MediaPipe人体检测阈值（高分辨率2560宽视频建议0.2-0.3，默认0.5检测不到人）")
    p.add_argument("--show", action="store_true", help="实时窗口播放（cv2.imshow，按q退出，不存mp4）")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERR] 打不开: {args.video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"=== 视频渲染 demo（真实B模型）===\n  {args.video}\n  {W}x{H} @ {fps:.1f}fps")
    print(f"  run_dir: {args.run_dir}")

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
        min_detection_confidence=args.min_detection,
        min_tracking_confidence=args.min_detection,
        min_pose_presence_confidence=args.min_detection))  # 高分辨率视频降阈值(Pre-VFall 2560宽)
    ext.__enter__()

    E = float(args.force_e)
    room = "demo"
    frames, poses, zones, masks, I_arr = [], [], [], [], []
    n_target = args.frames if args.frames > 0 else 10**9  # 0=全视频
    t = 0
    while t < n_target:
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
        poses.append(pr)
        frames.append(frame)
        t += 1
    cap.release()
    ext.__exit__(None, None, None)
    T = len(frames)
    if T == 0:
        print("[ERR] 无帧")
        return

    I_arr = np.array(I_arr, dtype=np.float32)
    print(f"  提取 {T} 帧 pose，开始 B 模型推理...")
    B, b_src = compute_B_realtime(poses, Path(args.run_dir))
    print(f"  B mean={B.mean():.3f} max={B.max():.3f} 来源={b_src}")

    _, _, alarm_on, _, ta_on, _, _ = fuse(B, E, 0.0, False, room, I=I_arr)
    ta_on_a = np.full(T, float(np.mean(ta_on))) if np.ndim(ta_on) == 0 else np.asarray(ta_on, float)
    # 门控关对比
    _, _, alarm_off, _, _, _, _ = fuse(B, E, 0.0, False, room, I=None)
    n_off, n_on = int(alarm_off.sum()), int(alarm_on.sum())
    print(f"  I mean={I_arr.mean():.3f}")
    print(f"  alarm 门控关={n_off}/{T}  门控开={n_on}/{T}  变化={n_on - n_off:+d}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (W + PANEL_W, H))
    hist = {"B": [], "E_eff": [], "I": [], "theta": [], "alarm": []}
    for t in range(T):
        frame = frames[t]
        z = zones[t]; m = masks[t]
        I_val = float(I_arr[t]); E_eff = E * I_val
        B_val = float(B[t]); theta = float(ta_on_a[t])
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
        cv2.putText(vis, f"frame {t}/{T}  E_eff={E_eff:.2f}  I={I_val:.2f}",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        b_color = (0, 0, 255) if B_val >= 0.6 else ((0, 165, 255) if B_val >= 0.3 else (0, 200, 0))
        cv2.putText(vis, f"B={B_val:.2f}", (W - 200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, b_color, 3)
        cv2.putText(vis, f"theta={theta:.2f}  alarm={'YES' if alm else 'no'}",
                    (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_alm, 2)
        cv2.putText(vis, f"real B model | green=Zone  red=mask  yellow=skeleton",
                    (8, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        draw_skeleton(vis, poses[t], W, H)

        hist["B"].append(B_val); hist["E_eff"].append(E_eff)
        hist["I"].append(I_val); hist["theta"].append(theta)
        hist["alarm"].append(alm)
        panel = draw_curves(hist, H)
        canvas = np.zeros((H, W + PANEL_W, 3), dtype=np.uint8)
        canvas[:, :W] = vis
        canvas[:, W:] = panel
        if args.show:
            cv2.imshow("gate demo (q=quit)", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            out.write(canvas)
    out.release()
    print(f"\n  视频 → {args.out}")
    print(f"  真实B模型曲线 + 门控开/关 alarm: {n_off}→{n_on}（{n_on - n_off:+d}）")
    Path(args.out_json).write_text(json.dumps({
        "video": args.video, "T": T, "B_source": b_src, "E": E,
        "B_mean": float(B.mean()), "B_max": float(B.max()),
        "I_mean": float(I_arr.mean()), "alarm_off": n_off, "alarm_on": n_on,
        "out": args.out}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
