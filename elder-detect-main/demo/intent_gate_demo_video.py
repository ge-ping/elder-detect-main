#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intent_gate_demo_video.py — 视频端到端 demo（纯算法环境掩码 + I 门控）

用 CAUCAFall Subject.1 的 avi 视频端到端：
  1. cv2 读帧
  2. MediaPipe 提 pose（髋中心2D速度矢量 + 人脚挖孔）
  3. EnvMaskBuilder bg_diff：MOG2 新增物 + HSV 反光 + VLM 静态基线 + 人脚挖孔（纯算法）
  4. IntentGate: I = 预测区域落在掩码内的占比
  5. 融合 fuse(B, E, P=0, I) 开/关门控对比
  6. 输出：掩码可视化 PNG（frame+mask 叠加）+ 风险曲线 PNG

⚠️ VLM 仅部署时一次性标定静态基线（vlm_baseline_calibrate.py），本 demo 运行时纯算法不调 VLM。

用法:
  python demo/intent_gate_demo_video.py \
    --video "D:/my_datasets/CAUCAFall/Dataset CAUCAFall/Dataset CAUCAFall/CAUCAFall/Subject.1/Fall backwards/FallBackwardsS1.avi" \
    --baseline env_baselines/Subject.1_Fall_backwards.json \
    --b-risk 0.55 --force-e 0.5 --frames 150
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pose.extractor import PoseExtractor, PoseExtractorConfig  # noqa: E402
from features.intent import MotionIntentProjector  # noqa: E402
from features.env_mask import EnvMaskBuilder  # noqa: E402
from features.intent_gate import IntentGate  # noqa: E402
from scripts.fuse_risk import fuse  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="视频端到端 demo（纯算法掩码 + I 门控）")
    p.add_argument("--video", required=True, help="avi 视频路径")
    p.add_argument("--baseline", default=None, help="VLM 静态基线 JSON（无则用默认地面ROI）")
    p.add_argument("--b-risk", type=float, default=0.55, help="B 常量基线（边缘值展示门控）")
    p.add_argument("--force-e", type=float, default=0.5, help="合成环境分 E")
    p.add_argument("--frames", type=int, default=150, help="最多跑多少帧")
    p.add_argument("--out-mask", default="demo/demo_video_masks.png")
    p.add_argument("--out-curve", default="demo/demo_video_curve.png")
    p.add_argument("--out-json", default="demo/demo_video.json")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERR] 打不开视频: {args.video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"=== 视频端到端 demo ===")
    print(f"  视频: {args.video}")
    print(f"  {W}x{H} @ {fps:.1f}fps  最多 {args.frames} 帧")
    print(f"  基线: {args.baseline or '(无,默认地面ROI)'}")

    # 掩码构建器：bg_diff 模式，加载 VLM 基线（若有）
    if args.baseline:
        mb = EnvMaskBuilder.from_baseline(args.baseline, mask_h=H, mask_w=W)
        print(f"  VLM基线: ground_top_y={mb.ground_top_y}, obstacles={len(mb.obstacles)}个")
    else:
        mb = EnvMaskBuilder(mode="bg_diff", mask_h=H, mask_w=W)

    proj = MotionIntentProjector()
    gate = IntentGate(mask_h=H, mask_w=W)
    ext_config = PoseExtractorConfig(
        model_path=str(PROJECT_ROOT / "model/mediapipe_pose/pose_landmarker_lite.task"),
        use_gpu=False)
    ext = PoseExtractor(ext_config)
    ext.__enter__()

    E = float(args.force_e)
    B_val = float(args.b_risk)
    room = "demo"

    I_arr, B_arr, alarm_off, alarm_on = [], [], None, None
    frames_vis, masks_vis = [], []   # 掩码可视化用
    vis_indices = set()               # 抽样帧
    n_target = args.frames
    # 均匀抽 5 帧做掩码可视化
    for k in np.linspace(0, max(n_target - 1, 0), 5, dtype=int):
        vis_indices.add(int(k))

    t = 0
    while t < n_target:
        ok, frame = cap.read()
        if not ok:
            break
        pr = ext.detect_frame(frame, int(t * 1000 / fps))
        if pr is None:
            # 无人 → I=0, B=B_val
            I_arr.append(0.0)
            B_arr.append(B_val)
            t += 1
            continue
        zone = proj.compute_per_frame(pr)
        mask = mb.build_per_frame(pr, frame_bgr=frame)
        I_val = gate.compute(zone, mask)
        I_arr.append(I_val)
        B_arr.append(B_val)
        if t in vis_indices:
            frames_vis.append(frame.copy())
            masks_vis.append(mask.copy())
        t += 1
    cap.release()
    ext.__exit__(None, None, None)

    T = len(I_arr)
    if T == 0:
        print("[ERR] 未读到任何帧")
        return
    I_arr = np.array(I_arr, dtype=np.float32)
    B_arr = np.array(B_arr, dtype=np.float32)

    # 融合开/关
    _, _, a_off, _, ta_off, _, _ = fuse(B_arr, E, 0.0, False, room, I=None)
    _, _, a_on, _, ta_on, _, _ = fuse(B_arr, E, 0.0, False, room, I=I_arr)
    n_off, n_on = int(a_off.sum()), int(a_on.sum())

    print(f"\n  跑了 {T} 帧  B={B_val}(常量)  E={E}(合成)  P=0")
    print(f"  I   mean={I_arr.mean():.3f}  max={I_arr.max():.3f}  I>0帧={int((I_arr>0).sum())}/{T}")
    print(f"  ── 报警帧对比 ──")
    print(f"  门控关: {n_off}/{T}  ({n_off/T*100:.1f}%)")
    print(f"  门控开: {n_on}/{T}  ({n_on/T*100:.1f}%)")
    print(f"  变化  : {n_on - n_off:+d} 帧")

    # ---- 掩码可视化 PNG ----
    n_vis = len(frames_vis)
    if n_vis > 0:
        fig, axes = plt.subplots(2, n_vis, figsize=(4 * n_vis, 8))
        if n_vis == 1:
            axes = axes.reshape(2, 1)
        fig.suptitle("bg_diff 环境掩码可视化（红=危险点：MOG2新增物+HSV反光+VLM基线障碍物）",
                     fontsize=12, fontweight="bold")
        for i in range(n_vis):
            fr = frames_vis[i]
            mk = masks_vis[i]
            overlay = fr.copy()
            overlay[mk > 0] = [0, 0, 255]  # 红色叠加危险点
            mix = cv2.addWeighted(fr, 0.6, overlay, 0.4, 0)
            axes[0, i].imshow(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(f"frame {vis_indices and list(vis_indices)[i]}", fontsize=9)
            axes[0, i].axis("off")
            axes[1, i].imshow(cv2.cvtColor(mix, cv2.COLOR_BGR2RGB))
            axes[1, i].set_title(f"mask 叠加 (I={I_arr[list(vis_indices)[i]]:.2f})", fontsize=9)
            axes[1, i].axis("off")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        Path(args.out_mask).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out_mask, dpi=100)
        plt.close()
        print(f"\n  掩码可视化 → {args.out_mask}")

    # ---- 风险曲线 PNG ----
    tt = np.arange(T)
    ta_off_a = np.full(T, float(np.mean(ta_off))) if np.ndim(ta_off) == 0 else np.asarray(ta_off, float)
    ta_on_a = np.full(T, float(np.mean(ta_on))) if np.ndim(ta_on) == 0 else np.asarray(ta_on, float)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(f"视频端到端 demo: bg_diff掩码 + I门控  |  B={B_val} E={E} P=0",
                 fontsize=13, fontweight="bold")
    ax = axes[0]
    ax.plot(tt, I_arr, "-", color="darkorange", lw=1.2, label="I (intent overlap)")
    ax.plot(tt, E * I_arr, "-", color="green", lw=1.2, label=f"E_eff=E×I (gate on)")
    ax.axhline(E, color="green", ls="--", lw=1, label=f"E={E} (gate off)")
    ax.set_ylabel("E / I"); ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("I 门控（bg_diff 真实掩码）：无碰撞→I=0→E归零", fontsize=10, loc="left")
    ax = axes[1]
    ax.plot(tt, B_arr, "b-", lw=1.2, label="B (const)")
    ax.plot(tt, ta_off_a, color="gray", ls="--", lw=1.2, label="θ_alarm (gate off)")
    ax.plot(tt, ta_on_a, color="black", ls="-", lw=1.2, label="θ_alarm (gate on)")
    ax.fill_between(tt, ta_on_a, B_arr, where=B_arr >= ta_on_a, alpha=0.25, color="red",
                    label="alarm region (on)")
    ax.set_ylabel("risk / θ"); ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    ax = axes[2]
    if n_off > 0:
        ax.scatter(tt[a_off], np.ones(n_off), color="red", marker="x", s=30, label=f"alarm off ({n_off})")
    if n_on > 0:
        ax.scatter(tt[a_on], np.ones(n_on) * 0.6, color="blue", marker="o", s=20, label=f"alarm on ({n_on})")
    ax.set_yticks([]); ax.set_xlabel("frame"); ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(f"报警帧：关 {n_off} → 开 {n_on}（{n_on - n_off:+d}）", fontsize=10, loc="left")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.out_curve, dpi=110)
    plt.close()
    print(f"  风险曲线   → {args.out_curve}")

    Path(args.out_json).write_text(json.dumps({
        "video": args.video, "T": T, "fps": fps, "W": W, "H": H,
        "B": B_val, "E": E, "I_mean": float(I_arr.mean()),
        "I_pos_frames": int((I_arr > 0).sum()),
        "alarm_off": n_off, "alarm_on": n_on, "delta": n_on - n_off,
        "baseline": args.baseline,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  明细       → {args.out_json}")
    print("=" * 60)


if __name__ == "__main__":
    main()
