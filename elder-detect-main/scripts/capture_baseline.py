#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
capture_baseline.py — 生成 absdiff 掩码用的「干净空房基准图」（完全不用 VLM）

思路（对应 demo_gate.py 的 absdiff 模式）：
  部署时，在老人家里地面干净、画面无人的时候，截取一张基准背景图。
  本脚本从一段「干净视频」里自动找出无人帧（用 MediaPipe pose 检测），
  对这些帧取中值（median）合成一张无人的干净基准图，存为 JPG。

用法：
  python scripts/capture_baseline.py "干净房间视频.mp4"
  python scripts/capture_baseline.py "干净房间视频.mp4" --out env_baselines/livingroom_clean.jpg
  python scripts/capture_baseline.py "干净房间视频.mp4" --max-scan 300 --min-empty 10

生成后，把该 JPG 喂给 demo_gate.py：
  python demo/demo_gate.py "监控视频.avi" --bg-image env_baselines/livingroom_clean.jpg

注意：扫描的是「无人帧」。如果视频里一直有人，就找不到足够干净帧，
      此时请直接用一张手动拍摄的干净房间照片作为 --bg-image。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pose.extractor import PoseExtractor, PoseExtractorConfig  # noqa: E402
import torch  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="生成 absdiff 掩码用的干净基准图（不用 VLM）")
    p.add_argument("video", help="一段『地面干净、尽量无人』的参考视频")
    p.add_argument("--out", default=None,
                   help="输出 JPG 路径（默认 env_baselines/<视频名>_clean.jpg）")
    p.add_argument("--max-scan", type=int, default=300,
                   help="最多扫描前 N 帧寻找无人帧（默认300）")
    p.add_argument("--min-empty", type=int, default=10,
                   help="至少需要这么多无人帧才合成基准（默认10）")
    p.add_argument("--resize-width", type=int, default=720,
                   help="pose 检测缩放宽度（默认720，仅影响标定速度）")
    p.add_argument("--min-detection", type=float, default=0.3,
                   help="MediaPipe 检测阈值（默认0.3）")
    p.add_argument("--gpu", action="store_true", help="强制 GPU")
    p.add_argument("--cpu", action="store_true", help="强制 CPU")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERR] 打不开视频: {args.video}")
        return

    if args.cpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = True
    else:
        use_gpu = torch.cuda.is_available()

    ext = PoseExtractor(PoseExtractorConfig(
        model_path=str(PROJECT_ROOT / "model/mediapipe_pose/pose_landmarker_lite.task"),
        use_gpu=use_gpu,
        min_detection_confidence=args.min_detection,
        min_tracking_confidence=args.min_detection,
        min_pose_presence_confidence=args.min_detection))
    ext.__enter__()

    imgs = []
    n = 0
    try:
        while n < args.max_scan:
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
    finally:
        ext.__exit__(None, None, None)
        cap.release()

    print(f"  扫描 {n} 帧，找到 {len(imgs)} 个无人帧")

    if len(imgs) < args.min_empty:
        print(f"[ERR] 无人帧不足（{len(imgs)} < {args.min_empty}）。"
              "请改用一张手动拍摄的干净房间照片作为 --bg-image。")
        return

    med = np.median(np.stack(imgs, axis=0), axis=0).astype(np.uint8)

    out = Path(args.out) if args.out else (
        PROJECT_ROOT / "env_baselines" / f"{Path(args.video).stem}_clean.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), med)
    print(f"[OK] 干净基准图已保存: {out}")
    print(f"     用法: python demo/demo_gate.py \"监控视频.avi\" --bg-image {out}")


if __name__ == "__main__":
    main()
