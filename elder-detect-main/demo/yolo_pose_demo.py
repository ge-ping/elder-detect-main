"""
YOLO 姿态后端可视化 demo（PoC 验证用）

目的：验证「YOLO 后端能跑 + GPU 速度 + 关键点质量」。
不喂任何已训模型（本次 PoC 不重训，YOLO 2D 与 3D 模型分布不一致，
强行推理无意义）。

用法:
  python demo\\yolo_pose_demo.py "视频.mp4"
  python demo\\yolo_pose_demo.py "视频.mp4" --model yolov8n-pose.pt --device cuda
  python demo\\yolo_pose_demo.py "视频.mp4" --no-save

输出 (demo/runs/yolo_<视频名>_<时间戳>/):
  annotated.mp4   标注视频
  run_meta.json   汇总（FPS / 分辨率 / 帧数 / 置信度门控统计）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pose.backend import create_pose_backend                                    # noqa: E402
from pose.coco_joints import YOLO_CORE_EDGES_BODYPART, COCO_CORE_INDICES       # noqa: E402
from pose.keypoint import BodyPart, PoseResult                                  # noqa: E402

# 骨架颜色（BGR）
EDGE_COLOR = (0, 200, 255)
KP_COLOR_HI = (0, 255, 0)     # 高置信 绿
KP_COLOR_LO = (0, 0, 255)     # 低置信 红


def draw(frame: np.ndarray, pr: PoseResult | None, min_conf: float) -> np.ndarray:
    """画 13 关节骨架 + 关键点（按置信度着色）。"""
    vis = frame.copy()
    if pr is None:
        cv2.putText(vis, "NO PERSON", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                     0.7, (0, 0, 255), 2)
        return vis

    h, w = vis.shape[:2]
    by_name = {kp.name: kp for kp in pr.landmarks}

    # 边
    for a, b in YOLO_CORE_EDGES_BODYPART:
        ka, kb = by_name.get(a), by_name.get(b)
        if ka is None or kb is None:
            continue
        pa = (int(ka.x * w), int(ka.y * h))
        pb = (int(kb.x * w), int(kb.y * h))
        cv2.line(vis, pa, pb, EDGE_COLOR, 2)

    # 点
    for kp in pr.landmarks:
        px, py = int(kp.x * w), int(kp.y * h)
        col = KP_COLOR_HI if kp.visibility >= min_conf else KP_COLOR_LO
        cv2.circle(vis, (px, py), 4, col, -1)

    # 低置信提示
    if pr.mean_confidence < min_conf:
        cv2.putText(vis, f"LOW CONF {pr.mean_confidence:.2f}<{min_conf:.2f}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    return vis


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO 姿态后端可视化 demo (PoC)")
    ap.add_argument("video", help="输入视频路径")
    ap.add_argument("--model", default="yolov8n-pose.pt", help="YOLO pose 权重")
    ap.add_argument("--device", default="cuda", help="cuda / cpu / 0")
    ap.add_argument("--conf", type=float, default=0.5, help="关键点置信度阈值")
    ap.add_argument("--resize-width", type=int, default=720, help="显示/保存宽度")
    ap.add_argument("--min-confidence", type=float, default=0.5,
                    help="帧级置信度门控：平均关节置信度低于此值→不更新 (方案A)")
    ap.add_argument("--no-save", action="store_true", help="关闭存盘")
    ap.add_argument("--out-dir", default=None, help="自定义输出目录")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERR] 无法打开视频: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    backend = create_pose_backend("yolo", model=args.model, device=args.device, conf=args.conf)
    print(f"=== YOLO 姿态 demo (PoC) ===\n  视频 : {args.video}\n  后端 : {backend}\n"
          f"  面板 : {args.resize_width}px宽 @ {fps:.1f}fps\n"
          f"  帧级置信度门控: 阈值={args.min_confidence}\n"
          f"  操作 : q=退出  空格=暂停")

    save_dir = None
    writer = None
    if not args.no_save:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        stem = Path(args.video).stem
        save_dir = Path(args.out_dir) if args.out_dir else (PROJECT_ROOT / "demo" / "runs" / f"yolo_{stem}_{stamp}")
        save_dir.mkdir(parents=True, exist_ok=True)

    n_frames = 0
    n_person = 0
    n_low_conf = 0
    detect_ms = 0.0          # 仅统计 detect_frame 耗时（排除显示）
    last_pr: PoseResult | None = None
    t = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # 缩放（仅改显示/保存尺寸，不影响送入后端的坐标归一化）
            if args.resize_width and frame.shape[1] != args.resize_width:
                scale = args.resize_width / frame.shape[1]
                small = cv2.resize(frame, (args.resize_width, int(frame.shape[0] * scale)))
            else:
                small = frame

            ts_ms = int(t * 1000 / fps)
            t0 = time.perf_counter()
            pr = backend.detect_frame(small, ts_ms)
            detect_ms += (time.perf_counter() - t0)

            has_person = pr is not None
            if has_person:
                n_person += 1
                last_pr = pr
                if pr.mean_confidence < args.min_confidence:
                    n_low_conf += 1

            vis = draw(small, pr if has_person else None, args.min_confidence)
            # FPS 叠层（detect_ms 单位为秒）
            inst_fps = (n_frames / detect_ms) if detect_ms > 0 else 0.0
            cv2.putText(vis, f"YOLO FPS(detect): {inst_fps:.1f}", (12, vis.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("YOLO Pose (PoC)", vis)
            if writer is None and save_dir is not None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(save_dir / "annotated.mp4"), fourcc, fps,
                    (vis.shape[1], vis.shape[0]))
            if writer is not None:
                writer.write(vis)

            n_frames += 1
            key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                while (cv2.waitKey(0) & 0xFF) != ord(" "):
                    pass

    finally:
        cap.release()
        cv2.destroyAllWindows()       # 先关窗口，避免"放完还卡着"的观感
        if writer:
            writer.release()
        if hasattr(backend, "__exit__"):
            backend.__exit__(None, None, None)

    avg_fps = (n_frames / detect_ms) if detect_ms > 0 else 0.0
    print(f"\n视频读取完毕，开始收尾...")
    print(f"  处理帧数 : {n_frames}/{total}")
    print(f"  有人帧   : {n_person}  ({100.0*n_person/max(n_frames,1):.1f}%)")
    print(f"  低置信帧 : {n_low_conf}")
    print(f"  YOLO detect 平均 FPS : {avg_fps:.1f}")
    print(f"  (对照: MediaPipe 在本机回退 CPU，通常 5~15 FPS)")

    if save_dir is not None:
        meta = {
            "backend": "yolo",
            "model": args.model,
            "device": args.device,
            "conf": args.conf,
            "video": str(args.video),
            "total_frames": total,
            "processed_frames": n_frames,
            "person_frames": n_person,
            "low_conf_frames": n_low_conf,
            "min_confidence": args.min_confidence,
            "avg_detect_fps": round(avg_fps, 2),
        }
        (save_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  完成 -> {save_dir}")
    print("收尾完成，已退出。")


if __name__ == "__main__":
    main()
