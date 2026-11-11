#!/usr/bin/env python
"""
env_extract_frames.py — EnvRisk R1 第一段：源视频 → 抽 K 帧 → 存图 + index

环境语义在单段视频里近似静止，每视频均匀抽 K 帧即够覆盖光照/房间/地面波动。
产物：
    frames/<ds>/<stem>/f{00..KK}.jpg
    frames/index.csv       列: ds, stem, n_frames_video, fps, frame_paths(;分隔)

用法:
    python scripts/env_extract_frames.py --video_root /path/to/videos        # 目录：递归扫描
    python scripts/env_extract_frames.py --video_root /path/to/video.mp4     # 也支持单个文件
    python scripts/env_extract_frames.py --video_root /path/to/videos --frames_per_video 5
依赖: opencv-python（cv2）；未安装时给出明确报错。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_config import (
    ENV_VLM_FRAMES_PER_VIDEO, ENV_VLM_IMG_LONG_SIDE,
)

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv")


def find_videos(root: Path) -> list[Path]:
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES:
            out.append(p)
    return out


def open_capture(path: Path):
    """仅用 cv2（随机访问稳）；未安装则明确报错。"""
    try:
        import cv2  # lazy import，避免无 cv2 环境 import 即崩
    except ImportError as e:
        raise RuntimeError(
            f"需要 opencv-python 来读视频帧。请安装: pip install opencv-python\n"
            f"(原错误: {e})"
        )
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    return cv2, cap


def extract_one(cv2, cap, video_path: Path, out_dir: Path,
                k: int, long_side: int) -> list[str]:
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if n <= 0:
        return []
    # 均匀采样 K 帧（首/中/尾覆盖）
    idxs = np.linspace(0, n - 1, k, dtype=int)
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_paths = []
    for j, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        # BGR→RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = long_side / max(h, w)
        if scale < 1.0:
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_AREA)
        # RGB→BGR 再写 jpg
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        fpath = out_dir / f"f{j:02d}.jpg"
        cv2.imwrite(str(fpath), bgr)
        rel_paths.append(str(fpath.relative_to(PROJECT_ROOT)))
    return rel_paths


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--video_root", required=True,
                   help="源视频根目录（递归扫描 mp4/avi/mov...）")
    p.add_argument("--out", default="frames", help="抽帧输出根（默认 frames/）")
    p.add_argument("--frames_per_video", type=int, default=ENV_VLM_FRAMES_PER_VIDEO)
    p.add_argument("--long_side", type=int, default=ENV_VLM_IMG_LONG_SIDE)
    args = p.parse_args()

    root = Path(args.video_root).resolve()
    if not root.exists():
        print(f"[ERR] video_root 不存在: {root}")
        return
    # 支持直接传单个视频文件（而非仅目录递归扫描）
    single = root.is_file()
    if single:
        if root.suffix.lower() not in VIDEO_SUFFIXES:
            print(f"[ERR] 不是支持的视频格式: {root}（支持 {VIDEO_SUFFIXES}）")
            return
        videos = [root]
    else:
        videos = find_videos(root)
    if not videos:
        print(f"[ERR] 在 {root} 下未找到任何视频文件")
        return

    out_root = PROJECT_ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    index_path = out_root / "index.csv"
    rows = []

    print(f"找到 {len(videos)} 个视频，抽帧 K={args.frames_per_video}，"
          f"长边={args.long_side} → {out_root}")

    for vi, vpath in enumerate(videos, 1):
        # ds = 相对 root 的一级目录名；单文件模式取所在目录名；扁平则 "_root"
        if single:
            ds = root.parent.name or "_root"
            stem = root.stem
        else:
            rel = vpath.relative_to(root)
            ds = rel.parts[0] if len(rel.parts) > 1 else "_root"
            stem = vpath.stem
        frame_dir = out_root / ds / stem
        cv2, cap = open_capture(vpath)
        try:
            n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            rels = extract_one(cv2, cap, vpath, frame_dir,
                               args.frames_per_video, args.long_side)
        finally:
            cap.release()
        if not rels:
            print(f"  [{vi}/{len(videos)}] {ds}/{stem} ... [SKIP 抽帧失败]")
            continue
        rows.append({
            "ds": ds, "stem": stem,
            "n_frames_video": n_video, "fps": round(fps, 2),
            "frame_paths": ";".join(rels),
        })
        print(f"  [{vi}/{len(videos)}] {ds}/{stem} "
              f"video_frames={n_video} fps={fps:.1f} → {len(rels)} 帧")

    with index_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ds", "stem", "n_frames_video", "fps", "frame_paths"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nDone. 成功 {len(rows)}/{len(videos)} 个视频，index → {index_path}")


if __name__ == "__main__":
    main()
