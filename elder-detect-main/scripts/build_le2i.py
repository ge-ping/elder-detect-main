#!/usr/bin/env python
"""
build_le2i.py — Le2i 数据集 → 按视频保存窗口，train/test 按视频切分

用法:
    python scripts/build_le2i.py

输出:
    datasets/Le2i/
    ├── videos/               ← 每个视频独立保存
    │   ├── Home_01_video_1_X.npy
    │   ├── Home_01_video_1_y.npy
    │   └── ...
    ├── X_train.npy, y_train.npy   ← 合并训练集
    ├── X_test.npy,  y_test.npy    ← 合并测试集
    └── split_log.txt              ← 切分记录
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pose import PoseExtractor, load_pose_config
from features.feature_builder import FrameFeatureBuilder
from features.gait import GaitSpeedExtractor
from sequence import WindowConfig, SlidingWindowBuilder
from scripts.build_config import (
    WINDOW_SIZE, STRIDE, TRAIN_RATIO, SEED, POSE_CONFIG, OUTPUT_ROOT,
)

LE2I_ROOT  = Path("D:/my_datasets/Le2i/Le2i/Le2i")
SUBSETS    = ["Home_01", "Home_02", "Coffee_room_01", "Coffee_room_02"]
OUTPUT_DIR = OUTPUT_ROOT / "Le2i"


# ==================================================================
# 标注解析
# ==================================================================

def parse_annotation(txt_path: Path) -> tuple[int, int] | None:
    try:
        lines = txt_path.read_text().strip().splitlines()
        if len(lines) < 2:
            return None
        return int(lines[0].strip()), int(lines[1].strip())
    except (ValueError, OSError):
        return None


def get_label(frame_idx_1based: int, fall_start: int, fall_end: int) -> int:
    if fall_start == 0 and fall_end == 0:
        return 0
    if frame_idx_1based < fall_start:
        return 0
    if frame_idx_1based <= fall_end:
        return 1
    return 2


# ==================================================================
# 单视频处理
# ==================================================================

def process_video(
    video_path: Path,
    fall_start: int,
    fall_end: int,
    extractor: PoseExtractor,
    feat_builder: FrameFeatureBuilder,
    windower: SlidingWindowBuilder,
) -> tuple[np.ndarray, np.ndarray]:
    frames, labels = [], []
    frame_idx = 0
    gait = GaitSpeedExtractor()

    for result in extractor.process_video(str(video_path)):
        frame_idx += 1
        if result is None:
            continue
        labels.append(get_label(frame_idx, fall_start, fall_end))
        gs = gait.compute_per_frame(result)
        frames.append(feat_builder.build(result, gs))

    if not frames:
        return (
            np.empty((0, WINDOW_SIZE, 29), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    feats = np.array(frames, dtype=np.float32)
    labs  = np.array(labels, dtype=np.int64)
    return windower.build_from_array(feats, labs)


# ==================================================================
# 主流程
# ==================================================================

def main():
    pose_cfg = load_pose_config(POSE_CONFIG)
    print(f"Pose model: {pose_cfg.model_path}")
    print(f"GPU: {pose_cfg.use_gpu}\n")

    feat_builder = FrameFeatureBuilder()
    windower = SlidingWindowBuilder(WindowConfig(window_size=WINDOW_SIZE, stride=STRIDE))

    videos_dir = OUTPUT_DIR / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # ---- 收集所有视频信息 ----
    video_records = []  # [(video_id, video_path, fall_start, fall_end)]
    total_videos, skipped = 0, 0

    for subset in SUBSETS:
        sd = LE2I_ROOT / subset / subset
        vd, ad = sd / "Videos", sd / "Annotations_files"
        if not vd.exists():
            continue
        for vp in sorted(vd.glob("video (*.avi")):
            total_videos += 1
            anno_path = ad / (vp.stem + ".txt")
            if not anno_path.exists():
                skipped += 1; continue
            parsed = parse_annotation(anno_path)
            if parsed is None:
                skipped += 1; continue
            video_id = f"{subset}_{vp.stem.replace(' ', '_')}"
            video_records.append((video_id, vp, parsed[0], parsed[1]))

    # ---- 视频级 train/test 切分 ----
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(video_records))
    n_train = int(len(video_records) * TRAIN_RATIO)
    train_idx = set(indices[:n_train])
    test_idx  = set(indices[n_train:])

    train_X, train_y = [], []
    test_X,  test_y  = [], []
    split_log: dict[str, list[str]] = {"train": [], "test": []}

    # ---- 逐视频处理 ----
    for i, (vid, vp, fs, fe) in enumerate(video_records):
        is_train = i in train_idx
        split = "train" if is_train else "test"
        split_log[split].append(vid)

        print(f"  [{i+1}/{len(video_records)}] {vid} ...", end=" ", flush=True)

        with PoseExtractor(pose_cfg) as extractor:
            X, y = process_video(vp, fs, fe, extractor, feat_builder, windower)

        # 保存单视频文件
        np.save(str(videos_dir / f"{vid}_X.npy"), X)
        np.save(str(videos_dir / f"{vid}_y.npy"), y)

        n = X.shape[0]
        t_frames = n + WINDOW_SIZE - 1 if n > 0 else 0
        uc = {int(lb): int(np.sum(y == lb)) for lb in np.unique(y)} if n > 0 else {}
        print(f"T={t_frames}, windows={n}, labels={uc} [{split}]")

        if n > 0:
            if is_train:
                train_X.append(X); train_y.append(y)
            else:
                test_X.append(X); test_y.append(y)

    # ---- 合并保存 ----
    def save_combined(prefix, X_list, y_list):
        if X_list:
            X_all = np.concatenate(X_list, axis=0)
            y_all = np.concatenate(y_list, axis=0)
        else:
            X_all = np.empty((0, WINDOW_SIZE, 29), dtype=np.float32)
            y_all = np.empty((0,), dtype=np.int64)
        np.save(str(OUTPUT_DIR / f"X_{prefix}.npy"), X_all)
        np.save(str(OUTPUT_DIR / f"y_{prefix}.npy"), y_all)
        return X_all, y_all

    Xt, yt = save_combined("train", train_X, train_y)
    Xe, ye = save_combined("test",  test_X,  test_y)

    # ---- 日志 ----
    log_path = OUTPUT_DIR / "split_log.json"
    log_path.write_text(json.dumps(split_log, indent=2, ensure_ascii=False))

    # ---- 摘要 ----
    print(f"\n{'='*60}")
    print(f"Le2i build complete.")
    print(f"  Videos: {len(video_records)} (skipped {skipped}/{total_videos})")
    print(f"  Split:  train={len(split_log['train'])} videos, test={len(split_log['test'])} videos")
    print(f"  Seed:   {SEED}")
    print(f"")
    print(f"  X_train: {Xt.shape}, y_train: {yt.shape}")
    for lb, name in [(0,"Non-fall"),(1,"Pre-impact"),(2,"Fall")]:
        print(f"    label {lb} ({name}): {int(np.sum(yt==lb))}")
    print(f"")
    print(f"  X_test:  {Xe.shape}, y_test:  {ye.shape}")
    for lb, name in [(0,"Non-fall"),(1,"Pre-impact"),(2,"Fall")]:
        print(f"    label {lb} ({name}): {int(np.sum(ye==lb))}")
    print(f"")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"    X_train.npy, y_train.npy")
    print(f"    X_test.npy,  y_test.npy")
    print(f"    videos/*.npy (per-video)")
    print(f"    split_log.json")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
