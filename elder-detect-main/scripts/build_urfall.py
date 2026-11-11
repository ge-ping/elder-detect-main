#!/usr/bin/env python
"""
build_urfall.py — UR-FALL 数据集 → (N,35,29) X.npy + y.npy

用法:
    python scripts/build_urfall.py

输出:
    datasets/UR-FALL/
    ├── X_train.npy, y_train.npy
    ├── X_test.npy,  y_test.npy
    ├── split_log.json
    └── videos/*.npy
"""
from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import cv2
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

URFALL_ROOT = Path("D:/my_datasets/UR-FALL/downloads")
OUTPUT_DIR  = OUTPUT_ROOT / "UR-FALL"


# ==================================================================
# CSV 解析
# ==================================================================

def load_labels(csv_path: Path) -> dict[str, dict[int, int]]:
    """解析 CSV → {sequence_name: {frame_num: csv_label}}"""
    labels: dict[str, dict[int, int]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            seq   = row[0].strip()
            frame = int(row[1])
            lbl   = int(row[2])
            labels.setdefault(seq, {})[frame] = lbl
    return labels


# ==================================================================
# 标签映射
# ==================================================================

def map_label(is_adl: bool, csv_label: int) -> int:
    """UR-FALL 原始标签 → 三分类

    ADL: 全部 → 0
    Fall: -1→0, 0→1, 1→2
    """
    if is_adl:
        return 0
    if csv_label == -1:
        return 0
    if csv_label == 0:
        return 1
    return 2  # csv_label == 1


# ==================================================================
# 单视频处理
# ==================================================================

def process_sequence(
    seq_name: str,
    zip_path: Path,
    seq_labels: dict[int, int],
    is_adl: bool,
    extractor: PoseExtractor,
    feat_builder: FrameFeatureBuilder,
    windower: SlidingWindowBuilder,
) -> tuple[np.ndarray, np.ndarray]:
    """单个 RGB zip → (N,35,29), (N,)"""
    frames: list[np.ndarray] = []
    labels: list[int] = []
    gait = GaitSpeedExtractor()

    with zipfile.ZipFile(zip_path) as zf:
        # 收集 PNG 条目并按帧号排序
        png_entries = []
        for name in zf.namelist():
            if not name.endswith(".png"):
                continue
            # 从文件名提取帧号: "...-XXX.png"
            try:
                fn = int(name.rsplit("-", 1)[-1].replace(".png", ""))
            except ValueError:
                continue
            png_entries.append((fn, name))
        png_entries.sort(key=lambda x: x[0])

        for frame_num, entry_name in png_entries:
            # 标签（先查，detect 失败则跳过整帧）
            csv_label = seq_labels.get(frame_num)
            if csv_label is None:
                continue

            # 读取 PNG → BGR
            data = zf.read(entry_name)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            # Pose 检测（timestamp 用帧号估算 ~30fps → ms）
            ts_ms = frame_num * 33
            result = extractor.detect_frame(img, timestamp_ms=ts_ms)
            if result is None:
                continue

            # 标签 + 特征（detect 成功后同时追加，保持对齐）
            label = map_label(is_adl, csv_label)
            labels.append(label)

            gs = gait.compute_per_frame(result)
            vec = feat_builder.build(result, gs)
            frames.append(vec)

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

    # 加载标签
    print("Loading labels...")
    adl_labels  = load_labels(URFALL_ROOT / "urfall-cam0-adls.csv")
    fall_labels = load_labels(URFALL_ROOT / "urfall-cam0-falls.csv")
    print(f"  ADL:  {len(adl_labels)} sequences")
    print(f"  Fall: {len(fall_labels)} sequences")

    feat_builder = FrameFeatureBuilder()
    windower = SlidingWindowBuilder(WindowConfig(window_size=WINDOW_SIZE, stride=STRIDE))

    videos_dir = OUTPUT_DIR / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # ---- 收集所有序列 ----
    all_seqs = []  # [(seq_id, seq_name, zip_path, is_adl)]
    for seq_name in sorted(adl_labels):
        zp = URFALL_ROOT / f"{seq_name}-cam0-rgb.zip"
        if zp.exists():
            all_seqs.append((seq_name, zp, True))
    for seq_name in sorted(fall_labels):
        zp = URFALL_ROOT / f"{seq_name}-cam0-rgb.zip"
        if zp.exists():
            all_seqs.append((seq_name, zp, False))

    print(f"  Found {len(all_seqs)} zip files\n")

    # ---- 视频级切分 ----
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(all_seqs))
    n_train = int(len(all_seqs) * TRAIN_RATIO)
    train_set = set(indices[:n_train])

    train_X, train_y = [], []
    test_X,  test_y  = [], []
    split_log: dict[str, list[str]] = {"train": [], "test": []}

    # ---- 逐视频处理 ----
    for i, (seq_name, zp, is_adl) in enumerate(all_seqs):
        is_train = i in train_set
        split = "train" if is_train else "test"
        split_log[split].append(seq_name)

        labels_dict = adl_labels if is_adl else fall_labels
        print(f"  [{i+1}/{len(all_seqs)}] {seq_name} ({'ADL' if is_adl else 'Fall'}) ...",
              end=" ", flush=True)

        with PoseExtractor(pose_cfg) as extractor:
            X, y = process_sequence(
                seq_name, zp, labels_dict[seq_name], is_adl,
                extractor, feat_builder, windower,
            )

        # 保存单视频
        safe_name = seq_name.replace("-", "_")
        np.save(str(videos_dir / f"{safe_name}_X.npy"), X)
        np.save(str(videos_dir / f"{safe_name}_y.npy"), y)

        n = X.shape[0]
        uc = {int(lb): int(np.sum(y == lb)) for lb in np.unique(y)} if n > 0 else {}
        print(f"windows={n}, labels={uc} [{split}]")

        if n > 0:
            (train_X if is_train else test_X).append(X)
            (train_y if is_train else test_y).append(y)

    # ---- 合并保存 ----
    def save_combined(prefix, X_list, y_list):
        if X_list:
            Xa = np.concatenate(X_list, axis=0)
            ya = np.concatenate(y_list, axis=0)
        else:
            Xa = np.empty((0, WINDOW_SIZE, 29), dtype=np.float32)
            ya = np.empty((0,), dtype=np.int64)
        np.save(str(OUTPUT_DIR / f"X_{prefix}.npy"), Xa)
        np.save(str(OUTPUT_DIR / f"y_{prefix}.npy"), ya)
        return Xa, ya

    Xt, yt = save_combined("train", train_X, train_y)
    Xe, ye = save_combined("test",  test_X,  test_y)

    # 日志
    (OUTPUT_DIR / "split_log.json").write_text(
        json.dumps(split_log, indent=2, ensure_ascii=False)
    )

    # 摘要
    print(f"\n{'='*60}")
    print(f"UR-FALL build complete.")
    print(f"  Sequences: {len(all_seqs)} ({sum(1 for _,_,a in all_seqs if a)} ADL + {sum(1 for _,_,a in all_seqs if not a)} Fall)")
    print(f"  Split: train={len(split_log['train'])}, test={len(split_log['test'])}")
    print(f"  Seed:  {SEED}")
    print(f"")
    print(f"  X_train: {Xt.shape}, y_train: {yt.shape}")
    for lb, nm in [(0,"Non-fall"),(1,"Pre-impact"),(2,"Fall")]:
        print(f"    {nm}: {int(np.sum(yt==lb))}")
    print(f"")
    print(f"  X_test:  {Xe.shape}, y_test:  {ye.shape}")
    for lb, nm in [(0,"Non-fall"),(1,"Pre-impact"),(2,"Fall")]:
        print(f"    {nm}: {int(np.sum(ye==lb))}")
    print(f"")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
