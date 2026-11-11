#!/usr/bin/env python
"""
build_upfall.py — UP-FALL 数据集 → (N,35,29) X.npy + y.npy

标签: CSV 最后一列 = 动作标签。1~5=失衡(label=1), 之前=正常, 之后=倒地(label=2)

用法:
    python scripts/build_upfall.py
"""
from __future__ import annotations

import json
import re
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

UPFALL_ROOT = Path("D:/my_datasets/UP-FALL")
OUTPUT_DIR  = OUTPUT_ROOT / "UP-FALL"
FALL_LABELS = {"1", "2", "3", "4", "5"}   # CSV 中代表失衡的标签


# ==================================================================
# 标签生成
# ==================================================================

def build_labels(csv_labels: list[str], activity_id: int) -> np.ndarray:
    """CSV 逐帧动作标签 → 三分类 labels

    ADL (activity_id > 5): 全 0
    Fall (activity_id ≤ 5): 找 1~5 区间 → 之前=0, 区间内=1, 之后=2
    """
    n = len(csv_labels)
    labels = np.zeros(n, dtype=np.int64)

    if activity_id > 5:   # ADL
        return labels

    # 找标签 ∈ {1,2,3,4,5} 的区间
    fall_indices = [i for i, l in enumerate(csv_labels) if l in FALL_LABELS]
    if not fall_indices:
        return labels

    first, last = fall_indices[0], fall_indices[-1]

    # Pre-impact: 失衡区间
    labels[first : last + 1] = 1

    # Fall: 失衡之后 → 倒地
    labels[last + 1 :] = 2

    return labels


# ==================================================================
# 单序列处理
# ==================================================================

def process_sequence(
    zip_path: Path,
    csv_labels: list[str],
    labels: np.ndarray,
    extractor: PoseExtractor,
    feat_builder: FrameFeatureBuilder,
    windower: SlidingWindowBuilder,
) -> tuple[np.ndarray, np.ndarray]:
    """单个 zip → (N,35,29), (N,)"""
    frames: list[np.ndarray] = []
    labs:   list[int] = []
    gait = GaitSpeedExtractor()

    with zipfile.ZipFile(zip_path) as zf:
        png_names = sorted([n for n in zf.namelist() if n.endswith(".png")])

        for idx, name in enumerate(png_names):
            if idx >= len(labels):
                break

            data = zf.read(name)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            ts_ms = idx * 33
            result = extractor.detect_frame(img, timestamp_ms=ts_ms)
            if result is None:
                continue

            gs = gait.compute_per_frame(result)
            vec = feat_builder.build(result, gs)
            frames.append(vec)
            labs.append(int(labels[idx]))

    if not frames:
        return (
            np.empty((0, WINDOW_SIZE, 29), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    feats = np.array(frames, dtype=np.float32)
    labs_arr = np.array(labs, dtype=np.int64)
    return windower.build_from_array(feats, labs_arr)


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

    # ---- 收集所有序列 ----
    seqs = []  # [(seq_id, zip_path, activity_id, csv_labels)]
    for zp_path in sorted(UPFALL_ROOT.glob("*Camera1.zip")):
        csv_path = UPFALL_ROOT / zp_path.name.replace("Camera1.zip", ".csv")
        if not csv_path.exists():
            continue

        # 解析 Activity 编号
        m = re.search(r"Activity(\d+)", zp_path.name)
        if not m:
            continue
        activity_id = int(m.group(1))

        # 读取 CSV 最后一列标签
        csv_labels: list[str] = []
        with open(csv_path) as f:
            lines = f.readlines()
            data_started = False
            for line in lines[1:]:  # 跳过表头
                parts = line.strip().split(",")
                # 数据行以时间戳开头
                if parts[0].startswith("201"):
                    csv_labels.append(parts[-1])
                    data_started = True
                elif data_started:
                    break  # 数据行结束

        if not csv_labels:
            continue

        seq_id = zp_path.stem.replace("Camera1", "")
        seqs.append((seq_id, zp_path, activity_id, csv_labels))

    print(f"Found {len(seqs)} sequences\n")

    # ---- 视频级切分 ----
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(seqs))
    n_train = int(len(seqs) * TRAIN_RATIO)
    train_set = set(indices[:n_train])

    train_X, train_y = [], []
    test_X,  test_y  = [], []
    split_log: dict[str, list[str]] = {"train": [], "test": []}

    # ---- 逐序列处理 ----
    for i, (seq_id, zp_path, activity_id, csv_labels) in enumerate(seqs):
        is_train = i in train_set
        split = "train" if is_train else "test"
        split_log[split].append(seq_id)

        # 生成三分类标签
        labels = build_labels(csv_labels, activity_id)
        uc = {int(lb): int(np.sum(labels == lb)) for lb in np.unique(labels)}

        act_type = "Fall" if activity_id <= 5 else "ADL"
        print(f"  [{i+1}/{len(seqs)}] {seq_id} ({act_type}, {len(csv_labels)} frames, labels={uc}) ...",
              end=" ", flush=True)

        with PoseExtractor(pose_cfg) as extractor:
            X, y = process_sequence(zp_path, csv_labels, labels,
                                    extractor, feat_builder, windower)

        safe_name = seq_id.replace(".", "_")
        np.save(str(videos_dir / f"{safe_name}_X.npy"), X)
        np.save(str(videos_dir / f"{safe_name}_y.npy"), y)

        n = X.shape[0]
        wuc = {int(lb): int(np.sum(y == lb)) for lb in np.unique(y)} if n > 0 else {}
        print(f"windows={n}, labels={wuc} [{split}]")

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

    (OUTPUT_DIR / "split_log.json").write_text(
        json.dumps(split_log, indent=2, ensure_ascii=False)
    )

    print(f"\n{'='*60}")
    print(f"UP-FALL build complete.")
    print(f"  Sequences: {len(seqs)}")
    print(f"  Split: train={len(split_log['train'])}, test={len(split_log['test'])}")
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
