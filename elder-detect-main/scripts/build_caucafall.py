#!/usr/bin/env python
"""
build_caucafall.py — CAUCAFall 数据集 → (N,35,29) X.npy + y.npy

标签: 从第一个 action_id="1" 往前推 35 帧为 Pre-impact (label=1)

用法:
    python scripts/build_caucafall.py
"""
from __future__ import annotations

import json
import sys
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

CAUCA_ROOT  = Path("D:/my_datasets/CAUCAFall/Dataset CAUCAFall/Dataset CAUCAFall/CAUCAFall")
PRE_WINDOW  = 35       # 从第一个 "1" 往前推的帧数
OUTPUT_DIR  = OUTPUT_ROOT / "CAUCAFall"


# ==================================================================
# 标签生成
# ==================================================================

def build_labels(action_ids: list[str]) -> np.ndarray:
    """action_ids → labels ∈ {0,1,2}

    规则:
      无 "1" → 全 0
      有 "1" → 首个 "1" 往前 35 帧 = 1, 首个 "1" 及之后 = 2, 其余 = 0
    """
    n = len(action_ids)
    labels = np.zeros(n, dtype=np.int64)

    # 找第一个 "1"
    first_fall = None
    for i, a in enumerate(action_ids):
        if a == "1":
            first_fall = i
            break

    if first_fall is None:
        return labels  # 全 0

    # Pre-impact: first_fall 往前 PRE_WINDOW 帧
    pre_start = max(0, first_fall - PRE_WINDOW)
    labels[pre_start:first_fall] = 1

    # Fall: first_fall 及之后
    for i in range(first_fall, n):
        if action_ids[i] == "1":
            labels[i] = 2

    return labels


# ==================================================================
# 单序列处理
# ==================================================================

def process_sequence(
    png_txt_pairs: list[tuple[Path, Path]],    # [(png_path, txt_path)] 按帧号排序
    labels: np.ndarray,                         # 预生成的逐帧标签
    extractor: PoseExtractor,
    feat_builder: FrameFeatureBuilder,
    windower: SlidingWindowBuilder,
) -> tuple[np.ndarray, np.ndarray]:
    """处理一个动作序列 → (N,35,29), (N,)"""
    frames: list[np.ndarray] = []
    labs:   list[int] = []
    gait = GaitSpeedExtractor()

    for idx, (png_path, _) in enumerate(png_txt_pairs):
        img = cv2.imread(str(png_path))
        if img is None:
            continue

        ts_ms = idx * 33  # ~30fps
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
    seqs = []  # [(seq_id, [(png, txt)])]
    for subj in sorted(CAUCA_ROOT.iterdir()):
        if not subj.is_dir(): continue
        for act in sorted(subj.iterdir()):
            if not act.is_dir(): continue
            # 收集 PNG/TXT 对
            pairs = []
            for png in sorted(act.glob("*.png")):
                txt = png.with_suffix(".txt")
                if txt.exists():
                    # 从文件名提取帧号用于排序
                    try:
                        fn = int(png.stem.split("-")[-1])
                    except ValueError:
                        fn = int(png.stem.rsplit("-", 1)[-1]) if "-" in png.stem else 0
                    pairs.append((fn, png, txt))
            if not pairs:
                continue
            pairs.sort(key=lambda x: x[0])
            seq_id = f"{subj.name}_{act.name}".replace(" ", "_")
            seqs.append((seq_id, [(p, t) for _, p, t in pairs]))

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
    for i, (seq_id, pairs) in enumerate(seqs):
        is_train = i in train_set
        split = "train" if is_train else "test"
        split_log[split].append(seq_id)

        # Pass 1: 读取所有 action_id → 生成 labels
        action_ids = []
        for _, txt_path in pairs:
            line = txt_path.read_text().strip()
            action_ids.append(line.split()[0] if line else "nofall")
        labels = build_labels(action_ids)

        uc = {int(lb): int(np.sum(labels == lb)) for lb in np.unique(labels)}
        print(f"  [{i+1}/{len(seqs)}] {seq_id} ({len(pairs)} frames, labels={uc}) ...",
              end=" ", flush=True)

        # Pass 2: MediaPipe + 特征
        with PoseExtractor(pose_cfg) as extractor:
            X, y = process_sequence(pairs, labels, extractor, feat_builder, windower)

        # 保存单序列
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
    print(f"CAUCAFall build complete.")
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
