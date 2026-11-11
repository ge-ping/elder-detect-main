#!/usr/bin/env python
"""
build_dataset.py — 视频数据集 → (N,35,29) X.npy + y.npy

用法:
    python scripts/build_dataset.py \
        --dataset_dir ./dataset \
        --output_dir ./data \
        --pose_config config/pose_config.yaml

数据集目录结构:
    dataset/
    ├── normal/     ← label=0
    │   ├── a.mp4
    │   └── b.mp4
    └── fall/       ← label=1
        ├── c.mp4
        └── d.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from pose import load_pose_config
from datasets import DatasetBuilder


def main():
    parser = argparse.ArgumentParser(description="构建 LSTM 训练数据集")
    parser.add_argument(
        "--dataset_dir", required=True,
        help="数据集根目录（含 normal/ fall/ 子文件夹）",
    )
    parser.add_argument(
        "--output_dir", default="data",
        help="输出目录（默认 data/）",
    )
    parser.add_argument(
        "--pose_config", default="config/pose_config.yaml",
        help="PoseExtractor YAML 配置",
    )
    parser.add_argument(
        "--window_size", type=int, default=35,
        help="滑动窗口帧数（默认 35）",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="窗口步长（默认 1）",
    )
    args = parser.parse_args()

    # 加载 Pose 配置
    pose_cfg = load_pose_config(args.pose_config)
    print(f"Pose model: {pose_cfg.model_path}")

    # 构建
    builder = DatasetBuilder(
        pose_config=pose_cfg,
        window_size=args.window_size,
        stride=args.stride,
    )

    X, y = builder.build_all(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
    )

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"Build complete.")
    print(f"  X.shape = {X.shape}  (N, {args.window_size}, 29)")
    print(f"  y.shape = {y.shape}")
    print(f"  X.dtype = {X.dtype}")
    print(f"  y.dtype = {y.dtype}")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        label_name = {0: "normal", 1: "fall"}.get(int(u), f"class_{u}")
        print(f"  label={u} ({label_name}): {c} samples")
    print(f"{'='*60}")

    # 随机样本预览
    if X.shape[0] > 0:
        idx = np.random.randint(0, X.shape[0])
        sample = X[idx]  # (35, 29)
        sample_label = y[idx]
        print(f"\nRandom sample[{idx}] label={sample_label}:")
        print(f"  frame 0:  gait={sample[0,0]:.2f}  angle={sample[0,1]:.1f}°  Rf={sample[0,2]:.3f}")
        print(f"  frame 17: gait={sample[17,0]:.2f}  angle={sample[17,1]:.1f}°  Rf={sample[17,2]:.3f}")
        print(f"  frame 34: gait={sample[34,0]:.2f}  angle={sample[34,1]:.1f}°  Rf={sample[34,2]:.3f}  <- last, label={sample_label}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
