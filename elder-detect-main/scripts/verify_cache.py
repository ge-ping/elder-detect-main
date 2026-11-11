#!/usr/bin/env python
"""
verify_cache.py — 对比缓存标签 vs 原始构建，不覆盖任何文件

验证逻辑:
  逐视频: cache 逐帧标签数 - 34 = 原始窗口数 → OK/MISMATCH
  合并: 原始 X_train/X_test shape 作为基准

用法:
    python scripts/verify_cache.py UR-FALL
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR  = PROJECT_ROOT / "pose_cache"
OUTPUT_DIR = PROJECT_ROOT / "datasets"


def verify(ds: str):
    cache_dir = CACHE_DIR / ds
    videos_dir = OUTPUT_DIR / ds / "videos"

    if not cache_dir.exists():
        print(f"no cache at {cache_dir}"); return
    if not videos_dir.exists():
        print(f"no original at {videos_dir}"); return

    npz_files = sorted(cache_dir.glob("*.npz"))
    print(f"\n{ds}: {len(npz_files)} cached videos")
    print(f"{'video':<45s} {'cache frames':>12} {'→windows':>10} {'orig win':>10} {' ':>6}")
    print("-" * 88)

    ok, mismatch, missing = 0, 0, 0
    for npz_path in npz_files:
        vid = npz_path.stem
        y_orig = videos_dir / f"{vid}_y.npy"
        if not y_orig.exists():
            y_orig = videos_dir / f"{vid.replace('-', '_')}_y.npy"

        data = np.load(npz_path)
        n_frames = data["labels"].shape[0]
        n_windows = max(0, n_frames - 34)

        if not y_orig.exists():
            print(f"{vid:<45s} {n_frames:>12} {n_windows:>10} {'N/A':>10} {'SKIP':>6}")
            missing += 1
            continue

        orig_n = np.load(str(y_orig)).shape[0]
        m = "OK" if n_windows == orig_n else f"MISMATCH({orig_n - n_windows:+d})"
        if n_windows == orig_n:
            ok += 1
        else:
            mismatch += 1
        print(f"{vid:<45s} {n_frames:>12} {n_windows:>10} {orig_n:>10} {m:>6}")

    print(f"\n  OK: {ok}, MISMATCH: {mismatch}, MISSING: {missing}")

    # 原始合并文件（基准）
    print(f"\n  Original merged (baseline):")
    for split in ["train", "test"]:
        xp = OUTPUT_DIR / ds / f"X_{split}.npy"
        if xp.exists():
            X = np.load(str(xp))
            y = np.load(str(OUTPUT_DIR / ds / f"y_{split}.npy"))
            bc = np.bincount(y)
            print(f"    {split}: X={X.shape}, y={y.shape}, labels(N/U/F)={bc.tolist()}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "UR-FALL"
    verify(ds)
