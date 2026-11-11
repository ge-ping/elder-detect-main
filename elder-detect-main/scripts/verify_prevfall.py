#!/usr/bin/env python
"""验证 Pre-VFall 缓存正确性"""
from pathlib import Path
from collections import Counter
import numpy as np

cache_dir = Path("pose_cache/Pre-VFall")
npz_files = sorted(cache_dir.glob("*.npz"))

if not npz_files:
    print("No cache files!"); exit(1)

total_detections, seqs_ok = 0, 0
label_counts = Counter()
sample_lm = None

for npz_path in npz_files:
    data = np.load(npz_path)
    n = data["landmarks"].shape[0]
    total_detections += n
    for lb in data["labels"]:
        label_counts[int(lb)] += 1
    if n > 0:
        seqs_ok += 1
        if sample_lm is None:
            sample_lm = data["landmarks"][0]

print(f"Sequences:        {len(npz_files)}")
print(f"  with persons:   {seqs_ok}")
print(f"  empty:          {len(npz_files) - seqs_ok}")
print(f"Total detections: {total_detections}")
print()

# 标签分布 vs CSV 原始分布
csv_expected = {0: 11754, 1: 8469, 2: 2334}
print(f"{'Label':>8} {'CSV(expected)':>14} {'Cache(actual)':>14} {'Match':>6}")
print("-" * 45)
for lb in [0, 1, 2]:
    exp = csv_expected[lb]
    act = label_counts.get(lb, 0)
    ok = "OK" if abs(exp - act) < exp * 0.1 else "MISMATCH"  # 10% tolerance for MediaPipe misses
    print(f"{lb:8d} {exp:14d} {act:14d} {ok:>6}")

print()
# 坐标合法性
if sample_lm is not None:
    print(f"Sample landmarks (33,3): x∈[{sample_lm[:,:,0].min():.3f},{sample_lm[:,:,0].max():.3f}]")
    print(f"                           y∈[{sample_lm[:,:,1].min():.3f},{sample_lm[:,:,1].max():.3f}]")
    print(f"  (expected: x,y ∈ [0,1] normalized)")

print("\nDone.")
