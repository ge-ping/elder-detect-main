#!/usr/bin/env python
"""
regen_risk_labels.py — 阶段 1.5: pose_cache/*.npz → <name>.risk.npy (sidecar)

    从已有缓存的逐帧 labels(0/1/2) 反推连续 risk 标签，
    按「指数平滑 ramp 方案 A（v2 统一版）」(model/lstm/risk_evolution.md)：

    RAMP_LO=0.3, RAMP_HI=0.7, TAU_U=12, TAU_ABN=20, HORIZON=25
    u_start = 第一个 labels>=1 的帧
    f_start = 第一个 labels==2 的帧 (None=永不跌倒)
    for t:
        labels==2                              -> 1.0
        0<=(f_start-t)<=HORIZON (f_start存在) -> RAMP_LO + (RAMP_HI-RAMP_LO)*exp(-(f_start-t)/TAU_U)   # ★pre-fall 统一 ramp，不依赖"1"
        f_start为None 且 u_start存在            -> min(RAMP_LO + (RAMP_HI-RAMP_LO)*(1-exp(-(t-u_start)/TAU_ABN)), RAMP_HI)
        u_start存在 且远离跌倒(H外)            -> RAMP_LO  (不稳保底)
        else                                  -> 0.0

★ v2 变更：pre-fall ramp 由 f_start 统一触发（替代原 u_start），
  使 Le2i/UR-FALL/CAUCAFall（cache 仅 0/2、无"1"）也获得 0.3→0.7 过渡带，
  消除「同类临跌倒姿态、不同数据集标成 0 vs 0.3~0.7」的矛盾监督。
★ 只读 npz，绝不回写。risk 存为同目录 sidecar `<stem>.risk.npy`。旧分类管线完全不受影响。

用法:
    python scripts/regen_risk_labels.py            # 全部 5 数据集
    python scripts/regen_risk_labels.py Le2i       # 单个
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_config import (
    RAMP_LO, RAMP_HI, TAU_U, TAU_ABN, RISK_HORIZON,
)

CACHE_DIR = PROJECT_ROOT / "pose_cache"
DATASETS = ["Le2i", "UR-FALL", "CAUCAFall", "UP-FALL", "Pre-VFall"]


def compute_risk(
    labels: np.ndarray,
    ramp_lo: float = RAMP_LO,
    ramp_hi: float = RAMP_HI,
    tau_u: float = TAU_U,
    tau_abn: float = TAU_ABN,
    horizon: int = RISK_HORIZON,
) -> np.ndarray:
    """逐帧 labels(0/1/2) → 连续 risk(0~1)，与帧数同长。

    方案 A（指数平滑 ramp，v2 统一版）：
      - 跌倒帧 (labels==2)                              = 1.0
      - 跌倒前 H 窗口内（无论是否有"1"标注）          = ramp_lo + (ramp_hi-ramp_lo)*exp(-(f_start-t)/tau_u)
      - 异常永不跌倒段 (仅 u_start、无 f_start)        = min(ramp_lo + (ramp_hi-ramp_lo)*(1-exp(-(t-u_start)/tau_abn)), ramp_hi)
      - 不稳但远在跌倒前 H 窗口外 (仅 u_start)        = ramp_lo  (保底)
      - 正常段                                         = 0.0

    ★ v2 关键变更：pre-fall ramp 由 f_start 统一触发（不再依赖 u_start），
      使 Le2i/UR-FALL/CAUCAFall（cache 仅 0/2、无"1"）也获得 0.3→0.7 过渡带，
      消除「同类临跌倒姿态、不同数据集标成 0 vs 0.3~0.7」的矛盾监督。
    """
    labels = np.asarray(labels).astype(np.int64)
    T = len(labels)
    risk = np.zeros(T, dtype=np.float32)

    u_start = next((i for i, v in enumerate(labels) if v >= 1), None)
    f_start = next((i for i, v in enumerate(labels) if v == 2), None)

    span = ramp_hi - ramp_lo
    for t in range(T):
        if labels[t] == 2:
            risk[t] = 1.0
        elif f_start is not None and 0 <= (f_start - t) <= horizon:
            # pre-fall window: ramp 0.3→0.7，与是否有"1"标注无关
            d = f_start - t
            risk[t] = ramp_lo + span * float(np.exp(-d / tau_u))
        elif f_start is None and u_start is not None and t >= u_start:
            # abnormal but never falls
            r = ramp_lo + span * (1.0 - float(np.exp(-(t - u_start) / tau_abn)))
            risk[t] = min(r, ramp_hi)
        elif u_start is not None and t >= u_start:
            # unstable but outside pre-fall window (far before fall) → floor
            risk[t] = ramp_lo
        else:
            risk[t] = 0.0
    return risk


def process_npz(npz_path: Path, horizon: int = RISK_HORIZON) -> dict:
    """读一个缓存 npz，算 risk，写 sidecar，返回统计。

    horizon: pre-fall ramp 预告窗(帧)，默认 RISK_HORIZON；
             可由 CLI --horizon 覆盖，改窗口无需改源码。
    """
    sidecar = npz_path.with_suffix(".risk.npy")
    data = np.load(npz_path, allow_pickle=False)
    if "labels" not in data.files:
        return {"status": "skip", "reason": "no labels"}
    labels = data["labels"]
    risk = compute_risk(labels, horizon=horizon)

    # 统计各带占比
    n = len(risk)
    n_zero = int(np.sum(risk < 0.3))
    n_mid = int(np.sum((risk >= 0.3) & (risk < 0.6)))
    n_high = int(np.sum(risk >= 0.6))

    np.save(str(sidecar), risk)
    return {
        "status": "ok",
        "T": n,
        "risk_stats": f"[0,0.3)={n_zero}  [0.3,0.6)={n_mid}  [0.6,1]={n_high}",
        "risk_min": float(risk.min()),
        "risk_max": float(risk.max()),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Regenerate risk sidecar labels from cache")
    parser.add_argument("dataset", nargs="?", default="all",
                        choices=["all"] + DATASETS)
    parser.add_argument("--horizon", "-H", type=int, default=RISK_HORIZON,
                        help=f"pre-fall ramp 预告窗(帧)，默认 {RISK_HORIZON}；"
                             f"覆盖 build_config.RISK_HORIZON，改窗口无需改源码")
    args = parser.parse_args()

    targets = DATASETS if args.dataset == "all" else [args.dataset]

    print(f"RISK params: RAMP_LO={RAMP_LO}, RAMP_HI={RAMP_HI}, "
          f"TAU_U={TAU_U}, TAU_ABN={TAU_ABN}, HORIZON={args.horizon}")
    print(f"Cache dir: {CACHE_DIR}\n")

    total_ok = 0
    total_skip = 0
    for ds in targets:
        cache_dir = CACHE_DIR / ds
        if not cache_dir.exists():
            print(f"  [SKIP] {ds}: no cache dir")
            continue
        npz_files = sorted(cache_dir.glob("*.npz"))
        if not npz_files:
            print(f"  [SKIP] {ds}: no .npz files")
            continue
        print(f"\n{'=' * 50}")
        print(f"{ds}: {len(npz_files)} videos")
        print(f"{'=' * 50}")
        for i, npz_path in enumerate(npz_files):
            res = process_npz(npz_path, horizon=args.horizon)
            if res["status"] == "ok":
                total_ok += 1
                print(f"  [{i+1}/{len(npz_files)}] {npz_path.stem}: "
                      f"T={res['T']}  {res['risk_stats']}")
            else:
                total_skip += 1
                print(f"  [{i+1}/{len(npz_files)}] {npz_path.stem}: SKIP({res['reason']})")

    print(f"\nDone. risk sidecar 生成: {total_ok}, skip: {total_skip}.")


if __name__ == "__main__":
    main()
