#!/usr/bin/env python
"""
build_skeleton_yolo.py — 阶段 2(YOLO 版): pose_cache_yolo/*.npz → 13关节2D骨架 → 窗口 → datasets/experiment/skeleton_wXX_coco2d/*

与 build_skeleton.py 完全并行独立：
  - 读 YOLO 2D 缓存（landmarks(T,13,2) 归一化坐标、visibility(T,13)）
  - skeleton_from_arrays_2d → 髋中心归一化到 [-1,1] → 取 COCO13（13 关节）
  - 每帧扁平 (13*2=26) → 窗口 → (N, window, 26)
  - ST-GCN 模型用 num_joints=13, in_channels=2 读取

用法:
    python scripts/build_skeleton_yolo.py                  # 全部 5 数据集，窗口=60
    python scripts/build_skeleton_yolo.py --window 35      # 公平对比窗口
    python scripts/build_skeleton_yolo.py Le2i             # 单个数据集
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.stgcn.graph import (
    COCO13_JOINTS, COCO13_EDGES, COCO13_JOINT_SET_NAME,
    skeleton_from_arrays_2d, NUM_COCO13_JOINTS,
)
from scripts.build_config import (
    STRIDE, TRAIN_RATIO, SEED, RISK_MODE,
    RAMP_LO, RAMP_HI, TAU_U, TAU_ABN,
)
from features.landmark_filter import LandmarkSmoother

CACHE_DIR = PROJECT_ROOT / "pose_cache_yolo"
DATASETS = ["Le2i", "UR-FALL", "CAUCAFall", "UP-FALL", "Pre-VFall"]

NUM_CH = 2  # YOLO 2D: x, y


# ==================================================================
# 窗口/标签对齐（与 build_skeleton.build_windows 同款逻辑）
# ==================================================================

def build_windows(feats: np.ndarray, labels: np.ndarray | None,
                  risk: np.ndarray | None, window_size: int, stride: int):
    """feats: (T, K) → (N, window, K)；标签取窗口最后一帧。"""
    T = feats.shape[0]
    K = feats.shape[1]
    if T < window_size:
        empty = np.empty((0, window_size, K), dtype=np.float32)
        return (empty,
                np.empty((0,), np.int64) if labels is not None else None,
                np.empty((0,), np.float32) if risk is not None else None)
    X, yc, yr = [], [], []
    for start in range(0, T - window_size + 1, stride):
        X.append(feats[start:start + window_size])
        if labels is not None:
            yc.append(int(labels[start + window_size - 1]))
        if risk is not None:
            yr.append(float(risk[start + window_size - 1]))
    X = np.stack(X, axis=0).astype(np.float32)
    yc = np.array(yc, np.int64) if labels is not None else None
    yr = np.array(yr, np.float32) if risk is not None else None
    return X, yc, yr


# ==================================================================
# 单视频: 缓存 → 2D 骨架 → 窗口
# ==================================================================

def process_cached(npz_path: Path, windower_size: int, risk: np.ndarray | None,
                    smoother: "LandmarkSmoother | None" = None):
    data = np.load(npz_path)
    lm2d = data["landmarks"]                     # (T,13,2)
    vis = data["visibility"]                      # (T,13)
    labels = data["labels"]                       # (T,)
    T = lm2d.shape[0]

    # 每帧 13 关节 2D 归一化 (T,13,2) → 扁平 (T,26)
    coords = skeleton_from_arrays_2d(lm2d, vis, COCO13_JOINTS, COCO13_EDGES)  # (T,13,2)
    if smoother is not None and smoother.enabled:
        # 坐标层 OneEuro 时序平滑（与推理侧 demo 同参，保证 train/infer 零漂移）
        coords = smoother.smooth_sequence(coords)                          # (T,13,2)
    feats = coords.reshape(T, -1).astype(np.float32)                       # (T,26)
    labs = np.asarray(labels, dtype=np.int64)

    return build_windows(feats, labs, risk, windower_size, STRIDE)


# ==================================================================
# Main
# ==================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="all",
                        choices=["all"] + DATASETS)
    parser.add_argument("--window", "-w", type=int, default=60,
                        help="窗口帧数 (默认 60)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出根目录 (默认 datasets/experiment/skeleton_w<window>_coco2d)")
    parser.add_argument("--smooth-landmarks", dest="smooth_landmarks",
                        action="store_true", default=True,
                        help="对归一化骨架做 OneEuro 时序平滑（默认开）")
    parser.add_argument("--no-smooth-landmarks", dest="smooth_landmarks",
                        action="store_false",
                        help="关闭坐标滤波")
    parser.add_argument("--min-cutoff", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--dcutoff", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    ws = args.window
    K = NUM_COCO13_JOINTS * NUM_CH   # 13*2 = 26
    if args.output:
        out_name = args.output
    else:
        suffix = "_smoothed" if args.smooth_landmarks else ""
        out_name = f"datasets/experiment/skeleton_w{ws}_coco2d{suffix}"
    output_root = Path(out_name)
    print(f"Task: YOLO 2D skeleton regression, Output: {output_root}, dim={K}, "
          f"window={ws}, stride={STRIDE}")

    smoother = LandmarkSmoother(
        enabled=args.smooth_landmarks,
        min_cutoff=args.min_cutoff, beta=args.beta, dcutoff=args.dcutoff,
        n_landmarks=NUM_COCO13_JOINTS, channels=NUM_CH, fps=args.fps)
    print(f"  Landmark smoothing (OneEuro): "
          f"{'ON' if smoother.enabled else 'OFF'}  "
          f"min_cutoff={args.min_cutoff} beta={args.beta} dcutoff={args.dcutoff}")

    targets = DATASETS if args.dataset == "all" else [args.dataset]
    rng = np.random.RandomState(SEED)

    for ds in targets:
        cache_dir = CACHE_DIR / ds
        if not cache_dir.exists():
            print(f"  [SKIP] {ds}: no cache dir (先跑 build_cache_yolo.py)")
            continue
        npz_files = sorted(cache_dir.glob("*.npz"))
        if not npz_files:
            print(f"  [SKIP] {ds}: no .npz files")
            continue

        out_dir = output_root / ds
        videos_dir = out_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)

        indices = rng.permutation(len(npz_files))
        n_train = int(len(npz_files) * TRAIN_RATIO)
        train_set = set(indices[:n_train])

        train_X, train_y, train_yr = [], [], []
        test_X, test_y, test_yr = [], [], []
        split_log = {"train": [], "test": []}

        print(f"\n{'='*50}\n{ds}: {len(npz_files)} videos  [yolo 2d skeleton]\n{'='*50}")

        for i, npz_path in enumerate(npz_files):
            vid = npz_path.stem
            is_train = i in train_set
            split = "train" if is_train else "test"

            sidecar = npz_path.with_suffix(".risk.npy")
            if not sidecar.exists():
                print(f"  [{i+1}/{len(npz_files)}] {vid} ... [SKIP no risk]")
                continue
            risk_arr = np.load(str(sidecar)).astype(np.float32)

            split_log[split].append(vid)

            print(f"  [{i+1}/{len(npz_files)}] {vid} ...", end=" ", flush=True)
            X, y, yr = process_cached(npz_path, ws, risk_arr, smoother)

            safe = vid.replace(".", "_")
            np.save(str(videos_dir / f"{safe}_X.npy"), X)
            np.save(str(videos_dir / f"{safe}_y_risk.npy"), yr)
            np.save(str(videos_dir / f"{safe}_y_cls.npy"), y)

            n = X.shape[0]
            uc = {int(lb): int(np.sum(y == lb)) for lb in np.unique(y)} if n > 0 else {}
            yr_stats = ""
            if yr is not None and n > 0:
                yr_stats = (f" risk[min={yr.min():.2f},max={yr.max():.2f},"
                            f"mean={yr.mean():.3f}]")
            print(f"windows={n}, labels={uc}{yr_stats} [{split}]")

            if n > 0:
                (train_X if is_train else test_X).append(X)
                (train_y if is_train else test_y).append(y)
                (train_yr if is_train else test_yr).append(yr)

        def save_combined(prefix, Xl, yl, yrl):
            Xa = np.concatenate(Xl) if Xl else np.empty((0, ws, K), dtype=np.float32)
            ya = np.concatenate(yl) if yl else np.empty((0,), np.int64)
            np.save(str(out_dir / f"X_{prefix}.npy"), Xa)
            np.save(str(out_dir / f"y_cls_{prefix}.npy"), ya)
            yra = np.concatenate(yrl) if yrl else np.empty((0,), np.float32)
            np.save(str(out_dir / f"y_risk_{prefix}.npy"), yra)
            return Xa, ya

        Xt, yt = save_combined("train", train_X, train_y, train_yr)
        Xe, ye = save_combined("test", test_X, test_y, test_yr)
        (out_dir / "split_log.json").write_text(json.dumps(split_log, indent=2))

        print(f"  dim={K}, X_train: {Xt.shape}, y_train: {yt.shape}")
        print(f"  X_test:  {Xe.shape}, y_test:  {ye.shape}")
        meta = {
            "task": "regression",
            "modality": "skeleton_2d",
            "joint_set": COCO13_JOINT_SET_NAME,
            "num_joints": NUM_COCO13_JOINTS,
            "input_channel": NUM_CH,
            "window_size": ws,
            "input_dim": K,
            "risk_mode": RISK_MODE,
            "ramp_lo": RAMP_LO, "ramp_hi": RAMP_HI,
            "tau_u": TAU_U, "tau_abn": TAU_ABN,
            "landmark_smoothing": smoother.to_dict(),
            "n_train": int(Xt.shape[0]), "n_test": int(Xe.shape[0]),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print("\nDone.")


if __name__ == "__main__":
    main()
