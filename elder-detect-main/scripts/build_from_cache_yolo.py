#!/usr/bin/env python
"""
build_from_cache_yolo.py — 阶段 2(YOLO 版): pose_cache_yolo/*.npz → 2D特征 → 窗口 → datasets/experiment/risk_wXX_coco2d/*

与 build_from_cache.py 完全并行独立：
  - 读 YOLO 2D 缓存（landmarks(T,13,2) 归一化坐标、visibility(T,13)）
  - features/feature_builder_2d.build_frame_features 算 29 维 2D 特征
    （gait_speed_2d / trunk_angle_2d x-y平面 / aspect_ratio / 13×2(x,y)）
  - 2D 坐标 OneEuro 时序平滑（与推理侧 demo 同参，零漂移）
  - 窗口/标签对齐逻辑与 LSTM 路径一致 → 训练数据 X 形状 (N, window, 29)，dim=29 不变

用法:
    python scripts/build_from_cache_yolo.py                # 全部，窗口=60
    python scripts/build_from_cache_yolo.py --window 35     # 公平对比窗口
    python scripts/build_from_cache_yolo.py Le2i            # 单个数据集
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.feature_builder_2d import build_frame_features, compute_gait_speed_2d
from features.landmark_filter import LandmarkSmoother
from sequence import WindowConfig, SlidingWindowBuilder
from scripts.build_config import (
    STRIDE, TRAIN_RATIO, SEED, RISK_MODE,
    RAMP_LO, RAMP_HI, TAU_U, TAU_ABN,
)

CACHE_DIR = PROJECT_ROOT / "pose_cache_yolo"
DATASETS = ["Le2i", "UR-FALL", "CAUCAFall", "UP-FALL", "Pre-VFall"]
DIM = 29  # 2D 特征布局与 FrameFeatureBuilder 同维


# ==================================================================
# 单视频: 缓存 → 2D 特征 → 窗口
# ==================================================================

def process_cached(npz_path: Path, windower, risk: np.ndarray | None,
                   smoother: "LandmarkSmoother | None" = None, fps: float = 25.0):
    data = np.load(npz_path)
    lm2d = data["landmarks"]                       # (T,13,2)
    vis = data["visibility"]                       # (T,13)
    labels = data["labels"]                        # (T,)
    T = lm2d.shape[0]

    # 坐标层 OneEuro 时序平滑（2D 双通道，压 YOLO 抖动，与推理侧 demo 同参保证零漂移）
    if smoother is not None and smoother.enabled:
        lm2d = smoother.smooth_sequence(lm2d)      # (T,13,2)

    dt = 1.0 / fps
    gait_2d = compute_gait_speed_2d(lm2d, dt)      # (T,)

    frames, labs = [], []
    for t in range(T):
        # 低可见度关节用邻域均值兜底（与 graph._fill_invalid 同源），避免 NaN 泄漏
        frame_vis = vis[t]
        if np.any(frame_vis < 0.6):
            lm_frame = lm2d[t].copy()
            for i in range(lm_frame.shape[0]):
                if frame_vis[i] < 0.6:
                    neigh = [j for j in range(lm_frame.shape[0]) if j != i and frame_vis[j] >= 0.6]
                    if neigh:
                        lm_frame[i] = lm_frame[neigh].mean(axis=0)
                    else:
                        lm_frame[i] = 0.0
        else:
            lm_frame = lm2d[t]
        vec = build_frame_features(lm_frame, gait_2d[t])
        frames.append(vec)
        labs.append(int(labels[t]))

    feats = np.array(frames, dtype=np.float32)
    labs_arr = np.array(labs, dtype=np.int64)
    return windower.build_from_array(feats, labs_arr, risk=risk)


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
                        help="输出根目录 (默认 datasets/experiment/risk_w<window>_coco2d)")
    parser.add_argument("--smooth-landmarks", dest="smooth_landmarks",
                        action="store_true", default=True,
                        help="对 2D 坐标做 OneEuro 时序平滑（默认开）")
    parser.add_argument("--no-smooth-landmarks", dest="smooth_landmarks",
                        action="store_false",
                        help="关闭坐标滤波")
    parser.add_argument("--min-cutoff", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--dcutoff", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    ws = args.window
    if args.output:
        out_name = args.output
    else:
        suffix = "_smoothed" if args.smooth_landmarks else ""
        out_name = f"datasets/experiment/risk_w{ws}_coco2d{suffix}"
    output_root = Path(out_name)
    print(f"Task: YOLO 2D feature regression, Output: {output_root}, dim={DIM}, window={ws}")

    smoother = LandmarkSmoother(
        enabled=args.smooth_landmarks,
        min_cutoff=args.min_cutoff, beta=args.beta, dcutoff=args.dcutoff,
        n_landmarks=13, channels=2, fps=args.fps)
    print(f"  Landmark smoothing (OneEuro 2D): "
          f"{'ON' if smoother.enabled else 'OFF'}  "
          f"min_cutoff={args.min_cutoff} beta={args.beta} dcutoff={args.dcutoff}")
    windower = SlidingWindowBuilder(WindowConfig(window_size=ws, stride=STRIDE))
    print(f"Window: {ws}, stride: {STRIDE}")

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

        print(f"\n{'='*50}\n{ds}: {len(npz_files)} videos  [yolo 2d feature]\n{'='*50}")

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
            X, y, yr = process_cached(npz_path, windower, risk_arr, smoother, args.fps)

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
            Xa = np.concatenate(Xl) if Xl else np.empty((0, ws, DIM), dtype=np.float32)
            ya = np.concatenate(yl) if yl else np.empty((0,), np.int64)
            np.save(str(out_dir / f"X_{prefix}.npy"), Xa)
            np.save(str(out_dir / f"y_cls_{prefix}.npy"), ya)
            yra = np.concatenate(yrl) if yrl else np.empty((0,), np.float32)
            np.save(str(out_dir / f"y_risk_{prefix}.npy"), yra)
            return Xa, ya

        Xt, yt = save_combined("train", train_X, train_y, train_yr)
        Xe, ye = save_combined("test", test_X, test_y, test_yr)
        (out_dir / "split_log.json").write_text(json.dumps(split_log, indent=2))

        print(f"  dim={DIM}, X_train: {Xt.shape}, y_train: {yt.shape}")
        print(f"  X_test:  {Xe.shape}, y_test:  {ye.shape}")
        yr_t = np.concatenate(train_yr) if train_yr else np.empty((0,))
        yr_e = np.concatenate(test_yr) if test_yr else np.empty((0,))
        print(f"  y_risk_train: {yr_t.shape}, y_risk_test: {yr_e.shape}")
        meta = {
            "task": "regression",
            "modality": "skeleton_2d_feature",
            "joint_set": "coco13",
            "input_dim": DIM,
            "window_size": ws,
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
