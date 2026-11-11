#!/usr/bin/env python
"""
build_skeleton.py — 阶段 2(ST-GCN 版): pose_cache/*.npz → 17关节3D骨架 → 窗口 → datasets/experiment/skeleton_wXX/*

与 build_from_cache.py 完全并行独立：
  - 不依赖 29 维手工特征 / FrameFeatureBuilder / GaitSpeedExtractor
  - 直接从 world_landmarks 取 3D 世界坐标 → 髋中心归一化到 [-1,1] → 取 FALL_JOINTS_17
  - 窗口/标签对齐逻辑与 LSTM 路径一致（仅把「每帧 29 维」换成「每帧 17×3 扁平」）
    → 训练数据 X 形状 (N, window, 51)，A/B 绝对公平

用法:
    python scripts/build_skeleton.py                 # 全部 5 数据集，窗口=STGCN_WINDOW(35)
    python scripts/build_skeleton.py --window 25     # 公平对比窗口
    python scripts/build_skeleton.py Le2i            # 单个数据集
    python scripts/build_skeleton.py --output datasets/experiment/skeleton_w25
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pose.keypoint import BodyPart
from model.stgcn.graph import (
    FALL_JOINTS_17, FALL_EDGES, FALL_JOINT_SET_NAME,
    skeleton_from_arrays, NUM_FALL_JOINTS,
)
from scripts.build_config import (
    STRIDE, TRAIN_RATIO, SEED, OUTPUT_ROOT,
    RAMP_LO, RAMP_HI, TAU_U, TAU_ABN, RISK_MODE,
    STGCN_WINDOW, STGCN_INPUT_CHANNEL, STGCN_JOINT_SET, STGCN_NUM_JOINTS,
)
from features.landmark_filter import LandmarkSmoother

CACHE_DIR = PROJECT_ROOT / "pose_cache"
DATASETS = ["Le2i", "UR-FALL", "CAUCAFall", "UP-FALL", "Pre-VFall"]


# ==================================================================
# 窗口/标签对齐（与 build_from_cache.build_from_array 同款逻辑）
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
# 单视频: 缓存 → 骨架 → 窗口
# ==================================================================

def process_cached(npz_path: Path, windower_size: int, risk: np.ndarray | None,
                    smoother: "LandmarkSmoother | None" = None):
    data = np.load(npz_path)
    world = data["world_landmarks"]          # (T,33,3)
    vis = data["visibility"]                  # (T,33)
    labels = data["labels"]                   # (T,)
    T = world.shape[0]

    # 每帧 17 关节 3D 归一化 (T,17,3) → 扁平 (T,51)
    coords = skeleton_from_arrays(world, vis, FALL_JOINTS_17, FALL_EDGES)  # (T,17,3)
    if smoother is not None and smoother.enabled:
        # 坐标层 OneEuro 时序平滑：压 mediapipe 抖动，减少中间带误判
        coords = smoother.smooth_sequence(coords)                          # (T,17,3)
    feats = coords.reshape(T, -1).astype(np.float32)                       # (T,51)
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
    parser.add_argument("--window", "-w", type=int, default=None,
                        help=f"窗口帧数 (默认 {STGCN_WINDOW})")
    parser.add_argument("--output", type=str, default=None,
                        help="输出根目录 (默认 datasets/experiment/skeleton_w<window>[_smoothed])")
    # 坐标层 OneEuro 时序平滑（默认开，压 mediapipe 抖动；训练/推理双侧同步保证零漂移）
    parser.add_argument("--smooth-landmarks", dest="smooth_landmarks",
                        action="store_true", default=True,
                        help="对归一化骨架做 OneEuro 时序平滑（压坐标抖动，防误报）；默认开")
    parser.add_argument("--no-smooth-landmarks", dest="smooth_landmarks",
                        action="store_false",
                        help="关闭坐标滤波（生成原始未滤波数据集，与旧 S1/S2 一致）")
    parser.add_argument("--min-cutoff", type=float, default=1.0,
                        help="OneEuro min_cutoff：越大越平滑（默认1.0）")
    parser.add_argument("--beta", type=float, default=0.02,
                        help="OneEuro beta：对快速运动的自适应平滑强度（默认0.02）")
    parser.add_argument("--dcutoff", type=float, default=1.0,
                        help="OneEuro dcutoff：导数低通截止（默认1.0）")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="坐标滤波时间步假设（默认25fps；cutoff 单位 Hz，必须与推理一致）")
    args = parser.parse_args()

    ws = args.window or STGCN_WINDOW
    K = NUM_FALL_JOINTS * STGCN_INPUT_CHANNEL   # 17*3 = 51
    if args.output:
        out_name = args.output
    else:
        suffix = "_smoothed" if args.smooth_landmarks else ""
        out_name = f"datasets/experiment/skeleton_w{ws}{suffix}"
    output_root = Path(out_name)
    print(f"Task: skeleton regression, Output: {output_root}, dim={K}, window={ws}, stride={STRIDE}")

    # 坐标层 OneEuro 平滑器（与推理侧 demo 同参，保证 train/infer 零漂移）
    smoother = LandmarkSmoother(
        enabled=args.smooth_landmarks,
        min_cutoff=args.min_cutoff, beta=args.beta, dcutoff=args.dcutoff,
        n_landmarks=NUM_FALL_JOINTS, channels=STGCN_INPUT_CHANNEL, fps=args.fps)
    print(f"  Landmark smoothing (OneEuro): "
          f"{'ON' if smoother.enabled else 'OFF'}  "
          f"min_cutoff={args.min_cutoff} beta={args.beta} dcutoff={args.dcutoff}")

    targets = DATASETS if args.dataset == "all" else [args.dataset]
    rng = np.random.RandomState(SEED)

    for ds in targets:
        cache_dir = CACHE_DIR / ds
        if not cache_dir.exists():
            print(f"  [SKIP] {ds}: no cache dir")
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

        print(f"\n{'='*50}\n{ds}: {len(npz_files)} videos  [skeleton]\n{'='*50}")

        for i, npz_path in enumerate(npz_files):
            vid = npz_path.stem
            is_train = i in train_set
            split = "train" if is_train else "test"

            # 回归：读 risk sidecar
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
        yr_t = np.concatenate(train_yr) if train_yr else np.empty((0,))
        yr_e = np.concatenate(test_yr) if test_yr else np.empty((0,))
        print(f"  y_risk_train: {yr_t.shape}, y_risk_test: {yr_e.shape}")
        meta = {
            "task": "regression",
            "modality": "skeleton_3d",
            "joint_set": STGCN_JOINT_SET,
            "num_joints": STGCN_NUM_JOINTS,
            "input_channel": STGCN_INPUT_CHANNEL,
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
