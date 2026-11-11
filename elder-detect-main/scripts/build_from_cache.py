#!/usr/bin/env python
"""
build_from_cache.py — 阶段 2: pose_cache/*.npz → 特征 → 窗口 → datasets/*/X.npy, y.npy

用法:
    python scripts/build_from_cache.py            # 全部
    python scripts/build_from_cache.py Le2i       # 单个
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pose.keypoint import Keypoint, PoseResult, BodyPart
from features.feature_builder import FrameFeatureBuilder
from features.gait import GaitSpeedExtractor
from features.landmark_filter import LandmarkSmoother
from sequence import WindowConfig, SlidingWindowBuilder
from scripts.build_config import (
    WINDOW_SIZE, STRIDE, TRAIN_RATIO, SEED, OUTPUT_ROOT,
    RAMP_LO, RAMP_HI, TAU_U, TAU_ABN, RISK_MODE,
)

CACHE_DIR = PROJECT_ROOT / "pose_cache"
DATASETS  = ["Le2i", "UR-FALL", "CAUCAFall", "UP-FALL", "Pre-VFall"]

# V2 可选: 从 sequence/velocity 导入
try:
    from sequence.velocity import add_velocity, add_acceleration
    _HAS_VELOCITY = True
except ImportError:
    _HAS_VELOCITY = False
    def add_velocity(x): return x
    def add_acceleration(x): return x


# ==================================================================
# 从缓存的 ndarray 重建 PoseResult
# ==================================================================

def _make_pose_result(lm33, wl33, vis33, ts, fw, fh):
    """单帧 ndarray → PoseResult"""
    landmarks = [
        Keypoint(x=float(lm33[i,0]), y=float(lm33[i,1]), z=float(lm33[i,2]),
                 visibility=float(vis33[i]), name=BodyPart(i))
        for i in range(33)
    ]
    world = [
        Keypoint(x=float(wl33[i,0]), y=float(wl33[i,1]), z=float(wl33[i,2]),
                 visibility=float(vis33[i]), name=BodyPart(i))
        for i in range(33)
    ]
    return PoseResult(landmarks=landmarks, world_landmarks=world,
                      timestamp_ms=int(ts), frame_width=int(fw), frame_height=int(fh))


# ==================================================================
# 单视频: 缓存 → 特征 → 窗口
# ==================================================================

def process_cached(npz_path: Path, feat_builder, windower,
                   use_velocity: bool = False, use_acceleration: bool = False,
                   risk: np.ndarray | None = None,
                   smoother2d=None, smoother3d=None):
    data = np.load(npz_path)
    T = data["landmarks"].shape[0]
    fw, fh = int(data["frame_width"]), int(data["frame_height"])
    labels = data["labels"]

    # ---- 坐标层 OneEuro 时序平滑（与 ST-GCN 同源，压 mediapipe 抖动防误报）----
    # LSTM 的 29 维特征混合了 2D image 坐标(x,y) 与 3D world 坐标：
    #   - skeleton 26 维来自 2D landmarks(x,y)
    #   - trunk_angle / gait_speed 来自 3D world_landmarks
    # 故 2D/3D 两套坐标都需平滑，否则 2D 骨架抖动仍泄漏进 26 维特征。
    lm = data["landmarks"]
    wl = data["world_landmarks"]
    if smoother2d is not None and smoother2d.enabled:
        xy = smoother2d.smooth_sequence(lm[..., :2])            # (T,33,2)
        lm = np.concatenate([xy, lm[..., 2:3]], axis=-1)        # 保留原 z 通道
    if smoother3d is not None and smoother3d.enabled:
        wl = smoother3d.smooth_sequence(wl)                     # (T,33,3)

    frames, labs = [], []
    gait = GaitSpeedExtractor()
    for t in range(T):
        result = _make_pose_result(
            lm[t], wl[t],
            data["visibility"][t], data["timestamps"][t], fw, fh)
        gs = gait.compute_per_frame(result)
        vec = feat_builder.build(result, gs)
        frames.append(vec)
        labs.append(int(labels[t]))

    feats = np.array(frames, dtype=np.float32)
    if use_velocity:
        feats = add_velocity(feats)
    if use_acceleration:
        feats = add_acceleration(feats)
    labs_arr = np.array(labs, dtype=np.int64)
    # 返回 (X, y_cls, y_risk)：risk 为 None 时 y_risk=None，原分类流程不变
    return windower.build_from_array(feats, labs_arr, risk=risk)


# ==================================================================
# Main
# ==================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="all",
                        choices=["all"] + DATASETS)
    parser.add_argument("--version", "-v", type=str, default=None,
                        help="快捷: v1|v2|v2.2|v3 (自动设置 output + 特征)")
    parser.add_argument("--use_velocity", action="store_true")
    parser.add_argument("--use_acceleration", action="store_true")
    parser.add_argument("--use_angles", action="store_true",
                        help="追加关节角度 (+7维)")
    parser.add_argument("--window_size", "-w", type=int, default=None,
                        help="窗口帧数 (默认 35, V5实验用)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出根目录 (覆盖 --version 的默认值)")
    parser.add_argument("--task", type=str, default="classification",
                        choices=["classification", "regression"],
                        help="classification=原3类; regression=连续risk(读<name>.risk.npy sidecar)")
    # 坐标层 OneEuro 时序平滑（默认开，压 mediapipe 抖动；训练/推理双侧同步保证零漂移）
    parser.add_argument("--smooth-landmarks", dest="smooth_landmarks",
                        action="store_true", default=True,
                        help="对 2D/3D 坐标做 OneEuro 时序平滑（压抖动防误报）；默认开")
    parser.add_argument("--no-smooth-landmarks", dest="smooth_landmarks",
                        action="store_false",
                        help="关闭坐标滤波（生成原始未滤波数据集，与旧 run_01/02/03 一致）")
    parser.add_argument("--min-cutoff", type=float, default=1.0,
                        help="OneEuro min_cutoff：越大越平滑（默认1.0）")
    parser.add_argument("--beta", type=float, default=0.02,
                        help="OneEuro beta：对快速运动的自适应平滑强度（默认0.02）")
    parser.add_argument("--dcutoff", type=float, default=1.0,
                        help="OneEuro dcutoff：导数低通截止（默认1.0）")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="坐标滤波时间步假设（默认25fps；cutoff 单位 Hz，必须与推理一致）")
    args = parser.parse_args()

    # --version 快捷映射（仅 classification）
    VERSION_MAP = {
        "v1":   (False, False, False, "datasets/experiment/v1"),
        "v2":   (True,  False, False, "datasets/experiment/v2"),
        "v2.2": (False, True,  False, "datasets/experiment/v2_2"),
        "v3":   (True,  True,  False, "datasets/experiment/v3"),
        "v4":   (False, False, True,  "datasets/experiment/v4"),
    }
    if args.version:
        v = args.version.lower()
        if v not in VERSION_MAP:
            print(f"Unknown version: {v}. Choose: {list(VERSION_MAP.keys())}")
            return
        use_vel, use_acc, use_ang, def_out = VERSION_MAP[v]
        args.use_velocity = args.use_velocity or use_vel
        args.use_acceleration = args.use_acceleration or use_acc
        args.use_angles = args.use_angles or use_ang
        if not args.output:
            args.output = def_out

    dim = 29 + (26 if args.use_velocity else 0) + (26 if args.use_acceleration else 0) + (7 if args.use_angles else 0)
    ws = args.window_size or WINDOW_SIZE
    # regression 默认输出目录（启用坐标平滑时加 _smoothed 后缀，与 build_skeleton 同源）
    if args.task == "regression" and not args.output and not args.version:
        suffix = "_smoothed" if args.smooth_landmarks else ""
        args.output = f"datasets/experiment/risk_w{ws}{suffix}"
    output_root = Path(args.output) if args.output else OUTPUT_ROOT
    print(f"Task: {args.task}, Output: {output_root}, dim={dim}")
    feat_builder = FrameFeatureBuilder(use_angles=args.use_angles)
    # 坐标层 OneEuro 平滑器（2D image + 3D world 双通道，与推理侧 demo 同参保证零漂移）
    smoother2d = LandmarkSmoother(
        enabled=args.smooth_landmarks,
        min_cutoff=args.min_cutoff, beta=args.beta, dcutoff=args.dcutoff,
        n_landmarks=33, channels=2, fps=args.fps)
    smoother3d = LandmarkSmoother(
        enabled=args.smooth_landmarks,
        min_cutoff=args.min_cutoff, beta=args.beta, dcutoff=args.dcutoff,
        n_landmarks=33, channels=3, fps=args.fps)
    print(f"  Landmark smoothing (OneEuro 2D+3D): "
          f"{'ON' if args.smooth_landmarks else 'OFF'}  "
          f"min_cutoff={args.min_cutoff} beta={args.beta} dcutoff={args.dcutoff}")
    windower = SlidingWindowBuilder(WindowConfig(window_size=ws, stride=STRIDE))
    print(f"Window: {ws}, stride: {STRIDE}")

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

        # 视频级切分
        indices = rng.permutation(len(npz_files))
        n_train = int(len(npz_files) * TRAIN_RATIO)
        train_set = set(indices[:n_train])

        train_X, train_y, train_yr = [], [], []
        test_X,  test_y,  test_yr  = [], [], []
        split_log = {"train": [], "test": []}

        print(f"\n{'='*50}")
        print(f"{ds}: {len(npz_files)} videos  [task={args.task}]")
        print(f"{'='*50}")

        for i, npz_path in enumerate(npz_files):
            vid = npz_path.stem
            is_train = i in train_set
            split = "train" if is_train else "test"

            # 回归：读 risk sidecar（缺失则跳过本视频，且不计入 split_log）
            risk_arr = None
            if args.task == "regression":
                sidecar = npz_path.with_suffix(".risk.npy")
                if not sidecar.exists():
                    print(f"  [{i+1}/{len(npz_files)}] {vid} ... [SKIP no risk]")
                    continue
                risk_arr = np.load(str(sidecar)).astype(np.float32)

            split_log[split].append(vid)

            print(f"  [{i+1}/{len(npz_files)}] {vid} ...", end=" ", flush=True)
            X, y, yr = process_cached(npz_path, feat_builder, windower,
                                     use_velocity=args.use_velocity,
                                     use_acceleration=args.use_acceleration,
                                     risk=risk_arr,
                                     smoother2d=smoother2d, smoother3d=smoother3d)

            safe = vid.replace(".", "_")
            np.save(str(videos_dir / f"{safe}_X.npy"), X)
            if args.task == "regression":
                np.save(str(videos_dir / f"{safe}_y_risk.npy"), yr)
                np.save(str(videos_dir / f"{safe}_y_cls.npy"), y)
            else:
                np.save(str(videos_dir / f"{safe}_y.npy"), y)

            n = X.shape[0]
            if args.task == "regression":
                uc = {int(lb): int(np.sum(y == lb)) for lb in np.unique(y)} if n > 0 else {}
                yr_stats = ""
                if yr is not None and n > 0:
                    yr_stats = (f" risk[min={yr.min():.2f},max={yr.max():.2f},"
                               f"mean={yr.mean():.3f}]")
                print(f"windows={n}, labels={uc}{yr_stats} [{split}]")
            else:
                uc = {int(lb): int(np.sum(y == lb)) for lb in np.unique(y)} if n > 0 else {}
                print(f"windows={n}, labels={uc} [{split}]")

            if n > 0:
                (train_X if is_train else test_X).append(X)
                (train_y if is_train else test_y).append(y)
                if yr is not None:
                    (train_yr if is_train else test_yr).append(yr)

        # 合并
        def save_combined(prefix, Xl, yl, yrl=None):
            Xa = np.concatenate(Xl) if Xl else np.empty((0, ws, dim), dtype=np.float32)
            ya = np.concatenate(yl) if yl else np.empty((0,), dtype=np.int64)
            np.save(str(out_dir / f"X_{prefix}.npy"), Xa)
            if args.task == "regression":
                np.save(str(out_dir / f"y_cls_{prefix}.npy"), ya)
            else:
                np.save(str(out_dir / f"y_{prefix}.npy"), ya)
            yra = None
            if args.task == "regression" and yrl is not None:
                yra = np.concatenate(yrl) if yrl else np.empty((0,), dtype=np.float32)
                np.save(str(out_dir / f"y_risk_{prefix}.npy"), yra)
            return Xa, ya

        Xt, yt = save_combined("train", train_X, train_y, train_yr)
        Xe, ye = save_combined("test",  test_X,  test_y,  test_yr)
        (out_dir / "split_log.json").write_text(json.dumps(split_log, indent=2))

        dim_out = Xt.shape[-1] if Xt.shape[0] > 0 else 0
        print(f"  dim={dim_out}, X_train: {Xt.shape}, y_train: {yt.shape}")
        print(f"  X_test:  {Xe.shape}, y_test:  {ye.shape}")
        if args.task == "regression":
            yr_t = np.concatenate(train_yr) if train_yr else np.empty((0,))
            yr_e = np.concatenate(test_yr) if test_yr else np.empty((0,))
            print(f"  y_risk_train: {yr_t.shape}, y_risk_test: {yr_e.shape}")
            meta = {
                "task": "regression",
                "window_size": ws,
                "input_dim": dim_out,
                "risk_mode": RISK_MODE,
                "ramp_lo": RAMP_LO, "ramp_hi": RAMP_HI,
                "tau_u": TAU_U, "tau_abn": TAU_ABN,
                "landmark_smoothing": smoother3d.to_dict(),
                "n_train": int(Xt.shape[0]), "n_test": int(Xe.shape[0]),
            }
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print("\nDone.")


if __name__ == "__main__":
    main()
