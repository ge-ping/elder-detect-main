#!/usr/bin/env python
"""
build_cache.py — 阶段 1: 四个数据集 → MediaPipe → pose_cache/*.npz

每个视频保存一个 .npz:
  landmarks, world_landmarks, visibility, timestamps, frame_width, frame_height, labels, video_id

用法:
    python scripts/build_cache.py          # 全部数据集
    python scripts/build_cache.py Le2i     # 单个数据集
"""
from __future__ import annotations

import csv
import io
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
from scripts.build_config import POSE_CONFIG

CACHE_DIR = PROJECT_ROOT / "pose_cache"
POSE_CFG  = load_pose_config(POSE_CONFIG)


# ==================================================================
# 工具: 从 PoseResult 提取 numpy 数组
# ==================================================================

def extract_arrays(result) -> dict:
    """单帧 PoseResult → numpy dict"""
    n = len(result.landmarks)
    lm = np.zeros((33, 3), dtype=np.float32)
    wl = np.zeros((33, 3), dtype=np.float32)
    vis = np.zeros(33, dtype=np.float32)
    for i, kp in enumerate(result.landmarks):
        idx = kp.name.value
        lm[idx] = [kp.x, kp.y, kp.z]
        vis[idx] = kp.visibility
    if result.world_landmarks:
        for kp in result.world_landmarks:
            wl[kp.name.value] = [kp.x, kp.y, kp.z]
    return {"landmarks": lm, "world_landmarks": wl, "visibility": vis}


# ==================================================================
# 1. Le2i
# ==================================================================

def build_le2i():
    root = Path("D:/my_datasets/Le2i/Le2i/Le2i")
    subsets = ["Home_01", "Home_02", "Coffee_room_01", "Coffee_room_02"]
    out = CACHE_DIR / "Le2i"; out.mkdir(parents=True, exist_ok=True)

    for subset in subsets:
        sd = root / subset / subset
        vd, ad = sd / "Videos", sd / "Annotations_files"
        if not vd.exists(): continue
        for vp in sorted(vd.glob("video (*.avi")):
            anno_path = ad / (vp.stem + ".txt")
            if not anno_path.exists(): continue
            lines = anno_path.read_text().strip().splitlines()
            if len(lines) < 2: continue
            fs, fe = int(lines[0]), int(lines[1])

            vid = f"{subset}_{vp.stem.replace(' ', '_')}"
            npz_path = out / f"{vid}.npz"
            if npz_path.exists():
                print(f"  [SKIP] {vid} (cached)")
                continue

            lms, wls, viss, tss, labels = [], [], [], [], []
            frame_idx = 0
            print(f"  {vid} ...", end=" ", flush=True)
            with PoseExtractor(POSE_CFG) as ex:
                for result in ex.process_video(str(vp)):
                    frame_idx += 1
                    if result is None: continue
                    arr = extract_arrays(result)
                    lms.append(arr["landmarks"]); wls.append(arr["world_landmarks"])
                    viss.append(arr["visibility"]); tss.append(result.timestamp_ms)
                    # 标签
                    if fs == 0 and fe == 0: lb = 0
                    elif frame_idx < fs:    lb = 0
                    elif frame_idx <= fe:   lb = 1
                    else:                   lb = 2
                    labels.append(lb)

            if not lms:
                print("NO DETECTIONS")
                continue
            np.savez_compressed(npz_path,
                landmarks=np.array(lms, dtype=np.float32),
                world_landmarks=np.array(wls, dtype=np.float32),
                visibility=np.array(viss, dtype=np.float32),
                timestamps=np.array(tss, dtype=np.int64),
                frame_width=result.frame_width if 'result' in dir() else 640,
                frame_height=result.frame_height if 'result' in dir() else 480,
                labels=np.array(labels, dtype=np.int64),
                video_id=vid)
            uc = {int(lb): labels.count(lb) for lb in set(labels)}
            print(f"T={len(lms)}, labels={uc}")


# ==================================================================
# 2. UR-FALL
# ==================================================================

def build_urfall():
    root = Path("D:/my_datasets/UR-FALL/downloads")
    out = CACHE_DIR / "UR-FALL"; out.mkdir(parents=True, exist_ok=True)

    # 解析两个全量 CSV → {sequence_name: {frame_num: csv_label}}
    csv_data = {}  # "fall-01" → {1: -1, 2: -1, ...}
    for csv_path in sorted(root.glob("*.csv")):
        with open(csv_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 3: continue
                # UR-FALL CSV: seq_name, frame_num, label, ...
                # 首行不包含数字帧号则跳过（无表头，全是数据行）
                seq_name = parts[0].strip()
                try:
                    frame_num = int(parts[1])
                except ValueError:
                    continue
                csv_label = int(parts[2])
                csv_data.setdefault(seq_name, {})[frame_num] = csv_label

    # 遍历 zip，匹配 CSV
    for zp_path in sorted(root.glob("*-cam0-rgb.zip")):
        # fall-01-cam0-rgb.zip → seq_name = "fall-01"
        seq_name = zp_path.name.replace("-cam0-rgb.zip", "")
        seq_labels = csv_data.get(seq_name, {})
        if not seq_labels: continue

        is_adl = seq_name.startswith("adl")
        vid = seq_name.replace("-", "_")

        npz_path = out / f"{vid}.npz"
        if npz_path.exists():
            print(f"  [SKIP] {vid} (cached)")
            continue

        # 生成逐帧标签（按 CSV 帧号排序）
        sorted_frames = sorted(seq_labels.items())  # [(1, -1), (2, -1), ...]
        n = len(sorted_frames)
        frame_labels = np.zeros(n, dtype=np.int64)
        if not is_adl:
            for i, (_, csv_lb) in enumerate(sorted_frames):
                if csv_lb == -1:   frame_labels[i] = 0
                elif csv_lb == 0:  frame_labels[i] = 1
                elif csv_lb == 1:  frame_labels[i] = 2

        lms, wls, viss, tss, labels = [], [], [], [], []
        print(f"  {vid} ({'ADL' if is_adl else 'Fall'}) ...", end=" ", flush=True)
        with zipfile.ZipFile(zp_path) as zf:
            pngs = sorted([n for n in zf.namelist() if n.endswith(".png")])
            with PoseExtractor(POSE_CFG) as ex:
                for idx, name in enumerate(pngs):
                    if idx >= len(frame_labels): break
                    img = cv2.imdecode(np.frombuffer(zf.read(name), np.uint8), cv2.IMREAD_COLOR)
                    if img is None: continue
                    result = ex.detect_frame(img, timestamp_ms=idx * 33)
                    if result is None: continue
                    arr = extract_arrays(result)
                    lms.append(arr["landmarks"]); wls.append(arr["world_landmarks"])
                    viss.append(arr["visibility"]); tss.append(result.timestamp_ms)
                    labels.append(int(frame_labels[idx]))

        if not lms:
            print("NO DETECTIONS")
            continue
        np.savez_compressed(npz_path,
            landmarks=np.array(lms, dtype=np.float32),
            world_landmarks=np.array(wls, dtype=np.float32),
            visibility=np.array(viss, dtype=np.float32),
            timestamps=np.array(tss, dtype=np.int64),
            frame_width=640, frame_height=480,
            labels=np.array(labels, dtype=np.int64), video_id=vid)
        uc = {int(lb): labels.count(lb) for lb in set(labels)}
        print(f"T={len(lms)}, labels={uc}")


# ==================================================================
# 3. CAUCAFall
# ==================================================================

def build_caucafall():
    root = Path("D:/my_datasets/CAUCAFall/Dataset CAUCAFall/Dataset CAUCAFall/CAUCAFall")
    out = CACHE_DIR / "CAUCAFall"; out.mkdir(parents=True, exist_ok=True)

    for subj in sorted(root.iterdir()):
        if not subj.is_dir(): continue
        for act in sorted(subj.iterdir()):
            if not act.is_dir(): continue
            pairs = []
            for png in sorted(act.glob("*.png")):
                txt = png.with_suffix(".txt")
                if txt.exists():
                    try: fn = int(png.stem.rsplit("-",1)[-1])
                    except: fn = 0
                    pairs.append((fn, png, txt))
            if not pairs: continue
            pairs.sort(key=lambda x: x[0])

            vid = f"{subj.name}_{act.name}".replace(" ", "_")
            npz_path = out / f"{vid}.npz"
            if npz_path.exists():
                print(f"  [SKIP] {vid} (cached)")
                continue

            # 读所有 action_id 生成标签
            action_ids = []
            for _, _, tp in pairs:
                line = tp.read_text().strip()
                action_ids.append(line.split()[0] if line else "nofall")
            n = len(action_ids)
            frame_labels = np.zeros(n, dtype=np.int64)
            first = next((i for i, a in enumerate(action_ids) if a == "1"), None)
            if first is not None:
                pre_start = max(0, first - 35)
                frame_labels[pre_start:first] = 1
                frame_labels[first:] = 2

            lms, wls, viss, tss, labels = [], [], [], [], []
            print(f"  {vid} ...", end=" ", flush=True)
            with PoseExtractor(POSE_CFG) as ex:
                for idx, (_, png_path, _) in enumerate(pairs):
                    img = cv2.imread(str(png_path))
                    if img is None: continue
                    result = ex.detect_frame(img, timestamp_ms=idx * 33)
                    if result is None: continue
                    arr = extract_arrays(result)
                    lms.append(arr["landmarks"]); wls.append(arr["world_landmarks"])
                    viss.append(arr["visibility"]); tss.append(result.timestamp_ms)
                    labels.append(int(frame_labels[idx]))

            if not lms:
                print("NO DETECTIONS")
                continue
            np.savez_compressed(npz_path,
                landmarks=np.array(lms, dtype=np.float32),
                world_landmarks=np.array(wls, dtype=np.float32),
                visibility=np.array(viss, dtype=np.float32),
                timestamps=np.array(tss, dtype=np.int64),
                frame_width=640, frame_height=480,
                labels=np.array(labels, dtype=np.int64), video_id=vid)
            uc = {int(lb): labels.count(lb) for lb in set(labels)}
            print(f"T={len(lms)}, labels={uc}")


# ==================================================================
# 4. UP-FALL
# ==================================================================

def build_upfall():
    root = Path("D:/my_datasets/UP-FALL")
    out = CACHE_DIR / "UP-FALL"; out.mkdir(parents=True, exist_ok=True)
    FALL_SET = {"1","2","3","4","5"}

    for zp_path in sorted(root.glob("*Camera1.zip")):
        csv_path = root / zp_path.name.replace("Camera1.zip", ".csv")
        if not csv_path.exists(): continue
        m = re.search(r"Activity(\d+)", zp_path.name)
        if not m: continue
        activity_id = int(m.group(1))

        # 读 CSV 标签
        csv_labels = []
        with open(csv_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if parts[0].startswith("201"):
                    csv_labels.append(parts[-1])

        vid = zp_path.stem.replace("Camera1", "")
        npz_path = out / f"{vid}.npz"
        if npz_path.exists():
            print(f"  [SKIP] {vid} (cached)")
            continue

        # 生成标签
        n = len(csv_labels)
        frame_labels = np.zeros(n, dtype=np.int64)
        if activity_id <= 5:
            fall_idx = [i for i, l in enumerate(csv_labels) if l in FALL_SET]
            if fall_idx:
                first, last = fall_idx[0], fall_idx[-1]
                frame_labels[first:last+1] = 1
                frame_labels[last+1:] = 2

        lms, wls, viss, tss, labels = [], [], [], [], []
        print(f"  {vid} ...", end=" ", flush=True)
        with zipfile.ZipFile(zp_path) as zf:
            pngs = sorted([n for n in zf.namelist() if n.endswith(".png")])
            with PoseExtractor(POSE_CFG) as ex:
                for idx, name in enumerate(pngs):
                    if idx >= len(frame_labels): break
                    img = cv2.imdecode(np.frombuffer(zf.read(name), np.uint8), cv2.IMREAD_COLOR)
                    if img is None: continue
                    result = ex.detect_frame(img, timestamp_ms=idx * 33)
                    if result is None: continue
                    arr = extract_arrays(result)
                    lms.append(arr["landmarks"]); wls.append(arr["world_landmarks"])
                    viss.append(arr["visibility"]); tss.append(result.timestamp_ms)
                    labels.append(int(frame_labels[idx]))

        if not lms:
            print("NO DETECTIONS")
            continue
        np.savez_compressed(npz_path,
            landmarks=np.array(lms, dtype=np.float32),
            world_landmarks=np.array(wls, dtype=np.float32),
            visibility=np.array(viss, dtype=np.float32),
            timestamps=np.array(tss, dtype=np.int64),
            frame_width=640, frame_height=480,
            labels=np.array(labels, dtype=np.int64), video_id=vid)
        uc = {int(lb): labels.count(lb) for lb in set(labels)}
        print(f"T={len(lms)}, labels={uc}")


# ==================================================================
# ==================================================================
# 5. Pre-VFall
# ==================================================================

def build_prevfall():
    root = Path("D:/my_datasets/Pre-VFall/Pre-VFall/Pre-VFall")
    out = CACHE_DIR / "Pre-VFall"; out.mkdir(parents=True, exist_ok=True)

    CLASS_MAP = {"Normal": 0, "Abnormal": 1, "Fall": 2}

    for act_dir in sorted(root.iterdir()):
        if not act_dir.is_dir(): continue
        act_name = act_dir.name

        for sub_dir in sorted(act_dir.iterdir()):
            if not sub_dir.is_dir(): continue
            sub_name = sub_dir.name

            # 收集所有 PNG 帧及其标签
            frames = []  # [(png_path, label)]
            for cls_name, label in CLASS_MAP.items():
                cls_dir = sub_dir / f"{sub_name}-{cls_name}"
                if not cls_dir.exists(): continue
                for png in sorted(cls_dir.glob("*.png")):
                    frames.append((png, label))
                for jpg in sorted(cls_dir.glob("*.jpg")):
                    frames.append((jpg, label))
                for jpeg in sorted(cls_dir.glob("*.jpeg")):
                    frames.append((jpeg, label))

            if not frames: continue
            # 按文件名排序（文件名通常含帧号）
            frames.sort(key=lambda x: x[0].name)

            vid = f"{act_name}_{sub_name}"
            npz_path = out / f"{vid}.npz"
            if npz_path.exists():
                print(f"  [SKIP] {vid} ({len(frames)} frames, cached)")
                continue

            lms, wls, viss, tss, labels = [], [], [], [], []
            print(f"  {vid} ({len(frames)} frames) ...", end=" ", flush=True)
            with PoseExtractor(POSE_CFG) as ex:
                for idx, (png_path, label) in enumerate(frames):
                    img = cv2.imread(str(png_path))
                    if img is None: continue
                    result = ex.detect_frame(img, timestamp_ms=idx * 33)
                    if result is None: continue
                    arr = extract_arrays(result)
                    lms.append(arr["landmarks"]); wls.append(arr["world_landmarks"])
                    viss.append(arr["visibility"]); tss.append(result.timestamp_ms)
                    labels.append(label)

            if not lms:
                print("NO DETECTIONS")
                continue
            np.savez_compressed(npz_path,
                landmarks=np.array(lms, dtype=np.float32),
                world_landmarks=np.array(wls, dtype=np.float32),
                visibility=np.array(viss, dtype=np.float32),
                timestamps=np.array(tss, dtype=np.int64),
                frame_width=640, frame_height=480,
                labels=np.array(labels, dtype=np.int64), video_id=vid)
            uc = {int(lb): labels.count(lb) for lb in set(labels)}
            print(f"detections={len(lms)}, labels={uc}")


# ==================================================================
# Main
# ==================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="all",
                        choices=["all","Le2i","UR-FALL","CAUCAFall","UP-FALL","Pre-VFall"])
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cache dir: {CACHE_DIR}\n")

    if args.dataset in ("all", "Le2i"):
        print("=== Le2i ==="); build_le2i()
    if args.dataset in ("all", "UR-FALL"):
        print("\n=== UR-FALL ==="); build_urfall()
    if args.dataset in ("all", "CAUCAFall"):
        print("\n=== CAUCAFall ==="); build_caucafall()
    if args.dataset in ("all", "UP-FALL"):
        print("\n=== UP-FALL ==="); build_upfall()
    if args.dataset in ("all", "Pre-VFall"):
        print("\n=== Pre-VFall ==="); build_prevfall()

    print("\nDone.")


if __name__ == "__main__":
    main()
