#!/usr/bin/env python
"""
build_cache_yolo.py — 阶段 1(YOLO 版): 五个数据集 → YOLO-pose(2D) → pose_cache_yolo/*.npz

与 build_cache.py 完全并行独立（绝不改 MediaPipe 路径）：
  - 用 YoloPoseBackend（ultralytics, CUDA）抽 2D 关键点
  - 每视频存一个 .npz: landmarks(T,13,2) [归一化 0~1]、visibility(T,13) [YOLO conf]、
                        timestamps、frame_width、frame_height、labels、video_id
  - 同时用 regen_risk_labels.compute_risk 从 labels 算 <name>.risk.npy sidecar
    （YOLO 与 MediaPipe 检测帧集合可能不同，故 risk 必须按 YOLO 自身 labels 现算，
     不能复制 MediaPipe 的 sidecar，否则帧数/对齐错位）

关节集 = COCO-13（见 pose/coco_joints.py COCO_CORE_INDICES），比 FALL_JOINTS_17 少脚跟/脚尖。

用法:
    python scripts/build_cache_yolo.py                # 全部 5 数据集
    python scripts/build_cache_yolo.py Le2i           # 单个数据集
    python scripts/build_cache_yolo.py --model yolov8s-pose.pt
"""
from __future__ import annotations

import csv
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pose.backend import create_pose_backend
from scripts.regen_risk_labels import compute_risk

CACHE_DIR = PROJECT_ROOT / "pose_cache_yolo"
DATASETS = ["Le2i", "UR-FALL", "CAUCAFall", "UP-FALL", "Pre-VFall"]


# ==================================================================
# 工具: 从 YOLO PoseResult 提取 numpy 数组（2D）
# ==================================================================

def extract_arrays_yolo(result) -> dict:
    """单帧 YOLO PoseResult → 2D numpy dict（13 关节）"""
    n = len(result.landmarks)
    lm = np.zeros((n, 2), dtype=np.float32)
    vis = np.zeros(n, dtype=np.float32)
    for i, kp in enumerate(result.landmarks):
        lm[i] = [kp.x, kp.y]      # 归一化 [0,1]
        vis[i] = kp.visibility     # YOLO conf
    return {"landmarks": lm, "visibility": vis}


def _save(out_dir: Path, vid: str, lms, viss, tss, labels, fw, fh, video_id, frame_idx_last):
    """汇总一视频的所有帧 → 存 npz + risk sidecar；无检测则返回 False。"""
    if len(lms) == 0:
        return False
    npz_path = out_dir / f"{vid}.npz"
    np.savez_compressed(
        npz_path,
        landmarks=np.array(lms, dtype=np.float32),      # (T,13,2)
        visibility=np.array(viss, dtype=np.float32),    # (T,13)
        timestamps=np.array(tss, dtype=np.int64),
        frame_width=np.int32(fw),
        frame_height=np.int32(fh),
        labels=np.array(labels, dtype=np.int64),
        video_id=vid,
    )
    # risk sidecar（必须按 YOLO 自身 labels 现算，保证与帧数同长、对齐）
    risk = compute_risk(np.array(labels, dtype=np.int64))
    np.save(str(npz_path.with_suffix(".risk.npy")), risk.astype(np.float32))
    uc = {int(lb): int(labels.count(lb)) for lb in set(labels)}
    print(f"T={len(lms)}, labels={uc}, risk[min={risk.min():.2f},max={risk.max():.2f}]")
    return True


# ==================================================================
# 1. Le2i（视频）
# ==================================================================

def _parse_le2i_fs_fe(anno_path: Path) -> tuple:
    """解析 Le2i 跌倒标注，返回 (fs, fe, fmt)。

    兼容两种格式：
      A) 前两行为单个整数 fs/fe（绝大多数视频）；
      B) 无表头，首行即逐帧 bbox 行 frame,label,x1,y1,x2,y2
         → 由 label==2（跌倒帧）推断 fs/fe；全程无跌倒则 (0,0,"B")。
    """
    lines = anno_path.read_text(errors="ignore").strip().splitlines()
    if len(lines) < 2:
        return 0, 0, "empty"

    def _is_int(s: str) -> bool:
        try:
            int(s.strip())
            return True
        except ValueError:
            return False

    if _is_int(lines[0]) and _is_int(lines[1]):
        return int(lines[0]), int(lines[1]), "A"
    # Format B：扫描逐帧标注找跌倒帧（label==2）
    fs = fe = 0
    for ln in lines:
        p = ln.split(",")
        if len(p) < 2:
            continue
        try:
            fr, lb = int(p[0]), int(p[1])
        except ValueError:
            continue
        if lb == 2:
            if fs == 0:
                fs = fr
            fe = fr
    return fs, fe, "B"


def build_le2i(backend):
    root = Path("D:/my_datasets/Le2i/Le2i/Le2i")
    subsets = ["Home_01", "Home_02", "Coffee_room_01", "Coffee_room_02"]
    out = CACHE_DIR / "Le2i"; out.mkdir(parents=True, exist_ok=True)

    for subset in subsets:
        sd = root / subset / subset
        vd, ad = sd / "Videos", sd / "Annotations_files"
        if not vd.exists():
            continue
        for vp in sorted(vd.glob("video (*.avi")):
            anno_path = ad / (vp.stem + ".txt")
            if not anno_path.exists():
                continue
            fs, fe, _fmt = _parse_le2i_fs_fe(anno_path)
            if _fmt == "B" and fs == 0 and fe == 0:
                # Format B 但逐帧标注里无 label==2（跌倒）→ 无法可靠标 fall，跳过
                print(f"  [SKIP] {vp.stem} (unparseable fall range)")
                continue

            vid = f"{subset}_{vp.stem.replace(' ', '_')}"
            npz_path = out / f"{vid}.npz"
            if npz_path.exists():
                print(f"  [SKIP] {vid} (cached)")
                continue

            lms, viss, tss, labels = [], [], [], []
            frame_idx = 0
            print(f"  {vid} ...", end=" ", flush=True)
            cap = cv2.VideoCapture(str(vp))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ts_ms = int(frame_idx * 1000 / fps)
                result = backend.detect_frame(frame, ts_ms)
                frame_idx += 1
                if result is None:
                    continue
                arr = extract_arrays_yolo(result)
                lms.append(arr["landmarks"]); viss.append(arr["visibility"])
                tss.append(ts_ms)
                if fs == 0 and fe == 0:
                    lb = 0
                elif frame_idx - 1 < fs:
                    lb = 0
                elif frame_idx - 1 <= fe:
                    lb = 1
                else:
                    lb = 2
                labels.append(lb)
            cap.release()
            if not _save(out, vid, lms, viss, tss, labels, fw, fh, vid, frame_idx):
                print("NO DETECTIONS")


# ==================================================================
# 2. UR-FALL（zip png）
# ==================================================================

def build_urfall(backend):
    root = Path("D:/my_datasets/UR-FALL/downloads")
    out = CACHE_DIR / "UR-FALL"; out.mkdir(parents=True, exist_ok=True)

    csv_data = {}
    for csv_path in sorted(root.glob("*.csv")):
        with open(csv_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                seq_name = parts[0].strip()
                try:
                    frame_num = int(parts[1])
                except ValueError:
                    continue
                csv_label = int(parts[2])
                csv_data.setdefault(seq_name, {})[frame_num] = csv_label

    for zp_path in sorted(root.glob("*-cam0-rgb.zip")):
        seq_name = zp_path.name.replace("-cam0-rgb.zip", "")
        seq_labels = csv_data.get(seq_name, {})
        if not seq_labels:
            continue

        is_adl = seq_name.startswith("adl")
        vid = seq_name.replace("-", "_")
        npz_path = out / f"{vid}.npz"
        if npz_path.exists():
            print(f"  [SKIP] {vid} (cached)")
            continue

        sorted_frames = sorted(seq_labels.items())
        n = len(sorted_frames)
        frame_labels = np.zeros(n, dtype=np.int64)
        if not is_adl:
            for i, (_, csv_lb) in enumerate(sorted_frames):
                if csv_lb == -1:
                    frame_labels[i] = 0
                elif csv_lb == 0:
                    frame_labels[i] = 1
                elif csv_lb == 1:
                    frame_labels[i] = 2

        lms, viss, tss, labels = [], [], [], []
        print(f"  {vid} ({'ADL' if is_adl else 'Fall'}) ...", end=" ", flush=True)
        with zipfile.ZipFile(zp_path) as zf:
            pngs = sorted([n for n in zf.namelist() if n.endswith(".png")])
            for idx, name in enumerate(pngs):
                if idx >= len(frame_labels):
                    break
                img = cv2.imdecode(np.frombuffer(zf.read(name), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                result = backend.detect_frame(img, idx * 33)
                if result is None:
                    continue
                arr = extract_arrays_yolo(result)
                lms.append(arr["landmarks"]); viss.append(arr["visibility"])
                tss.append(idx * 33); labels.append(int(frame_labels[idx]))
        if not _save(out, vid, lms, viss, tss, labels, 640, 480, vid, len(pngs)):
            print("NO DETECTIONS")


# ==================================================================
# 3. CAUCAFall（png + txt）
# ==================================================================

def build_caucafall(backend):
    root = Path("D:/my_datasets/CAUCAFall/Dataset CAUCAFall/Dataset CAUCAFall/CAUCAFall")
    out = CACHE_DIR / "CAUCAFall"; out.mkdir(parents=True, exist_ok=True)

    for subj in sorted(root.iterdir()):
        if not subj.is_dir():
            continue
        for act in sorted(subj.iterdir()):
            if not act.is_dir():
                continue
            pairs = []
            for png in sorted(act.glob("*.png")):
                txt = png.with_suffix(".txt")
                if txt.exists():
                    try:
                        fn = int(png.stem.rsplit("-", 1)[-1])
                    except ValueError:
                        fn = 0
                    pairs.append((fn, png, txt))
            if not pairs:
                continue
            pairs.sort(key=lambda x: x[0])

            vid = f"{subj.name}_{act.name}".replace(" ", "_")
            npz_path = out / f"{vid}.npz"
            if npz_path.exists():
                print(f"  [SKIP] {vid} (cached)")
                continue

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

            lms, viss, tss, labels = [], [], [], []
            print(f"  {vid} ...", end=" ", flush=True)
            for idx, (_, png_path, _) in enumerate(pairs):
                img = cv2.imread(str(png_path))
                if img is None:
                    continue
                result = backend.detect_frame(img, idx * 33)
                if result is None:
                    continue
                arr = extract_arrays_yolo(result)
                lms.append(arr["landmarks"]); viss.append(arr["visibility"])
                tss.append(idx * 33); labels.append(int(frame_labels[idx]))
            if not _save(out, vid, lms, viss, tss, labels, 640, 480, vid, len(pairs)):
                print("NO DETECTIONS")


# ==================================================================
# 4. UP-FALL（zip png + csv）
# ==================================================================

def build_upfall(backend):
    root = Path("D:/my_datasets/UP-FALL")
    out = CACHE_DIR / "UP-FALL"; out.mkdir(parents=True, exist_ok=True)
    FALL_SET = {"1", "2", "3", "4", "5"}

    for zp_path in sorted(root.glob("*Camera1.zip")):
        csv_path = root / zp_path.name.replace("Camera1.zip", ".csv")
        if not csv_path.exists():
            continue
        m = __import__("re").search(r"Activity(\d+)", zp_path.name)
        if not m:
            continue
        activity_id = int(m.group(1))

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

        n = len(csv_labels)
        frame_labels = np.zeros(n, dtype=np.int64)
        if activity_id <= 5:
            fall_idx = [i for i, l in enumerate(csv_labels) if l in FALL_SET]
            if fall_idx:
                first, last = fall_idx[0], fall_idx[-1]
                frame_labels[first:last + 1] = 1
                frame_labels[last + 1:] = 2

        lms, viss, tss, labels = [], [], [], []
        print(f"  {vid} ...", end=" ", flush=True)
        with zipfile.ZipFile(zp_path) as zf:
            pngs = sorted([n for n in zf.namelist() if n.endswith(".png")])
            for idx, name in enumerate(pngs):
                if idx >= len(frame_labels):
                    break
                img = cv2.imdecode(np.frombuffer(zf.read(name), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                result = backend.detect_frame(img, idx * 33)
                if result is None:
                    continue
                arr = extract_arrays_yolo(result)
                lms.append(arr["landmarks"]); viss.append(arr["visibility"])
                tss.append(idx * 33); labels.append(int(frame_labels[idx]))
        if not _save(out, vid, lms, viss, tss, labels, 640, 480, vid, len(pngs)):
            print("NO DETECTIONS")


# ==================================================================
# 5. Pre-VFall（png/jpg 文件夹）
# ==================================================================

def build_prevfall(backend):
    root = Path("D:/my_datasets/Pre-VFall/Pre-VFall/Pre-VFall")
    out = CACHE_DIR / "Pre-VFall"; out.mkdir(parents=True, exist_ok=True)
    CLASS_MAP = {"Normal": 0, "Abnormal": 1, "Fall": 2}

    for act_dir in sorted(root.iterdir()):
        if not act_dir.is_dir():
            continue
        act_name = act_dir.name
        for sub_dir in sorted(act_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            sub_name = sub_dir.name
            frames = []
            for cls_name, label in CLASS_MAP.items():
                cls_dir = sub_dir / f"{sub_name}-{cls_name}"
                if not cls_dir.exists():
                    continue
                for png in sorted(cls_dir.glob("*.png")):
                    frames.append((png, label))
                for jpg in sorted(cls_dir.glob("*.jpg")):
                    frames.append((jpg, label))
                for jpeg in sorted(cls_dir.glob("*.jpeg")):
                    frames.append((jpeg, label))
            if not frames:
                continue
            frames.sort(key=lambda x: x[0].name)

            vid = f"{act_name}_{sub_name}"
            npz_path = out / f"{vid}.npz"
            if npz_path.exists():
                print(f"  [SKIP] {vid} ({len(frames)} frames, cached)")
                continue

            lms, viss, tss, labels = [], [], [], []
            print(f"  {vid} ({len(frames)} frames) ...", end=" ", flush=True)
            for idx, (png_path, label) in enumerate(frames):
                img = cv2.imread(str(png_path))
                if img is None:
                    continue
                result = backend.detect_frame(img, idx * 33)
                if result is None:
                    continue
                arr = extract_arrays_yolo(result)
                lms.append(arr["landmarks"]); viss.append(arr["visibility"])
                tss.append(idx * 33); labels.append(label)
            if not _save(out, vid, lms, viss, tss, labels, 640, 480, vid, len(frames)):
                print("NO DETECTIONS")


# ==================================================================
# Main
# ==================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="all",
                        choices=["all", "Le2i", "UR-FALL", "CAUCAFall", "UP-FALL", "Pre-VFall"])
    parser.add_argument("--model", default="yolov8n-pose.pt",
                        help="YOLO pose 模型 (默认 yolov8n-pose.pt，可换 yolov8s-pose.pt)")
    parser.add_argument("--device", default="cuda", help="推理设备 (默认 cuda)")
    parser.add_argument("--conf", type=float, default=0.5, help="检测置信度阈值")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"YOLO cache dir: {CACHE_DIR}")
    print(f"Model: {args.model}, device: {args.device}, conf: {args.conf}\n")

    backend = create_pose_backend(
        "yolo", model=args.model, device=args.device, conf=args.conf)
    print("YOLO backend 已加载。\n")

    if args.dataset in ("all", "Le2i"):
        print("=== Le2i ==="); build_le2i(backend)
    if args.dataset in ("all", "UR-FALL"):
        print("\n=== UR-FALL ==="); build_urfall(backend)
    if args.dataset in ("all", "CAUCAFall"):
        print("\n=== CAUCAFall ==="); build_caucafall(backend)
    if args.dataset in ("all", "UP-FALL"):
        print("\n=== UP-FALL ==="); build_upfall(backend)
    if args.dataset in ("all", "Pre-VFall"):
        print("\n=== Pre-VFall ==="); build_prevfall(backend)

    print("\nDone.")


if __name__ == "__main__":
    main()
