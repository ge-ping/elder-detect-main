#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
env_perceive.py — EnvRisk 感知 + 映射 → 写 env sidecar（R1 优先，R0 回退）
==============================================================
统一入口。对每个视频：

  1) R1（真实视频感知，优先）：若 env_labels/<ds>/<stem>.json 存在
     （由 env_extract_frames.py + env_label_vlm.py 产出），取其 room/light/floor
     + VLM 直出 fall_risk，按 §8 映射口径算 risk_环境。
     —— 此时环境标签来自「看视频画面」，不是文件名。
  2) R0（离线规则，回退）：无 R1 标签时，房间按 VIDEO_ROOM_MAP[stem]
     或 DATASET_ROOM_DEFAULT[ds] 的元标签假设，光照可经 --video-root 抽帧估算；
     算 risk_环境（规则表）。

→ 写 sidecar：
    pose_cache/<ds>/<stem>.env_risk.npy   (T,) float32，每帧 = 该视频 risk_环境（恒定）
    pose_cache/<ds>/<stem>.env_meta.json   可解释追溯（含 source 字段）

硬约束（envrisk_v0_design.md §1）：原 *.npz 只读、零改动；仅新增 sidecar。

用法:
    # R1：先抽帧+打标（需 API key 或 --mock），再统一写 sidecar
    python scripts/env_extract_frames.py --video-root /path/to/videos
    python scripts/env_label_vlm.py --index frames/index.csv --mock
    python scripts/env_perceive.py            # 自动优先用 R1 标签

    # R0：无视频/无标签，纯元标签回退
    python scripts/env_perceive.py --datasets UR-FALL
    python scripts/env_perceive.py --dry-run
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_config import (
    DATASET_ROOM_DEFAULT, VIDEO_ROOM_MAP,
)
from model.env.env_risk_model import (
    HomeFastModel, risk_from_label, save_env_sidecar,
)

MODEL = HomeFastModel()
CACHE = PROJECT_ROOT / "pose_cache"
LABELS = PROJECT_ROOT / "env_labels"


def load_r1_label(ds: str, stem: str, labels_root: Path) -> dict | None:
    """读 R1 视频级聚合标签；不存在返回 None。"""
    jp = labels_root / ds / f"{stem}.json"
    if not jp.exists():
        return None
    try:
        return json.loads(jp.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def light_from_video(video_path: Path, n_sample: int = 5) -> str:
    """抽 n_sample 帧算平均亮度 → 离散枚举（对齐 design §4.3）。需 cv2。"""
    try:
        import cv2  # noqa: F401
    except Exception:  # noqa: BLE001
        return "bright"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "bright"
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or 1
    idxs = [int(i * n / n_sample) for i in range(n_sample)]
    ls = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ls.append(float(g.mean()) / 255.0)
    cap.release()
    if not ls:
        return "bright"
    L = sum(ls) / len(ls)
    if L < 0.25:
        return "dark"
    if L < 0.45:
        return "dim"
    return "bright"


def find_video(video_root: Path, stem: str):
    cands = list(video_root.rglob(f"*{stem}*"))
    cands = [c for c in cands if c.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm")]
    return cands[0] if cands else None


def build_meta(ds, stem, room, light, risk, source, **extra):
    m = {
        "stem": stem, "dataset": ds,
        "room_type": room, "lighting": light,
        "risk_env": round(float(risk), 3),
        "source": source,
        "n_frames": None, "has_video": False,
    }
    m.update(extra)
    return m


def process(ds: str, stem: str, npz: Path, video_root, labels_root,
            dry: bool):
    # ---- R1 优先：真实视频感知标签（HOME FAST 细标签） ----
    agg = load_r1_label(ds, stem, labels_root)
    if agg is not None:
        if "corridor_clear" in agg:
            # 新格式：HOME FAST 细标签 → 25 题评分
            sc = risk_from_label(agg)
            room = agg.get("room_type", "unknown")
            light = agg.get("lighting", "bright")
            risk = sc["normalized"]
            meta = build_meta(
                ds, stem, room, light, risk, "R1_vlm",
                homefast_total=sc["total"],
                homefast_n_observed=sc["n_observed"],
                homefast_level=sc["level"],
                vlm_confidence=agg.get("confidence"),
                need_review=agg.get("need_review"),
            )
        else:
            # 旧 room/light/floor 格式不兼容 HOME FAST → 回退 R0
            room = agg.get("room_type", "unknown")
            light = agg.get("lighting", "bright")
            risk = 0.0
            meta = build_meta(
                ds, stem, room, light, risk, "R1_oldformat-skip",
                note="旧格式 room/light/floor 不兼容 HOME FAST，回退无环境隐患",
            )
        src = "R1"
    else:
        # ---- R0 回退：无 R1 标签 → 无已知环境隐患（E=0） ----
        #   HOME FAST 是问卷量表，自动化批处理无法代替人工问卷；
        #   完整 R0 问卷模式请走 demo/demo_env_risk.py --answers。
        room = VIDEO_ROOM_MAP.get(stem) or DATASET_ROOM_DEFAULT.get(ds, "unknown")
        light = "bright"
        has_video = False
        if video_root:
            vp = find_video(video_root, stem)
            if vp:
                light = light_from_video(vp)
                has_video = True
        risk = 0.0
        meta = build_meta(
            ds, stem, room, light, risk, "R0_none",
            has_video=has_video,
            note="无 R1 标签，HOME FAST 假设无已知环境隐患",
        )
        src = "R0"

    if not dry:
        ok, T = save_env_sidecar(npz, risk, meta)
        if not ok:
            print(f"  [WARN] {ds}/{stem}: 写 sidecar 失败（{meta.get('error')}）")
    return risk, meta, src


def main():
    p = argparse.ArgumentParser(description="EnvRisk 感知 → 写 env sidecar（R1 优先 / R0 回退）")
    p.add_argument("--cache", default=str(CACHE), help="pose_cache 根目录")
    p.add_argument("--labels-root", default=str(LABELS), help="R1 标签根目录（env_labels）")
    p.add_argument("--datasets", nargs="*", default=None, help="只处理指定数据集子目录")
    p.add_argument("--video-root", default=None, help="可选：启用 R0 帧亮度分支（需 opencv）")
    p.add_argument("--dry-run", action="store_true", help="只打印不落盘")
    args = p.parse_args()

    cache = Path(args.cache)
    labels_root = Path(args.labels_root)
    video_root = Path(args.video_root) if args.video_root else None
    if video_root and not video_root.exists():
        print(f"[ERR] video_root 不存在: {video_root}")
        return

    r1_ready = labels_root.exists() and any(labels_root.rglob("*.json"))
    npzs = sorted(glob.glob(str(cache / "*" / "*.npz")))
    if args.datasets:
        ds_set = set(args.datasets)
        npzs = [f for f in npzs if Path(f).parent.name in ds_set]
    if not npzs:
        print(f"[ERR] 在 {cache} 下未找到 npz")
        return

    print(f"EnvPerceive — 处理 {len(npzs)} 个视频"
          f"{' [DRY-RUN]' if args.dry_run else ''}")
    print(f"  环境标签来源: "
          f"{'R1(HOME FAST细标签,优先)+R0(无标签→无环境隐患,回退)' if r1_ready else 'R0(无R1标签→无环境隐患)'}")
    print(f"{'数据集':<10}{'stem':<40}{'src':<5}{'room':<10}{'light':<7}{'risk_环境':<10}")
    print("-" * 82)

    ok = 0
    for f in npzs:
        npz = Path(f)
        ds = npz.parent.name
        stem = npz.stem
        risk, meta, src = process(ds, stem, npz, video_root, labels_root,
                                   args.dry_run)
        ok += 1
        s = stem if len(stem) <= 38 else stem[:35] + "..."
        print(f"{ds:<10}{s:<40}{src:<5}{meta['room_type']:<10}{meta['lighting']:<7}{risk:<10.3f}")

    if not args.dry_run:
        print(f"\nDone. 写 sidecar {ok} 个 → {cache}")
        print("  每视频: <stem>.env_risk.npy (T,) + <stem>.env_meta.json")
    else:
        print(f"\n[DRY-RUN] 仅预览，未落盘。共 {ok} 个。")


if __name__ == "__main__":
    main()
