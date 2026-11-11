#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prevfall_calibrate.py — Pre-VFall 多场景 VLM 静态基线标定（测泛化）

对 Pre-VFall 6 个跌倒类型各挑首个场景，抽首帧 → VLM(方舟)标定
→ env_baselines/Pre-VFall/<scene>.json

文件规范：env_baselines/<dataset>/<scene>.json
VLM 仅此一次标定，运行时纯算法不再调 VLM。

用法:
  python scripts/prevfall_calibrate.py
  python scripts/prevfall_calibrate.py --mock   # 无网测试
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 复用 vlm_baseline_calibrate 的方舟调用 + 解析 + .env 加载
from scripts.vlm_baseline_calibrate import _call_ark, _parse, _load_dotenv  # noqa: E402

_load_dotenv()

PREVFALL_ROOT = Path("D:/my_datasets/Pre-VFall/Pre-VFall/Pre-VFall")
OUT_DIR = PROJECT_ROOT / "env_baselines" / "Pre-VFall"
TMP_DIR = PROJECT_ROOT / "temp" / "_prevfall_frames"

# 6 个跌倒类型 × 首个场景（测泛化：不同病因/方向/布局）
SCENES = [
    ("confusion_delirium",   "CLFM1"),
    ("confusion_nph",        "CGFM1"),
    ("dizziness_fall_forward", "FDFFM1"),
    ("dizziness_fall_side",  "FDSFM1"),
    ("weakness_fall_forward", "FWFFM1"),
    ("weakness_fall_side",   "FWSFM1"),
]


def extract_first_frame(mp4: Path, out_png: Path) -> bool:
    """从 mp4 抽首帧存 png。"""
    cap = cv2.VideoCapture(str(mp4))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), frame)
    return True


def main():
    ap = argparse.ArgumentParser(description="Pre-VFall 多场景 VLM 基线标定")
    ap.add_argument("--mock", action="store_true", help="无网测试(占位基线)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    model = os.environ.get("ARK_VLM_MODEL", "Doubao-Seed-1.6-vision")

    print(f"=== Pre-VFall 多场景 VLM 基线标定 ===")
    print(f"  数据源: {PREVFALL_ROOT}")
    print(f"  输出: {OUT_DIR}/<scene>.json")
    print(f"  模型: {model}")
    print(f"  场景数: {len(SCENES)}")
    print("-" * 70)

    results = []
    for ftype, scene in SCENES:
        mp4 = PREVFALL_ROOT / ftype / f"{scene}.mp4"
        if not mp4.exists():
            print(f"[SKIP] {scene}: {mp4} 不存在")
            continue
        png = TMP_DIR / f"{scene}.png"
        if not extract_first_frame(mp4, png):
            print(f"[SKIP] {scene}: 抽首帧失败")
            continue

        print(f"  {scene} ({ftype}) ...", end=" ", flush=True)
        if args.mock:
            result = {"ground_top_y": 0.6, "obstacles": [{"bbox": [0.1, 0.7, 0.15, 0.1], "type": "[mock]"}],
                      "confidence": 1.0, "mock": True}
        else:
            try:
                text = _call_ark(png, model)
                result = _parse(text)
                if result is None:
                    print(f"[ERR] JSON 解析失败")
                    continue
            except Exception as e:
                print(f"[ERR] {e}")
                continue

        result["scene"] = scene
        result["dataset"] = "Pre-VFall"
        result["fall_type"] = ftype
        result["frame_source"] = str(mp4)
        result["model"] = model

        out = OUT_DIR / f"{scene}.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        gty = result.get("ground_top_y")
        nobs = len(result.get("obstacles", []))
        print(f"ground_top_y={gty}  obstacles={nobs}  → {out.name}")
        results.append({"scene": scene, "type": ftype, "ground_top_y": gty, "obstacles": nobs})

    print("-" * 70)
    print(f"=== 汇总（{len(results)}/{len(SCENES)} 场景标定成功）===")
    print(f"{'场景':<8} {'跌倒类型':<24} {'ground_top_y':>12} {'障碍物数':>8}")
    for r in results:
        print(f"{r['scene']:<8} {r['type']:<24} {str(r['ground_top_y']):>12} {r['obstacles']:>8}")
    print(f"\n基线文件 → {OUT_DIR}/")
    print("下一步：跑 scripts/prevfall_demo.py 测门控泛化")


if __name__ == "__main__":
    main()
