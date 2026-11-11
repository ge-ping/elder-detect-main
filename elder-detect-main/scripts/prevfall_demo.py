#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prevfall_demo.py — Pre-VFall 多场景视频 demo 测门控泛化

对 6 个已标定场景各跑 intent_gate_demo_video3（真实B模型 + bg_diff掩码 + I门控），
汇总各场景门控开/关 alarm 对比，验证门控在不同跌倒类型/场景的泛化性。

输出：demo/Pre-VFall/<scene>.mp4 + <scene>.json + 汇总表

用法:
  python scripts/prevfall_demo.py
  python scripts/prevfall_demo.py --frames 100
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PY = r"C:\Users\R9000P\.conda\envs\elder\python.exe"

PREVFALL = Path("D:/my_datasets/Pre-VFall/Pre-VFall/Pre-VFall")
SCENES = [
    ("confusion_delirium",   "CLFM1"),
    ("confusion_nph",        "CGFM1"),
    ("dizziness_fall_forward", "FDFFM1"),
    ("dizziness_fall_side",  "FDSFM1"),
    ("weakness_fall_forward", "FWFFM1"),
    ("weakness_fall_side",   "FWSFM1"),
]


def main():
    ap = argparse.ArgumentParser(description="Pre-VFall 多场景视频 demo 测泛化")
    ap.add_argument("--frames", type=int, default=100, help="每场景跑多少帧")
    ap.add_argument("--run-dir", default="model/lstm/risk_v1/run_03")
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / "demo" / "Pre-VFall"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Pre-VFall 多场景门控泛化验证 ===")
    print(f"  场景数: {len(SCENES)}  每场景帧数: {args.frames}  B模型: {args.run_dir}")
    print(f"  输出: {out_dir}/<scene>.mp4")
    print("=" * 78)

    results = []
    for ftype, scene in SCENES:
        video = PREVFALL / ftype / f"{scene}.mp4"
        baseline = PROJECT_ROOT / "env_baselines" / "Pre-VFall" / f"{scene}.json"
        out_mp4 = out_dir / f"{scene}.mp4"
        out_json = out_dir / f"{scene}.json"
        if not video.exists() or not baseline.exists():
            print(f"[SKIP] {scene}: video/baseline 缺失")
            continue
        cmd = [PY, str(PROJECT_ROOT / "demo" / "intent_gate_demo_video2.py"),
               "--video", str(video), "--baseline", str(baseline),
               "--force-e", "0.5",
               "--frames", str(args.frames),
               "--out", str(out_mp4), "--out-json", str(out_json)]
        print(f"\n--- {scene} ({ftype}) ---")
        r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [ERR] {r.stderr[-400:]}")
            continue
        # 读 json 汇总
        if out_json.exists():
            d = json.loads(out_json.read_text(encoding="utf-8"))
            row = {k: d.get(k) for k in ["T", "I_mean", "alarm_off", "alarm_on", "delta"]}
            row.update({"scene": scene, "type": ftype})
            results.append(row)
            print(f"  T={row.get('T')} I_mean={row.get('I_mean')} "
                  f"alarm {row.get('alarm_off')}→{row.get('alarm_on')} ({row.get('delta')})")

    # 汇总
    print("\n" + "=" * 78)
    print("  泛化汇总（门控开/关 alarm 对比）")
    print("=" * 78)
    print(f"{'场景':<8}{'跌倒类型':<24}{'T':>5}{'I_mean':>8}"
          f"{'alarm关':>8}{'alarm开':>8}{'变化':>7}")
    print("-" * 78)
    tot_off = tot_on = 0
    for r in results:
        im = r.get("I_mean") or 0.0
        print(f"{r['scene']:<8}{r['type']:<24}{r.get('T',0):>5}{im:>8.3f}"
              f"{r.get('alarm_off',0):>8}{r.get('alarm_on',0):>8}{r.get('delta',0):>+7}")
        tot_off += r.get("alarm_off", 0); tot_on += r.get("alarm_on", 0)
    print("-" * 78)
    print(f"{'合计':<8}{'':<24}{'':>5}{'':>8}{'':>8}{tot_off:>8}{tot_on:>8}{tot_on - tot_off:>+7}")
    print("=" * 78)
    print(f"  门控开启后报警帧 {tot_off} → {tot_on}（{tot_on - tot_off:+d}）")
    print("  → 减少=门控抑制误报；变化小=跌倒B高该报还报；增加=需排查")
    print(f"\n  视频 → {out_dir}/<scene>.mp4（6个，可播放对比）")


if __name__ == "__main__":
    main()
