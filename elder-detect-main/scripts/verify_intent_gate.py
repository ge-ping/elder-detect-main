#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_intent_gate.py — 验证 I 门控降误报效果

对 pose_cache 的 npz，分别用 开/关 --enable-gate 跑融合，
对比报警帧数，验证「老人离障碍物远(I≈0)时 E 不再激活报警」。

用法:
  # 无模型，B 用常量基线（快速验证门控效果）:
  python scripts/verify_intent_gate.py --b-risk 0.5 --limit 10
  # 指定数据集:
  python scripts/verify_intent_gate.py --b-risk 0.5 --datasets Le2i
  # 指定预测窗口:
  python scripts/verify_intent_gate.py --b-risk 0.5 --intent-horizon 20

需项目完整环境（numpy + features 依赖 + pose_cache npz）。
设计见 docs/环境交互风险预测_设计.md §11。
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fuse_risk import load_env, compute_P, compute_I, fuse, CACHE  # noqa: E402
from scripts.build_config import INTENT_HORIZON, INTENT_VEL_WINDOW  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="验证 I 门控降误报效果（开/关对比）")
    p.add_argument("--b-risk", type=float, default=0.5,
                   help="B 常量基线（无模型时用，默认 0.5 落在 warning 区）")
    p.add_argument("--meta", default=None, help="个体基础 JSON")
    p.add_argument("--cache", default=str(CACHE))
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=0, help="每个数据集最多处理几个（0=全部）")
    p.add_argument("--intent-horizon", type=int, default=INTENT_HORIZON,
                   help=f"预测未来帧数（默认 {INTENT_HORIZON}）")
    p.add_argument("--intent-vel-window", type=int, default=INTENT_VEL_WINDOW,
                   help=f"速度矢量滑窗帧数（默认 {INTENT_VEL_WINDOW}）")
    p.add_argument("--out", default="verify_intent_gate.json", help="汇总输出路径")
    args = p.parse_args()

    cache = Path(args.cache)
    meta_all = json.loads(Path(args.meta).read_text(encoding="utf-8")) if args.meta else {}
    npzs = sorted(glob.glob(str(cache / "*" / "*.npz")))
    if args.datasets:
        ds_set = set(args.datasets)
        npzs = [f for f in npzs if Path(f).parent.name in ds_set]
    if args.limit:
        npzs = npzs[: args.limit]
    if not npzs:
        print(f"[ERR] 在 {cache} 下未找到 npz")
        return

    print("=" * 78)
    print("  I 门控降误报验证 —— 开/关 --enable-gate 报警帧对比")
    print("=" * 78)
    header = f"{'视频':<30} {'T':>5} {'E':>6} {'I_mean':>8} {'alarm关':>8} {'alarm开':>8} {'变化':>8}"
    print(header)
    print("-" * 78)

    rows = []
    tot_off = tot_on = 0
    for f in npzs:
        npz = Path(f)
        ds = npz.parent.name
        stem = npz.stem
        E, room, e_src, T = load_env(npz)
        P, P_ok = compute_P(meta_all.get(stem))
        B = np.full(T, float(args.b_risk), dtype=np.float32)

        # 关 gate（现有行为）
        _, _, alarm_off, _, _, _, _ = fuse(B, E, P, P_ok, room, I=None)
        n_off = int(alarm_off.sum())

        # 开 gate
        I_arr, i_src = compute_I(npz, E, args.intent_horizon, args.intent_vel_window)
        i_mean = float("nan")
        n_on = n_off
        if I_arr is not None:
            if len(I_arr) > T:
                I_arr = I_arr[:T]
            elif len(I_arr) < T:
                I_arr = np.concatenate(
                    [I_arr, np.zeros(T - len(I_arr), dtype=np.float32)])
            _, _, alarm_on, _, _, _, _ = fuse(B, E, P, P_ok, room, I=I_arr)
            n_on = int(alarm_on.sum())
            i_mean = float(I_arr.mean())

        tot_off += n_off
        tot_on += n_on
        delta = n_on - n_off
        tag = f"{ds}/{stem}"[:28]
        print(f"{tag:<30} {T:>5} {E:>6.3f} {i_mean:>8.3f} "
              f"{n_off:>8} {n_on:>8} {delta:>+8}")
        rows.append({
            "dataset": ds, "stem": stem, "T": T, "E": float(E),
            "I_mean": i_mean if not np.isnan(i_mean) else None,
            "alarm_off": n_off, "alarm_on": n_on, "delta": delta,
            "I_source": i_src,
        })

    print("-" * 78)
    print(f"{'合计':<30} {'':>5} {'':>6} {'':>8} "
          f"{tot_off:>8} {tot_on:>8} {tot_on - tot_off:>+8}")
    print("=" * 78)
    reduced = tot_off - tot_on
    direction = "减少" if reduced > 0 else ("增加" if reduced < 0 else "不变")
    print(f"  门控开启后报警帧{direction} {abs(reduced)}（{tot_off} → {tot_on}）")
    print("  → 减少 = 门控成功抑制误报；增加 = 需排查（I 可能误激活）")

    Path(args.out).write_text(json.dumps(
        {"total_alarm_off": tot_off, "total_alarm_on": tot_on,
         "reduced": reduced, "per_video": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  明细 → {args.out}")


if __name__ == "__main__":
    main()
