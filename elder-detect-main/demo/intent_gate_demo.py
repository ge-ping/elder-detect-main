#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intent_gate_demo.py — 环境交互风险预测完整 demo

端到端跑单个视频：
  B = 真实 RiskRegressor 模型推理（姿态风险，含 gait 速度）
  E = 合成环境分（--force-e，因 env_labels 未覆盖全量 npz）
  I = compute_I 真实 pose 逐帧干涉系数（运动意图 × 环境掩码门控）
  P = 0（本 demo 不考虑生理因素）
  融合：fuse(B, E, P=0, I) 开/关门控对比

输出：逐帧风险曲线 PNG + 控制台摘要 + JSON。

用法:
  python demo/intent_gate_demo.py
  python demo/intent_gate_demo.py --npz pose_cache/CAUCAFall/Subject.10_Fall_backwards.npz
  python demo/intent_gate_demo.py --run-dir model/lstm/risk_v1/run_03 --force-e 0.5
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端，直接存 PNG
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fuse_risk import compute_B_from_model, compute_I, fuse  # noqa: E402
from scripts.build_config import FUSE_LV_WATCH, FUSE_LV_WARNING, FUSE_LV_CRITICAL  # noqa: E402


def synth_fall_risk(T: int) -> np.ndarray:
    """B 模型不可用时的合成跌倒 risk 曲线：前70%正常→爬升→跌倒=1.0。"""
    r = np.full(T, 0.15, dtype=np.float32)
    fall_start = int(T * 0.7)
    fall_peak = int(T * 0.9)
    n = max(fall_peak - fall_start, 1)
    r[fall_start:fall_peak] = 0.3 + 0.6 * (1 - np.exp(-np.arange(n) / 5.0))
    r[fall_peak:] = 1.0
    return r


def main():
    p = argparse.ArgumentParser(description="环境交互风险预测完整 demo")
    p.add_argument("--npz", default=None, help="pose npz 路径（默认取第一个 CAUCAFall Fall 视频）")
    p.add_argument("--run-dir", default="model/lstm/risk_v1/run_03",
                   help="RiskRegressor 目录（算真实 B）")
    p.add_argument("--force-e", type=float, default=0.5,
                   help="合成环境分 E（env_labels 未覆盖时用）")
    p.add_argument("--b-risk", type=float, default=None,
                   help="B 常量基线（提供则不用模型，用于演示门控对边缘 B 的降误报）")
    p.add_argument("--out", default="demo/intent_gate_demo.png", help="输出 PNG 路径")
    p.add_argument("--out-json", default="demo/intent_gate_demo.json", help="输出 JSON 路径")
    args = p.parse_args()

    # 选视频
    npz = Path(args.npz) if args.npz else None
    if npz is None:
        cand = sorted(glob.glob(str(PROJECT_ROOT / "pose_cache" / "CAUCAFall" / "*Fall*.npz")))
        if not cand:
            print("[ERR] 未找到 CAUCAFall Fall npz，请用 --npz 指定")
            return
        npz = Path(cand[0])
    if not npz.exists():
        print(f"[ERR] npz 不存在: {npz}")
        return

    T = int(np.load(str(npz), allow_pickle=False)["landmarks"].shape[0])
    E = float(args.force_e)
    room = "demo"
    print("=" * 72)
    print(f"  环境交互风险预测 demo —— {npz.parent.name}/{npz.stem}")
    print(f"  T={T} 帧  E={E}(合成)  P=0(不考虑)  run_dir={args.run_dir}")
    print("=" * 72)

    # ---- B：--b-risk 常量 优先；否则真实模型，失败 fallback 合成 ----
    if args.b_risk is not None:
        B = np.full(T, float(args.b_risk), dtype=np.float32)
        b_src = f"const({args.b_risk})"
    else:
        B, b_src = compute_B_from_model(npz, Path(args.run_dir))
        if B is None:
            print(f"  [WARN] B 模型推理失败，用合成跌倒曲线")
            B = synth_fall_risk(T)
            b_src = "synth"
        B = B[:T] if len(B) >= T else np.concatenate([B, np.zeros(T - len(B), dtype=np.float32)])

    # ---- I：真实 pose 逐帧门控 ----
    I_arr, i_src = compute_I(npz, E)
    if I_arr is None:
        print("[ERR] compute_I 失败，无法演示门控")
        return
    if len(I_arr) > T:
        I_arr = I_arr[:T]
    elif len(I_arr) < T:
        I_arr = np.concatenate([I_arr, np.zeros(T - len(I_arr), dtype=np.float32)])

    # ---- 融合：开/关门控对比（P=0 不考虑）----
    s_off, _, a_off, _, ta_off, _, _ = fuse(B, E, 0.0, False, room, I=None)
    s_on, _, a_on, _, ta_on, _, _ = fuse(B, E, 0.0, False, room, I=I_arr)

    n_off, n_on = int(a_off.sum()), int(a_on.sum())
    print(f"\n  B   mean={B.mean():.3f}  max={B.max():.3f}  来源={b_src}")
    print(f"  I   mean={I_arr.mean():.3f}  max={I_arr.max():.3f}  I>0帧={int((I_arr>0).sum())}/{T}")
    print(f"  E   ={E:.3f} (合成)  →  E_eff(开) mean={(E*I_arr).mean():.3f}")
    print(f"  ── 报警帧对比 ──")
    print(f"  门控关: {n_off}/{T}  ({n_off/T*100:.1f}%)")
    print(f"  门控开: {n_on}/{T}  ({n_on/T*100:.1f}%)")
    print(f"  变化  : {n_on - n_off:+d} 帧  ({'降误报' if n_on < n_off else '需排查'})")
    print(f"  θ_alarm 关 mean={ta_off.mean():.3f}  开 mean={ta_on.mean():.3f}")

    # ---- 画图 ----
    t = np.arange(T)
    # theta_alarm 关门控时为标量（E_eff=E 常数 → θ 常数），统一成 (T,) 数组
    ta_off = np.full(T, float(np.mean(ta_off))) if np.ndim(ta_off) == 0 else np.asarray(ta_off, float)
    ta_on = np.full(T, float(np.mean(ta_on))) if np.ndim(ta_on) == 0 else np.asarray(ta_on, float)
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    fig.suptitle(f"Fall Risk Demo: B + E×I gate  |  {npz.parent.name}/{npz.stem}  (P off)",
                 fontsize=13, fontweight="bold")

    # 子图1: B 姿态风险
    ax = axes[0]
    ax.plot(t, B, "b-", lw=1.2, label="B (pose risk, model)")
    ax.axhline(FUSE_LV_WARNING, color="orange", ls=":", lw=1, label=f"warn {FUSE_LV_WARNING}")
    ax.axhline(FUSE_LV_CRITICAL, color="red", ls="--", lw=1, label=f"alarm {FUSE_LV_CRITICAL}")
    ax.set_ylabel("B risk"); ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(f"B source = {b_src}", fontsize=10, loc="left")

    # 子图2: I + E_effective
    ax = axes[1]
    ax.plot(t, I_arr, "-", color="darkorange", lw=1.2, label="I (intent, overlap ratio)")
    ax.plot(t, E * I_arr, "-", color="green", lw=1.2, label=f"E_eff = E×I (gate on)")
    ax.axhline(E, color="green", ls="--", lw=1, label=f"E = {E} (gate off, const)")
    ax.set_ylabel("E / I"); ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("I 门控：无碰撞→I=0→E_eff归零；有碰撞→I>0→E_eff激活", fontsize=10, loc="left")

    # 子图3: risk_short + θ_alarm 开/关
    ax = axes[2]
    ax.plot(t, s_on, "r-", lw=1.2, label="risk_short (= B, 开关相同)")
    ax.plot(t, ta_off, color="gray", ls="--", lw=1.2, label="θ_alarm (gate off)")
    ax.plot(t, ta_on, color="black", ls="-", lw=1.2, label="θ_alarm (gate on)")
    ax.fill_between(t, ta_on, s_on, where=s_on >= ta_on, alpha=0.25, color="red",
                    label="alarm region (on)")
    ax.set_ylabel("risk / threshold"); ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("门控开→θ_alarm 回升（无碰撞时不敏感）→ 报警区缩小", fontsize=10, loc="left")

    # 子图4: alarm 标记
    ax = axes[3]
    if n_off > 0:
        ax.scatter(t[a_off], np.ones(n_off), color="red", marker="x", s=30, label=f"alarm off ({n_off})")
    if n_on > 0:
        ax.scatter(t[a_on], np.ones(n_on) * 0.6, color="blue", marker="o", s=20, label=f"alarm on ({n_on})")
    ax.set_yticks([]); ax.set_xlabel("frame")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(f"报警帧：关 {n_off} → 开 {n_on}（{n_on - n_off:+d}）", fontsize=10, loc="left")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=110)
    print(f"\n  曲线图 → {args.out}")

    Path(args.out_json).write_text(json.dumps({
        "video": f"{npz.parent.name}/{npz.stem}", "T": T, "B_source": b_src,
        "E": E, "I_mean": float(I_arr.mean()), "I_pos_frames": int((I_arr > 0).sum()),
        "alarm_off": n_off, "alarm_on": n_on, "delta": n_on - n_off,
        "theta_alarm_off_mean": float(ta_off.mean()), "theta_alarm_on_mean": float(ta_on.mean()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  明细   → {args.out_json}")
    print("=" * 72)


if __name__ == "__main__":
    main()
