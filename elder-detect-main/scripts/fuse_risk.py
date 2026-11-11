#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fuse_risk.py — Phase-2 三因子复合风险融合（行为姿态B + 环境E + 个体基础P）
================================================================================
加载各分量 → 加权融合 → 联动放大 → 四级分级 + 分期 → 输出。

三因子：
  B  risk_行为姿态   ← RiskRegressor(run_03/04) 滑窗推理，或 --b-sidecar / --b-risk
                        （pose 输入已含 gait 速度与摆动）
  E  risk_环境(HOME FAST) ← env_labels/<ds>/<stem>.json（R1 VLM 细标签 → risk_from_label）；无标签→0
  P  risk_个体基础   ← 元数据规则（--meta JSON）

★ 输出要求（用户 2026-07-12）：**每种风险都单独输出，最后输出总的**。
  即先逐分量打印 B / E(HOME FAST 环境) / P'，再打印 总风险 = risk_short(短期) 与 risk_long(长期)
  （含等级 safe/watch/warning/critical + 触发因素）。

融合公式（design §6，阈值调节范式，权重来自 build_config）：
  risk_short(t) = B                              # B 唯一「此刻正在摔」信号（已含 gait）
  θ_alarm      = clamp( THR_ALARM_BASE − CTX_GAIN·ctx , FLOOR , BASE )   # ctx = CTX_W_E·E + CTX_W_P·P
  alarm(t)     ⇔ B(t) ≥ θ_alarm               # E/P 只调阈值，不进分数（治高频误报+个体敏感度）
  risk_long     = clamp( FUSE_W_E_ST·E + FUSE_W_P·P , 0,1 )           # E = HOME FAST 单分，P = 问卷个体基础

用法:
  # 完整（需 torch + 已训模型）:
  python scripts/fuse_risk.py --run-dir runs/risk_w35/run_03
  # 演示 / 无模型（B 用常量基线，仍输出全部分量+总）:
  python scripts/fuse_risk.py --b-risk 0.1
  # B 来自预先算好的 risk sidecar:
  python scripts/fuse_risk.py --b-sidecar pose_cache/<ds>/<stem>.risk.npy
  # 指定数据集 / 单视频:
  python scripts/fuse_risk.py --b-risk 0.1 --datasets Le2i --limit 3
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # 三因子融合核心依赖（B/E/P/对齐均用 numpy）

from scripts.build_config import (
    DATASET_ROOM_DEFAULT,
    FUSE_W_B, FUSE_W_E_ST, FUSE_W_P,
    CTX_W_E, CTX_W_P, CTX_GAIN, CTX_MAX,
    THR_ALARM_BASE, THR_ALARM_FLOOR, THR_WARN_BASE, THR_WARN_FLOOR,
    FUSE_LV_WATCH, FUSE_LV_WARNING, FUSE_LV_CRITICAL,
    HOME_FAST_LEVEL_LOW, HOME_FAST_LEVEL_MID,
    P_AGE_75, P_AGE_85, P_FALL_HIST, P_AID, P_ALONE, P_DIZZY, P_MAX,
    GATE_ENABLED_DEFAULT, INTENT_HORIZON, INTENT_VEL_WINDOW,
)

from model.env.env_risk_model import risk_from_label

CACHE = PROJECT_ROOT / "pose_cache"

_FUTURE_RISK_STATUSES = {
    "insufficient_quality",
    "collecting",
    "baseline_ready",
    "calibration_required",
    "calibrated",
}


def clamp(x, lo=0.0, hi=1.0):
    return float(min(max(x, lo), hi))


def load_future_risk(path):
    """读取纵向预警结果，阻止未校准结果携带概率。"""
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    status = result.get("status")
    if status not in _FUTURE_RISK_STATUSES:
        raise ValueError(f"未知纵向预警状态：{status}")
    if not str(result.get("person_id", "")).strip():
        raise ValueError("纵向预警结果缺少 person_id")
    risks = (result.get("risk_7d"), result.get("risk_30d"))
    if status != "calibrated" and any(value is not None for value in risks):
        raise ValueError("未校准的纵向结果不能包含跌倒概率")
    if status == "calibrated":
        if any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
            for value in risks
        ):
            raise ValueError("已校准概率必须是 0 到 1 之间的 JSON 数值")
    return result


def attach_future_risk(payload, path):
    """按需把独立纵向结果附加到融合汇总，不改变原风险计算。"""
    if not path:
        return payload
    combined = dict(payload)
    combined["future_risk"] = load_future_risk(path)
    return combined


# ----------------------------------------------------------------------------
# 分量加载
# ----------------------------------------------------------------------------

def load_env(npz: Path):
    """读 HOME FAST 环境标签 → 单一环境风险标量 E_hf ∈ [0,1]。

    返回 (E_hf, room, src)：
      - 优先读 R1 真实标注 env_labels/<ds>/<stem>.json（HOME FAST 细标签）→
        risk_from_label() 算出归一化 E = T/N_obs；
      - 旧 room/light/floor 格式不兼容 HOME FAST → 回退 E=0（标 old-format-skip）；
      - 无标签 → E=0（标 none）。
    HOME FAST 把「静态(房间+光照)+动态(脚下)」合一为整体居家环境评估，
    因此融合层不再区分 E_static/E_im，统一用 E_hf 进 risk_long 与阈值上下文。
    """
    ds = npz.parent.name
    stem = npz.stem
    room = DATASET_ROOM_DEFAULT.get(ds, "unknown")
    label_path = PROJECT_ROOT / "env_labels" / ds / f"{stem}.json"
    E = 0.0
    src = "none"
    if label_path.exists():
        try:
            agg = json.loads(label_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            agg = {}
        if "corridor_clear" in agg:                 # 新 HOME FAST 细标签格式
            sc = risk_from_label(agg)
            E = sc["normalized"]
            room = agg.get("room_type", room)
            src = f"R1({agg.get('model', '?')})"
        elif "room_type" in agg and "floor_hazard" in agg:  # 旧格式
            src = "old-format-skip"
    # 帧数 T（用于 B 对齐到 npz 长度）
    T = 1
    try:
        if npz.exists():
            T = int(np.load(str(npz), allow_pickle=False)["landmarks"].shape[0])
    except Exception:
        pass
    return E, room, src, T


def compute_B_from_model(npz: Path, run_dir: Path):
    """B(t)：加载 RiskRegressor，对 npz 滑窗推理。需 torch + features 模块。"""
    try:
        import torch
        from model.lstm.risk_model import RiskRegressor
        from features.feature_builder import FrameFeatureBuilder
        from features.gait import GaitSpeedExtractor
        from sequence import WindowConfig, SlidingWindowBuilder
        from scripts.build_config import WINDOW_SIZE, STRIDE
    except Exception as e:  # noqa: BLE001
        print(f"    [WARN] B: 无法加载模型依赖({e})，跳过模型推理")
        return None, "unavailable"

    try:
        info = json.loads((run_dir / "model_info.json").read_text(encoding="utf-8"))
        input_dim = int(info.get("input_dim", 29))
        window = int(info.get("window", WINDOW_SIZE))
        model = RiskRegressor(input_dim=input_dim)
        model.load_state_dict(torch.load(str(run_dir / "best_model.pt"),
                                      map_location="cpu", weights_only=True))
        model.eval()
    except Exception as e:  # noqa: BLE001
        print(f"    [WARN] B: 模型加载失败({e})")
        return None, "unavailable"

    try:
        from scripts.build_from_cache import _make_pose_result
        d = np.load(str(npz), allow_pickle=False)
        T = int(d["landmarks"].shape[0])
        fw, fh = int(d["frame_width"]), int(d["frame_height"])
        gait = GaitSpeedExtractor()
        fb = FrameFeatureBuilder()
        feats = []
        for t in range(T):
            res = _make_pose_result(d["landmarks"][t], d["world_landmarks"][t],
                                     d["visibility"][t], d["timestamps"][t], fw, fh)
            gs = gait.compute_per_frame(res)
            feats.append(fb.build(res, gs))
        X = np.array(feats, dtype=np.float32)
        wb = SlidingWindowBuilder(WindowConfig(window_size=window, stride=STRIDE))
        # 用分类 windower 仅取 X 窗口（labels 占位）
        wins = wb.build_from_array(X, np.zeros(T, dtype=np.int64))[0]  # (N, W, dim)
        with torch.no_grad():
            out = model(torch.from_numpy(wins).float()).numpy().reshape(-1)
        # 窗口 → 中心帧对齐
        centers = np.arange(window // 2, T - (window // 2), STRIDE)
        B = np.zeros(T, dtype=np.float32)
        m = min(len(out), len(centers))
        B[centers[:m]] = out[:m]
        return B, "model"
    except Exception as e:  # noqa: BLE001
        print(f"    [WARN] B: 推理失败({e})")
        return None, "unavailable"

def compute_P(meta: dict | None):
    """risk_个体基础 P ∈ [0,1]，报告 §6.2。缺元数据返回 (0.0, False)。"""
    if not meta:
        return 0.0, False
    s = (P_AGE_75 * int(meta.get("age", 0) >= 75)
          + P_AGE_85 * int(meta.get("age", 0) >= 85)
          + P_FALL_HIST * int(meta.get("fall_history_12m", 0) >= 1)
          + P_AID * int(bool(meta.get("uses_walking_aid", False)))
          + P_ALONE * int(bool(meta.get("living_alone", False)))
          + P_DIZZY * int(bool(meta.get("dizziness", False))))
    return clamp(s / P_MAX), True


def compute_I(npz: Path, E_semantic: float,
              horizon: int = INTENT_HORIZON, vel_window: int = INTENT_VEL_WINDOW):
    """I 门控：逐帧计算干涉系数 I(t) = 预测区域 Zone(t) 落在环境掩码 M(t) 内的占比。

    需 pose npz（landmarks/world_landmarks）。无碰撞 → I=0 → E_effective=0（环境风险归零）。

    Args:
        npz: pose_cache/<ds>/<stem>.npz
        E_semantic: HOME FAST 视频级环境分（v0 用 E>0 判定 has_hazard）
        horizon: 预测未来帧数
        vel_window: 速度矢量滑窗帧数

    Returns:
        (I_array, src)：I_array=(T,) float32；失败返回 (None, "unavailable")
    """
    try:
        from features.intent import MotionIntentProjector
        from features.env_mask import EnvMaskBuilder
        from features.intent_gate import IntentGate
        from scripts.build_from_cache import _make_pose_result
    except Exception as e:  # noqa: BLE001
        print(f"    [WARN] I门控: 依赖加载失败({e})，跳过")
        return None, "unavailable"

    try:
        d = np.load(str(npz), allow_pickle=False)
        T = int(d["landmarks"].shape[0])
        fw, fh = int(d["frame_width"]), int(d["frame_height"])
    except Exception as e:  # noqa: BLE001
        print(f"    [WARN] I门控: npz读取失败({e})")
        return None, "unavailable"

    projector = MotionIntentProjector(horizon=horizon, vel_window=vel_window)
    mask_builder = EnvMaskBuilder()
    gate = IntentGate()
    has_hazard = E_semantic > 0.0   # v0 粗判定：有环境隐患 → 地面ROI填1

    I = np.zeros(T, dtype=np.float32)
    for t in range(T):
        res = _make_pose_result(d["landmarks"][t], d["world_landmarks"][t],
                                 d["visibility"][t], d["timestamps"][t], fw, fh)
        zone = projector.compute_per_frame(res)
        mask = mask_builder.build_per_frame(res, has_hazard)
        I[t] = gate.compute(zone, mask)
    return I, "computed"


# ----------------------------------------------------------------------------
# 融合 + 分级 + 分期
# ----------------------------------------------------------------------------

def level_of(r: float) -> str:
    if r < FUSE_LV_WATCH:
        return "safe"
    if r < FUSE_LV_WARNING:
        return "watch"
    if r < FUSE_LV_CRITICAL:
        return "warning"
    return "critical"

def fuse(B, E, P, P_available, room, I=None):
    """三因子融合 + 可选 I 门控。

    Args:
        I: 干涉系数逐帧数组 (T,)，开启 ``--enable-gate`` 时传入。
           开启时 ``E_effective(t) = E × I(t)``（无碰撞 → E 归零），
           仅影响 risk_short 的阈值上下文；risk_long 仍用 E_semantic。
           None 时退化为现有行为（E 为标量调阈值，向后兼容）。
    """
    T = len(B) if B is not None else 1
    if B is None:
        B = np.zeros(T, dtype=np.float32)

    # 短期：B 是唯一「此刻正在摔」信号（环境 E 与个体 P 不进分数）
    short = np.clip(FUSE_W_B * B, 0.0, 1.0)   # FUSE_W_B=1.0 → = B

    # 环境有效分：I 门控开启时逐帧 E_eff(t) = E × I(t)，否则 E_eff = E（标量广播）
    if I is not None:
        E_eff = np.clip(E * I, 0.0, 1.0)          # 逐帧 (T,)
    else:
        E_eff = E                                  # 标量（现有行为）

    # 上下文灵敏度：E_eff 与 P 只调阈值，不进分数
    #   （治设计评审 #2/#3：场景绝不凭空把良性弯腰推过报警线）
    E_ctx = np.clip(E_eff, 0.0, 1.0)
    Pv = P if P_available else 0.0
    ctx = np.clip(CTX_W_E * E_ctx + CTX_W_P * Pv, 0.0, CTX_MAX)
    # 报警/预警阈值随上下文下调（越危险越敏感），但有**硬下限**——
    # 任何上下文都救不回 B<下限 的良性动作（弯腰/蹲下/捡物永不报警）。
    theta_alarm = np.clip(THR_ALARM_BASE - CTX_GAIN * ctx,
                             THR_ALARM_FLOOR, THR_ALARM_BASE)
    theta_warn = np.clip(THR_WARN_BASE - CTX_GAIN * ctx,
                          THR_WARN_FLOOR, THR_WARN_BASE)
    alarm = short >= theta_alarm
    warn = (short >= theta_warn) & (~alarm)

    # 长期：静态环境（房间+光照） + 个体基础 P（问卷），不变（用 E_semantic，不受 I 门控）
    Pw = FUSE_W_P if P_available else 0.0
    long = clamp(FUSE_W_E_ST * E + Pw * Pv)

    # 触发因素（可解释：哪些上下文压低了阈值）
    triggers = []
    if E >= 0.5:
        triggers.append(f"{room}_scene")
    if P_available and P >= 0.5:
        triggers.append("high_individual_risk")
    if float(ctx.max()) > 0:
        triggers.append(f"threshold_lowered(θ_alarm={float(np.mean(theta_alarm)):.2f})")

    return short, long, alarm, warn, theta_alarm, theta_warn, triggers

def fmt(arr):
    a = np.asarray(arr, dtype=np.float32)
    return f"mean={a.mean():.3f}  max={a.max():.3f}"

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Phase-2 三因子融合（每种风险 + 总风险）")
    p.add_argument("--run-dir", default=None, help="已训 RiskRegressor 目录（算 B）")
    p.add_argument("--b-sidecar", default=None, help="预计算 B 的 <stem>.risk.npy")
    p.add_argument("--b-risk", type=float, default=None, help="B 常量基线（无模型时用）")
    p.add_argument("--meta", default=None, help="个体基础 JSON：{stem:{age,fall_history_12m,...}}")
    p.add_argument("--cache", default=str(CACHE))
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=0, help="每个数据集最多处理几个（0=全部）")
    p.add_argument("--out", default="fuse_metrics.json", help="汇总输出路径")
    p.add_argument("--future-risk-json", default=None,
                   help="可选：附加纵向个人基线预警 latest_risk.json，不参与B/E/P/I计算")
    # ---- I 门控（环境交互风险预测，见 docs/环境交互风险预测_设计.md）----
    p.add_argument("--enable-gate", action="store_true",
                   default=GATE_ENABLED_DEFAULT,
                   help="开启 I 门控：E_effective(t)=E×I(t)，无碰撞→E归零（默认关，向后兼容）")
    p.add_argument("--intent-horizon", type=int, default=INTENT_HORIZON,
                   help=f"预测未来帧数（默认 {INTENT_HORIZON}，≈1.5s@23fps）")
    p.add_argument("--intent-vel-window", type=int, default=INTENT_VEL_WINDOW,
                   help=f"速度矢量滑窗帧数（默认 {INTENT_VEL_WINDOW}）")
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

    results = []
    print("=" * 78)
    print("  Phase-2 三因子复合风险融合 —— 每种风险单独输出，最后输出总风险")
    print("=" * 78)

    for f in npzs:
        npz = Path(f)
        ds = npz.parent.name
        stem = npz.stem

        # ---- 各分量 ----
        E_hf, room, e_src, T = load_env(npz)
        P, P_ok = compute_P(meta_all.get(stem))

        # B 来源判定
        B, b_src = None, "none"
        if args.run_dir:
            B, b_src = compute_B_from_model(npz, Path(args.run_dir))
        if B is None and args.b_sidecar:
            sp = Path(args.b_sidecar)
            cand = sp if sp.exists() else cache / ds / f"{stem}.risk.npy"
            if cand.exists():
                B = np.load(str(cand)).astype(np.float32)[:T]
                b_src = "sidecar"
        if B is None and args.b_risk is not None:
            B = np.full(T, float(args.b_risk), dtype=np.float32)
            b_src = "const"

        # B 长度对齐到 T（sidecar 截/补零）
        if B is not None and len(B) != T:
            if len(B) > T:
                B = B[:T]
            else:
                B = np.concatenate([B, np.zeros(T - len(B), dtype=np.float32)])

        # I 门控（可选）：开启时逐帧算 I(t)，E_effective=E×I
        I_arr, i_src = None, "off"
        if args.enable_gate:
            I_arr, i_src = compute_I(npz, E_hf,
                                       args.intent_horizon, args.intent_vel_window)
            # I 长度对齐到 T
            if I_arr is not None and len(I_arr) != T:
                if len(I_arr) > T:
                    I_arr = I_arr[:T]
                else:
                    I_arr = np.concatenate(
                        [I_arr, np.zeros(T - len(I_arr), dtype=np.float32)])

        short, long, alarm, warn, th_alarm, th_warn, triggers = \
            fuse(B, E_hf, P, P_ok, room, I=I_arr)
        # ---- 输出：每种风险单独列 ----
        print(f"\n● 视频: {ds}/{stem}  (T={T}, room={room})")
        print(f"  ── 各分量风险 ──")
        Pp = P if P_ok else 0.0
        print(f"  B  risk_行为姿态  : {fmt(B)}   来源={b_src}")
        E_level = ("低" if E_hf*25 <= HOME_FAST_LEVEL_LOW else "中" if E_hf*25 <= HOME_FAST_LEVEL_MID else "高")
        print(f"  E  risk_环境(HOME FAST): {E_hf:.3f}   来源={e_src}  分级={E_level}（≈{E_hf*25:.0f}/25）")
        if args.enable_gate and I_arr is not None:
            print(f"  I  干涉系数        : mean={I_arr.mean():.3f}  max={I_arr.max():.3f}  "
                  f"归零帧={int((I_arr==0).sum())}/{T}  来源={i_src}")
        print(f"  P  risk_个体基础  : {P:.3f}  (P'={Pp:.3f})  来源={'元数据' if P_ok else '缺省(权重归零)'}")
        # ---- 输出：总风险（阈值调节范式）----
        theta_a = float(np.mean(th_alarm))
        theta_w = float(np.mean(th_warn))
        n_alarm = int(alarm.sum())
        n_warn_f = int(warn.sum())
        if n_alarm > 0:
            lvl = "critical/danger"
        elif n_warn_f > 0:
            lvl = "warning"
        elif float(short.mean()) >= FUSE_LV_WATCH:
            lvl = "watch"
        else:
            lvl = "safe"
        print(f"  ── 总风险 ──")
        print(f"  risk_short(短期B): {fmt(short)}  等级={lvl}")
        print(f"  有效报警阈值 θ_alarm: {theta_a:.2f}  "
              f"(基准 0.60, 下限 0.45；因上下文{'下调' if theta_a < 0.60 else '未动'})")
        print(f"  有效预警阈值 θ_warn : {theta_w:.2f}")
        print(f"  risk_long (长期总): {long:.3f}    等级={level_of(long)}")
        print(f"  报警帧(≥θ_alarm): {n_alarm}/{len(short)}   "
              f"预警帧: {n_warn_f}  触发因素: {triggers or '无'}")
        results.append({
            "dataset": ds, "stem": stem, "room": room, "T": T,
            "B_source": b_src,
            "B_mean": float(np.asarray(B).mean()), "E_homefast": float(E_hf),
            "gate_enabled": bool(args.enable_gate),
            "I_mean": float(I_arr.mean()) if I_arr is not None else None,
            
            "P": P, "risk_short_mean": float(short.mean()),
            "risk_short_max": float(short.max()),
            "risk_long": float(long),
            "alarm_frames": n_alarm, "triggers": triggers,
        })

    # 汇总
    if results:
        import statistics as st
        agg = {
            "n_videos": len(results),
            "B_mean_avg": round(st.mean(r["B_mean"] for r in results), 3),
            "E_homefast_avg": round(st.mean(r["E_homefast"] for r in results), 3),
            "risk_short_mean_avg": round(st.mean(r["risk_short_mean"] for r in results), 3),
            "risk_long_avg": round(st.mean(r["risk_long"] for r in results), 3),
            "total_alarm_frames": sum(r["alarm_frames"] for r in results),
        }
        output_payload = attach_future_risk(
            {"aggregate": agg, "per_video": results}, args.future_risk_json)
        Path(args.out).write_text(json.dumps(
            output_payload, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print("\n" + "=" * 78)
        print("  汇总（总风险平均）:")
        print(f"    B均={agg['B_mean_avg']}  "
              f"E(HOME FAST)均={agg['E_homefast_avg']}  risk_short均={agg['risk_short_mean_avg']}  "
              f"risk_long均={agg['risk_long_avg']}")
        print(f"    全部报警帧合计: {agg['total_alarm_frames']}")
        print(f"  明细 → {args.out}")

if __name__ == "__main__":
    main()
