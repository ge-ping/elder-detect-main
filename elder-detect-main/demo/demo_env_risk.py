#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demo/demo_env_risk.py
========================
环境风险（EnvRisk）独立 demo —— 传入一个视频，输出 HOME FAST 量表评分卡。

设计原则（与 env_perceive / fuse_risk 完全一致）：
  - 环境一个模型：HomeFastModel（HOME FAST 25 题规则表，零训练、零推理成本）。
  - 标签来源 R1（VLM 看画面，打 HOME FAST 细标签）/ R0（问卷直传）共用同一套规则表。
  - 本 demo 只算「环境分数」，不做姿态 RiskRegressor、不做融合。

分数清单（输出内容）：
  ① 原始标签        VLM 细标签（corridor_clear / toilet_grab_bar / pet_* / lighting ...）
  ② 25 题逐题判定   每题 0(安全)/1(隐患)/—(N/A 未观察或场景不符)
  ③ 总分 / 归一化  T(0~25) / E = T / N_obs ∈ [0,1]
  ④ 风险分级        0~4 低 / 5~7 中 / ≥8 高（HOME FAST 官方标准）
  ⑤ 融合联动（可选 --p）→ E 进 ctx → θ_alarm / θ_warn（展示 E 如何调灵敏度）

默认零 token：
  - 若 env_labels/<ds>/<stem>.json 已存在（之前跑过 R1 真实标注）→ 直接复用，不烧 API。
  - 仅 --r1 才会真正抽帧 + 调 VLM（耗 token，谨慎使用）。
  - 也可用 --answers 直接传问卷答（R0 模式，零 token）。

运行:
    # 传入已标注过的视频（免费复用 R1 标签）
    python demo/demo_env_risk.py --video "test/video/Lecture room/video (2).avi"

    # 直接传 HOME FAST 问卷答（R0，零 token）：题号->0/1，N/A 题不写
    python demo/demo_env_risk.py --answers '{"1":1,"3":1,"7":1,"10":1,"11":1,"12":1,"13":1,"14":1}'

    # 传入新视频，跑真实 VLM 感知（耗 token！需 .env 里 ARK_VLM_MODEL 为有效 endpoint）
    python demo/demo_env_risk.py --video some_clip.mp4 --r1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_config import (
    HOME_FAST_ITEMS, HOME_FAST_NA_SCENE, HOME_FAST_N,
    HOME_FAST_COVERAGE_FLOOR,
    HOME_FAST_LEVEL_LOW, HOME_FAST_LEVEL_MID,
    CTX_W_E, CTX_W_P, CTX_GAIN, CTX_MAX,
    THR_ALARM_BASE, THR_ALARM_FLOOR, THR_WARN_BASE, THR_WARN_FLOOR,
)
from model.env.env_risk_model import HomeFastModel

MODEL = HomeFastModel()
LABELS_ROOT = PROJECT_ROOT / "env_labels"
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def clamp(x, lo, hi):
    return float(min(max(x, lo), hi))


def band(r: float) -> str:
    if r < 0.30:
        return "低"
    if r < 0.60:
        return "中"
    return "高"


# ---------------------------------------------------------------
# 标签获取
# ---------------------------------------------------------------
def load_r1_label(ds: str, stem: str) -> dict | None:
    """复用已存在的 R1 真实标签（免费）。不存在返回 None。"""
    jp = LABELS_ROOT / ds / f"{stem}.json"
    if not jp.exists():
        return None
    try:
        return json.loads(jp.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def derive_videos(video_arg: str):
    """单文件 → [(ds, stem, path)]；目录 → 遍历所有视频。"""
    p = Path(video_arg).resolve()
    out = []
    if p.is_file():
        out.append((p.parent.name, p.stem, p))
    elif p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.suffix.lower() in VIDEO_SUFFIXES:
                out.append((p.name, f.stem, f))
    return out


def run_r1_pipeline(video_arg: str):
    """抽帧 + 真实调 VLM（耗 token）。复用现有脚本，保持一致性。"""
    print("  [--r1] 运行真实 VLM 感知管线（抽帧 + 标注，将消耗 token）...")
    subprocess.run(
        [sys.executable, "scripts/env_extract_frames.py", "--video_root", str(video_arg)],
        cwd=str(PROJECT_ROOT), check=True,
    )
    subprocess.run(
        [sys.executable, "scripts/env_label_vlm.py", "--index", "frames/index.csv"],
        cwd=str(PROJECT_ROOT), check=True,
    )


def _load_answers(s: str) -> dict:
    """解析 --answers：文件路径或内联 JSON → {题号(int): 0/1}。"""
    p = Path(s)
    raw = p.read_text(encoding="utf-8") if p.exists() else s
    d = json.loads(raw)
    return {int(k): int(v) for k, v in d.items()}


# ---------------------------------------------------------------
# 分数计算 + 打印
# ---------------------------------------------------------------
def _mark(v) -> str:
    if v is None:
        return "— N/A"
    return "✓ 安全(0)" if v == 0 else "✗ 隐患(1)"


def score_card(labels: dict, source: str, room: str = "unknown",
             agg: dict | None = None, p: float | None = None,
             is_answers: bool = False):
    """打印 HOME FAST 评分卡。

    labels : VLM 细标签 dict（is_answers=False）或 {题号:0/1} 问卷答（is_answers=True）
    """
    if is_answers:
        # 问卷模式：补齐 25 题，未传题默认 0（安全）；
        # 只计传入题的「部分覆盖」语义留给 R1（from_labels 已剔除 N/A）
        answers = {i: int(labels.get(i, 0)) for i in range(1, HOME_FAST_N + 1)}
    else:
        answers = MODEL.from_labels(labels, room)

    sc = MODEL.score(answers)
    total, n_obs, E, level = sc["total"], sc["n_observed"], sc["normalized"], sc["level"]
    coverage, low_cov = sc["coverage"], sc["low_coverage"]
    # need_review 合并：VLM 置信度不足(R1 agg) OR 覆盖度不足 任一即触发
    vlm_nr = bool(agg.get("need_review")) if (agg and not is_answers) else False
    need_review = vlm_nr or low_cov

    line = "=" * 68
    print(line)
    print(f"  HOME FAST 环境风险评分卡   来源={source}")
    print(line)
    if not is_answers and agg:
        vr = agg.get("confidence")
        if vr is not None:
            rev = "⚠需复核" if need_review else "无需复核"
            print(f"  【原始 VLM 细标签】 room={room!r}  lighting={agg.get('lighting')!r}  "
                  f"confidence={vr}  need_review={need_review}({rev})")
            if low_cov:
                print(f"    ⚠ 覆盖度不足：N_obs={n_obs}/{HOME_FAST_N}={coverage:.2f} "
                      f"< {HOME_FAST_COVERAGE_FLOOR:.2f}，E 仅反映已观察部分、不可全信")
            # 打印关键 bool 标签
            keys = ["corridor_clear", "floor_intact", "floor_antislip", "mat_fixed",
                     "toilet_grab_bar", "bath_accessible", "bath_grab_bar",
                     "bath_antislip_mat", "indoor_stair_rail", "stair_edge_clear",
                     "entry_safe", "shoe_antislip", "pet_present", "pet_safe"]
            shown = {k: agg.get(k) for k in keys if k in agg}
            if shown:
                print(f"    {shown}")
    print("-" * 68)
    print("  【25 题逐题判定】（✓=安全0分 / ✗=隐患1分 / —=N/A 未观察或场景不符）")
    for qid, module, q, visual, na in HOME_FAST_ITEMS:
        v = answers.get(qid)
        print(f"   [{qid:>2}] {module:<6} {q:<22} {_mark(v)}")
    print("-" * 68)
    print("  【总分 / 归一化 / 分级】")
    print(f"    绝对总分 T      = {total:>2} / 25")
    print(f"    已观察题数 N_obs = {n_obs} / {HOME_FAST_N}   "
          f"（已剔除 N/A 题，覆盖度={coverage:.2f}）")
    print(f"    归一化 E = T/N_obs = {E:.3f}    ← 进融合的环境风险分")
    print(f"    风险分级        = {level}   "
          f"（HOME FAST: 0~{HOME_FAST_LEVEL_LOW}低 / "
          f"{HOME_FAST_LEVEL_LOW+1}~{HOME_FAST_LEVEL_MID}中 / ≥{HOME_FAST_LEVEL_MID+1}高）")
    print("-" * 68)

    if p is not None:
        p = float(min(max(p, 0.0), 1.0))
        ctx = clamp(CTX_W_E * E + CTX_W_P * p, 0.0, CTX_MAX)
        theta_alarm = clamp(THR_ALARM_BASE - CTX_GAIN * ctx, THR_ALARM_FLOOR, THR_ALARM_BASE)
        theta_warn = clamp(THR_WARN_BASE - CTX_GAIN * ctx, THR_WARN_FLOOR, THR_WARN_BASE)
        print("  【融合联动（给定 P=个体基础，展示 E 如何一起调灵敏度）】")
        print(f"    ctx       = clamp({CTX_W_E:.2f}·E + {CTX_W_P:.2f}·P, 0, {CTX_MAX:.2f})"
              f" = {ctx:.3f}")
        print(f"    θ_alarm   = clamp({THR_ALARM_BASE:.2f} - {CTX_GAIN:.2f}·ctx, "
              f"{THR_ALARM_FLOOR:.2f}, {THR_ALARM_BASE:.2f}) = {theta_alarm:.3f}")
        print(f"    θ_warn    = clamp({THR_WARN_BASE:.2f} - {CTX_GAIN:.2f}·ctx, "
              f"{THR_WARN_FLOOR:.2f}, {THR_WARN_BASE:.2f}) = {theta_warn:.3f}")
        print(f"    → 本环境+个体下：B≥{theta_alarm:.3f} 即报警（基准 {THR_ALARM_BASE:.2f}，"
              f"被下调更灵敏）；B≥{theta_warn:.3f} 触发预警喝止。")
        print("-" * 68)
    else:
        print("  （未给 --p，跳过融合联动；加 --p <0~1> 看 E 如何调阈值）")
        print("-" * 68)
    print()
    return {"E": E, "total": total, "n_observed": n_obs,
            "coverage": coverage, "low_coverage": low_cov,
            "need_review": need_review, "level": level}


def process_video(ds, stem, vpath, args):
    print(f"\n● 视频: {ds}/{stem}  ({vpath.name})")
    if args.r1:
        run_r1_pipeline(str(vpath))
    agg = load_r1_label(ds, stem)
    if agg is None:
        print("    ⚠ 无已存在的 R1 标签。请加 --r1 跑真实感知，或用 --answers 问卷模式。")
        return None
    # 旧格式（room/light/floor）不兼容 HOME FAST
    if "floor_hazard" in agg and "corridor_clear" not in agg:
        print("    ⚠ 检测到旧格式 R1 标签(room/light/floor)，HOME FAST 不兼容，请 --r1 重打。")
        return None
    room = agg.get("room_type", "unknown")
    source = f"R1(VLM真实感知, endpoint={agg.get('model')})"
    return score_card(agg, source, room, agg=agg, p=args.p)


def main():
    p = argparse.ArgumentParser(
        description="环境风险 EnvRisk demo —— HOME FAST 量表评分卡")
    p.add_argument("--video", default=None, help="视频文件或目录（传入即视频模式）")
    p.add_argument("--r1", action="store_true",
                   help="(耗token) 先跑真实 VLM 感知管线（抽帧+标注）再算分")
    p.add_argument("--answers", default=None,
                   help="R0 问卷模式：HOME FAST 答 JSON（文件路径或内联），"
                        "如 '{\"1\":1,\"3\":1}'（题号->0/1，N/A题不写）")
    p.add_argument("--p", type=float, default=None,
                   help="个体基础风险 P∈[0,1]，给定后展示 E 如何一起调阈值")
    args = p.parse_args()

    if args.video:
        vids = derive_videos(args.video)
        if not vids:
            print(f"[ERR] 在 {args.video} 下未找到视频文件")
            return
        print(f"EnvRisk demo — 处理 {len(vids)} 个视频（HOME FAST 评分卡）")
        for ds, stem, vpath in vids:
            process_video(ds, stem, vpath, args)
    elif args.answers:
        try:
            answers = _load_answers(args.answers)
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] --answers 解析失败: {e}")
            return
        print("EnvRisk demo — R0 问卷模式（HOME FAST 答直传）")
        score_card(answers, "R0(问卷直传, 免费)", room="questionnaire",
                   p=args.p, is_answers=True)
    else:
        print("用法:")
        print('  python demo/demo_env_risk.py --video "test/video/Lecture room/video (2).avi"')
        print("  python demo/demo_env_risk.py --video clip.mp4 --r1        # 耗token")
        print('  python demo/demo_env_risk.py --answers \'{"1":1,"3":1,"7":1,"10":1,"11":1,"12":1,"13":1,"14":1}\'')
        print("  （--p <0~1> 可叠加个体基础，看 E 如何调阈值）")
        print("\n（不传 --video 也不传 --answers 时显示本帮助）")


if __name__ == "__main__":
    main()
