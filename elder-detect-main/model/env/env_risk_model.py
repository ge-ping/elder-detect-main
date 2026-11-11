#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model/env/env_risk_model.py
===========================
环境风险模型（EnvRisk）——「环境一个模型」的模型接口 + v0 规则实现。

设计定位（见 model/lstm/envrisk_v0_design.md）：
  - 双模型原则：姿态 = RiskRegressor（Phase-1 已训），环境 = EnvRisk（本文件）。
  - 二者解耦，各自独立出分，仅在融合层（fuse_risk.py）晚期加权相加。
  - 本文件**只负责输出 risk_环境 ∈ [0,1]**，不碰姿态、不做融合。

v0 = R0/R1 共用规则（依据 HOME FAST 居家跌倒意外筛查工具，25 题）：
  - 每题「否(安全隐患)」= 1 分；「是 / 不适用(N/A)」= 0 分。
  - 绝对总分 T(0~25) 越高 → 居家跌倒环境隐患越多、风险越高。
  - 风险分级：0~4 低 / 5~7 中 / >=8 高（HOME FAST 官方标准）。
  - 归一化环境分 E = T / N_obs（N_obs = 非 N/A 题数）∈ [0,1]，
    用于融合（进 risk_long 长期监护 + 调报警/预警阈值上下文）。
  - 单视频只覆盖所见房间：未见的题标记 N/A（不计入分母），覆盖度透明。

视觉增强版（用户 2026-07-13 选定）：
  - 22 题视觉可判项由 R1(VLM 看画面) 自动打细标签 → from_labels() 映射成 25 题 0/1 答；
  - 仅 5/6/17 三题纯体验走问卷（VLM 不判，标记 N/A）；
  - R0 回退：直接传问卷答 dict（题号->0/1）即可。

R0 与 R1 唯一区别仅在于标签来源：
  - R0：人工/问卷直接给 25 题 0/1 答；
  - R1：VLM 看画面打细标签 → from_labels() 换算成 25 题 0/1 答。
二者都走 HomeFastModel.score()，规则只有一份（单一真相源）。

依赖：仅 scripts.build_config（常量）。numpy 仅用于 save_env_sidecar（写 .npy）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# ---- 项目根（model/env/ → 上溯两级到 elder_decete） ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_config import (
    HOME_FAST_N,
    HOME_FAST_COVERAGE_FLOOR,
    HOME_FAST_LEVEL_LOW,
    HOME_FAST_LEVEL_MID,
    HOME_FAST_ITEMS,
    HOME_FAST_VLM_TAGS,
    HOME_FAST_SHARED_TAG,
    HOME_FAST_NA_SCENE,
)


def _is_na(qid: int, room: str) -> bool:
    """某题是否因「当前房间不在其场景」而 N/A（不计入分母）。"""
    scene = HOME_FAST_NA_SCENE.get(qid)
    if not scene:
        return False
    # room_type 命中场景词（如 bathroom / out_door / stairs）
    return scene not in (room or "")


class EnvRiskModel:
    """环境风险模型统一接口。子类实现 ``compute_env``。

    对外契约：``compute_env(answers) -> float``，
    其中 ``answers`` 为 {题号: 0/1}（仅非 N/A 题）。
    满足「环境一个模型」结构要求；v0 用 :class:`HomeFastModel` 实现，
    远期 R1 用 :class:`TrainedEnvRisk`（训练小头）替换实现即可，调用方零改动。
    """

    def compute_env(self, answers: dict) -> float:
        """返回视频级标量 risk_环境 ∈ [0,1]（= T / N_obs）。"""
        raise NotImplementedError


class HomeFastModel(EnvRiskModel):
    """R0/R1 共用的 HOME FAST 规则实现（v0 默认）。

    纯规则表 + 线性计分，无训练、无推理成本。
    """

    # --------------------------------------------------------
    # 核心：25 题答 → 总分 / 归一化 / 分级
    # --------------------------------------------------------
    @staticmethod
    def score(answers: dict) -> dict:
        """给定 {题号: 0/1}（仅非 N/A 题），返回评分摘要。

        返回：
          total      : 绝对总分 T(0~25)，越高越危险
          n_observed : 实际计分题数 N_obs（已剔除 N/A）
          n_total    : 量表总题数（HOME_FAST_N=25）
          coverage   : 覆盖度 = N_obs / 25（单视频常 <1）
          low_coverage: 覆盖度 < HOME_FAST_COVERAGE_FLOOR → E 不可全信
          normalized  : 归一化环境分 E = T / N_obs ∈ [0,1]（进融合）
          level       : 风险分级 低/中/高（HOME FAST 官方标准）
        """
        n_obs = len(answers)
        total = sum(answers.values())
        normalized = (total / n_obs) if n_obs else 0.0
        if total <= HOME_FAST_LEVEL_LOW:
            level = "低"
        elif total <= HOME_FAST_LEVEL_MID:
            level = "中"
        else:
            level = "高"
        # 覆盖度守卫：单视频只覆盖部分房间时，E 仅反映已观察部分、不可全信
        coverage = (n_obs / HOME_FAST_N) if HOME_FAST_N else 0.0
        low_coverage = coverage < HOME_FAST_COVERAGE_FLOOR
        return {
            "total": total,
            "n_observed": n_obs,
            "n_total": HOME_FAST_N,
            "coverage": round(coverage, 4),
            "low_coverage": low_coverage,
            "normalized": float(normalized),
            "level": level,
        }

    @staticmethod
    def from_labels(labels: dict, room: str = "unknown") -> dict:
        """R1 映射（设计文档 §8，纯口径 B）：VLM 给出 HOME FAST 细标签
        (corridor_clear / floor_intact / toilet_grab_bar / pet_* / lighting ...)，
        换算成 25 题每题 0/1 答（N/A 跳过）。

        - ``labels``: env_label_vlm.py 输出的视频级聚合标签
          {room_type, lighting, corridor_clear, floor_intact, ...,
           pet_present, pet_safe, confidence, ...}
        - ``room``  : 房间类型（用于 N/A 场景判定；来自 labels['room_type']）
        - 标签取值语义：bool 字段 True=安全条件满足(0分) / False=隐患(1分)；
          lighting 非 bright=隐患(1分)；pet 仅 present 且非 safe=1分。
        - R0 与 R1 唯一区别仅在于标签来源；二者都走 ``score()``。
        """
        ans: dict = {}
        for tag, (qid, kind) in HOME_FAST_VLM_TAGS.items():
            # 题 12 复用题 11 的 bath_accessible 标签（循环结束统一处理）
            if kind == "light_not_bright_is_risk":
                lv = labels.get("lighting")
                if lv is None:
                    continue
                ans[7] = 1 if lv != "bright" else 0
                continue
            if kind == "pet_present":
                present = bool(labels.get("pet_present", False))
                safe = labels.get("pet_safe", True)
                ans[25] = 1 if (present and not safe) else 0
                continue
            if kind == "pet_safe":
                continue  # 已在 pet_present 处理
            # ---- bool_false_is_risk ----
            val = labels.get(tag)
            if val is None:
                continue  # VLM 未观察到该标签 → 跳过（不算分）
            if _is_na(qid, room):
                continue  # 当前房间不在该题场景 → N/A
            ans[qid] = 1 if val is False else 0
        # 题 12 复用题 11 的 bath_accessible 标签
        if 11 in ans and 12 not in ans:
            ans[12] = ans[11]
        return ans

    def compute_env(self, answers: dict) -> float:
        """视频级标量 risk_环境 = T / N_obs ∈ [0,1]。"""
        return self.score(answers)["normalized"]


def risk_from_label(agg: dict) -> dict:
    """R1 聚合标签 → HOME FAST 评分摘要。

    ``agg``: env_label_vlm.py 输出的视频级聚合标签
      {room_type, lighting, corridor_clear, ..., confidence, need_review, ...}
    返回 score() 的摘要 dict（含 total / n_observed / normalized / level /
    coverage / low_coverage），并把 need_review 合并为
    「VLM 置信度不足 OR 覆盖度不足」二者任一即触发。
    """
    room = agg.get("room_type", "unknown")
    answers = HomeFastModel.from_labels(agg, room)
    sc = HomeFastModel.score(answers)
    # need_review 合并：置信度(aggregate 阶段) 或 覆盖度 任一不足 → 需复核
    need_review = bool(agg.get("need_review", False)) or sc.get("low_coverage", False)
    sc["need_review"] = need_review
    sc["vlm_confidence"] = agg.get("confidence")
    return sc


def save_env_sidecar(npz_path, risk: float, meta: dict):
    """统一写环境 sidecar：<stem>.env_risk.npy + <stem>.env_meta.json。

    - 若 ``npz_path`` 存在：把标量 risk broadcast 成 (T,) 常数数组（T=帧数）。
    - 若不存在（视频命名与 npz 对不上）：存 (1,) 标量并标 warn。
    返回 (ok:bool, T:int)。要求 numpy（sidecar 本质是 .npy）。
    """
    npz = Path(npz_path)
    out_npy = npz.parent / f"{npz.stem}.env_risk.npy"
    out_json = npz.parent / f"{npz.stem}.env_meta.json"
    risk = float(min(max(risk, 0.0), 1.0))
    try:
        import numpy as np
        if npz.exists():
            T = int(np.load(str(npz), allow_pickle=False)["landmarks"].shape[0])
            arr = np.full(T, risk, dtype=np.float32)
            meta["n_frames"] = T
        else:
            arr = np.array([risk], dtype=np.float32)
            meta["n_frames"] = -1
            meta["warn"] = "no_npz"
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_npy), arr)
        out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                           encoding="utf-8")
        return True, (T if npz.exists() else -1)
    except Exception as e:  # noqa: BLE001
        meta["error"] = str(e)
        try:
            out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        return False, -1


# 便捷单例（demo / 推理脚本直接用 env_risk 即可）
env_risk = HomeFastModel()
