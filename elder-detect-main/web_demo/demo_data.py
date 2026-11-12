"""为无长期随访数据的演示环境生成明确标记的示例画像。"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import math
from typing import Any


def build_sample_profile(person: dict[str, Any] | None = None) -> dict[str, Any]:
    """返回可直接驱动页面的示例画像，所有风险概率保持未校准状态。"""
    history: list[dict[str, Any]] = []
    start = date(2026, 10, 15)
    for index in range(28):
        cadence = 92.0 + 2.4 * math.sin(index / 3.2)
        if index >= 23:
            cadence -= (index - 22) * 0.65
        distance = 0.65 + 0.18 * math.sin(index / 2.8)
        if index >= 23:
            distance += (index - 22) * 0.34
        history.append({
            "date": (start + timedelta(days=index)).isoformat(),
            "value": round(cadence, 2),
            "cadence_spm": round(cadence, 2),
            "step_time_cv": round(0.071 + 0.006 * math.sin(index / 4.0), 3),
            "trunk_sway_rms_deg": round(1.31 + 0.11 * math.sin(index / 3.6), 2),
            "emos_proxy": round(0.29 - 0.012 * math.sin(index / 3.1), 3),
            "baseline_median": 92.0,
            "baseline_low": 86.0,
            "baseline_high": 98.0,
            "distance": round(distance, 2),
            "quality": "合格",
        })

    profile: dict[str, Any] = {
        "data_mode": "sample",
        "mode_label": "示例数据",
        "person": {
            "id": "elder_001",
            "name": "李奶奶",
            "age": 76,
            "sex": "女",
            "living_context": "居家独居，家属每日探访",
            "psychological": {
                "label": "跌倒担忧：轻度",
                "role": "仅作辅助协变量，不参与未校准偏离指数加权",
            },
        },
        "baseline": {
            "valid_days": 14,
            "required_days": 14,
            "progress": 1.0,
            "anchor_start": "2026-10-15",
            "anchor_end": "2026-10-28",
            "updated_at": "2026-11-11T08:00:00+08:00",
            "method": "中位数 + MAD；MAD 为零时退化为 IQR",
            "slow_reference": "低偏离有效观测用于慢速参考线，固定14日锚点不自动覆盖",
        },
        "metrics": {
            "cadence_spm": 88.7,
            "step_time_cv": 0.083,
            "trunk_sway_rms_deg": 1.56,
            "emos_proxy": 0.25,
            "deviation_index": 0.31,
            "distance": 2.54,
            "gait_speed_proxy": 0.33,
            "stance_width_norm": 0.71,
        },
        "risk": {
            "status": "baseline_ready",
            "level": "关注变化",
            "risk_7d": None,
            "risk_30d": None,
            "message": "等待真实纵向标签校准；当前只展示个人基线偏离，不输出跌倒概率",
            "persistent_deviation_days": 0,
        },
        "quality": {
            "status": "合格",
            "valid_frames": 486,
            "valid_steps": 38,
            "walking_duration_s": 31.4,
            "mean_confidence": 0.87,
            "note": "质量阈值是工程门控，不是临床诊断阈值",
        },
        "history": history,
        "disclaimer": "本演示用于说明个人画像与统计偏离流程，不替代临床评估或紧急处置。",
    }
    if person:
        for key in ("id", "name", "age"):
            if person.get(key) not in (None, ""):
                profile["person"][key] = person[key]
    return profile


def profile_with_real_observation(
    result: dict[str, Any], person: dict[str, Any], observed_at: str
) -> dict[str, Any]:
    """建立仅含本次真实观测的画像，不继承任何示例长期统计。"""
    profile = deepcopy(build_sample_profile(person))
    profile["data_mode"] = "real_observation"
    profile["mode_label"] = "本次视频真实姿态摘要"
    profile["person"]["living_context"] = "本次未采集生活环境信息"
    profile["person"]["psychological"] = {
        "label": "本次未采集心理辅助信息",
        "role": "心理因素仅可作为后续校准模型的辅助协变量",
    }
    features = {
        key: value
        for key, value in result.get("features", {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    profile["metrics"] = {
        **features,
        "deviation_index": None,
        "distance": None,
    }
    quality = result.get("quality", {})
    profile["quality"] = {
        "status": "本次摘要已生成",
        **quality,
        "note": "仍需累计14个质量合格的真实观测日建立个人基线",
    }
    profile["baseline"] = {
        "valid_days": 1,
        "required_days": 14,
        "progress": 1 / 14,
        "anchor_start": None,
        "anchor_end": None,
        "anchor_label": "待建立",
        "updated_at": observed_at,
        "method": "待累计14个真实有效观测日后使用中位数 + MAD",
        "slow_reference": "基线建立后才启用慢速参考线",
    }
    profile["risk"] = {
        "status": "collecting",
        "level": "基线采集中",
        "risk_7d": None,
        "risk_30d": None,
        "message": "当前为第1个真实有效日，尚不能计算个人偏离或未来跌倒概率",
        "persistent_deviation_days": 0,
    }
    profile["history"] = [{
        "date": observed_at,
        **features,
        "quality": "本次视频",
    }]
    profile["disclaimer"] = "本次指标来自上传视频；个人基线尚未建立，不能据此作临床判断。"
    return profile
