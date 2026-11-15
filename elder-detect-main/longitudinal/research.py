"""离线研究接口：用服务产出的同日协变量应用交叉验证折模型。

与研究模型配套的函数不要求 ``calibrated=True``，但返回值明确标记
``calibrated=False``、``research_only=True``；绝不能把它当作临床概率。
模型必须声明输入契约；契约缺失、来源/精度/配置不匹配或协变量不可用时
返回具体状态且风险值为空。
"""
from __future__ import annotations

from typing import Any

from .model_contract import match_model_input
from .service import FutureRiskResult
from .survival import CoxModelArtifact, predict_cumulative_risk_research


def _base_payload(
    artifact: CoxModelArtifact,
    result: FutureRiskResult,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "calibrated": False,
        "research_only": True,
        "model_bundle": artifact.bundle_id,
        "horizon_days": artifact.horizon_days,
        "features": list(artifact.feature_names),
        "risk": None,
        "covariate_status": dict(getattr(result, "research_covariate_status", {}) or {}),
        "ds_pcd_status": (getattr(result, "ds_pcd", None) or {}).get("status"),
    }


def research_prediction(
    result: FutureRiskResult,
    artifact: CoxModelArtifact,
) -> dict[str, Any]:
    """在契约匹配后，用服务结果中的协变量产生研究预测。

    返回 ``status=ok`` 时 ``risk`` 为研究数值；任何不匹配/缺失状态都保留
    ``risk=None``。``calibrated`` 恒为 false，服务概率门槛不受影响。
    """
    covariates = dict(getattr(result, "research_covariates", {}) or {})
    match = match_model_input(
        artifact.input_contract,
        provenance=getattr(result, "provenance", {}) or {},
        covariates=covariates,
        feature_names=artifact.feature_names,
        ds_pcd=getattr(result, "ds_pcd", None),
    )
    payload = _base_payload(artifact, result)
    payload["contract_match"] = match.to_dict()
    if match.status != "ok":
        payload["status"] = match.status
        payload["risk"] = None
        if match.missing:
            payload["missing"] = list(match.missing)
        return payload
    risk = predict_cumulative_risk_research(
        {name: float(covariates[name]) for name in artifact.feature_names},
        artifact,
    )
    payload.update({
        "status": "ok",
        "risk": float(risk),
        "matched": dict(match.matched),
    })
    return payload
