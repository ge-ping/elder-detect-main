"""Landmark Cox 模型产物校验与无额外依赖推理。"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CoxModelArtifact:
    schema_version: int
    bundle_id: str
    horizon_days: int
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    feature_centers: tuple[float, ...]
    baseline_survival: float
    calibrated: bool
    trained_at: str
    evidence_ids: tuple[str, ...]
    training_summary: dict[str, Any] = field(default_factory=dict)
    model_type: str = "landmark_cox"
    input_contract: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CoxModelArtifact":
        schema_version = int(data.get("schema_version", 0))
        if schema_version != 2:
            raise ValueError("Cox 模型 schema_version 必须是2")
        if data.get("model_type") != "landmark_cox":
            raise ValueError("model_type 必须是 landmark_cox")
        bundle_id = str(data.get("bundle_id", "")).strip()
        if not bundle_id:
            raise ValueError("Cox 模型缺少 bundle_id")
        horizon = int(data["horizon_days"])
        if horizon not in (7, 30):
            raise ValueError("预测窗口只能是7天或30天")
        names = tuple(str(value) for value in data["feature_names"])
        coefficients = tuple(float(value) for value in data["coefficients"])
        centers = tuple(float(value) for value in data["feature_centers"])
        if not names or not (len(names) == len(coefficients) == len(centers)):
            raise ValueError("feature_names、coefficients 与 feature_centers 长度必须一致且非空")
        if len(set(names)) != len(names):
            raise ValueError("feature_names 不能重复")
        if not all(math.isfinite(value) for value in coefficients):
            raise ValueError("coefficients 必须全部是有限数值")
        if not all(math.isfinite(value) for value in centers):
            raise ValueError("feature_centers 必须全部是有限数值")
        baseline_survival = float(data["baseline_survival"])
        if not math.isfinite(baseline_survival) or not 0.0 < baseline_survival <= 1.0:
            raise ValueError("baseline_survival 必须位于 (0,1]")
        calibrated = data.get("calibrated", False)
        if type(calibrated) is not bool:
            raise ValueError("calibrated 必须是 JSON boolean")
        raw_contract = data.get("input_contract")
        if raw_contract is not None and not isinstance(raw_contract, dict):
            raise ValueError("input_contract 必须是对象或 null")
        return cls(
            schema_version=schema_version,
            bundle_id=bundle_id,
            horizon_days=horizon,
            feature_names=names,
            coefficients=coefficients,
            feature_centers=centers,
            baseline_survival=baseline_survival,
            calibrated=calibrated,
            trained_at=str(data.get("trained_at", "")),
            evidence_ids=tuple(str(value) for value in data.get("evidence_ids", ())),
            training_summary=dict(data.get("training_summary", {})),
            input_contract=(
                copy.deepcopy(raw_contract) if raw_contract is not None else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "model_type": self.model_type,
            "bundle_id": self.bundle_id,
            "horizon_days": self.horizon_days,
            "feature_names": list(self.feature_names),
            "coefficients": list(self.coefficients),
            "feature_centers": list(self.feature_centers),
            "baseline_survival": self.baseline_survival,
            "calibrated": self.calibrated,
            "trained_at": self.trained_at,
            "evidence_ids": list(self.evidence_ids),
            "training_summary": dict(self.training_summary),
        }
        if self.input_contract is not None:
            payload["input_contract"] = copy.deepcopy(self.input_contract)
        return payload


def _risk_from_artifact(
    features: dict[str, float],
    artifact: CoxModelArtifact,
) -> float:
    missing = [name for name in artifact.feature_names if name not in features]
    if missing:
        raise ValueError(f"缺少模型特征：{', '.join(missing)}")
    values = [float(features[name]) for name in artifact.feature_names]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("模型输入特征必须全部是有限数值")
    linear_predictor = sum(
        coefficient * (value - center)
        for coefficient, value, center in zip(
            artifact.coefficients, values, artifact.feature_centers
        )
    )
    hazard_multiplier = math.exp(max(-50.0, min(50.0, linear_predictor)))
    survival = math.exp(math.log(artifact.baseline_survival) * hazard_multiplier)
    return max(0.0, min(1.0, 1.0 - survival))


def predict_cumulative_risk(
    features: dict[str, float],
    artifact: CoxModelArtifact,
) -> float:
    """服务路径：只允许输出已校准模型的跌倒概率。"""
    if not artifact.calibrated:
        raise ValueError("模型尚未校准，不能输出跌倒概率")
    return _risk_from_artifact(features, artifact)


def predict_cumulative_risk_research(
    features: dict[str, float],
    artifact: CoxModelArtifact,
) -> float:
    """离线研究路径：允许未校准模型输出研究预测。

    公式与 ``predict_cumulative_risk`` 相同，只是不要求 ``calibrated=True``；
    服务概率的校准门槛不受影响。
    """
    return _risk_from_artifact(features, artifact)


def _load_artifact(path: str | Path) -> CoxModelArtifact:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return CoxModelArtifact.from_dict(data)


def load_model_bundle(
    seven_day_path: str | Path,
    thirty_day_path: str | Path,
) -> dict[int, CoxModelArtifact]:
    """加载严格配对的7天和30天模型产物。"""
    artifacts = [_load_artifact(seven_day_path), _load_artifact(thirty_day_path)]
    bundle = {artifact.horizon_days: artifact for artifact in artifacts}
    if set(bundle) != {7, 30}:
        raise ValueError("模型包必须同时包含7天和30天模型")
    seven, thirty = bundle[7], bundle[30]
    if seven.bundle_id != thirty.bundle_id:
        raise ValueError("7天和30天模型必须来自同一训练批次")
    if seven.feature_names != thirty.feature_names:
        raise ValueError("7天和30天模型的特征顺序必须一致")
    if seven.calibrated != thirty.calibrated:
        raise ValueError("7天和30天模型的校准状态必须一致")
    if seven.evidence_ids != thirty.evidence_ids:
        raise ValueError("7天和30天模型的证据标识必须一致")
    def semantic_contract(contract: dict[str, Any] | None):
        if contract is None:
            return None
        return {
            key: value for key, value in contract.items()
            if key != "contract_id"
        }
    if semantic_contract(seven.input_contract) != semantic_contract(thirty.input_contract):
        raise ValueError("7天和30天模型的输入契约必须语义一致")
    return bundle
