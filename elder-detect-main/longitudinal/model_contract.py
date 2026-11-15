"""模型输入契约：版本化、可序列化、按输入语义匹配。

契约只记录会影响对应协变量计算的输入语义：模态、特征定义/版本、
RMS 配置、DS-PCD 配置及精度模式。模型文件本身的数值、校准标志和
训练信息不在契约内，导出副本原样继承。
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

CONTRACT_SCHEMA_VERSION = 1

# 全局字段的匹配状态名；顺序决定同时出现多个不匹配时返回的首要状态。
_GLOBAL_FIELD_STATUS = {
    "modality": "modality_mismatch",
    "feature_version": "feature_version_mismatch",
    "score_version": "score_version_mismatch",
    "rms_distance_config_fingerprint": "rms_config_mismatch",
    # 兼容阶段 8 v1 契约：整份 LongitudinalConfig 指纹。
    "rms_config_fingerprint": "rms_config_mismatch",
}
_GLOBAL_FIELDS = tuple(_GLOBAL_FIELD_STATUS)


@dataclass(frozen=True)
class ContractMatch:
    """一次契约匹配的机器可读结果。"""

    status: str
    reasons: tuple[str, ...] = ()
    mismatches: tuple[dict[str, Any], ...] = ()
    missing: tuple[str, ...] = ()
    matched: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": self.status}
        if self.reasons:
            payload["reasons"] = list(self.reasons)
        if self.mismatches:
            payload["mismatches"] = [dict(item) for item in self.mismatches]
        if self.missing:
            payload["missing"] = list(self.missing)
        if self.matched:
            payload["matched"] = dict(self.matched)
        return payload


def _non_empty_string(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"输入契约缺少 {name}")
    return text


def validate_input_contract(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """校验并返回规范化契约；结构错误抛 ``ValueError``。"""
    if not isinstance(raw, Mapping):
        raise ValueError("输入契约必须是对象")
    version = raw.get("schema_version")
    if type(version) is not int or version != CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            f"输入契约 schema_version 必须是 {CONTRACT_SCHEMA_VERSION}"
        )
    modality = _non_empty_string(raw.get("modality"), "modality")
    feature_version = _non_empty_string(raw.get("feature_version"), "feature_version")
    raw_features = raw.get("features")
    if not isinstance(raw_features, Mapping) or not raw_features:
        raise ValueError("输入契约缺少 features")
    features: dict[str, dict[str, Any]] = {}
    for name, spec in raw_features.items():
        key = _non_empty_string(name, "feature name")
        if not isinstance(spec, Mapping):
            raise ValueError(f"特征 {key} 的契约必须是对象")
        definition = _non_empty_string(spec.get("definition"), f"features.{key}.definition")
        source = _non_empty_string(spec.get("source"), f"features.{key}.source")
        requires_ds_pcd = spec.get("requires_ds_pcd", False)
        if type(requires_ds_pcd) is not bool:
            raise ValueError(f"features.{key}.requires_ds_pcd 必须是 JSON boolean")
        features[key] = {
            "definition": definition,
            "source": source,
            "requires_ds_pcd": requires_ds_pcd,
        }
    raw_ds = raw.get("ds_pcd")
    if raw_ds is not None and not isinstance(raw_ds, Mapping):
        raise ValueError("输入契约的 ds_pcd 必须是对象")
    ds_pcd: dict[str, Any] = {}
    if raw_ds is not None:
        required = raw_ds.get("required", False)
        if type(required) is not bool:
            raise ValueError("ds_pcd.required 必须是 JSON boolean")
        ds_pcd["required"] = required
        if required:
            ds_pcd["precision_mode"] = _non_empty_string(
                raw_ds.get("precision_mode"), "ds_pcd.precision_mode"
            )
            scoped_fingerprint = str(
                raw_ds.get("scoring_config_fingerprint", "") or ""
            ).strip()
            legacy_fingerprint = str(raw_ds.get("config_fingerprint", "") or "").strip()
            if scoped_fingerprint:
                ds_pcd["scoring_config_fingerprint"] = scoped_fingerprint
            elif legacy_fingerprint:
                ds_pcd["config_fingerprint"] = legacy_fingerprint
            else:
                raise ValueError("输入契约缺少 ds_pcd.scoring_config_fingerprint")
            raw_ds_features = raw_ds.get("features", ())
            if raw_ds_features is not None:
                if not isinstance(raw_ds_features, Sequence) or isinstance(raw_ds_features, (str, bytes)):
                    raise ValueError("ds_pcd.features 必须是数组")
                ds_pcd["features"] = [str(item) for item in raw_ds_features]
            raw_directions = raw_ds.get("directions")
            if raw_directions is not None:
                if not isinstance(raw_directions, Mapping):
                    raise ValueError("ds_pcd.directions 必须是对象")
                ds_pcd["directions"] = {
                    str(key): float(value) for key, value in raw_directions.items()
                }
            for key in ("k_ref", "gap_reset_days"):
                if key in raw_ds:
                    ds_pcd[key] = raw_ds[key]
    requires_ds_any = any(item["requires_ds_pcd"] for item in features.values())
    if requires_ds_any and not ds_pcd.get("required", False):
        raise ValueError("使用 C 的模型必须在 ds_pcd 中声明 required=true")
    contract: dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "modality": modality,
        "feature_version": feature_version,
        "features": features,
        "ds_pcd": ds_pcd or {"required": False},
    }
    contract_id = str(raw.get("contract_id", "") or "").strip()
    if contract_id:
        contract["contract_id"] = contract_id
    for key in (
        "population", "score_version", "rms_distance_config_fingerprint",
        "rms_config_fingerprint", "precision_policy",
    ):
        value = str(raw.get(key, "") or "").strip()
        if value:
            contract[key] = value
    # population 是模型适用范围声明，不作为硬性输入匹配条件；
    # 保留在契约中供人工审核，实际匹配依赖模态、特征版本和配置指纹。
    return contract


def build_input_contract(
    *,
    modality: str,
    feature_version: str,
    features: Mapping[str, Mapping[str, Any]],
    population: str | None = None,
    score_version: str | None = None,
    rms_distance_config_fingerprint: str | None = None,
    rms_config_fingerprint: str | None = None,
    ds_pcd: Mapping[str, Any] | None = None,
    contract_id: str | None = None,
    precision_policy: str | None = None,
) -> dict[str, Any]:
    """构造并校验一份输入契约。"""
    raw: dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "modality": modality,
        "feature_version": feature_version,
        "features": {key: dict(value) for key, value in features.items()},
        "ds_pcd": dict(ds_pcd or {"required": False}),
    }
    for key, value in (
        ("population", population), ("score_version", score_version),
        ("rms_distance_config_fingerprint", rms_distance_config_fingerprint),
        ("rms_config_fingerprint", rms_config_fingerprint),
        ("contract_id", contract_id), ("precision_policy", precision_policy),
    ):
        if value:
            raw[key] = value
    return validate_input_contract(raw)


def _actual_value(
    provenance: Mapping[str, Any] | None,
    ds_pcd: Mapping[str, Any] | None,
    field_name: str,
) -> Any:
    source = provenance or {}
    value = source.get(field_name)
    if value is None and ds_pcd:
        if field_name == "precision_mode":
            value = ds_pcd.get("precision_mode")
        elif field_name == "ds_pcd_config_fingerprint":
            value = ds_pcd.get("config_fingerprint")
        elif field_name == "ds_pcd_scoring_config_fingerprint":
            value = ds_pcd.get("scoring_config_fingerprint")
    return value


def match_model_input(
    contract: Mapping[str, Any] | None,
    *,
    provenance: Mapping[str, Any] | None,
    covariates: Mapping[str, Any],
    feature_names: Sequence[str],
    ds_pcd: Mapping[str, Any] | None = None,
) -> ContractMatch:
    """按输入语义检查契约、来源、协变量和数值。

    返回的 ``status`` 是首要原因，``mismatches`` 保留全部不匹配项；不匹配时
    调用方不得计算风险值。契约缺失由调用方单独返回 ``contract_missing``。
    """
    if contract is None:
        return ContractMatch("contract_missing", reasons=("模型缺少输入契约",))
    try:
        normalized = validate_input_contract(contract)
    except ValueError as exc:
        return ContractMatch("contract_invalid", reasons=(str(exc),))
    features = normalized["features"]
    undeclared = [name for name in feature_names if name not in features]
    if undeclared:
        return ContractMatch(
            "contract_invalid",
            reasons=(f"契约未声明模型特征：{', '.join(undeclared)}",),
            missing=tuple(undeclared),
        )

    mismatches: list[dict[str, Any]] = []
    missing_provenance: list[str] = []
    for field_name in _GLOBAL_FIELDS:
        expected = normalized.get(field_name)
        if not expected:
            continue
        actual = _actual_value(provenance, ds_pcd, field_name)
        if actual is None or str(actual).strip() == "":
            missing_provenance.append(field_name)
            mismatches.append({
                "field": field_name, "expected": expected, "actual": None,
                "reason": "missing_provenance",
            })
        elif str(actual) != str(expected):
            mismatches.append({
                "field": field_name, "expected": expected, "actual": actual,
                "reason": "value_mismatch",
            })
    if mismatches:
        primary = (
            "missing_provenance" if missing_provenance
            else _GLOBAL_FIELD_STATUS[mismatches[0]["field"]]
        )
        return ContractMatch(
            primary,
            reasons=tuple(
                f"{item['field']} 期望 {item['expected']!r}，实际 {item['actual']!r}"
                if item["actual"] is not None
                else f"输入结果缺少来源字段 {item['field']}"
                for item in mismatches
            ),
            mismatches=tuple(mismatches),
            missing=tuple(missing_provenance),
        )

    missing_covariates: list[str] = []
    invalid_covariates: list[str] = []
    for name in feature_names:
        value = covariates.get(name)
        if value is None:
            missing_covariates.append(name)
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            invalid_covariates.append(name)
            continue
        if not math.isfinite(numeric):
            invalid_covariates.append(name)
    if missing_covariates:
        return ContractMatch(
            "missing_covariates",
            reasons=tuple(f"缺少协变量 {name}" for name in missing_covariates),
            missing=tuple(missing_covariates),
        )
    if invalid_covariates:
        return ContractMatch(
            "invalid_covariates",
            reasons=tuple(f"协变量不是有限数值：{name}" for name in invalid_covariates),
            missing=tuple(invalid_covariates),
        )

    ds_contract = normalized.get("ds_pcd") or {}
    if ds_contract.get("required"):
        expected_precision = ds_contract.get("precision_mode")
        actual_precision = _actual_value(provenance, ds_pcd, "precision_mode")
        if actual_precision is None or str(actual_precision).strip() == "":
            return ContractMatch(
                "missing_provenance",
                reasons=("输入结果缺少 precision_mode",),
                missing=("precision_mode",),
            )
        if str(actual_precision) != str(expected_precision):
            return ContractMatch(
                "precision_mode_mismatch",
                reasons=(
                    f"精度模式期望 {expected_precision!r}，实际 {actual_precision!r}",
                ),
                mismatches=({
                    "field": "precision_mode", "expected": expected_precision,
                    "actual": actual_precision, "reason": "value_mismatch",
                },),
            )
        expected_fingerprint = (
            ds_contract.get("scoring_config_fingerprint")
            or ds_contract.get("config_fingerprint")
        )
        if ds_contract.get("scoring_config_fingerprint"):
            actual_fingerprint = _actual_value(
                provenance, ds_pcd, "ds_pcd_scoring_config_fingerprint"
            )
        else:
            actual_fingerprint = _actual_value(
                provenance, ds_pcd, "ds_pcd_config_fingerprint"
            )
        if actual_fingerprint is None or str(actual_fingerprint).strip() == "":
            return ContractMatch(
                "missing_provenance",
                reasons=("输入结果缺少 DS-PCD 配置指纹",),
                missing=("ds_pcd_config_fingerprint",),
            )
        if str(actual_fingerprint) != str(expected_fingerprint):
            return ContractMatch(
                "ds_pcd_config_mismatch",
                reasons=(
                    "DS-PCD 配置指纹与模型训练输入不一致："
                    f"期望 {expected_fingerprint}，实际 {actual_fingerprint}",
                ),
                mismatches=({
                    "field": "ds_pcd_config_fingerprint",
                    "expected": expected_fingerprint,
                    "actual": actual_fingerprint,
                    "reason": "value_mismatch",
                },),
            )

    matched: dict[str, Any] = {"schema_version": normalized["schema_version"]}
    for field_name in _GLOBAL_FIELDS:
        value = _actual_value(provenance, ds_pcd, field_name)
        if value is not None:
            matched[field_name] = value
    if normalized.get("population"):
        matched["population"] = _actual_value(provenance, ds_pcd, "population")
    if ds_contract.get("required"):
        matched["precision_mode"] = _actual_value(
            provenance, ds_pcd, "precision_mode"
        )
        if ds_contract.get("scoring_config_fingerprint"):
            matched["ds_pcd_scoring_config_fingerprint"] = _actual_value(
                provenance, ds_pcd, "ds_pcd_scoring_config_fingerprint"
            )
        else:
            matched["ds_pcd_config_fingerprint"] = _actual_value(
                provenance, ds_pcd, "ds_pcd_config_fingerprint"
            )
    return ContractMatch("ok", matched=matched)


def contract_summary(contract: Mapping[str, Any] | None) -> dict[str, Any]:
    """用于报告/导出映射的精简契约摘要。"""
    if not contract:
        return {}
    try:
        normalized = validate_input_contract(contract)
    except ValueError:
        return {"status": "invalid", "raw": copy.deepcopy(dict(contract))}
    summary = {
        "schema_version": normalized["schema_version"],
        "modality": normalized["modality"],
        "feature_version": normalized["feature_version"],
        "features": sorted(normalized["features"]),
        "ds_pcd_required": bool(normalized["ds_pcd"].get("required")),
    }
    for key in (
        "population", "score_version", "contract_id",
        "rms_distance_config_fingerprint", "rms_config_fingerprint",
        "precision_policy",
    ):
        if normalized.get(key):
            summary[key] = normalized[key]
    if normalized["ds_pcd"].get("required"):
        summary["precision_mode"] = normalized["ds_pcd"].get("precision_mode")
        if normalized["ds_pcd"].get("scoring_config_fingerprint"):
            summary["ds_pcd_scoring_config_fingerprint"] = normalized["ds_pcd"].get(
                "scoring_config_fingerprint"
            )
        else:
            summary["ds_pcd_config_fingerprint"] = normalized["ds_pcd"].get(
                "config_fingerprint"
            )
    return summary
