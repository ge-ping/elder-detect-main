"""个人稳健基线的建立与序列化。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
from typing import Any

import numpy as np

from .config import LongitudinalConfig
from .quality import assess_quality
from .schema import DailyObservation


def config_fingerprint(config: LongitudinalConfig) -> str:
    """生成用于阻止量纲混用的稳定配置指纹。"""
    payload = json.dumps(asdict(config), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BaselineFeatureStats:
    median: float
    scale: float
    scale_method: str
    count: int
    active: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaselineFeatureStats":
        return cls(
            median=float(data["median"]),
            scale=float(data["scale"]),
            scale_method=str(data["scale_method"]),
            count=int(data["count"]),
            active=bool(data["active"]),
        )


@dataclass(frozen=True)
class PersonalBaseline:
    person_id: str
    created_at: str
    valid_days: int
    valid_dates: tuple[str, ...]
    features: dict[str, BaselineFeatureStats]
    config_fingerprint: str
    updated_at: str = ""
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    slow_reference: dict[str, float] = field(default_factory=dict)
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "created_at": self.created_at,
            "valid_days": self.valid_days,
            "valid_dates": list(self.valid_dates),
            "features": {
                name: stats.to_dict() for name, stats in self.features.items()
            },
            "config_fingerprint": self.config_fingerprint,
            "updated_at": self.updated_at,
            "config_snapshot": dict(self.config_snapshot),
            "slow_reference": dict(self.slow_reference),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonalBaseline":
        return cls(
            person_id=str(data["person_id"]),
            created_at=str(data["created_at"]),
            valid_days=int(data["valid_days"]),
            valid_dates=tuple(str(value) for value in data["valid_dates"]),
            features={
                str(name): BaselineFeatureStats.from_dict(stats)
                for name, stats in data["features"].items()
            },
            config_fingerprint=str(data["config_fingerprint"]),
            updated_at=str(data.get("updated_at", data["created_at"])),
            config_snapshot=dict(data.get("config_snapshot", {})),
            slow_reference={
                str(name): float(value)
                for name, value in data.get("slow_reference", {}).items()
            },
            version=int(data.get("version", 1)),
        )


def _feature_stats(values: list[float]) -> BaselineFeatureStats:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array)) if len(array) else 0.0
    mad = float(np.median(np.abs(array - median))) if len(array) else 0.0
    mad_scale = 1.4826 * mad
    if mad_scale > 1e-12:
        return BaselineFeatureStats(median, mad_scale, "mad", len(values), True)
    if len(array):
        q25, q75 = np.percentile(array, [25, 75])
        iqr_scale = float((q75 - q25) / 1.349)
    else:
        iqr_scale = 0.0
    if iqr_scale > 1e-12:
        return BaselineFeatureStats(median, iqr_scale, "iqr", len(values), True)
    return BaselineFeatureStats(median, 0.0, "constant", len(values), False)


def build_baseline(
    observations: list[DailyObservation],
    config: LongitudinalConfig,
) -> PersonalBaseline:
    """从质量合格的不同观测日建立个人基线。"""
    valid_by_date: dict[str, DailyObservation] = {}
    for item in observations:
        if assess_quality(item, config).is_valid:
            valid_by_date[item.observation_date] = item
    valid = [valid_by_date[key] for key in sorted(valid_by_date)]
    if len(valid) < config.min_baseline_days:
        raise ValueError(
            f"至少需要 {config.min_baseline_days} 个有效观测日，当前只有 {len(valid)} 个"
        )
    valid = valid[:config.min_baseline_days]
    person_ids = {item.person_id for item in valid}
    if len(person_ids) != 1:
        raise ValueError("建立基线的观测必须属于同一个 person_id")

    features = {
        name: _feature_stats([
            item.features[name] for item in valid if name in item.features
        ])
        for name in config.core_features
    }
    created_at = datetime.now(timezone.utc).isoformat()
    return PersonalBaseline(
        person_id=valid[0].person_id,
        created_at=created_at,
        valid_days=len(valid),
        valid_dates=tuple(item.observation_date for item in valid),
        features=features,
        config_fingerprint=config_fingerprint(config),
        updated_at=created_at,
        config_snapshot=asdict(config),
        slow_reference={name: stats.median for name, stats in features.items()},
    )


def update_slow_reference(
    baseline: PersonalBaseline,
    observation: DailyObservation,
    *,
    distance: float,
    threshold: float = 3.0,
    alpha: float = 0.05,
) -> PersonalBaseline:
    """仅用低偏离观测缓慢更新参考线，锚点统计保持不变。"""
    if distance >= threshold:
        return baseline
    updated = dict(baseline.slow_reference)
    for name, previous in updated.items():
        if name in observation.features:
            updated[name] = (1.0 - alpha) * previous + alpha * observation.features[name]
    return replace(
        baseline,
        slow_reference=updated,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
