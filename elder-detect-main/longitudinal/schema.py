"""纵向观测的数据契约。"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone as utc_timezone
import math
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def _finite_mapping(name: str, values: dict[str, Any] | None) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in (values or {}).items():
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}.{key} 必须是有限数值") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name}.{key} 必须是有限数值")
        result[str(key)] = number
    return result


@dataclass(frozen=True)
class QualityMetrics:
    """每日姿态摘要的数据质量。"""

    valid_frames: int = 0
    valid_steps: int = 0
    walking_duration_s: float = 0.0
    mean_confidence: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "QualityMetrics":
        data = data or {}
        def nonnegative_integer(name: str) -> int:
            value = data.get(name, 0)
            if isinstance(value, bool):
                raise ValueError(f"quality.{name} 必须是非负整数")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"quality.{name} 必须是非负整数") from exc
            if not math.isfinite(number) or number < 0 or not number.is_integer():
                raise ValueError(f"quality.{name} 必须是非负整数")
            return int(number)

        def finite_number(name: str, *, maximum: float | None = None) -> float:
            try:
                number = float(data.get(name, 0.0))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"quality.{name} 必须是有效数值") from exc
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"quality.{name} 必须是非负有限数值")
            if maximum is not None and number > maximum:
                raise ValueError(f"quality.{name} 必须位于 0 到 {maximum:g} 之间")
            return number

        return cls(
            valid_frames=nonnegative_integer("valid_frames"),
            valid_steps=nonnegative_integer("valid_steps"),
            walking_duration_s=finite_number("walking_duration_s"),
            mean_confidence=finite_number("mean_confidence", maximum=1.0),
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "valid_frames": self.valid_frames,
            "valid_steps": self.valid_steps,
            "walking_duration_s": self.walking_duration_s,
            "mean_confidence": self.mean_confidence,
        }


@dataclass(frozen=True)
class DailyObservation:
    """单个老人在一个自然日内的步态和姿态摘要。"""

    person_id: str
    observed_at: datetime
    features: dict[str, float]
    quality: QualityMetrics
    psychological: dict[str, float] = field(default_factory=dict)
    source: str = ""
    timezone: str = "Asia/Shanghai"
    local_date: str = ""

    @property
    def observation_date(self) -> str:
        return self.local_date

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DailyObservation":
        person_id = str(data.get("person_id", "")).strip()
        if not person_id:
            raise ValueError("person_id 不能为空")
        raw_time = str(data.get("observed_at", "")).strip()
        if not raw_time:
            raise ValueError("observed_at 不能为空")
        try:
            parsed = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("observed_at 必须是 ISO 8601 日期时间") from exc
        timezone_name = str(data.get("timezone", "Asia/Shanghai"))
        try:
            local_zone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"未知时区：{timezone_name}") from exc
        if parsed.tzinfo is None:
            local_time = parsed.replace(tzinfo=local_zone)
        else:
            local_time = parsed.astimezone(local_zone)
        observed_at = local_time.astimezone(utc_timezone.utc)
        return cls(
            person_id=person_id,
            observed_at=observed_at,
            features=_finite_mapping("features", data.get("features")),
            quality=QualityMetrics.from_dict(data.get("quality")),
            psychological=_finite_mapping("psychological", data.get("psychological")),
            source=str(data.get("source", "")),
            timezone=timezone_name,
            local_date=local_time.date().isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "observed_at": self.observed_at.isoformat(),
            "features": dict(self.features),
            "quality": self.quality.to_dict(),
            "psychological": dict(self.psychological),
            "source": self.source,
            "timezone": self.timezone,
        }
