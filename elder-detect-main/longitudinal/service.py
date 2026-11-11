"""纵向个人画像、偏离与生存模型的统一编排。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .baseline import PersonalBaseline, config_fingerprint
from .config import LongitudinalConfig
from .deviation import DeviationResult, compute_deviation, count_persistent_deviation
from .quality import assess_quality
from .schema import DailyObservation
from .storage import ProfileStore
from .survival import CoxModelArtifact, predict_cumulative_risk


@dataclass(frozen=True)
class FutureRiskResult:
    person_id: str
    status: str
    baseline_valid_days: int
    observation_date: str
    deviation_index: float | None
    distance: float | None
    persistent_deviation_days: int
    feature_deviations: dict[str, float]
    risk_7d: float | None
    risk_30d: float | None
    quality_reasons: tuple[str, ...]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "status": self.status,
            "baseline_valid_days": self.baseline_valid_days,
            "observation_date": self.observation_date,
            "deviation_index": self.deviation_index,
            "distance": self.distance,
            "persistent_deviation_days": self.persistent_deviation_days,
            "feature_deviations": dict(self.feature_deviations),
            "risk_7d": self.risk_7d,
            "risk_30d": self.risk_30d,
            "quality_reasons": list(self.quality_reasons),
            "message": self.message,
        }


class LongitudinalRiskService:
    """对单个个人画像执行一次未来风险评估。"""

    def __init__(
        self,
        store: ProfileStore,
        config: LongitudinalConfig,
        models: dict[int, CoxModelArtifact] | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self.models = dict(models or {})
        if self.models and set(self.models) != {7, 30}:
            raise ValueError("校准模型必须同时包含7天和30天窗口")

    def _save(self, result: FutureRiskResult) -> FutureRiskResult:
        self.store.save_latest_result(result)
        return result

    def evaluate(
        self,
        person_id: str,
        observation: DailyObservation,
    ) -> FutureRiskResult:
        if person_id != self.store.person_id or observation.person_id != person_id:
            raise ValueError("person_id 与画像或观测不一致")

        quality = assess_quality(observation, self.config)
        if not quality.is_valid:
            return self._save(FutureRiskResult(
                person_id=person_id,
                status="insufficient_quality",
                baseline_valid_days=self._valid_day_count(observation),
                observation_date=observation.observation_date,
                deviation_index=None,
                distance=None,
                persistent_deviation_days=0,
                feature_deviations={},
                risk_7d=None,
                risk_30d=None,
                quality_reasons=quality.reasons,
                message="本次数据质量不足，未更新个人基线或风险结果",
            ))

        raw_baseline = self.store.load_baseline()
        if raw_baseline is None:
            return self._save(FutureRiskResult(
                person_id=person_id,
                status="collecting",
                baseline_valid_days=self._valid_day_count(observation),
                observation_date=observation.observation_date,
                deviation_index=None,
                distance=None,
                persistent_deviation_days=0,
                feature_deviations={},
                risk_7d=None,
                risk_30d=None,
                quality_reasons=(),
                message=f"正在收集个人基线，需要至少 {self.config.min_baseline_days} 个有效观测日",
            ))

        baseline = PersonalBaseline.from_dict(raw_baseline)
        if baseline.person_id != person_id:
            raise ValueError("基线 person_id 与当前个人不一致")
        if baseline.config_fingerprint != config_fingerprint(self.config):
            raise ValueError("个人基线与当前纵向配置不一致，请重新建立基线")

        history_results = self._post_baseline_deviations(baseline, observation)
        current = history_results[-1]
        persistent_days = count_persistent_deviation(
            history_results, self.config.persistent_distance_threshold
        )

        if self.models:
            model_features = self._model_features(
                observation, current, history_results, persistent_days
            )
            risk_7d = predict_cumulative_risk(model_features, self.models[7])
            risk_30d = predict_cumulative_risk(model_features, self.models[30])
            status = "calibrated"
            message = "已使用校准 Landmark Cox 模型计算未来7天和30天累计跌倒概率"
        else:
            risk_7d = None
            risk_30d = None
            if persistent_days >= self.config.persistent_days:
                status = "calibration_required"
                message = "检测到持续个人基线偏离；当前结果未经跌倒概率校准"
            else:
                status = "baseline_ready"
                message = "个人基线可用；当前结果为统计偏离监测，不是跌倒概率"

        return self._save(FutureRiskResult(
            person_id=person_id,
            status=status,
            baseline_valid_days=baseline.valid_days,
            observation_date=observation.observation_date,
            deviation_index=current.deviation_index,
            distance=current.distance,
            persistent_deviation_days=persistent_days,
            feature_deviations=current.feature_deviations,
            risk_7d=risk_7d,
            risk_30d=risk_30d,
            quality_reasons=(),
            message=message,
        ))

    def _merged_observations(
        self,
        current: DailyObservation,
    ) -> list[DailyObservation]:
        by_date = {
            item.observation_date: item for item in self.store.load_observations()
        }
        by_date[current.observation_date] = current
        return [by_date[key] for key in sorted(by_date)]

    def _valid_day_count(self, current: DailyObservation) -> int:
        return sum(
            assess_quality(item, self.config).is_valid
            for item in self._merged_observations(current)
        )

    def _post_baseline_deviations(
        self,
        baseline: PersonalBaseline,
        current: DailyObservation,
    ) -> list[DeviationResult]:
        baseline_dates = set(baseline.valid_dates)
        results = [
            compute_deviation(item.features, baseline)
            for item in self._merged_observations(current)
            if item.observation_date not in baseline_dates
            and assess_quality(item, self.config).is_valid
        ]
        if not results:
            results.append(compute_deviation(current.features, baseline))
        return results

    @staticmethod
    def _model_features(
        observation: DailyObservation,
        current: DeviationResult,
        history: list[DeviationResult],
        persistent_days: int,
    ) -> dict[str, float]:
        recent = history[-7:]
        trend = (
            (recent[-1].distance - recent[0].distance) / (len(recent) - 1)
            if len(recent) > 1 else 0.0
        )
        features = {
            "distance": current.distance,
            "deviation_index": current.deviation_index,
            "persistent_deviation_days": float(persistent_days),
            "trend_7d": trend,
        }
        for name, value in current.feature_deviations.items():
            features[name] = value
            features[f"z_{name}"] = value
        for name, value in observation.psychological.items():
            features[f"psych_{name}"] = value
        return features
