"""纵向个人画像、偏离与生存模型的统一编排。"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date
from datetime import timedelta
from typing import Any

from .baseline import (
    PersonalBaseline,
    config_fingerprint,
    rms_distance_config_fingerprint,
)
from .config import LongitudinalConfig
from .deviation import DeviationResult, compute_deviation, count_persistent_deviation
from .ds_pcd import (
    BaselineUnavailable,
    DayRecord,
    DsPcdBaseline,
    DsPcdConfig,
    config_fingerprint as ds_pcd_fingerprint,
    score_timeline,
    scoring_config_fingerprint as ds_pcd_scoring_fingerprint,
)
from .model_contract import match_model_input
from .quality import assess_quality
from .schema import DailyObservation
from .storage import ProfileStore
from .survival import CoxModelArtifact, predict_cumulative_risk


_RESEARCH_COVARIATE_NAMES = (
    "rms_distance", "cusum", "distance", "deviation_index",
    "trend_7d", "persistent_deviation_days",
)


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
    ds_pcd: dict[str, Any] | None = None
    as_of: str = ""
    baseline_end_date: str | None = None
    baseline_source: str = "stored"
    trend_7d_window: dict[str, Any] = field(default_factory=dict)
    baseline_contains_prediction: bool = False
    research_covariates: dict[str, float | None] = field(default_factory=dict)
    research_covariate_status: dict[str, Any] = field(default_factory=dict)
    valid_day_counts: dict[str, int] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    model_input_status: dict[str, Any] = field(default_factory=dict)

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
            "ds_pcd": self.ds_pcd,
            "as_of": self.as_of or self.observation_date,
            "baseline_end_date": self.baseline_end_date,
            "baseline_source": self.baseline_source,
            "trend_7d_window": dict(self.trend_7d_window),
            "baseline_contains_prediction": self.baseline_contains_prediction,
            "research_covariates": dict(self.research_covariates),
            "research_covariate_status": dict(self.research_covariate_status),
            "valid_day_counts": dict(self.valid_day_counts),
            "provenance": dict(self.provenance),
            "model_input_status": dict(self.model_input_status),
        }


def trend_7d_calendar(
    history: list[tuple[str, float]],
    as_of: str,
) -> tuple[float, dict[str, Any]]:
    """最近 7 个日历日内的变化率；稀疏观测按实际日期间隔折算。"""
    end = _date.fromisoformat(as_of)
    window = [
        (day, value)
        for day, value in history
        if _date.fromisoformat(day) > end - timedelta(days=7)
    ]
    if len(window) < 2:
        return 0.0, {
            "window_start": (end - timedelta(days=7)).isoformat(),
            "window_end": as_of,
            "points": len(window),
            "span_days": 0,
        }
    first_day, first_value = window[0]
    last_day, last_value = window[-1]
    span = (_date.fromisoformat(last_day) - _date.fromisoformat(first_day)).days
    trend = (last_value - first_value) / span if span > 0 else 0.0
    return float(trend), {
        "window_start": (end - timedelta(days=7)).isoformat(),
        "window_end": as_of,
        "points": len(window),
        "span_days": span,
    }


class LongitudinalRiskService:
    """对单个个人画像执行一次未来风险评估。

    默认行为与旧版 RMS 规则一致：使用已保存的个人基线、计算偏离距离与
    持续偏离天数。带上 ``ds_pcd_config`` 时，同一个有效观测日还会计算
    DS-PCD 的方向统计量、个人尾概率、CUSUM 与主要贡献特征。

    评分只使用预测日及以前的观测；传入更晚的观测不会改变历史结果。
    """

    def __init__(
        self,
        store: ProfileStore,
        config: LongitudinalConfig,
        models: dict[int, CoxModelArtifact] | None = None,
        *,
        ds_pcd_config: DsPcdConfig | None = None,
        ds_pcd_alarm_threshold: float | None = None,
        modality: str = "video",
        population: str = "community-dwelling older adults",
        feature_version: str = "video-pose-v1",
        score_version: str = "legacy-rms-v1",
    ) -> None:
        self.store = store
        self.config = config
        self.models = dict(models or {})
        self.ds_pcd_config = ds_pcd_config
        self.ds_pcd_alarm_threshold = ds_pcd_alarm_threshold
        self.modality = str(modality)
        self.population = str(population)
        self.feature_version = str(feature_version)
        self.score_version = str(score_version)
        if self.models and set(self.models) != {7, 30}:
            raise ValueError("校准模型必须同时包含7天和30天窗口")

    def _save(self, result: FutureRiskResult) -> FutureRiskResult:
        self.store.save_latest_result(result)
        return result

    def evaluate(
        self,
        person_id: str,
        observation: DailyObservation,
        *,
        as_of: str | None = None,
        persist: bool = True,
        baseline_override: PersonalBaseline | None = None,
        ds_pcd_baseline_override: DsPcdBaseline | None = None,
        strict_baseline_order: bool = True,
    ) -> FutureRiskResult:
        if person_id != self.store.person_id or observation.person_id != person_id:
            raise ValueError("person_id 与画像或观测不一致")
        as_of_date = as_of or observation.observation_date
        if observation.observation_date != as_of_date:
            raise ValueError("as_of 必须等于传入观测的日期")
        history = self._merged_observations(observation, as_of_date)
        quality = assess_quality(observation, self.config)
        valid_days = sum(assess_quality(item, self.config).is_valid for item in history)
        if not quality.is_valid:
            result = FutureRiskResult(
                person_id=person_id,
                status="insufficient_quality",
                baseline_valid_days=valid_days,
                observation_date=observation.observation_date,
                deviation_index=None,
                distance=None,
                persistent_deviation_days=0,
                feature_deviations={},
                risk_7d=None,
                risk_30d=None,
                quality_reasons=quality.reasons,
                message="本次数据质量不足，未更新个人基线或风险结果",
                ds_pcd=None,
                as_of=as_of_date,
                research_covariate_status={
                    "rms_distance": "unavailable:insufficient_quality",
                    "cusum": "unavailable:insufficient_quality",
                    "covariates": {
                        name: "unavailable:insufficient_quality"
                        for name in _RESEARCH_COVARIATE_NAMES
                    },
                },
                valid_day_counts={
                    "n_valid_days": valid_days,
                    "n_valid_days_before_landmark": sum(
                        1 for item in history
                        if item.observation_date < as_of_date
                        and assess_quality(item, self.config).is_valid
                    ),
                },
                provenance=self._provenance(observation, None, None, as_of_date),
            )
            return self._save(result) if persist else result

        baseline, baseline_source = self._baseline_for(
            person_id, history, as_of_date, baseline_override
        )
        if baseline is None:
            result = FutureRiskResult(
                person_id=person_id,
                status="collecting",
                baseline_valid_days=valid_days,
                observation_date=observation.observation_date,
                deviation_index=None,
                distance=None,
                persistent_deviation_days=0,
                feature_deviations={},
                risk_7d=None,
                risk_30d=None,
                quality_reasons=(),
                message=f"正在收集个人基线，需要至少 {self.config.min_baseline_days} 个有效观测日",
                ds_pcd=None,
                as_of=as_of_date,
                baseline_source="missing",
                research_covariate_status={
                    "rms_distance": "unavailable:baseline_not_ready",
                    "cusum": "unavailable:baseline_not_ready",
                    "covariates": {
                        name: "unavailable:baseline_not_ready"
                        for name in _RESEARCH_COVARIATE_NAMES
                    },
                },
                valid_day_counts={
                    "n_valid_days": valid_days,
                    "n_valid_days_before_landmark": sum(
                        1 for item in history
                        if item.observation_date < as_of_date
                        and assess_quality(item, self.config).is_valid
                    ),
                },
                provenance=self._provenance(observation, None, None, as_of_date),
            )
            return self._save(result) if persist else result
        baseline_contains_prediction = bool(
            baseline.valid_dates and max(baseline.valid_dates) >= as_of_date
        )
        if baseline_contains_prediction:
            if strict_baseline_order or max(baseline.valid_dates) > as_of_date:
                raise ValueError(
                    "个人基线包含预测日或之后的数据；请按预测日截断历史或重建基线"
                )
            baseline_source = "stored_contains_prediction"

        dated_history = self._post_baseline_deviations(baseline, history)
        if dated_history:
            current = dated_history[-1][1]
            points = [result for _, result in dated_history]
        else:
            current = compute_deviation(observation.features, baseline)
            points = [current]
        persistent_days = count_persistent_deviation(
            points, self.config.persistent_distance_threshold
        )
        trend, trend_window = self._trend_7d(dated_history, as_of_date)

        ds_pcd = self._ds_pcd_for(
            person_id, history, as_of_date, ds_pcd_baseline_override
        )
        cusum = (
            float(ds_pcd["cusum"])
            if ds_pcd and ds_pcd.get("status") == "scored"
            and ds_pcd.get("cusum") is not None else None
        )
        research_covariates = {
            "rms_distance": current.distance,
            "cusum": cusum,
            "distance": current.distance,
            "deviation_index": current.deviation_index,
            "trend_7d": float(trend),
            "persistent_deviation_days": float(persistent_days),
        }
        if cusum is not None:
            cusum_status = "available"
        elif ds_pcd is None:
            cusum_status = "missing:ds_pcd_disabled"
        else:
            cusum_status = "missing:" + str(ds_pcd.get("status", "unknown"))
        covariate_status = {
            "rms_distance": (
                "available" if research_covariates["rms_distance"] is not None
                else "unavailable:no_distance"
            ),
            "cusum": cusum_status,
            "distance": (
                "available" if research_covariates["distance"] is not None
                else "unavailable:no_distance"
            ),
            "deviation_index": (
                "available" if research_covariates["deviation_index"] is not None
                else "unavailable:no_deviation_index"
            ),
            "trend_7d": "available",
            "persistent_deviation_days": "available",
        }
        research_status = {
            "covariates": covariate_status,
            "rms_distance": covariate_status["rms_distance"],
            "cusum": cusum_status,
            "precision_mode": (ds_pcd or {}).get("precision_mode"),
            "source": observation.source,
            "as_of": as_of_date,
            "n_valid_days_includes_landmark": True,
        }
        valid_day_counts = {
            "n_valid_days": valid_days,
            "n_valid_days_before_landmark": sum(
                1 for item in history
                if item.observation_date < as_of_date
                and assess_quality(item, self.config).is_valid
            ),
        }
        provenance = self._provenance(
            observation, baseline, ds_pcd, as_of_date,
        )
        risk_7d = None
        risk_30d = None
        model_input_status: dict[str, Any] = {}
        if self.models:
            for artifact in self.models.values():
                if not artifact.calibrated:
                    raise ValueError("模型尚未校准，不能用于服务跌倒概率")
            model_features = self._model_features(
                observation, current, points, persistent_days, trend, cusum,
            )
            missing = sorted({
                name for artifact in self.models.values()
                for name in artifact.feature_names if name not in model_features
            })
            for horizon, artifact in sorted(self.models.items()):
                if artifact.input_contract is None:
                    model_input_status[str(horizon)] = {
                        "status": "legacy_no_contract",
                    }
                    continue
                match = match_model_input(
                    artifact.input_contract,
                    provenance=provenance,
                    covariates=research_covariates,
                    feature_names=artifact.feature_names,
                    ds_pcd=ds_pcd,
                )
                model_input_status[str(horizon)] = match.to_dict()
            mismatched = {
                horizon: item for horizon, item in model_input_status.items()
                if item.get("status") not in ("ok", "legacy_no_contract")
            }
            if missing:
                status = "model_inputs_incomplete"
                message = "模型输入缺少特征：" + ", ".join(missing)
            elif mismatched:
                status = "model_input_mismatch"
                message = "模型输入与契约不匹配：" + ", ".join(
                    f"{horizon}={item.get('status')}"
                    for horizon, item in sorted(mismatched.items())
                )
            else:
                risk_7d = predict_cumulative_risk(model_features, self.models[7])
                risk_30d = predict_cumulative_risk(model_features, self.models[30])
                status = "calibrated"
                message = "已使用校准 Landmark Cox 模型计算未来7天和30天累计跌倒概率"
        else:
            if persistent_days >= self.config.persistent_days:
                status = "calibration_required"
                message = "检测到持续个人基线偏离；当前结果未经跌倒概率校准"
            else:
                status = "baseline_ready"
                message = "个人基线可用；当前结果为统计偏离监测，不是跌倒概率"
        result = FutureRiskResult(
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
            ds_pcd=ds_pcd,
            as_of=as_of_date,
            baseline_end_date=baseline.valid_dates[-1] if baseline.valid_dates else None,
            baseline_source=baseline_source,
            trend_7d_window=trend_window,
            baseline_contains_prediction=baseline_contains_prediction,
            research_covariates=research_covariates,
            research_covariate_status=research_status,
            valid_day_counts=valid_day_counts,
            provenance=provenance,
            model_input_status=model_input_status,
        )
        return self._save(result) if persist else result

    # ------------------------------------------------------------------
    # 历史与基线
    # ------------------------------------------------------------------

    def _merged_observations(
        self,
        current: DailyObservation,
        as_of: str,
    ) -> list[DailyObservation]:
        """只保留预测日及以前的观测；同一日期只保留一条。"""
        cutoff = _date.fromisoformat(as_of)
        by_date: dict[str, DailyObservation] = {}
        for item in self.store.load_observations():
            if _date.fromisoformat(item.observation_date) <= cutoff:
                by_date[item.observation_date] = item
        by_date[current.observation_date] = current
        return [by_date[key] for key in sorted(by_date)]

    def _provenance(
        self,
        observation: DailyObservation,
        baseline: PersonalBaseline | None,
        ds_pcd: dict[str, Any] | None,
        as_of: str,
    ) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "population": self.population,
            "feature_version": self.feature_version,
            "score_version": self.score_version,
            "rms_config_fingerprint": config_fingerprint(self.config),
            "rms_distance_config_fingerprint": rms_distance_config_fingerprint(self.config),
            "detector": "ds_pcd" if self.ds_pcd_config is not None else "rms",
            "precision_mode": (ds_pcd or {}).get("precision_mode"),
            "ds_pcd_config_fingerprint": (ds_pcd or {}).get("config_fingerprint"),
            "ds_pcd_scoring_config_fingerprint": (ds_pcd or {}).get(
                "scoring_config_fingerprint"
            ),
            "baseline_version": baseline.version if baseline is not None else None,
            "ds_pcd_baseline_version": (ds_pcd or {}).get("baseline_version"),
            "source": observation.source,
            "as_of": as_of,
        }

    def _baseline_for(
        self,
        person_id: str,
        observations: list[DailyObservation],
        as_of: str,
        override: PersonalBaseline | None,
    ) -> tuple[PersonalBaseline | None, str]:
        if override is not None:
            baseline = override
            source = "override"
        else:
            raw = self.store.load_baseline()
            if raw is None:
                return None, "missing"
            baseline = PersonalBaseline.from_dict(raw)
            source = "stored"
        if baseline.person_id != person_id:
            raise ValueError("基线 person_id 与当前个人不一致")
        if baseline.config_fingerprint != config_fingerprint(self.config):
            raise ValueError("个人基线与当前纵向配置不一致，请重新建立基线")
        return baseline, source

    def _post_baseline_deviations(
        self,
        baseline: PersonalBaseline,
        observations: list[DailyObservation],
    ) -> list[tuple[str, DeviationResult]]:
        baseline_dates = set(baseline.valid_dates)
        history: list[tuple[str, DeviationResult]] = []
        for item in observations:
            if item.observation_date in baseline_dates:
                continue
            if not assess_quality(item, self.config).is_valid:
                continue
            history.append((item.observation_date, compute_deviation(item.features, baseline)))
        return history

    @staticmethod
    def _trend_7d(
        history: list[tuple[str, DeviationResult]],
        as_of: str,
    ) -> tuple[float, dict[str, Any]]:
        return trend_7d_calendar(
            [(day, result.distance) for day, result in history], as_of
        )

    # ------------------------------------------------------------------
    # DS-PCD
    # ------------------------------------------------------------------

    def _ds_pcd_for(
        self,
        person_id: str,
        observations: list[DailyObservation],
        as_of: str,
        override: DsPcdBaseline | None,
    ) -> dict[str, Any] | None:
        if self.ds_pcd_config is None:
            return None
        config = self.ds_pcd_config
        payload: dict[str, Any] = {
            "detector": "ds_pcd",
            "precision_mode": config.precision_mode,
            "k_ref": config.k_ref,
            "gap_reset_days": config.gap_reset_days,
            "config_fingerprint": ds_pcd_fingerprint(config),
            "scoring_config_fingerprint": ds_pcd_scoring_fingerprint(config),
        }
        if override is not None:
            baseline = override
            source = "override"
        else:
            raw = self.store.load_ds_pcd_baseline()
            if raw is None:
                payload.update({"status": "no_baseline"})
                return payload
            baseline = DsPcdBaseline.from_dict(raw)
            source = "stored"
        payload["baseline_source"] = source
        payload["baseline_end"] = baseline.baseline_end
        payload["baseline_version"] = baseline.version
        if baseline.person_id != person_id:
            payload.update({"status": "person_mismatch"})
            return payload
        if baseline.config_fingerprint != ds_pcd_fingerprint(config):
            payload.update({"status": "config_mismatch"})
            return payload
        if baseline.baseline_end >= as_of:
            payload.update({"status": "baseline_not_before_prediction"})
            return payload
        records = []
        for item in observations:
            assessment = assess_quality(item, self.config)
            records.append(DayRecord(
                date=item.observation_date,
                features=dict(item.features),
                feature_se=dict(item.feature_se),
                valid=assessment.is_valid,
                note=";".join(assessment.reasons),
            ))
        try:
            timeline = score_timeline(
                records, person_id, config, baseline=baseline,
                alarm_threshold=self.ds_pcd_alarm_threshold,
            )
        except BaselineUnavailable as exc:
            payload.update({"status": exc.reason, "message": str(exc)})
            return payload
        row = timeline.score_on(as_of)
        if row is None:
            payload.update({"status": "not_scored", "message": "预测日不在监测区间"})
            return payload
        scored = [
            item for item in timeline.rows
            if item.status == "scored" and item.date <= as_of
        ]
        payload.update(row.to_dict())
        payload.update({
            "baseline_dates": list(baseline.baseline_dates),
            "scored_days_to_as_of": len(scored),
            "cusum_resets_to_as_of": sum(1 for item in scored if item.cusum_reset),
            "alarms_to_as_of": (
                sum(1 for item in scored if item.alarm)
                if self.ds_pcd_alarm_threshold is not None else None
            ),
            "alarm_threshold": self.ds_pcd_alarm_threshold,
        })
        return payload

    @staticmethod
    def _model_features(
        observation: DailyObservation,
        current: DeviationResult,
        history: list[DeviationResult],
        persistent_days: int,
        trend: float,
        cusum: float | None = None,
    ) -> dict[str, float]:
        features = {
            "distance": current.distance,
            "rms_distance": current.distance,
            "deviation_index": current.deviation_index,
            "persistent_deviation_days": float(persistent_days),
            "trend_7d": float(trend),
        }
        if cusum is not None:
            features["cusum"] = float(cusum)
        for name, value in current.feature_deviations.items():
            features[name] = value
            features[f"z_{name}"] = value
        for name, value in observation.psychological.items():
            features[f"psych_{name}"] = value
        return features
