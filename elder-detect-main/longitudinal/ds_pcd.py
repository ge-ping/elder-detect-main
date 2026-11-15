"""DS-PCD：方向先验 + 测量误差感知的个人基线序贯偏离检验。

本模块只做统计层，不依赖视频或穿戴设备的具体来源：

* 观测模型 ``x_t = mu + eta_t + eps_t``，``eta_t ~ N(0, tau^2)``，
  ``eps_t ~ N(0, s_t^2)``；``s_t`` 是当次会话的测量标准误。
* 基线用 DerSimonian–Laird 随机效应估计 ``mu``、``tau^2``、``Var(mu)``；
  ``precision_mode="none"`` 时令 ``s=0``，退化为不含会话精度的版本。
* 单日标准化后计算方向约束平方统计量 ``T`` 与双侧 ``D``，并通过
  个人化零分布表得到尾概率与正态分数 ``q``。
* CUSUM ``C_t = max(0, C_{t-1} + q_t - k_ref)``；相邻有效观测跨过
  ``gap_reset_days`` 个日历时重置 ``C``，个人基线保留。

所有随机性都来自稳定摘要派生的种子，零分布表随基线保存，因此同一
输入、同一基线的得分可复现。模块不写文件、不读文件。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date as _date
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


_NORMAL = NormalDist()
SEED_VERSION = "ds-pcd-v1"


class BaselineUnavailable(ValueError):
    """基线无法建立时抛出，``reason`` 供调用方记录。"""

    def __init__(self, reason: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.reason = reason
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "message": str(self), "details": self.details}


def stable_seed(material: str) -> int:
    """稳定种子：不依赖 Python 内置 hash，跨进程与跨版本可复现。"""
    return int.from_bytes(sha256(material.encode("utf-8")).digest()[:8], "big")


def _as_date(value: str | _date) -> _date:
    if isinstance(value, _date):
        return value
    return _date.fromisoformat(str(value))


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class DsPcdConfig:
    """DS-PCD 的预注册参数；方向向量不从数据学习。"""

    features: tuple[str, ...] = ("gait_speed_proxy", "cadence_spm", "step_time_mean_s")
    directions: Mapping[str, float] = field(default_factory=lambda: {
        "gait_speed_proxy": -1.0,
        "cadence_spm": -1.0,
        "step_time_mean_s": 1.0,
    })
    precision_mode: str = "session_se"
    min_baseline_days: int = 14
    min_baseline_days_after_outlier: int = 10
    outlier_z: float = 4.0
    shrink_lambda: float = 0.5
    null_samples: int = 20000
    quantile_points: int = 1000
    k_ref: float = 0.5
    gap_reset_days: int = 7
    q_min: float = -4.0
    q_max: float = 6.0
    min_features: int = 2
    # tau_floor / tau_floor_relative 以特征原始单位给出日间波动下限
    # （例如步速 0.005 m/s），拟合时平方为方差下限。
    tau_floor: Mapping[str, float] = field(default_factory=dict)
    tau_floor_relative: float = 1e-3
    se_floor: Mapping[str, float] = field(default_factory=dict)
    se_floor_relative: float = 1e-6
    bootstrap_samples: int = 200
    min_bouts: int = 5

    def __post_init__(self) -> None:
        features = tuple(str(name) for name in self.features)
        if not features or len(set(features)) != len(features):
            raise ValueError("DS-PCD features 必须非空且不重复")
        missing = [name for name in features if name not in self.directions]
        extra = [name for name in self.directions if name not in features]
        if missing or extra:
            raise ValueError(f"DS-PCD 方向向量与特征不一致：缺少 {missing}，多余 {extra}")
        for name in features:
            if float(self.directions[name]) not in (-1.0, 0.0, 1.0):
                raise ValueError(f"DS-PCD 方向只能是 -1、0 或 1：{name}")
        if self.precision_mode not in ("session_se", "none"):
            raise ValueError("precision_mode 只能是 session_se 或 none")
        if not 2 <= self.min_features <= len(features):
            raise ValueError("min_features 必须介于 2 与特征数之间")
        if self.min_baseline_days < 2:
            raise ValueError("min_baseline_days 至少为 2")
        if not 1 <= self.min_baseline_days_after_outlier <= self.min_baseline_days:
            raise ValueError("min_baseline_days_after_outlier 必须介于 1 与 min_baseline_days 之间")
        if not 0.0 <= self.shrink_lambda <= 1.0:
            raise ValueError("shrink_lambda 必须位于 0 到 1 之间")
        if self.null_samples < 200 or self.quantile_points < 50:
            raise ValueError("零分布抽样数与分位点数过小")
        if not 0.0 < self.gap_reset_days:
            raise ValueError("gap_reset_days 必须为正")
        if self.q_min >= self.q_max:
            raise ValueError("q_min 必须小于 q_max")
        if self.bootstrap_samples < 20:
            raise ValueError("bootstrap_samples 至少为 20")
        if self.min_bouts < 3:
            raise ValueError("min_bouts 至少为 3")

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "directions": {name: float(self.directions[name]) for name in self.features},
            "precision_mode": self.precision_mode,
            "min_baseline_days": self.min_baseline_days,
            "min_baseline_days_after_outlier": self.min_baseline_days_after_outlier,
            "outlier_z": self.outlier_z,
            "shrink_lambda": self.shrink_lambda,
            "null_samples": self.null_samples,
            "quantile_points": self.quantile_points,
            "k_ref": self.k_ref,
            "gap_reset_days": self.gap_reset_days,
            "q_min": self.q_min,
            "q_max": self.q_max,
            "min_features": self.min_features,
            "tau_floor": dict(self.tau_floor),
            "tau_floor_relative": self.tau_floor_relative,
            "se_floor": dict(self.se_floor),
            "se_floor_relative": self.se_floor_relative,
            "bootstrap_samples": self.bootstrap_samples,
            "min_bouts": self.min_bouts,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DsPcdConfig":
        payload = dict(data)
        payload["features"] = tuple(payload.get("features", ()))
        payload["directions"] = dict(payload.get("directions", {}))
        for key in ("tau_floor", "se_floor"):
            payload[key] = dict(payload.get(key, {}))
        return cls(**payload)


def config_fingerprint(config: DsPcdConfig) -> str:
    payload = json.dumps(config.to_dict(), ensure_ascii=False, sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()


def scoring_config_fingerprint(config: DsPcdConfig) -> str:
    """C 的计算实际依赖的 DS-PCD 配置子集指纹。

    ``bootstrap_samples`` 与 ``min_bouts`` 属于上游精度估计配置，不进入
    ``score_timeline`` 的 C 计算；其余字段会影响基线 q 表、方向统计量或
    CUSUM，因此保留。
    """
    payload = config.to_dict()
    for key in ("bootstrap_samples", "min_bouts"):
        payload.pop(key, None)
    return sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# 记录与结果
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DayRecord:
    """单个有效观测日。``feature_se`` 是当次会话的测量标准误。"""

    date: str
    features: Mapping[str, float] = field(default_factory=dict)
    feature_se: Mapping[str, float] = field(default_factory=dict)
    valid: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        _as_date(self.date)

    @property
    def day(self) -> _date:
        return _as_date(self.date)


@dataclass
class DayScore:
    """一个观测日的 DS-PCD 结果。"""

    date: str
    status: str
    mask: tuple[str, ...] = ()
    z: dict[str, float] = field(default_factory=dict)
    statistic_t: float | None = None
    p_value: float | None = None
    q: float | None = None
    omnibus_d: float | None = None
    omnibus_p: float | None = None
    top_contributors: tuple[str, ...] = ()
    contributions: dict[str, float] = field(default_factory=dict)
    se_used: dict[str, float] = field(default_factory=dict)
    cusum: float | None = None
    cusum_reset: bool = False
    gap_days: int | None = None
    days_since_baseline_end: int | None = None
    alarm: bool | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "status": self.status,
            "mask": list(self.mask),
            "z": dict(self.z),
            "T": self.statistic_t,
            "p_value": self.p_value,
            "q": self.q,
            "omnibus_D": self.omnibus_d,
            "omnibus_p": self.omnibus_p,
            "top_contributors": list(self.top_contributors),
            "contributions": dict(self.contributions),
            "se_used": dict(self.se_used),
            "cusum": self.cusum,
            "cusum_reset": self.cusum_reset,
            "gap_days": self.gap_days,
            "days_since_baseline_end": self.days_since_baseline_end,
            "alarm": self.alarm,
            "note": self.note,
        }


@dataclass
class TimelineScore:
    """一次完整时间线的重放结果（基线 + 每日得分 + 汇总）。"""

    person_id: str
    baseline: "DsPcdBaseline"
    rows: list[DayScore]
    summary: dict[str, Any] = field(default_factory=dict)

    def score_on(self, day: str | _date) -> DayScore | None:
        key = _as_date(day).isoformat()
        for row in self.rows:
            if row.date == key:
                return row
        return None

    def scored_rows(self) -> list[DayScore]:
        return [row for row in self.rows if row.status == "scored"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "baseline": self.baseline.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
            "summary": dict(self.summary),
        }


# ---------------------------------------------------------------------------
# 基线与零分布
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DsPcdBaseline:
    """DS-PCD 个人基线；只含预注册参数与本人数据。"""

    person_id: str
    created_at: str
    feature_names: tuple[str, ...]
    directions: dict[str, float]
    precision_mode: str
    baseline_dates: tuple[str, ...]
    valid_days: int
    excluded_dates: tuple[str, ...]
    mu: dict[str, float]
    tau2: dict[str, float]
    var_mu: dict[str, float]
    correlation: tuple[tuple[float, ...], ...]
    null: dict[str, dict[str, Any]]
    config_fingerprint: str
    seed_material: str
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def baseline_end(self) -> str:
        return self.baseline_dates[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "person_id": self.person_id,
            "created_at": self.created_at,
            "feature_names": list(self.feature_names),
            "directions": dict(self.directions),
            "precision_mode": self.precision_mode,
            "baseline_dates": list(self.baseline_dates),
            "valid_days": self.valid_days,
            "excluded_dates": list(self.excluded_dates),
            "mu": dict(self.mu),
            "tau2": dict(self.tau2),
            "var_mu": dict(self.var_mu),
            "correlation": [list(row) for row in self.correlation],
            "null": {
                key: {name: list(value[name]) for name in ("p_grid", "T", "D")}
                | {"tail_p": value["tail_p"], "samples": value["samples"]}
                for key, value in self.null.items()
            },
            "config_fingerprint": self.config_fingerprint,
            "seed_material": self.seed_material,
            "metadata": dict(self.metadata),
            "config": dict(self.config),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DsPcdBaseline":
        return cls(
            person_id=str(data["person_id"]),
            created_at=str(data["created_at"]),
            feature_names=tuple(str(name) for name in data["feature_names"]),
            directions={str(k): float(v) for k, v in data["directions"].items()},
            precision_mode=str(data["precision_mode"]),
            baseline_dates=tuple(str(value) for value in data["baseline_dates"]),
            valid_days=int(data["valid_days"]),
            excluded_dates=tuple(str(value) for value in data.get("excluded_dates", ())),
            mu={str(k): float(v) for k, v in data["mu"].items()},
            tau2={str(k): float(v) for k, v in data["tau2"].items()},
            var_mu={str(k): float(v) for k, v in data["var_mu"].items()},
            correlation=tuple(tuple(float(v) for v in row) for row in data["correlation"]),
            null={
                str(key): {
                    "p_grid": [float(v) for v in value["p_grid"]],
                    "T": [float(v) for v in value["T"]],
                    "D": [float(v) for v in value["D"]],
                    "tail_p": float(value["tail_p"]),
                    "samples": int(value["samples"]),
                }
                for key, value in data["null"].items()
            },
            config_fingerprint=str(data["config_fingerprint"]),
            seed_material=str(data["seed_material"]),
            version=int(data.get("version", 1)),
            metadata=dict(data.get("metadata", {})),
            config=dict(data.get("config", {})),
        )


def mask_key(names: Iterable[str]) -> str:
    return "|".join(names)


def _feature_scale(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    scale = float(np.median(np.abs(array))) if len(array) else 0.0
    return scale if scale > 1e-12 else 1.0


def _robust_exclusion(
    values: dict[str, np.ndarray], outlier_z: float
) -> np.ndarray:
    """按每个特征的稳健 z 标记需要剔除的基线日。"""
    count = len(next(iter(values.values())))
    keep = np.ones(count, dtype=bool)
    for array in values.values():
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median)))
        if mad > 1e-12:
            scale = 1.4826 * mad
        else:
            q25, q75 = np.percentile(array, [25, 75])
            scale = (q75 - q25) / 1.349
        if scale <= 1e-12:
            continue
        z = (array - median) / scale
        keep &= np.abs(z) <= outlier_z
    return keep


def _fit_feature(
    values: np.ndarray,
    ses: np.ndarray | None,
    *,
    precision_mode: str,
    tau_floor: float,
    se_floor: float,
) -> tuple[float, float, float]:
    """单特征的随机效应基线；返回 (mu, tau2, var_mu)。"""
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    variance_floor = max(float(tau_floor), 1e-12) ** 2
    if precision_mode == "session_se":
        if ses is None or np.any(~np.isfinite(ses)) or np.any(ses <= 0):
            raise ValueError("session_se 模式需要全部基线日的正测量标准误")
        s = np.maximum(np.asarray(ses, dtype=np.float64), se_floor)
        v = s ** 2
        w = 1.0 / v
        mu_fe = float((w * x).sum() / w.sum())
        q_stat = float((w * (x - mu_fe) ** 2).sum())
        denominator = float(w.sum() - (w ** 2).sum() / w.sum())
        tau2 = max(0.0, (q_stat - (n - 1)) / denominator) if denominator > 1e-12 else 0.0
        tau2 = max(tau2, variance_floor)
        ws = 1.0 / (tau2 + v)
        mu = float((ws * x).sum() / ws.sum())
        var_mu = float(1.0 / ws.sum())
    else:
        mu = float(x.mean())
        if n > 1:
            tau2 = max(float(x.var(ddof=1)), variance_floor)
        else:
            tau2 = variance_floor
        var_mu = tau2 / n
    return mu, tau2, var_mu


def _shrink_correlation(residuals: np.ndarray, shrink_lambda: float) -> np.ndarray:
    k = residuals.shape[1]
    if residuals.shape[0] < 3:
        return np.eye(k)
    centered = residuals - residuals.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / (len(centered) - 1)
    sd = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    if np.any(sd <= 1e-12):
        return np.eye(k)
    empirical = covariance / sd[:, None] / sd[None, :]
    empirical = np.clip(empirical, -1.0, 1.0)
    matrix = shrink_lambda * np.eye(k) + (1.0 - shrink_lambda) * empirical
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 1e-6, None)
    matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    diagonal = np.sqrt(np.clip(np.diag(matrix), 1e-12, None))
    matrix = matrix / diagonal[:, None] / diagonal[None, :]
    return (matrix + matrix.T) / 2.0


def _null_table(
    correlation: np.ndarray,
    features: Sequence[str],
    directions: Mapping[str, float],
    *,
    samples: int,
    points: int,
    seed: int,
) -> dict[str, Any]:
    k = len(features)
    jitter = 1e-10 * np.eye(k)
    cholesky = np.linalg.cholesky(correlation + jitter)
    rng = np.random.default_rng(seed)
    draws = rng.standard_normal((samples, k)) @ cholesky.T
    statistic = directional_statistic(draws, features, directions)
    omnibus = np.sum(draws ** 2, axis=1)
    grid = (np.arange(points) + 0.5) / points
    return {
        "p_grid": grid.tolist(),
        "T": np.quantile(statistic, grid).tolist(),
        "D": np.quantile(omnibus, grid).tolist(),
        "tail_p": 0.5 / (samples + 1.0),
        "samples": int(samples),
    }


def fit_baseline(
    records: Iterable[DayRecord],
    person_id: str,
    config: DsPcdConfig,
    *,
    created_at: str | None = None,
    baseline_dates: Sequence[str] | None = None,
) -> DsPcdBaseline:
    """拟合个人基线。

    默认取最早 ``min_baseline_days`` 个合格观测日；给出 ``baseline_dates``
    时强制使用这些日期（用于与 RMS 基线共用同一批基线日）。
    """
    features = tuple(config.features)
    ordered: list[DayRecord] = []
    seen: set[str] = set()
    for record in records:
        if not record.valid:
            continue
        key = record.date
        if key in seen:
            raise ValueError(f"重复观测日：{key}")
        seen.add(key)
        ordered.append(record)
    ordered.sort(key=lambda item: item.date)

    eligible = []
    skipped: list[dict[str, str]] = []
    for record in ordered:
        missing = [name for name in features if not _finite(record.features.get(name))]
        if missing:
            skipped.append({"date": record.date, "reason": "missing_features:" + ",".join(missing)})
            continue
        if config.precision_mode == "session_se":
            bad_se = [
                name for name in features
                if not _finite(record.feature_se.get(name)) or float(record.feature_se[name]) <= 0
            ]
            if bad_se:
                skipped.append({"date": record.date, "reason": "missing_se:" + ",".join(bad_se)})
                continue
        eligible.append(record)

    if baseline_dates is not None:
        by_date = {record.date: record for record in eligible}
        missing = [day for day in baseline_dates if day not in by_date]
        if missing:
            raise BaselineUnavailable(
                "baseline_dates_unavailable",
                f"指定基线日在当前精度模式下不可用：{missing[:5]}",
                {"missing_dates": missing[:50]},
            )
        candidates = [by_date[day] for day in baseline_dates]
        if len(candidates) < config.min_baseline_days_after_outlier:
            raise BaselineUnavailable(
                "insufficient_days",
                f"指定基线日只有 {len(candidates)} 个，少于 {config.min_baseline_days_after_outlier}",
            )
    else:
        if len(eligible) < config.min_baseline_days:
            raise BaselineUnavailable(
                "insufficient_days",
                f"至少需要 {config.min_baseline_days} 个合格基线日，当前 {len(eligible)} 个",
                {"eligible_days": len(eligible), "skipped": skipped[:20]},
            )
        candidates = eligible[: config.min_baseline_days]
    values = {
        name: np.asarray([float(record.features[name]) for record in candidates])
        for name in features
    }
    keep = _robust_exclusion(values, config.outlier_z)
    excluded_dates = tuple(
        record.date for record, kept in zip(candidates, keep) if not kept
    )
    kept_records = [record for record, kept in zip(candidates, keep) if kept]
    if len(kept_records) < config.min_baseline_days_after_outlier:
        raise BaselineUnavailable(
            "insufficient_after_outlier",
            f"剔除异常基线日后只剩 {len(kept_records)} 个，少于 {config.min_baseline_days_after_outlier}",
            {"excluded_dates": list(excluded_dates)},
        )

    mu: dict[str, float] = {}
    tau2: dict[str, float] = {}
    var_mu: dict[str, float] = {}
    residuals = np.zeros((len(kept_records), len(features)))
    for index, name in enumerate(features):
        x = np.asarray([float(record.features[name]) for record in kept_records])
        floor = max(
            float(config.tau_floor.get(name, 0.0)),
            config.tau_floor_relative * _feature_scale(x),
        )
        se_floor = max(
            float(config.se_floor.get(name, 0.0)),
            config.se_floor_relative * _feature_scale(x),
        )
        ses = None
        if config.precision_mode == "session_se":
            ses = np.asarray([float(record.feature_se[name]) for record in kept_records])
        feature_mu, feature_tau2, feature_var_mu = _fit_feature(
            x, ses, precision_mode=config.precision_mode,
            tau_floor=floor, se_floor=se_floor,
        )
        mu[name] = feature_mu
        tau2[name] = feature_tau2
        var_mu[name] = feature_var_mu
        denominator = np.sqrt(feature_tau2 + (np.asarray(ses) ** 2 if ses is not None else 0.0))
        residuals[:, index] = (x - feature_mu) / denominator

    correlation = _shrink_correlation(residuals, config.shrink_lambda)
    seed_material = "|".join((
        SEED_VERSION, person_id, ",".join(record.date for record in kept_records),
        config_fingerprint(config),
    ))
    null = {
        mask_key(features): _null_table(
            correlation, features, config.directions,
            samples=config.null_samples, points=config.quantile_points,
            seed=stable_seed(seed_material + "|" + mask_key(features)),
        )
    }
    return DsPcdBaseline(
        person_id=str(person_id),
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        feature_names=features,
        directions={name: float(config.directions[name]) for name in features},
        precision_mode=config.precision_mode,
        baseline_dates=tuple(record.date for record in kept_records),
        valid_days=len(kept_records),
        excluded_dates=excluded_dates,
        mu=mu,
        tau2=tau2,
        var_mu=var_mu,
        correlation=tuple(tuple(float(value) for value in row) for row in correlation),
        null=null,
        config_fingerprint=config_fingerprint(config),
        seed_material=seed_material,
        config=config.to_dict(),
        metadata={
            "candidate_days": len(eligible),
            "excluded_outlier_days": list(excluded_dates),
            "skipped_days": skipped[:100],
            "se_summary": {
                name: {
                    "median": float(np.median([float(r.feature_se[name]) for r in kept_records])),
                    "max": float(np.max([float(r.feature_se[name]) for r in kept_records])),
                }
                for name in features
            } if config.precision_mode == "session_se" else {},
            "correlation_pairs": [
                {"pair": [features[i], features[j]], "value": float(correlation[i, j])}
                for i in range(len(features)) for j in range(i + 1, len(features))
            ],
        },
    )


# ---------------------------------------------------------------------------
# 单日统计量
# ---------------------------------------------------------------------------


def directional_statistic(
    z: np.ndarray,
    features: Sequence[str],
    directions: Mapping[str, float],
) -> np.ndarray:
    """方向约束平方统计量：d≠0 取 (max(0,d z))²，d=0 取 0.5 z²。"""
    values = np.asarray(z, dtype=np.float64)
    result = np.zeros(values.shape[:-1], dtype=np.float64)
    for index, name in enumerate(features):
        direction = float(directions[name])
        component = values[..., index]
        if direction == 0.0:
            result = result + 0.5 * component ** 2
        else:
            result = result + np.maximum(0.0, direction * component) ** 2
    return result


def omnibus_statistic(z: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(z, dtype=np.float64) ** 2, axis=-1)


def tail_probability(
    value: float, quantiles: Sequence[float], p_grid: Sequence[float], tail_p: float
) -> float:
    """由个人零分布分位表得到上尾概率。"""
    if value >= quantiles[-1]:
        return float(tail_p)
    if value <= quantiles[0]:
        return float(1.0 - tail_p)
    cdf = float(np.interp(value, quantiles, p_grid))
    return float(min(max(1.0 - cdf, tail_p), 1.0 - tail_p))


def normal_score(p_value: float, q_min: float, q_max: float) -> float:
    upper = min(max(float(p_value), 1e-12), 1.0 - 1e-12)
    return float(min(max(_NORMAL.inv_cdf(1.0 - upper), q_min), q_max))


def _table_for(
    baseline: DsPcdBaseline, config: DsPcdConfig, mask: tuple[str, ...],
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    key = mask_key(mask)
    if key in cache:
        return cache[key]
    if key in baseline.null:
        table = baseline.null[key]
    else:
        positions = [baseline.feature_names.index(name) for name in mask]
        matrix = np.asarray(baseline.correlation, dtype=np.float64)[np.ix_(positions, positions)]
        table = _null_table(
            matrix, mask, config.directions,
            samples=config.null_samples, points=config.quantile_points,
            seed=stable_seed(baseline.seed_material + "|" + key),
        )
    cache[key] = table
    return table


def _contributions(
    z: dict[str, float], config: DsPcdConfig
) -> dict[str, float]:
    contributions: dict[str, float] = {}
    for name, value in z.items():
        direction = float(config.directions[name])
        if direction == 0.0:
            contributions[name] = 0.5 * value ** 2
        else:
            contributions[name] = max(0.0, direction * value) ** 2
    return contributions


def score_record(
    baseline: DsPcdBaseline,
    record: DayRecord,
    config: DsPcdConfig,
    *,
    cache: dict[str, dict[str, Any]] | None = None,
) -> DayScore:
    """计算单日标准化值、T、尾概率、q 与贡献特征。"""
    cache = cache if cache is not None else {}
    if baseline.config_fingerprint != config_fingerprint(config):
        raise ValueError("DS-PCD 基线与当前配置不一致")
    present = [
        name for name in config.features
        if _finite(record.features.get(name))
    ]
    if len(present) < config.min_features:
        return DayScore(
            date=record.date, status="excluded_missing_features",
            note="可用特征少于 min_features",
        )
    if config.precision_mode == "session_se":
        present = [
            name for name in present
            if _finite(record.feature_se.get(name)) and float(record.feature_se[name]) > 0
        ]
        if len(present) < config.min_features:
            return DayScore(
                date=record.date, status="excluded_missing_se",
                note="缺少会话测量标准误",
            )
    mask = tuple(present)
    positions = [baseline.feature_names.index(name) for name in mask]
    se_used = {
        name: float(record.feature_se.get(name, 0.0))
        for name in mask
    }
    z: dict[str, float] = {}
    for name in mask:
        mu = baseline.mu[name]
        tau2 = baseline.tau2[name]
        var_mu = baseline.var_mu[name]
        if config.precision_mode == "session_se":
            s2 = se_used[name] ** 2
        else:
            s2 = 0.0
        denominator = math.sqrt(max(tau2 + s2 + var_mu, 1e-18))
        z[name] = (float(record.features[name]) - mu) / denominator
    vector = np.asarray([z[name] for name in mask], dtype=np.float64)
    table = _table_for(baseline, config, mask, cache)
    statistic = float(directional_statistic(vector, mask, config.directions))
    omnibus = float(omnibus_statistic(vector))
    p_value = tail_probability(statistic, table["T"], table["p_grid"], table["tail_p"])
    omnibus_p = tail_probability(omnibus, table["D"], table["p_grid"], table["tail_p"])
    contributions = _contributions(z, config)
    ordered = sorted(
        (name for name in mask if contributions[name] > 0),
        key=lambda name: contributions[name], reverse=True,
    )
    total = sum(contributions.values())
    top = tuple(
        name for name in ordered[:3]
        if total > 0 and contributions[name] / total >= 0.1
    )
    return DayScore(
        date=record.date,
        status="scored",
        mask=mask,
        z=z,
        statistic_t=statistic,
        p_value=p_value,
        q=normal_score(p_value, config.q_min, config.q_max),
        omnibus_d=omnibus,
        omnibus_p=omnibus_p,
        top_contributors=top,
        contributions=contributions,
        se_used=se_used,
    )


# ---------------------------------------------------------------------------
# 时间线与 CUSUM
# ---------------------------------------------------------------------------


def cusum_replay(
    rows: Sequence[DayScore],
    config: DsPcdConfig,
    *,
    alarm_threshold: float | None = None,
    baseline_end: str | None = None,
) -> list[DayScore]:
    """按日历顺序重放 CUSUM；长空档重置累计值，基线不变。"""
    previous: _date | None = None
    current = 0.0
    result: list[DayScore] = []
    for row in sorted(rows, key=lambda item: item.date):
        if row.status != "scored" or row.q is None:
            result.append(row)
            continue
        day = _as_date(row.date)
        gap = (day - previous).days if previous is not None else None
        reset = bool(gap is not None and gap > config.gap_reset_days)
        if reset:
            current = 0.0
        current = max(0.0, current + float(row.q) - config.k_ref)
        result.append(replace(
            row,
            cusum=current,
            cusum_reset=reset,
            gap_days=gap,
            days_since_baseline_end=(
                (day - _as_date(baseline_end)).days if baseline_end else None
            ),
            alarm=(
                None if alarm_threshold is None else bool(current >= alarm_threshold)
            ),
        ))
        previous = day
    return result


def score_timeline(
    records: Iterable[DayRecord],
    person_id: str,
    config: DsPcdConfig,
    *,
    baseline: DsPcdBaseline | None = None,
    alarm_threshold: float | None = None,
) -> TimelineScore:
    """建立（或复用）基线并重放整条时间线。"""
    ordered = sorted(
        (record for record in records if record.valid), key=lambda item: item.date
    )
    if not ordered:
        raise BaselineUnavailable("no_valid_days", "没有有效观测日")
    baseline = baseline or fit_baseline(ordered, person_id, config)
    if baseline.config_fingerprint != config_fingerprint(config):
        raise ValueError("DS-PCD 基线与当前配置不一致")
    if baseline.person_id != str(person_id):
        raise ValueError("DS-PCD 基线的 person_id 不一致")
    baseline_dates = set(baseline.baseline_dates)
    baseline_end = baseline.baseline_end
    monitoring = [
        record for record in ordered
        if record.date > baseline_end and record.date not in baseline_dates
    ]
    cache: dict[str, dict[str, Any]] = {}
    rows = [
        score_record(baseline, record, config, cache=cache)
        for record in monitoring
    ]
    rows = cusum_replay(
        rows, config, alarm_threshold=alarm_threshold, baseline_end=baseline_end
    )
    scored = [row for row in rows if row.status == "scored"]
    summary = {
        "person_id": str(person_id),
        "baseline_dates": list(baseline.baseline_dates),
        "baseline_end": baseline_end,
        "valid_days": len(ordered),
        "monitoring_days": len(rows),
        "scored_days": len(scored),
        "excluded_missing_features": sum(1 for row in rows if row.status == "excluded_missing_features"),
        "excluded_missing_se": sum(1 for row in rows if row.status == "excluded_missing_se"),
        "cusum_resets": sum(1 for row in scored if row.cusum_reset),
        "max_cusum": max((row.cusum or 0.0) for row in scored) if scored else 0.0,
        "alarms": sum(1 for row in scored if row.alarm) if alarm_threshold is not None else None,
        "alarm_threshold": alarm_threshold,
        "mask_counts": _mask_counts(scored),
    }
    return TimelineScore(
        person_id=str(person_id), baseline=baseline, rows=rows, summary=summary,
    )


def _mask_counts(rows: Sequence[DayScore]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = mask_key(row.mask) or "none"
        counts[key] = counts.get(key, 0) + 1
    return counts


def score_many(
    records_by_person: Mapping[str, Iterable[DayRecord]],
    config: DsPcdConfig,
    *,
    alarm_threshold: float | None = None,
) -> dict[str, TimelineScore]:
    return {
        person: score_timeline(
            records, person, config, alarm_threshold=alarm_threshold,
        )
        for person, records in records_by_person.items()
    }
