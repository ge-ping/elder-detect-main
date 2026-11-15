"""阶段 12：半合成个人序列生成器（S 分支）。

原则（对应 C1–C3、C6–C8）：

* 生成器只由**训练折 donor** 的真实日表与逐段资料估计分布与噪声；测试折
  donor 不参与任何共享参数。每个 fold 的生成器参数都单独保存。
* 事件时间由潜在健康过程与预注册的临床系数生成，不调用 RMS、T、q、C 或
  任何模型预测；同一事件时间再导出 1/3/7 天嵌套标签。
* 只有训练数据支持的量（个人水平分布、日间波动、自相关、会话精度、块间
  变化幅度、年龄分布、既往跌倒比例、事件率锚点）才拟合，其余为写入
  ``protocol.json`` 的研究假设。
* 真实观测日与虚拟序列用独立 ``series_id``；虚拟记录标 ``source=synthetic``。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


FEATURE_NAMES: tuple[str, ...] = (
    "gait_speed_mean",
    "stride_length_mean_cm",
    "stride_duration_bout_cv",
)
LOADINGS: dict[str, float] = {
    "gait_speed_mean": -1.0,
    "stride_length_mean_cm": -1.0,
    "stride_duration_bout_cv": 1.0,
}
MECHANISMS: tuple[str, ...] = ("level_dominant", "sustained_change", "change_no_effect")
NOISE_FAMILIES: tuple[str, ...] = ("gaussian_ar1", "heavy_tail_spike")
TRAJECTORY_MIX: dict[str, tuple[str, ...]] = {
    "level_dominant": ("stable", "slow_linear"),
    "sustained_change": ("slow_linear", "rapid_step"),
    "change_no_effect": ("transient", "reverse"),
}
SCENARIO_TRAJECTORY: dict[str, tuple[str, float | None]] = {
    "stable_null": ("stable", None),
    "gradual_prior_direction": ("slow_linear", None),
    "sudden_change": ("rapid_step", None),
    "reverse_change": ("reverse", None),
    "transient_noise_recovery": ("transient", None),
    "heterogeneous_precision": ("slow_linear", "wide_precision"),
}


@dataclass(frozen=True)
class GeneratorConfig:
    """生成器的固定结构与已声明假设（不做性能调参）。"""

    baseline_days: int = 14
    monitoring_days: int = 30
    tail_days: int = 7
    horizons_days: tuple[int, ...] = (1, 3, 7)
    primary_horizon_days: int = 7
    beta_level_per_sd: float = 0.5
    beta_change_per_sd: float = 0.8
    beta_age_per_sd: float = 0.3
    beta_prior_fall: float = 0.8
    rapid_step_window: tuple[int, int] = (10, 25)
    transient_length_days: int = 7
    bad_day_share: float = 0.05
    bad_day_inflation: float = 2.5
    heavy_tail_df: float = 5.0
    personal_variability_log_sd: float = 0.25
    precision_log_sd: float = 0.25
    min_n_bouts: int = 3

    @property
    def total_days(self) -> int:
        return self.baseline_days + self.monitoring_days + self.tail_days

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_days": self.baseline_days,
            "monitoring_days": self.monitoring_days,
            "tail_days": self.tail_days,
            "horizons_days": list(self.horizons_days),
            "primary_horizon_days": self.primary_horizon_days,
            "declared_assumptions": {
                "beta_level_per_sd": self.beta_level_per_sd,
                "beta_change_per_sd": self.beta_change_per_sd,
                "beta_age_per_sd": self.beta_age_per_sd,
                "beta_prior_fall": self.beta_prior_fall,
                "rapid_step_window": list(self.rapid_step_window),
                "transient_length_days": self.transient_length_days,
                "bad_day_share": self.bad_day_share,
                "bad_day_inflation": self.bad_day_inflation,
                "heavy_tail_df": self.heavy_tail_df,
                "personal_variability_log_sd": self.personal_variability_log_sd,
                "precision_log_sd": self.precision_log_sd,
            },
            "mechanisms": list(MECHANISMS),
            "noise_families": list(NOISE_FAMILIES),
            "trajectory_mix": {k: list(v) for k, v in TRAJECTORY_MIX.items()},
        }


@dataclass(frozen=True)
class GeneratorParams:
    """一个训练折的生成器参数（可由 ``to_dict`` 完整重建）。"""

    feature_names: tuple[str, ...]
    population_mean: tuple[float, ...]
    between_person_sd: tuple[float, ...]
    within_person_sd: tuple[float, ...]
    residual_correlation: tuple[tuple[float, ...], ...]
    ar1_phi: float
    se_intercept: tuple[float, ...]
    se_log_n_slope: tuple[float, ...]
    se_residual_sd: tuple[float, ...]
    se_floor: tuple[float, ...]
    nbouts_quantiles: tuple[float, ...]
    age_mean: float
    age_sd: float
    prior_fall_rate: float
    event_rate_anchor_7d: float
    event_rate_anchor_30d: float
    change_magnitude: tuple[float, ...]
    lambda0: dict[str, float]
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "population_mean": list(self.population_mean),
            "between_person_sd": list(self.between_person_sd),
            "within_person_sd": list(self.within_person_sd),
            "residual_correlation": [list(row) for row in self.residual_correlation],
            "ar1_phi": self.ar1_phi,
            "se_intercept": list(self.se_intercept),
            "se_log_n_slope": list(self.se_log_n_slope),
            "se_residual_sd": list(self.se_residual_sd),
            "se_floor": list(self.se_floor),
            "nbouts_quantiles": list(self.nbouts_quantiles),
            "age_mean": self.age_mean,
            "age_sd": self.age_sd,
            "prior_fall_rate": self.prior_fall_rate,
            "event_rate_anchor_7d": self.event_rate_anchor_7d,
            "event_rate_anchor_30d": self.event_rate_anchor_30d,
            "change_magnitude": list(self.change_magnitude),
            "lambda0": dict(self.lambda0),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeneratorParams":
        return cls(
            feature_names=tuple(payload["feature_names"]),
            population_mean=tuple(float(v) for v in payload["population_mean"]),
            between_person_sd=tuple(float(v) for v in payload["between_person_sd"]),
            within_person_sd=tuple(float(v) for v in payload["within_person_sd"]),
            residual_correlation=tuple(
                tuple(float(v) for v in row) for row in payload["residual_correlation"]
            ),
            ar1_phi=float(payload["ar1_phi"]),
            se_intercept=tuple(float(v) for v in payload["se_intercept"]),
            se_log_n_slope=tuple(float(v) for v in payload["se_log_n_slope"]),
            se_residual_sd=tuple(float(v) for v in payload["se_residual_sd"]),
            se_floor=tuple(float(v) for v in payload["se_floor"]),
            nbouts_quantiles=tuple(float(v) for v in payload["nbouts_quantiles"]),
            age_mean=float(payload["age_mean"]),
            age_sd=float(payload["age_sd"]),
            prior_fall_rate=float(payload["prior_fall_rate"]),
            event_rate_anchor_7d=float(payload["event_rate_anchor_7d"]),
            event_rate_anchor_30d=float(payload["event_rate_anchor_30d"]),
            change_magnitude=tuple(float(v) for v in payload["change_magnitude"]),
            lambda0=dict(payload["lambda0"]),
            provenance=dict(payload.get("provenance", {})),
        )


def _shrink_correlation(matrix: np.ndarray, shrink: float = 0.3) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    k = matrix.shape[0]
    shrunk = (1.0 - shrink) * matrix + shrink * np.eye(k)
    # 保证正定
    values, vectors = np.linalg.eigh(shrunk)
    values = np.clip(values, 1e-6, None)
    shrunk = vectors @ np.diag(values) @ vectors.T
    scale = np.sqrt(np.diag(shrunk))
    return shrunk / np.outer(scale, scale)


def fit_generator_params(
    *,
    person_daily: pd.DataFrame,
    landmarks: pd.DataFrame,
    donors: Sequence[str],
    feature_columns: Mapping[str, str],
    config: GeneratorConfig = GeneratorConfig(),
    provenance: Mapping[str, Any] | None = None,
) -> GeneratorParams:
    """由训练折 donor 的真实逐日资料估计分布与噪声参数。"""
    names = list(feature_columns)
    frame = person_daily.loc[person_daily["participantid"].isin(set(donors))].copy()
    frame = frame.sort_values(["participantid", "day"], kind="stable")
    values = frame[names].to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1)
    frame = frame.loc[valid].copy()
    values = frame[names].to_numpy(dtype=float)
    if len(frame) < 30:
        raise ValueError(f"训练折可用日数过少：{len(frame)}")

    person_stats = frame.assign(**{f"_{name}": values[:, index] for index, name in enumerate(names)}).groupby(
        "participantid"
    )[[f"_{name}" for name in names]].agg(["mean", "std", "count"])
    person_mean = person_stats.xs("mean", level=1, axis=1)
    person_sd = person_stats.xs("std", level=1, axis=1)
    person_count = person_stats.xs("count", level=1, axis=1)
    usable = person_count.min(axis=1).ge(5)
    person_mean = person_mean.loc[usable]
    person_sd = person_sd.loc[usable]
    if len(person_mean) < 5:
        raise ValueError("训练折可用 donor 过少")

    population_mean = person_mean.mean(axis=0).to_numpy(dtype=float)
    between_sd = person_mean.std(axis=0, ddof=1).to_numpy(dtype=float)
    observed_within_sd = person_sd.median(axis=0).to_numpy(dtype=float)
    # 观测模型：观测到的个人内方差 = 真实日间方差 tau^2 + 会话测量方差 s^2。
    # 若直接把观测 SD 当作 tau，就会把测量误差重复计一次（观测 = tau + noise + se）。
    measurement_variance = []
    for name in names:
        se_column = f"se_{name}"
        if se_column in frame.columns:
            se_values = pd.to_numeric(frame[se_column], errors="coerce").to_numpy(dtype=float)
            usable = se_values[np.isfinite(se_values) & (se_values > 0)]
            measurement_variance.append(float(np.mean(usable ** 2)) if len(usable) else 0.0)
        else:
            measurement_variance.append(0.0)
    measurement_variance = np.asarray(measurement_variance)
    true_within_var = np.maximum(
        observed_within_sd ** 2 - measurement_variance,
        0.05 * observed_within_sd ** 2,
    )
    within_sd = np.sqrt(true_within_var)
    between_sd = np.maximum(between_sd, 1e-6)
    within_sd = np.maximum(within_sd, 1e-6)

    # 个人内标准化残差的相关与滞后 1 自相关
    residuals = []
    lag_pairs = []
    for person, group in frame.groupby("participantid", sort=True):
        if person not in person_mean.index:
            continue
        block = group[names].to_numpy(dtype=float)
        mu = person_mean.loc[person].to_numpy(dtype=float)
        sd = np.maximum(person_sd.loc[person].to_numpy(dtype=float), 1e-6)
        standardized = (block - mu) / sd
        residuals.append(standardized)
        days = pd.to_datetime(group["day"]).to_numpy(dtype="datetime64[D]")
        gaps = np.diff(days).astype(int)
        for position in range(len(standardized) - 1):
            if gaps[position] == 1:
                lag_pairs.append((standardized[position], standardized[position + 1]))
    stacked = np.vstack(residuals)
    correlation = np.corrcoef(stacked, rowvar=False)
    correlation = _shrink_correlation(np.atleast_2d(correlation))
    phi = 0.0
    if lag_pairs:
        pairs = np.asarray(lag_pairs, dtype=float)
        left = pairs[:, 0, :]
        right = pairs[:, 1, :]
        phi_values = []
        for index in range(len(names)):
            if np.std(left[:, index]) > 1e-9 and np.std(right[:, index]) > 1e-9:
                phi_values.append(float(np.corrcoef(left[:, index], right[:, index])[0, 1]))
        phi = float(np.clip(np.median(phi_values), 0.0, 0.9)) if phi_values else 0.0

    # 会话精度模型：log(se) = a + b log(n_bouts) + noise
    se_columns = [f"se_{name}" for name in names]
    se_intercept, se_slope, se_residual = [], [], []
    for name, se_column in zip(names, se_columns):
        if se_column not in frame.columns:
            raise ValueError(f"缺少逐日标准误列：{se_column}")
        se = pd.to_numeric(frame[se_column], errors="coerce").to_numpy(dtype=float)
        bouts = pd.to_numeric(frame["n_bouts"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(se) & (se > 0) & np.isfinite(bouts) & (bouts >= 1)
        if mask.sum() < 30:
            raise ValueError(f"{name} 的可用标准误过少：{int(mask.sum())}")
        x = np.log(bouts[mask])
        y = np.log(se[mask])
        design = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ beta
        se_intercept.append(float(beta[0]))
        se_slope.append(float(beta[1]))
        se_residual.append(float(np.clip(residual.std(ddof=1), 0.05, 1.0)))
    nbouts = pd.to_numeric(frame["n_bouts"], errors="coerce").dropna().to_numpy(dtype=float)
    quantiles = (
        tuple(float(np.quantile(nbouts, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95))
        if len(nbouts) else (3.0, 10.0, 50.0, 200.0, 400.0)
    )
    floors = []
    for name in names:
        sensible = pd.to_numeric(frame[f"se_{name}"], errors="coerce").dropna()
        floors.append(float(sensible.quantile(0.01)) if len(sensible) else 1e-4)

    # 相邻采集块的变化幅度（个人 z 单位），用于轨迹幅度
    magnitudes: list[float] = []
    landmark_frame = landmarks.loc[landmarks["participantid"].isin(set(donors))]
    for person in sorted(set(landmark_frame["participantid"])):
        if person not in person_mean.index:
            continue
        person_landmarks = landmark_frame.loc[landmark_frame["participantid"] == person]
        for row in person_landmarks.itertuples(index=False):
            block = frame.loc[
                (frame["participantid"] == person)
                & (frame["day"] >= getattr(row, "monitoring_start_day", "1900-01-01"))
                & (frame["day"] <= getattr(row, "landmark_day", "2100-01-01"))
            ]
            if len(block) < 2:
                continue
            mu = person_mean.loc[person].to_numpy(dtype=float)
            sd = np.maximum(person_sd.loc[person].to_numpy(dtype=float), 1e-6)
            change = (block[names].mean().to_numpy(dtype=float) - mu) / sd
            magnitudes.append(float(np.median(np.abs(change))))
    magnitudes = [value for value in magnitudes if math.isfinite(value)]
    change_magnitude = tuple(
        float(value) for value in np.quantile(magnitudes, np.linspace(0.1, 0.9, 9))
    ) if magnitudes else (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    ages = landmarks.loc[
        landmarks["participantid"].isin(set(donors)), ["participantid", "age"]
    ].drop_duplicates("participantid")["age"].astype(float)
    prior_fall = landmarks.loc[
        landmarks["participantid"].isin(set(donors)), "prior_fall_any"
    ].astype(float)
    for horizon in (7, 30):
        label = f"y_{horizon}d_num" if f"y_{horizon}d_num" in landmarks.columns else "event_30d"
        if label in landmarks.columns:
            share = float(
                pd.to_numeric(
                    landmarks.loc[landmarks["participantid"].isin(set(donors)), label],
                    errors="coerce",
                ).dropna().mean()
            )
        else:
            share = float("nan")
        if horizon == 7:
            anchor_7d = share
        else:
            anchor_30d = share
    if "event_30d" in landmarks.columns:
        anchor_30d = float(
            pd.to_numeric(
                landmarks.loc[landmarks["participantid"].isin(set(donors)), "event_30d"],
                errors="coerce",
            ).dropna().mean()
        )
    if not math.isfinite(anchor_7d):
        anchor_7d = 0.02
    if not math.isfinite(anchor_30d):
        anchor_30d = 0.09

    return GeneratorParams(
        feature_names=tuple(names),
        population_mean=tuple(float(v) for v in population_mean),
        between_person_sd=tuple(float(v) for v in between_sd),
        within_person_sd=tuple(float(v) for v in within_sd),
        residual_correlation=tuple(tuple(float(v) for v in row) for row in correlation),
        ar1_phi=phi,
        se_intercept=tuple(se_intercept),
        se_log_n_slope=tuple(se_slope),
        se_residual_sd=tuple(se_residual),
        se_floor=tuple(floors),
        nbouts_quantiles=tuple(float(v) for v in quantiles),
        age_mean=float(ages.mean()),
        age_sd=float(ages.std(ddof=1)) if len(ages) > 1 else 5.0,
        prior_fall_rate=float(prior_fall.mean()),
        event_rate_anchor_7d=float(anchor_7d),
        event_rate_anchor_30d=float(anchor_30d),
        change_magnitude=tuple(float(v) for v in change_magnitude),
        lambda0={},
        provenance={
            "donors": int(len(person_mean)),
            "daily_rows": int(len(frame)),
            "observed_within_sd": [float(v) for v in observed_within_sd],
            "measurement_variance": [float(v) for v in measurement_variance],
            "true_within_sd": [float(v) for v in within_sd],
            "variance_decomposition": "observed within-person variance minus mean session measurement variance",
            "fitted_within_fold": True,
            **(dict(provenance) if provenance else {}),
        },
    )


def _trajectory(
    kind: str, *, rng: np.random.Generator, delta: float, config: GeneratorConfig
) -> np.ndarray:
    """返回从基线首日到尾部末日的潜在健康过程 h(t)（个人 SD 单位）。"""
    total = config.total_days
    baseline_end_index = config.baseline_days - 1
    h = np.zeros(total, dtype=float)
    if kind == "stable":
        return h
    if kind == "slow_linear":
        slope = delta / config.monitoring_days
        for index in range(config.baseline_days, total):
            h[index] = slope * (index - baseline_end_index)
        return h
    if kind == "rapid_step":
        low, high = config.rapid_step_window
        onset = int(rng.integers(low, high + 1))
        for index in range(config.baseline_days, total):
            if index - baseline_end_index >= onset:
                h[index] = delta
        return h
    if kind == "reverse":
        slope = -delta / config.monitoring_days
        for index in range(config.baseline_days, total):
            h[index] = slope * (index - baseline_end_index)
        return h
    if kind == "transient":
        low, high = config.rapid_step_window
        onset = int(rng.integers(low, high + 1))
        for index in range(config.baseline_days, total):
            offset = index - baseline_end_index
            if onset <= offset < onset + config.transient_length_days:
                h[index] = delta
        return h
    raise ValueError(f"未知轨迹类型：{kind}")


def _correlated_noise(
    count: int,
    correlation: np.ndarray,
    *,
    family: str,
    phi: float,
    rng: np.random.Generator,
    config: GeneratorConfig,
) -> np.ndarray:
    """平稳 AR(1) 噪声；重尾族用 t 新息并叠加“坏日”放大。"""
    k = correlation.shape[0]
    cholesky = np.linalg.cholesky(correlation)
    innovations = np.empty((count, k), dtype=float)
    if family == "gaussian_ar1":
        z = rng.standard_normal((count, k))
    elif family == "heavy_tail_spike":
        z = rng.standard_t(config.heavy_tail_df, size=(count, k))
        z = z / math.sqrt(config.heavy_tail_df / (config.heavy_tail_df - 2.0))
    else:
        raise ValueError(f"未知噪声族：{family}")
    innovations = z @ cholesky.T
    out = np.empty_like(innovations)
    previous = np.zeros(k)
    scale = math.sqrt(max(1.0 - phi ** 2, 1e-6))
    for index in range(count):
        previous = phi * previous + scale * innovations[index]
        out[index] = previous
    if family == "heavy_tail_spike":
        mask = rng.random(count) < config.bad_day_share
        out[mask] *= config.bad_day_inflation
    return out


def _person_latent(
    params: GeneratorParams,
    *,
    rng: np.random.Generator,
    trajectory: str,
    config: GeneratorConfig,
) -> dict[str, Any]:
    """抽个人协变量与潜在健康过程（不生成观测值，供事件过程与标定复用）。"""
    age = float(np.clip(rng.normal(params.age_mean, params.age_sd), 65.0, 99.0))
    prior_fall = int(rng.random() < params.prior_fall_rate)
    quantiles = np.asarray(params.change_magnitude, dtype=float)
    probs = np.linspace(0.1, 0.9, len(quantiles))
    magnitude = float(np.interp(rng.random(), probs, quantiles)) if len(quantiles) > 1 else float(quantiles[0])
    magnitude = min(max(magnitude, 0.15), 2.5)
    h = _trajectory(trajectory, rng=rng, delta=magnitude, config=config)
    return {
        "age": age,
        "age_z": (age - params.age_mean) / max(params.age_sd, 1e-6),
        "prior_fall_any": prior_fall,
        "delta": magnitude,
        "h": h,
    }


def simulate_person(
    params: GeneratorParams,
    *,
    series_id: str,
    mechanism: str,
    noise_family: str,
    trajectory: str,
    rng: np.random.Generator,
    config: GeneratorConfig = GeneratorConfig(),
    fold: int = 0,
    cohort: str = "pd",
    forcing_lambda0: float | None = None,
    precision_scale: float = 1.0,
    bias_drift_sd: float = 0.0,
) -> dict[str, Any]:
    """生成一名虚拟受试者的完整 14+30+7 天序列与事件时间。

    ``bias_drift_sd>0`` 时叠加与健康无关、跨日缓慢漂移的测量偏差（压力测试）：
    它只进入观测值，不进入潜在健康过程与事件风险。
    """
    k = len(params.feature_names)
    correlation = np.asarray(params.residual_correlation, dtype=float)
    between = np.asarray(params.between_person_sd, dtype=float)
    within = np.asarray(params.within_person_sd, dtype=float)

    latent = _person_latent(params, rng=rng, trajectory=trajectory, config=config)
    age = latent["age"]
    age_z = latent["age_z"]
    prior_fall = latent["prior_fall_any"]
    delta = latent["delta"]
    h = latent["h"]

    personal_level = np.asarray(params.population_mean) + rng.standard_normal(k) @ (
        np.linalg.cholesky(correlation).T * between
    )
    personal_scale = within * np.exp(
        rng.normal(0.0, config.personal_variability_log_sd, size=k)
    )

    noise = _correlated_noise(
        config.total_days, correlation, family=noise_family, phi=params.ar1_phi,
        rng=rng, config=config,
    )
    nbouts = _draw_nbouts(params, rng, count=config.total_days)
    se = np.empty((config.total_days, k), dtype=float)
    for index in range(k):
        log_se = (
            params.se_intercept[index]
            + params.se_log_n_slope[index] * np.log(np.maximum(nbouts, 1.0))
            + rng.normal(0.0, params.se_residual_sd[index], size=config.total_days)
        )
        se[:, index] = np.maximum(np.exp(log_se), params.se_floor[index]) * precision_scale

    loadings = np.asarray([LOADINGS[name] for name in params.feature_names])
    signal = h[:, None] * loadings[None, :]
    measurement = rng.standard_normal((config.total_days, k)) * se
    observed = personal_level[None, :] + personal_scale[None, :] * (signal + noise) + measurement
    if bias_drift_sd > 0:
        bias = np.zeros((config.total_days, k), dtype=float)
        state = np.zeros(k, dtype=float)
        step = np.sqrt(max(1.0 - 0.95 ** 2, 1e-6))
        for index in range(config.total_days):
            state = 0.95 * state + step * rng.standard_normal(k)
            bias[index] = state
        observed = observed + personal_scale[None, :] * bias_drift_sd * bias

    # 事件过程：从监测首日算起，到尾部末日
    days = np.arange(config.total_days)
    monitoring_start = config.baseline_days
    h_start = float(h[monitoring_start - 1])
    change = np.maximum(0.0, h - h_start)
    change_coefficient = (
        config.beta_change_per_sd if mechanism == "sustained_change" else 0.0
    )
    level_coefficient = config.beta_level_per_sd
    lambda0 = forcing_lambda0 if forcing_lambda0 is not None else params.lambda0.get(
        f"{mechanism}|{noise_family}", params.lambda0.get("default", 0.0)
    )
    linear = (
        math.log(max(lambda0, 1e-12))
        + config.beta_age_per_sd * age_z
        + config.beta_prior_fall * prior_fall
        + level_coefficient * h
        + change_coefficient * change
    )
    hazard = np.exp(np.clip(linear, -30.0, 10.0))
    event_day = None
    probability = 1.0 - np.exp(-hazard[monitoring_start:])
    draws = rng.random(len(probability))
    hits = np.nonzero(draws < probability)[0]
    if len(hits):
        event_day = int(monitoring_start + hits[0])

    snapshot_index = monitoring_start + config.monitoring_days - 1
    labels: dict[str, float] = {}
    for horizon in config.horizons_days:
        if event_day is None:
            labels[f"y_{horizon}d"] = 0.0
        else:
            gap = (event_day - snapshot_index)
            labels[f"y_{horizon}d"] = float(0 < gap <= horizon)
    event_gap = None if event_day is None else int(event_day - snapshot_index)
    first_event_offset = None if event_day is None else int(event_day - monitoring_start)
    return {
        "series_id": f"{series_id}",
        "fold": int(fold),
        "cohort": cohort,
        "mechanism": mechanism,
        "noise_family": noise_family,
        "trajectory": trajectory,
        "source": "synthetic",
        "age": age,
        "age_z": age_z,
        "prior_fall_any": prior_fall,
        "delta": delta,
        "h_snapshot": float(h[snapshot_index]),
        "h_max": float(np.max(h)),
        "event_day": event_day,
        "event_gap_from_snapshot": event_gap,
        "first_event_offset_from_monitoring": first_event_offset,
        **labels,
        "observed": observed,
        "se": se,
        "nbouts": nbouts,
        "day_index": days,
        "h_trajectory": h,
    }


def _draw_nbouts(params: GeneratorParams, rng: np.random.Generator, *, count: int) -> np.ndarray:
    quantiles = np.asarray(params.nbouts_quantiles, dtype=float)
    probs = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    knots = np.unique(np.clip(quantiles, 1.0, None))
    if len(knots) < 5:
        knots = np.asarray([1.0, 5.0, 20.0, 100.0, 400.0])
        probs = np.array([0.05, 0.25, 0.5, 0.75, 0.95])[: len(knots)]
    uniform = rng.random(count)
    values = np.interp(uniform, probs[: len(knots)], knots)
    values = np.round(values).clip(1, None)
    return values


def calibrate_lambda0(
    params: GeneratorParams,
    *,
    mechanism: str,
    noise_family: str,
    target_rate: float,
    config: GeneratorConfig = GeneratorConfig(),
    seed: int = 20261115,
    sample: int = 2000,
    horizon: int | None = None,
) -> tuple[float, dict[str, float]]:
    """二分求解基线风险，使快照后窗口的事件比例等于真实训练折锚点。

    只抽个人协变量与潜在健康过程，不生成观测值；因此标定不引入精度或噪声
    参数的自由度。
    """
    horizon = horizon or config.primary_horizon_days
    rng = np.random.default_rng(seed)
    mix = TRAJECTORY_MIX[mechanism]
    latents = [
        _person_latent(
            params, rng=rng, trajectory=mix[index % len(mix)], config=config
        )
        for index in range(sample)
    ]
    monitoring_start = config.baseline_days
    snapshot_index = monitoring_start + config.monitoring_days - 1
    change_coefficient = (
        config.beta_change_per_sd if mechanism == "sustained_change" else 0.0
    )
    # 固定均匀数使事件率随 lambda0 单调，二分可用。
    uniforms = np.random.default_rng(seed + 1).random(sample)

    def window_rate(lambda0: float) -> float:
        probabilities = []
        for latent in latents:
            h = latent["h"]
            h_start = float(h[monitoring_start - 1])
            change = np.maximum(0.0, h - h_start)
            linear = (
                math.log(max(lambda0, 1e-12))
                + config.beta_age_per_sd * latent["age_z"]
                + config.beta_prior_fall * int(latent["prior_fall_any"])
                + config.beta_level_per_sd * h
                + change_coefficient * change
            )
            hazard = np.exp(np.clip(linear, -30.0, 10.0))[monitoring_start:]
            monitoring = hazard[: config.monitoring_days]
            window = hazard[config.monitoring_days: config.monitoring_days + horizon]
            # 标签要求事件落在快照之后的窗口内，且此前没有事件；
            # 只用尾部风险会高估可观测的 7 天事件比例。
            probabilities.append(
                math.exp(-float(monitoring.sum())) * (1.0 - math.exp(-float(window.sum())))
            )
        probabilities = np.asarray(probabilities)
        return float((uniforms < probabilities).mean())

    # P(尾部事件且此前无事件) 是 lambda0 的单峰函数，峰值在
    # lambda* ≈ log((H+D)/D)/H；只在峰左侧（可辨识的上升支）二分。
    horizon_total = config.monitoring_days + horizon
    peak = math.log(horizon_total / horizon) / config.monitoring_days
    low, high = 1e-9, peak
    rate_high = window_rate(high)
    if rate_high < target_rate:
        return high, {
            "rate_at_high": rate_high, "target": target_rate,
            "reason": "target_above_attainable_on_identifiable_branch",
        }
    for _ in range(60):
        middle = math.sqrt(low * high)
        rate = window_rate(middle)
        if rate > target_rate:
            high = middle
        else:
            low = middle
        if abs(rate - target_rate) < 5e-4:
            break
    solution = float(math.sqrt(low * high))
    return solution, {
        "rate_at_solution": window_rate(solution),
        "target": target_rate,
        "sample": sample,
    }


def simulate_batch(
    params: GeneratorParams,
    *,
    count: int,
    fold: int,
    mechanism: str,
    noise_family: str,
    seed: int,
    config: GeneratorConfig = GeneratorConfig(),
    prefix: str = "sim",
    keep_series: bool = False,
    trajectory_override: str | None = None,
    precision_scale: float = 1.0,
    lambda0: float | None = None,
    bias_drift_sd: float = 0.0,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """批量生成；返回快照表与（可选的）逐日序列。"""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    series: list[dict[str, Any]] = []
    mix = TRAJECTORY_MIX[mechanism]
    for index in range(count):
        trajectory = trajectory_override or mix[index % len(mix)]
        person = simulate_person(
            params, series_id=f"{prefix}-{fold}-{mechanism}-{noise_family}-{index:05d}",
            mechanism=mechanism, noise_family=noise_family, trajectory=trajectory,
            rng=rng, config=config, fold=fold, forcing_lambda0=lambda0,
            precision_scale=precision_scale, bias_drift_sd=bias_drift_sd,
        )
        snapshot = {
            key: value for key, value in person.items()
            if key not in ("observed", "se", "nbouts", "day_index", "h_trajectory")
        }
        rows.append(snapshot)
        if keep_series:
            series.append(person)
    return pd.DataFrame(rows), series


def person_records(person: Mapping[str, Any], feature_names: Sequence[str]) -> list:
    """把生成结果转成 DS-PCD 的 ``DayRecord`` 列表（含基线日）。"""
    from longitudinal.ds_pcd import DayRecord

    records = []
    start = date(2020, 1, 1)
    observed = np.asarray(person["observed"], dtype=float)
    se = np.asarray(person["se"], dtype=float)
    for index in range(observed.shape[0]):
        day = (start + timedelta(days=int(index))).isoformat()
        features = {
            name: float(observed[index, position])
            for position, name in enumerate(feature_names)
        }
        feature_se = {
            name: float(se[index, position])
            for position, name in enumerate(feature_names)
        }
        records.append(DayRecord(date=day, features=features, feature_se=feature_se))
    return records


def feature_distribution_summary(
    real_daily: pd.DataFrame,
    synthetic: pd.DataFrame,
    feature_names: Sequence[str],
) -> dict:
    """真实与虚拟输入的分布对照（均值、SD、分位与相关）。"""
    summary: dict[str, Any] = {}
    for name in feature_names:
        real = pd.to_numeric(real_daily.get(name), errors="coerce").dropna()
        sim = pd.to_numeric(synthetic.get(name), errors="coerce").dropna()
        summary[name] = {
            "real": {
                "n": int(len(real)),
                "mean": float(real.mean()),
                "sd": float(real.std(ddof=1)),
                "p10": float(real.quantile(0.1)),
                "p50": float(real.median()),
                "p90": float(real.quantile(0.9)),
            },
            "synthetic": {
                "n": int(len(sim)),
                "mean": float(sim.mean()),
                "sd": float(sim.std(ddof=1)),
                "p10": float(sim.quantile(0.1)),
                "p50": float(sim.median()),
                "p90": float(sim.quantile(0.9)),
            },
        }
    return summary
