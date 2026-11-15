"""研究层 Landmark Cox：观测权重 + 完整基线生存阶梯 + 1/3/7 天查询。

项目原有的 ``longitudinal.cox_fit.fit_landmark_cox`` 只有单一时点基线生存，
也不支持观测权重；本轮（阶段 13–14）需要：

* 同一训练折的**同一生存曲线**同时查询 1、3、7 天累计风险，避免把 7 天
  概率除以 7 或改元数据窗口；
* 半合成训练行的总权重按源人/家族归一（C7），不能用无权重复制冒充；
* 研究模型保存 ``research_only=true``、``calibrated=false`` 与窗口标识，
  不进入服务临床概率出口。

加权 Breslow 偏似然的定义（与 R ``survival::coxph(weights=)`` 的计数过程
解释一致）：事件行 i 的对数似然贡献乘以 ``w_i``，基线风险增量
``dH0(t) = sum_{i in D(t)} w_i / sum_{j in R(t)} w_j exp(x_j b)``。
所有权重为 1 时退化为原有未加权拟合。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
MODEL_TYPE = "landmark_cox_horizon_research"
DEFAULT_HORIZONS: tuple[int, ...] = (1, 3, 7)


@dataclass(frozen=True)
class HorizonCoxFit:
    """一次加权 Landmark Cox 拟合；基线生存以阶梯形式保存。"""

    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    centers: tuple[float, ...]
    scales: tuple[float, ...]
    baseline_times: tuple[float, ...]
    baseline_survival: tuple[float, ...]
    log_likelihood: float | None
    iterations: int
    converged: bool
    status: str
    events: int
    effective_events: float
    rows: int
    total_weight: float
    horizons_days: tuple[int, ...]
    active_features: tuple[str, ...]
    notes: dict[str, Any] = field(default_factory=dict)

    def survival_at(self, horizon: float) -> float:
        """阶梯基线生存：取不晚于 ``horizon`` 的最后一个事件时点。"""
        if not self.baseline_times:
            return 1.0
        times = np.asarray(self.baseline_times, dtype=float)
        position = int(np.searchsorted(times, float(horizon), side="right")) - 1
        if position < 0:
            return 1.0
        return float(self.baseline_survival[position])

    def linear_predictor(self, features: Mapping[str, float] | Sequence[float]) -> float:
        """线性预测量：``coefficients`` 与 ``centers`` 都在原始单位。"""
        values = _as_vector(features, self.feature_names)
        return float(sum(
            self.coefficients[index] * (values[index] - self.centers[index])
            for index in range(len(self.feature_names))
        ))

    def risk_at(self, features: Mapping[str, float] | Sequence[float], horizon: float) -> float:
        linear = self.linear_predictor(features)
        multiplier = math.exp(max(-50.0, min(50.0, linear)))
        survival = self.survival_at(horizon) ** multiplier
        return float(max(0.0, min(1.0, 1.0 - survival)))

    def risks(self, features: Mapping[str, float] | Sequence[float]) -> dict[str, float]:
        return {f"{h}d": self.risk_at(features, h) for h in self.horizons_days}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_type": MODEL_TYPE,
            "bundle_id": self.notes.get("bundle_id", ""),
            "research_only": True,
            "calibrated": False,
            "horizons_days": list(self.horizons_days),
            "feature_names": list(self.feature_names),
            "coefficients": list(self.coefficients),
            "feature_centers": list(self.centers),
            "feature_scales": list(self.scales),
            "baseline_times_days": list(self.baseline_times),
            "baseline_survival": list(self.baseline_survival),
            "log_likelihood": self.log_likelihood,
            "iterations": self.iterations,
            "converged": bool(self.converged),
            "status": self.status,
            "events": self.events,
            "effective_events": self.effective_events,
            "rows": self.rows,
            "total_weight": self.total_weight,
            "active_features": list(self.active_features),
            "notes": dict(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HorizonCoxFit":
        if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError("HorizonCoxFit schema_version 不匹配")
        if payload.get("model_type") != MODEL_TYPE:
            raise ValueError("model_type 不匹配")
        if payload.get("research_only") is not True or payload.get("calibrated") is not False:
            raise ValueError("研究模型必须 research_only=true 且 calibrated=false")
        return cls(
            feature_names=tuple(str(v) for v in payload["feature_names"]),
            coefficients=tuple(float(v) for v in payload["coefficients"]),
            centers=tuple(float(v) for v in payload["feature_centers"]),
            scales=tuple(float(v) for v in payload["feature_scales"]),
            baseline_times=tuple(float(v) for v in payload["baseline_times_days"]),
            baseline_survival=tuple(float(v) for v in payload["baseline_survival"]),
            log_likelihood=payload.get("log_likelihood"),
            iterations=int(payload.get("iterations", 0)),
            converged=bool(payload.get("converged", False)),
            status=str(payload.get("status", "")),
            events=int(payload.get("events", 0)),
            effective_events=float(payload.get("effective_events", 0.0)),
            rows=int(payload.get("rows", 0)),
            total_weight=float(payload.get("total_weight", 0.0)),
            horizons_days=tuple(int(v) for v in payload.get("horizons_days", DEFAULT_HORIZONS)),
            active_features=tuple(str(v) for v in payload.get("active_features", ())),
            notes=dict(payload.get("notes", {})),
        )

    def fingerprint(self) -> str:
        payload = {
            "coefficients": list(self.coefficients),
            "centers": list(self.centers),
            "scales": list(self.scales),
            "baseline_times": list(self.baseline_times),
            "baseline_survival": list(self.baseline_survival),
            "features": list(self.feature_names),
            "horizons": list(self.horizons_days),
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def _as_vector(features: Mapping[str, float] | Sequence[float], names: Sequence[str]) -> np.ndarray:
    if isinstance(features, Mapping):
        missing = [name for name in names if name not in features]
        if missing:
            raise ValueError(f"缺少模型特征：{', '.join(missing)}")
        values = [float(features[name]) for name in names]
    else:
        values = [float(value) for value in features]
        if len(values) != len(names):
            raise ValueError("特征向量长度与特征名不一致")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("模型输入必须是有限数值")
    return np.asarray(values, dtype=float)


def _risk_sets(order: np.ndarray, times: np.ndarray, weights: np.ndarray,
               covariates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(times)
    k = covariates.shape[1]
    w_sum = np.zeros(n)
    wx_sum = np.zeros((n, k))
    wxx_sum = np.zeros((n, k, k))
    running_w = 0.0
    running_wx = np.zeros(k)
    running_wxx = np.zeros((k, k))
    for position in range(n - 1, -1, -1):
        index = order[position]
        weight = weights[index]
        x = covariates[index]
        running_w += weight
        running_wx += weight * x
        running_wxx += weight * np.outer(x, x)
        w_sum[position] = running_w
        wx_sum[position] = running_wx
        wxx_sum[position] = running_wxx
    return w_sum, wx_sum, wxx_sum


def _objective(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    beta: np.ndarray,
    ridge: float,
    weights: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """加权 Breslow 偏似然（减去岭惩罚）及其梯度、Hessian。"""
    linear = covariates @ beta
    shift = float(np.max(linear)) if len(linear) else 0.0
    hazard_weights = weights * np.exp(linear - shift)
    order = np.argsort(times, kind="stable")
    sorted_times = times[order]
    w_sum, wx_sum, wxx_sum = _risk_sets(order, times, hazard_weights, covariates)
    loglik = 0.0
    gradient = -(ridge * beta)
    hessian = -(ridge * np.eye(covariates.shape[1]))
    block_times: list[list[float]] = []
    for position in range(len(times)):
        index = order[position]
        if not events[index]:
            continue
        weight = weights[index]
        loglik += weight * linear[index]
        gradient += weight * covariates[index]
        time = float(sorted_times[position])
        if block_times and block_times[-1][0] == time:
            block_times[-1][1] += weight
        else:
            block_times.append([time, float(weight)])
    for time, weight_sum in block_times:
        position = int(np.searchsorted(sorted_times, time, side="left"))
        if position >= len(sorted_times):
            continue
        total = w_sum[position]
        if total <= 0 or not math.isfinite(total):
            return -math.inf, gradient, hessian
        mean_x = wx_sum[position] / total
        loglik -= weight_sum * (math.log(total) + shift)
        gradient -= weight_sum * mean_x
        hessian -= weight_sum * (wxx_sum[position] / total - np.outer(mean_x, mean_x))
    loglik -= 0.5 * ridge * float(beta @ beta)
    return float(loglik), gradient, hessian


def fit_horizon_cox(
    durations: Sequence[float],
    events: Sequence[int],
    covariates: np.ndarray,
    feature_names: Sequence[str],
    *,
    horizon_days: float = 7.0,
    ridge: float = 1e-4,
    max_iter: int = 100,
    tol: float = 1e-8,
    weights: Sequence[float] | None = None,
    horizons_days: Sequence[int] = DEFAULT_HORIZONS,
) -> HorizonCoxFit:
    """拟合加权 Landmark Cox 并保存完整基线生存阶梯。"""
    times = np.asarray(durations, dtype=float)
    event_array = np.asarray(events, dtype=int)
    matrix = np.asarray(covariates, dtype=float)
    names = tuple(str(name) for name in feature_names)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("协变量矩阵列数必须与特征名一致")
    if len(times) != len(event_array) or len(times) != matrix.shape[0]:
        raise ValueError("duration、event 与协变量行数必须一致")
    if np.any(~np.isfinite(matrix)) or np.any(~np.isfinite(times)):
        raise ValueError("协变量与随访时间必须是有限数值")
    if np.any(times <= 0):
        raise ValueError("随访时间必须为正")
    weight_array = (
        np.ones(len(times)) if weights is None else np.asarray(weights, dtype=float)
    )
    if len(weight_array) != len(times):
        raise ValueError("权重长度必须与行数一致")
    if np.any(~np.isfinite(weight_array)) or np.any(weight_array < 0):
        raise ValueError("权重必须是有限非负数值")
    total_weight = float(weight_array.sum())
    events_count = int(event_array.sum())
    effective_events = float(weight_array[event_array == 1].sum())
    rows = len(times)
    horizons = tuple(int(value) for value in horizons_days)

    base_notes: dict[str, Any] = {
        "ridge": ridge,
        "objective": "weighted_penalized_partial_log_likelihood_breslow",
        "horizons_days": list(horizons),
        "research_only": True,
        "calibrated": False,
        "fit_horizon_days": float(horizon_days),
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if rows == 0 or events_count == 0 or effective_events <= 0:
        return HorizonCoxFit(
            feature_names=names, coefficients=(0.0,) * len(names),
            centers=(0.0,) * len(names), scales=(1.0,) * len(names),
            baseline_times=(), baseline_survival=(),
            log_likelihood=None, iterations=0, converged=False,
            status="no_events", events=events_count, effective_events=effective_events,
            rows=rows, total_weight=total_weight, horizons_days=horizons,
            active_features=(),
            notes={**base_notes, "reason": "训练折没有加权事件"},
        )

    weight_sum = max(total_weight, 1e-12)
    centers = (weight_array[:, None] * matrix).sum(axis=0) / weight_sum
    variance = (weight_array[:, None] * (matrix - centers) ** 2).sum(axis=0) / weight_sum
    scales = np.sqrt(np.maximum(variance, 0.0))
    active = [index for index in range(matrix.shape[1]) if scales[index] > 1e-12]
    if not active:
        return HorizonCoxFit(
            feature_names=names, coefficients=(0.0,) * len(names),
            centers=tuple(float(value) for value in centers),
            scales=tuple(float(value) for value in scales),
            baseline_times=(), baseline_survival=(),
            log_likelihood=None, iterations=0, converged=False,
            status="no_active_features", events=events_count,
            effective_events=effective_events, rows=rows,
            total_weight=total_weight, horizons_days=horizons, active_features=(),
            notes={**base_notes, "reason": "训练折协变量为常数"},
        )
    standardized = (matrix[:, active] - centers[active]) / scales[active]
    beta = np.zeros(len(active))
    loglik, gradient, hessian = _objective(
        times, event_array, standardized, beta, ridge, weight_array
    )
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        try:
            step = np.linalg.solve(-hessian, gradient)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        current = 1.0
        accepted = False
        for _ in range(30):
            candidate = beta + current * step
            candidate_loglik, candidate_gradient, candidate_hessian = _objective(
                times, event_array, standardized, candidate, ridge, weight_array
            )
            if math.isfinite(candidate_loglik) and candidate_loglik >= loglik - 1e-12:
                beta = candidate
                loglik = candidate_loglik
                gradient = candidate_gradient
                hessian = candidate_hessian
                accepted = True
                break
            current *= 0.5
        if not accepted:
            break
        if float(np.max(np.abs(gradient))) < tol and float(np.max(np.abs(step))) < 1e-6:
            converged = True
            break

    # 完整基线生存阶梯（Breslow）
    linear = standardized @ beta
    shift = float(np.max(linear)) if len(linear) else 0.0
    hazard_weights = weight_array * np.exp(linear - shift)
    order = np.argsort(times, kind="stable")
    sorted_times = times[order]
    w_sum, _, _ = _risk_sets(order, times, hazard_weights, standardized)
    blocks: list[list[float]] = []
    for position in range(len(times)):
        index = order[position]
        if not event_array[index]:
            continue
        weight = float(weight_array[index])
        time = float(sorted_times[position])
        if blocks and blocks[-1][0] == time:
            blocks[-1][1] += weight
        else:
            blocks.append([time, weight])
    cumulative = 0.0
    baseline_times: list[float] = []
    baseline_survival: list[float] = []
    for time, weight_sum in blocks:
        position = int(np.searchsorted(sorted_times, time, side="left"))
        total = w_sum[position]
        if total <= 0:
            continue
        cumulative += weight_sum * math.exp(-shift) / total
        baseline_times.append(time)
        baseline_survival.append(float(math.exp(-cumulative)))

    coefficients = [0.0] * len(names)
    for position, index in enumerate(active):
        coefficients[index] = float(beta[position] / scales[index])
    finite = all(math.isfinite(value) for value in coefficients)
    status = "ok"
    if not finite:
        status = "non_finite_coefficients"
    elif not baseline_times:
        status = "degenerate_baseline"
    elif not converged:
        status = "not_converged"
    elif max(abs(value) for value in beta) > 10.0:
        status = "unstable_coefficients"

    # 短窗可估性：某个窗口内没有事件时基线在该窗口退化，风险估计为 0。
    degenerate_horizons = [
        horizon for horizon in horizons if horizon < min(baseline_times)
    ] if baseline_times else list(horizons)
    notes = {
        **base_notes,
        "penalty_at_solution": 0.5 * ridge * float(beta @ beta),
        "gradient_max_abs": float(np.max(np.abs(gradient))) if len(gradient) else 0.0,
        "standardized_beta": [float(value) for value in beta],
        "degenerate_horizons_days": degenerate_horizons,
        "degenerate_note": (
            "该窗口内训练折没有事件，累计风险估计为 0；零估计不等于真实风险为零"
            if degenerate_horizons else ""
        ),
        "centering": "weighted training-fold mean/sd, standardized then re-expressed in original units",
    }
    return HorizonCoxFit(
        feature_names=names,
        coefficients=tuple(coefficients),
        centers=tuple(float(value) for value in centers),
        scales=tuple(float(value) for value in scales),
        baseline_times=tuple(baseline_times),
        baseline_survival=tuple(baseline_survival),
        log_likelihood=float(loglik),
        iterations=int(iterations),
        converged=bool(converged),
        status=status,
        events=events_count,
        effective_events=effective_events,
        rows=rows,
        total_weight=total_weight,
        horizons_days=horizons,
        active_features=tuple(names[index] for index in active),
        notes=notes,
    )


def predict_horizon_risk(
    fit: HorizonCoxFit, matrix: np.ndarray, horizon_days: float
) -> np.ndarray:
    """按行查询累计风险；概率单调性由同一阶梯保证。"""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    survival = fit.survival_at(horizon_days)
    coefficients = np.asarray(fit.coefficients, dtype=float)
    centers = np.asarray(fit.centers, dtype=float)
    out = np.empty(matrix.shape[0])
    for row in range(matrix.shape[0]):
        linear = float((matrix[row, : len(coefficients)] - centers) @ coefficients)
        multiplier = math.exp(max(-50.0, min(50.0, linear)))
        out[row] = 1.0 - survival ** multiplier
    return out


def _cluster_meat(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    beta: np.ndarray,
    weights: np.ndarray,
    clusters: np.ndarray,
) -> np.ndarray:
    """计数过程得分的影响函数分解，按簇求和得到三明治 meat。

    ``beta`` 视为固定；每行的贡献
    ``d_i w_i (x_i - x̄_R(t_i)) - sum_{t<=T_i} w_i e^{x_i b}(x_i - x̄_R(t)) dH0(t)``
    汇总后等于 Cox 得分。
    """
    linear = covariates @ beta
    shift = float(np.max(linear)) if len(linear) else 0.0
    hazard_weights = weights * np.exp(linear - shift)
    order = np.argsort(times, kind="stable")
    sorted_times = times[order]
    w_sum, wx_sum, _ = _risk_sets(order, times, hazard_weights, covariates)
    blocks: list[list[float]] = []
    for position in range(len(times)):
        index = order[position]
        if not events[index]:
            continue
        weight = float(weights[index])
        time = float(sorted_times[position])
        if blocks and blocks[-1][0] == time:
            blocks[-1][1] += weight
        else:
            blocks.append([time, weight])
    k = covariates.shape[1]
    influence = np.zeros((len(times), k))
    for position in range(len(times)):
        index = order[position]
        if not events[index]:
            continue
        mean_x = wx_sum[position] / w_sum[position]
        influence[index] += weights[index] * (covariates[index] - mean_x)
    for time, weight_sum in blocks:
        position = int(np.searchsorted(sorted_times, time, side="left"))
        total = w_sum[position]
        if total <= 0:
            continue
        mean_x = wx_sum[position] / total
        increment = weight_sum * math.exp(-shift) / total
        at_risk = sorted_times >= time
        centered = covariates[at_risk] - mean_x
        contribution = (
            hazard_weights[at_risk][:, None] * centered * increment
        )
        influence[at_risk] -= contribution
    meat = np.zeros((k, k))
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        total = influence[mask].sum(axis=0)
        meat += np.outer(total, total)
    return meat


def joint_test_coefficients(
    fit: HorizonCoxFit,
    durations: Sequence[float],
    events: Sequence[int],
    covariates: np.ndarray,
    clusters: Sequence[Any],
    coefficients: Sequence[str],
    *,
    ridge: float = 1e-4,
) -> dict[str, Any]:
    """人员聚类稳健的联合 Wald 检验（只检验已拟合模型中的系数块）。

    这不是似然比检验：带惩罚点估计、按合成行放大的样本量都不能套普通 LRT。
    """
    names = list(fit.feature_names)
    positions = []
    for name in coefficients:
        if name not in names:
            return {"status": "not_estimable", "reason": f"模型不含系数 {name}"}
        positions.append(names.index(name))
    times = np.asarray(durations, dtype=float)
    event_array = np.asarray(events, dtype=int)
    matrix = np.asarray(covariates, dtype=float)
    cluster_array = np.asarray(clusters)
    if fit.status != "ok" or not fit.active_features:
        return {"status": "not_estimable", "reason": f"拟合状态 {fit.status}"}
    active = [index for index in range(matrix.shape[1]) if fit.scales[index] > 1e-12]
    if any(fit.scales[index] <= 1e-12 for index in positions):
        return {"status": "not_estimable", "reason": "待检系数为常数特征"}
    active_positions = [active.index(index) for index in positions]
    standardized = (matrix[:, active] - np.asarray(fit.centers)[active]) / np.asarray(fit.scales)[active]
    beta = np.asarray([fit.coefficients[index] * fit.scales[index] for index in active])
    weights = np.ones(len(times))
    _, _, hessian = _objective(times, event_array, standardized, beta, ridge, weights)
    try:
        bread = np.linalg.inv(-hessian)
    except np.linalg.LinAlgError:
        return {"status": "not_estimable", "reason": "Hessian 不可逆"}
    meat = _cluster_meat(times, event_array, standardized, beta, weights, cluster_array)
    covariance = bread @ meat @ bread
    block = covariance[np.ix_(active_positions, active_positions)]
    estimate = beta[active_positions]
    try:
        statistic = float(estimate @ np.linalg.solve(block, estimate))
    except np.linalg.LinAlgError:
        return {"status": "not_estimable", "reason": "协方差块不可逆"}
    if not math.isfinite(statistic) or statistic < 0:
        return {"status": "not_estimable", "reason": "统计量非有限"}
    from scipy import stats

    degrees = len(positions)
    p_value = float(stats.chi2.sf(statistic, degrees))
    return {
        "status": "ok",
        "test": "cluster_robust_wald_on_fitted_coefficients",
        "not_a_likelihood_ratio_test": True,
        "coefficients": list(coefficients),
        "standardized_estimates": [float(value) for value in estimate],
        "robust_covariance": block.tolist(),
        "chi2": statistic,
        "df": degrees,
        "p_value": p_value,
        "clusters": int(len(np.unique(cluster_array))),
        "note": "点估计来自带固定岭惩罚的拟合；协方差为人员聚类三明治估计。",
    }
