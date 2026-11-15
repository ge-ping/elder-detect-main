"""最小 Landmark Cox 拟合器（NumPy/SciPy，Breslow 处理并列事件）。

项目现有训练脚本依赖 lifelines，但本环境没有安装，也不应为此新增依赖。
本模块只实现固定时窗 Landmark Cox 所需的部分：Breslow 偏似然、
Newton–Raphson 求解、岭正则化（很小的固定值，仅保证数值稳定）与
基线生存。产物仍使用 ``longitudinal.survival.CoxModelArtifact`` 的
schema，``calibrated`` 保持 False。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class CoxFit:
    """一次拟合的结果与诊断信息。"""

    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    centers: tuple[float, ...]
    baseline_survival: float | None
    log_likelihood: float | None
    iterations: int
    converged: bool
    status: str
    events: int
    rows: int
    active_features: tuple[str, ...]
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "coefficients": list(self.coefficients),
            "centers": list(self.centers),
            "baseline_survival": self.baseline_survival,
            "log_likelihood": self.log_likelihood,
            "iterations": self.iterations,
            "converged": self.converged,
            "status": self.status,
            "events": self.events,
            "rows": self.rows,
            "active_features": list(self.active_features),
            "notes": dict(self.notes),
        }


def _risk_set_sums(
    order: np.ndarray, times: np.ndarray, weights: np.ndarray, covariates: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按时间升序累计风险集的和、加权协变量和与加权外积和。"""
    # order: 时间升序索引；从后往前累计 “时间 >= t” 的风险集
    n = len(times)
    k = covariates.shape[1]
    w_sum = np.zeros(n, dtype=np.float64)
    wx_sum = np.zeros((n, k), dtype=np.float64)
    wxx_sum = np.zeros((n, k, k), dtype=np.float64)
    running_w = 0.0
    running_wx = np.zeros(k, dtype=np.float64)
    running_wxx = np.zeros((k, k), dtype=np.float64)
    for position in range(n - 1, -1, -1):
        index = order[position]
        w = weights[index]
        x = covariates[index]
        running_w += w
        running_wx += w * x
        running_wxx += w * np.outer(x, x)
        w_sum[position] = running_w
        wx_sum[position] = running_wx
        wxx_sum[position] = running_wxx
    return w_sum, wx_sum, wxx_sum


def _partial_likelihood(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    beta: np.ndarray,
    ridge: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """带岭惩罚的 Breslow 偏似然及其梯度、Hessian。

    返回的 ``loglik``、``gradient``、``hessian`` 对应同一个目标函数
    ``l(beta) - ridge/2 * ||beta||^2``；步长搜索使用这个值。
    """
    linear = covariates @ beta
    shift = float(np.max(linear)) if len(linear) else 0.0
    weights = np.exp(linear - shift)
    order = np.argsort(times, kind="stable")
    sorted_times = times[order]
    w_sum, wx_sum, wxx_sum = _risk_set_sums(
        order, times, weights, covariates
    )
    # 位置到时间 >= t 的风险集；同一时间的事件取该时间的风险集
    unique_events: list[list[float]] = []
    gradient = -(ridge * beta)
    hessian = -(ridge * np.eye(covariates.shape[1]))
    loglik = 0.0
    for position in range(len(times)):
        if not events[order[position]]:
            continue
        time = sorted_times[position]
        index = order[position]
        loglik += linear[index]
        gradient += covariates[index]
        if unique_events and unique_events[-1][0] == time:
            unique_events[-1][1] += 1
        else:
            unique_events.append([float(time), 1])
    for time, count in unique_events:
        # 该时间在升序数组中的最后一个位置，风险集为 time >= t 的行
        position = int(np.searchsorted(sorted_times, time, side="left"))
        if position >= len(sorted_times):
            continue
        total_w = w_sum[position]
        if total_w <= 0 or not math.isfinite(total_w):
            return -math.inf, gradient, hessian
        mean_x = wx_sum[position] / total_w
        loglik -= count * (math.log(total_w) + shift)
        gradient -= count * mean_x
        hessian -= count * (
            wxx_sum[position] / total_w - np.outer(mean_x, mean_x)
        )
    loglik -= 0.5 * ridge * float(beta @ beta)
    return float(loglik), gradient, hessian


def fit_landmark_cox(
    durations: Sequence[float],
    events: Sequence[int],
    covariates: np.ndarray,
    feature_names: Sequence[str],
    *,
    horizon_days: float,
    ridge: float = 1e-4,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> CoxFit:
    """拟合固定时窗的 Landmark Cox 模型（Breslow 并列处理）。"""
    times = np.asarray(durations, dtype=np.float64)
    event_array = np.asarray(events, dtype=np.int64)
    matrix = np.asarray(covariates, dtype=np.float64)
    names = tuple(str(name) for name in feature_names)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("协变量矩阵列数必须与特征名一致")
    if len(times) != len(event_array) or len(times) != matrix.shape[0]:
        raise ValueError("duration、event 与协变量行数必须一致")
    if np.any(~np.isfinite(matrix)) or np.any(~np.isfinite(times)):
        raise ValueError("协变量与随访时间必须是有限数值")
    if np.any(times <= 0):
        raise ValueError("随访时间必须为正")
    events_count = int(event_array.sum())
    rows = len(times)
    if rows == 0 or events_count == 0:
        return CoxFit(
            feature_names=names, coefficients=(0.0,) * len(names),
            centers=(0.0,) * len(names), baseline_survival=None,
            log_likelihood=None, iterations=0, converged=False,
            status="no_events", events=events_count, rows=rows,
            active_features=(), notes={"reason": "训练折没有事件"},
        )
    centers = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    active = [index for index in range(matrix.shape[1]) if scales[index] > 1e-12]
    if not active:
        return CoxFit(
            feature_names=names, coefficients=(0.0,) * len(names),
            centers=tuple(float(value) for value in centers),
            baseline_survival=None, log_likelihood=None, iterations=0,
            converged=False, status="no_active_features", events=events_count,
            rows=rows, active_features=(),
            notes={"reason": "训练折协变量为常数"},
        )
    standardized = (matrix[:, active] - centers[active]) / scales[active]
    beta = np.zeros(len(active), dtype=np.float64)
    loglik, gradient, hessian = _partial_likelihood(
        times, event_array, standardized, beta, ridge
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
        # 步长减半，保证惩罚似然不下降
        current = 1.0
        accepted = False
        for _ in range(30):
            candidate = beta + current * step
            candidate_loglik, candidate_gradient, candidate_hessian = _partial_likelihood(
                times, event_array, standardized, candidate, ridge
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
        if float(np.max(np.abs(gradient))) < tol:
            converged = True
            break
    linear = standardized @ beta
    shift = float(np.max(linear)) if len(linear) else 0.0
    weights = np.exp(linear - shift)
    order = np.argsort(times, kind="stable")
    sorted_times = times[order]
    w_sum, _, _ = _risk_set_sums(order, times, weights, standardized)
    unique_event_times: list[list[float]] = []
    for position in range(len(times)):
        if not event_array[order[position]]:
            continue
        time = float(sorted_times[position])
        if unique_event_times and unique_event_times[-1][0] == time:
            unique_event_times[-1][1] += 1
        else:
            unique_event_times.append([time, 1])
    cumulative_hazard = 0.0
    for time, count in unique_event_times:
        if time > horizon_days:
            continue
        position = int(np.searchsorted(sorted_times, time, side="left"))
        total = w_sum[position]
        if total <= 0:
            continue
        cumulative_hazard += count * math.exp(-shift) / total
    baseline_survival = (
        float(math.exp(-cumulative_hazard)) if cumulative_hazard > 0 else None
    )
    coefficients = [0.0] * len(names)
    for position, index in enumerate(active):
        coefficients[index] = float(beta[position] / scales[index])
    finite = all(math.isfinite(value) for value in coefficients)
    status = "ok"
    if not finite or baseline_survival is None:
        status = "degenerate_baseline" if finite else "non_finite_coefficients"
    elif not converged:
        status = "not_converged"
    elif max(abs(value) for value in beta) > 10.0:
        status = "unstable_coefficients"
    return CoxFit(
        feature_names=names,
        coefficients=tuple(coefficients),
        centers=tuple(float(value) for value in centers),
        baseline_survival=baseline_survival,
        log_likelihood=float(loglik),
        iterations=iterations,
        converged=bool(converged),
        status=status,
        events=events_count,
        rows=rows,
        active_features=tuple(names[index] for index in active),
        notes={
            "ridge": ridge,
            "objective": "penalized_partial_log_likelihood",
            "penalty_at_solution": 0.5 * ridge * float(beta @ beta),
            "max_iterations": max_iter,
            "gradient_max_abs": (
                float(np.max(np.abs(gradient))) if len(gradient) else 0.0
            ),
            "standardized_beta": [float(value) for value in beta],
            "fitted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
