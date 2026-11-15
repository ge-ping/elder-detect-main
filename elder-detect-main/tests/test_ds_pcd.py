"""DS-PCD 统计层单元测试（合成数据，不依赖真实数据）。"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.ds_pcd import (  # noqa: E402
    BaselineUnavailable,
    DayRecord,
    DsPcdBaseline,
    DsPcdConfig,
    config_fingerprint,
    directional_statistic,
    fit_baseline,
    normal_score,
    score_record,
    score_timeline,
    stable_seed,
    tail_probability,
)


FEATURES = ("gait_speed_proxy", "cadence_spm", "step_time_mean_s")
DIRECTIONS = {"gait_speed_proxy": -1.0, "cadence_spm": -1.0, "step_time_mean_s": 1.0}


def make_config(**overrides) -> DsPcdConfig:
    payload = {
        "features": FEATURES,
        "directions": DIRECTIONS,
        "precision_mode": "none",
        "null_samples": 2000,
        "quantile_points": 200,
        "tau_floor_relative": 1e-3,
        "outlier_z": 100.0,
    }
    payload.update(overrides)
    return DsPcdConfig(**payload)


def make_records(
    values: np.ndarray,
    start: date = date(2024, 1, 1),
    *,
    ses: np.ndarray | None = None,
) -> list[DayRecord]:
    records = []
    for index, row in enumerate(values):
        feature_se = {}
        if ses is not None:
            feature_se = {name: float(ses[index, k]) for k, name in enumerate(FEATURES)}
        records.append(DayRecord(
            date=(start + timedelta(days=index)).isoformat(),
            features={name: float(row[k]) for k, name in enumerate(FEATURES)},
            feature_se=feature_se,
        ))
    return records


def test_dl_without_se_equals_mean_and_variance():
    rng = np.random.default_rng(7)
    values = rng.normal(
        loc=np.array([1.0, 90.0, 0.62]),
        scale=np.array([0.05, 2.0, 0.01]),
        size=(14, 3),
    )
    baseline = fit_baseline(make_records(values), "p", make_config())
    for k, name in enumerate(FEATURES):
        assert baseline.mu[name] == pytest.approx(values[:, k].mean(), abs=1e-9)
        assert baseline.tau2[name] == pytest.approx(values[:, k].var(ddof=1), rel=1e-6)
        assert baseline.var_mu[name] == pytest.approx(
            values[:, k].var(ddof=1) / 14, rel=1e-6
        )


def test_dl_with_se_pulls_estimate_towards_precise_days():
    rng = np.random.default_rng(2)
    values = np.tile([1.0, 90.0, 0.62], (14, 1)) + rng.normal(0, 0.002, (14, 3))
    ses = np.full((14, 3), 1.0)
    ses[:, 0] = 0.001
    ses[0, 0] = 5.0
    values[0, 0] = 100.0
    baseline = fit_baseline(
        make_records(values, ses=ses), "p",
        make_config(precision_mode="session_se"),
    )
    # 精确日权重更大，异常但极不精确的基线日影响被压低
    assert baseline.mu["gait_speed_proxy"] == pytest.approx(1.0, abs=0.02)
    assert baseline.var_mu["gait_speed_proxy"] < 0.001


def test_standardization_is_centred_and_scaled():
    rng = np.random.default_rng(11)
    config = make_config()
    z_values = []
    for person in range(600):
        base = rng.normal(
            loc=np.array([1.0, 90.0, 0.62]),
            scale=np.array([0.05, 2.0, 0.01]),
            size=(14, 3),
        )
        records = make_records(base, start=date(2024, 1, 1) + timedelta(days=person * 40))
        baseline = fit_baseline(records, f"p{person}", config)
        future = rng.normal(
            loc=np.array([1.0, 90.0, 0.62]),
            scale=np.array([0.05, 2.0, 0.01]),
        )
        record = DayRecord(
            date=records[-1].date,
            features={name: float(future[k]) for k, name in enumerate(FEATURES)},
        )
        # 使用基线后的第二天，确保落在监测区间
        record = DayRecord(
            date=(date.fromisoformat(records[-1].date) + timedelta(days=1)).isoformat(),
            features=record.features,
        )
        row = score_record(baseline, record, config)
        z_values.append([row.z[name] for name in FEATURES])
    z_values = np.asarray(z_values)
    assert np.allclose(z_values.mean(axis=0), 0.0, atol=0.12)
    # 14 日基线的 tau^2 是插件估计：E[tau^2/tau_hat^2] = 13/11，
    # 因此 z 的 SD 略大于 1（约 1.09）；这是已知的小样本近似，不是错误。
    spread = z_values.std(axis=0, ddof=1)
    assert np.all(spread > 1.0)
    assert np.all(spread < 1.16)


def test_null_statistic_matches_independent_large_sample():
    config = make_config(null_samples=20000, quantile_points=500)
    values = np.array([[1.0 + 0.01 * j, 90.0 + 0.1 * j, 0.62 + 0.001 * j] for j in range(14)])
    baseline = fit_baseline(make_records(values), "p", config)
    # 独立模拟：相关系数用基线里保存的矩阵，按方向统计量计算
    rng = np.random.default_rng(3)
    correlation = np.asarray(baseline.correlation)
    draws = rng.multivariate_normal(np.zeros(3), correlation, size=200000)
    statistic = directional_statistic(draws, FEATURES, DIRECTIONS)
    quantiles = np.asarray(baseline.null["|".join(FEATURES)]["T"])
    grid = (np.arange(500) + 0.5) / 500
    expected = np.quantile(statistic, grid)
    assert np.allclose(quantiles, expected, rtol=0.08, atol=0.02)
    assert statistic.mean() == pytest.approx(1.5, abs=0.05)


def test_tail_probability_monotone_and_clipped():
    quantiles = [0.1, 1.0, 3.0, 8.0]
    grid = [0.125, 0.375, 0.625, 0.875]
    assert tail_probability(-1.0, quantiles, grid, 1e-4) == pytest.approx(1 - 1e-4)
    assert tail_probability(100.0, quantiles, grid, 1e-4) == pytest.approx(1e-4)
    values = [tail_probability(value, quantiles, grid, 1e-4) for value in (0.1, 1.0, 3.0, 8.0)]
    assert values == sorted(values, reverse=True)


def test_direction_prior_separates_decline_from_improvement():
    values = np.array([[1.0 + 0.01 * j, 90.0 + 0.1 * j, 0.62 + 0.001 * j] for j in range(14)])
    config = make_config()
    baseline = fit_baseline(make_records(values), "p", config)
    denominator = 1.0
    decline = {
        "gait_speed_proxy": baseline.mu["gait_speed_proxy"] - 3 * denominator,
        "cadence_spm": baseline.mu["cadence_spm"] - 3 * denominator,
        "step_time_mean_s": baseline.mu["step_time_mean_s"] + 3 * denominator,
    }
    improvement = {
        "gait_speed_proxy": baseline.mu["gait_speed_proxy"] + 3 * denominator,
        "cadence_spm": baseline.mu["cadence_spm"] + 3 * denominator,
        "step_time_mean_s": baseline.mu["step_time_mean_s"] - 3 * denominator,
    }
    day = baseline.baseline_end
    next_day = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    declined = score_record(baseline, DayRecord(next_day, decline), config)
    improved = score_record(baseline, DayRecord(next_day, improvement), config)
    assert declined.statistic_t > 10
    assert improved.statistic_t == pytest.approx(0.0, abs=1e-12)
    assert declined.q > 2
    assert declined.top_contributors


def test_cusum_replay_formula_and_gap_reset_deterministic():
    values = np.array([[1.0 + 0.01 * j, 90.0 + 0.1 * j, 0.62 + 0.001 * j] for j in range(14)])
    config = make_config()
    baseline = fit_baseline(make_records(values), "p", config)
    records = make_records(values, start=date(2024, 1, 1))
    abnormal_values = {
        name: baseline.mu[name] + (2.0 if name == "step_time_mean_s" else -2.0)
        for name in FEATURES
    }
    abnormal = [
        DayRecord((date(2024, 1, day)).isoformat(), abnormal_values)
        for day in (15, 16)
    ]
    later = DayRecord(
        (date(2024, 1, 26)).isoformat(),
        {name: baseline.mu[name] + (1.5 if name == "step_time_mean_s" else -1.5)
         for name in FEATURES},
    )
    timeline = score_timeline(records + abnormal + [later], "p", config)
    scored = timeline.scored_rows()
    assert len(scored) == 3
    assert scored[0].cusum == pytest.approx(max(0.0, scored[0].q - config.k_ref))
    assert scored[1].cusum == pytest.approx(
        max(0.0, scored[0].cusum + scored[1].q - config.k_ref)
    )
    assert scored[1].cusum_reset is False
    assert scored[2].cusum_reset is True  # 10 天空档
    assert scored[2].cusum == pytest.approx(max(0.0, scored[2].q - config.k_ref))
    first = [row.to_dict() for row in scored]
    second = [row.to_dict() for row in score_timeline(
        records + abnormal + [later], "p", config).scored_rows()]
    assert first == second


def test_baseline_roundtrip_and_partial_mask():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(14, 3)) + np.array([1.0, 90.0, 0.62])
    config = make_config()
    baseline = fit_baseline(make_records(values), "p", config)
    restored = DsPcdBaseline.from_dict(baseline.to_dict())
    day = (date.fromisoformat(baseline.baseline_end) + timedelta(days=1)).isoformat()
    record = DayRecord(day, {"gait_speed_proxy": 0.9, "cadence_spm": 90.0})
    first = score_record(baseline, record, config)
    second = score_record(restored, record, config)
    assert first.mask == ("gait_speed_proxy", "cadence_spm")
    assert first.to_dict() == second.to_dict()
    # 只剩一个特征时不足 min_features
    single = score_record(restored, DayRecord(day, {"gait_speed_proxy": 0.9}), config)
    assert single.status == "excluded_missing_features"


def test_missing_se_and_zero_se_are_rejected_in_session_mode():
    rng = np.random.default_rng(8)
    values = rng.normal(size=(14, 3)) + np.array([1.0, 90.0, 0.62])
    config = make_config(precision_mode="session_se")
    with pytest.raises(BaselineUnavailable) as info:
        fit_baseline(make_records(values), "p", config)
    assert info.value.reason == "insufficient_days"
    ses = np.full((14, 3), 0.01)
    ses[0, 0] = 0.0
    with pytest.raises(BaselineUnavailable):
        fit_baseline(make_records(values, ses=ses), "p", config)


def test_fingerprint_changes_with_precision_mode_and_seed_is_stable():
    assert config_fingerprint(make_config()) != config_fingerprint(
        make_config(precision_mode="session_se")
    )
    assert stable_seed("abc") == stable_seed("abc") != stable_seed("abd")
    values = np.array([[1.0 + 0.01 * j, 90.0 + 0.1 * j, 0.62 + 0.001 * j] for j in range(14)])
    baseline = fit_baseline(make_records(values), "p", make_config())
    assert len(baseline.null["|".join(FEATURES)]["T"]) == 200
    assert baseline.baseline_dates[0] == "2024-01-01"
    assert baseline.baseline_end == "2024-01-14"


def test_normal_score_clipped():
    assert normal_score(1.0, -4.0, 6.0) == pytest.approx(-4.0)
    assert normal_score(1e-12, -4.0, 6.0) == pytest.approx(6.0)
    assert normal_score(0.5, -4.0, 6.0) == pytest.approx(0.0, abs=1e-9)
