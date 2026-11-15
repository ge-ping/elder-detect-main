"""阶段 12：逐段派生特征、会话标准误与训练侧插补。

日表只提供速度、步长与跨步周期的**均值**；段间跨步周期变异与三个特征的
会话测量标准误必须从逐段资料重算（阶段 11 已核验日表 ``*_var_d`` 不能当作
段间方差使用）。

约定：

* 步速与步长只用时长 >30 秒的步行段，与 ``ws_30_avg_d`` / ``strlen_30_avg_d``
  口径一致；段间跨步周期变异用当日全部步行段。
* 标准误只覆盖“有限步行段造成的抽样不确定性”，不含传感器系统偏差。
* 标准误与变异指标都在当日冻结：段数不足时用同人**既往**日期的合并标准差
  回退（因果口径），不使用未来日期，也不把未知精度置零。
* 插补只发生在训练折，且只允许线性插值填补同人相邻观测日之间的零星缺测；
  长空档保持缺失（S 分支在虚拟序列里补足，不改真实记录）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from longitudinal.uncertainty import bootstrap_mean_se
from .precision import PrecisionConfig, estimate_mean_se, scan_bouts, measurement_day
from .sources import BOUTS, default_cache_dir


BOUT_COLUMNS_EXTENDED = (
    "participantid", "datatype", "visit_number", "wbday", "visit_date", "duration",
    "averagecadence", "averagestridespeed", "averagestrideduration",
    "averagestridelength", "initialcontact_event_number",
)
MIN_BOUTS_CV = 3
CV_BOOTSTRAP_SAMPLES = 200


@dataclass(frozen=True)
class BoutDerivedConfig:
    """逐段派生口径。"""

    long_bout_min_duration_s: float = 30.0
    min_bouts_se: int = 5
    min_bouts_cv: int = MIN_BOUTS_CV
    bootstrap_samples: int = CV_BOOTSTRAP_SAMPLES
    lag1_threshold: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return {
            "long_bout_min_duration_s": self.long_bout_min_duration_s,
            "min_bouts_se": self.min_bouts_se,
            "min_bouts_cv": self.min_bouts_cv,
            "bootstrap_samples": self.bootstrap_samples,
            "lag1_threshold": self.lag1_threshold,
        }


def _bootstrap_cv_se(
    values: np.ndarray, *, samples: int, seed_material: str
) -> dict[str, Any]:
    """段间变异的抽样标准误：对当日步行段做有放回重采样。"""
    array = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    n = len(array)
    if n < MIN_BOUTS_CV:
        return {"se": None, "method": "unavailable", "reason": "too_few_bouts", "n": n}
    mean = float(array.mean())
    if mean <= 0:
        return {"se": None, "method": "unavailable", "reason": "degenerate_mean", "n": n}
    if float(array.std(ddof=1)) <= 0:
        return {"se": 0.0, "method": "zero_variance", "reason": None, "n": n}
    rng = np.random.default_rng(abs(hash(seed_material)) % (2**32))
    indices = rng.integers(0, n, size=(samples, n))
    resampled = array[indices]
    means = resampled.mean(axis=1)
    sds = resampled.std(axis=1, ddof=1)
    cvs = np.where(means > 0, sds / means, np.nan)
    se = float(np.nanstd(cvs, ddof=1))
    if not math.isfinite(se):
        return {"se": None, "method": "unavailable", "reason": "non_finite", "n": n}
    return {"se": se, "method": "bout_bootstrap", "reason": None, "n": n}


def _current_pool(
    history: list[dict[str, float]]
) -> tuple[float | None, int, int]:
    """同人既往“段数充足”日的合并段内标准差与来源计数。"""
    weight = 0.0
    variance = 0.0
    days = 0
    bouts = 0
    for item in history:
        n = item.get("n", 0.0)
        sd = item.get("sd", float("nan"))
        if n >= 1 and math.isfinite(sd) and sd > 0:
            degrees = float(n) - 1.0
            weight += degrees
            variance += degrees * float(sd) ** 2
            days += 1
            bouts += int(n)
    if weight <= 0:
        return None, 0, 0
    return math.sqrt(variance / weight), days, bouts


def daily_bout_features(
    bouts: pd.DataFrame,
    *,
    config: BoutDerivedConfig = BoutDerivedConfig(),
) -> pd.DataFrame:
    """从逐段表计算逐日特征、标准误与因果回退来源。"""
    frame = bouts.copy()
    frame["participantid"] = frame["participantid"].astype(str)
    frame["visit_number"] = frame["visit_number"].astype(str).str.lower()
    if "day" not in frame.columns:
        frame["day"] = measurement_day(frame["visit_date"])
    frame["day"] = pd.to_datetime(frame["day"]).dt.normalize()
    for column in ("duration", "averagestridespeed", "averagestrideduration",
                   "averagestridelength"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame["datatype"].astype(str).eq("wb")].copy()
    frame["stride_length_cm"] = frame["averagestridelength"] * 100.0

    rows: list[dict[str, Any]] = []
    for (person, visit, day), group in frame.groupby(
        ["participantid", "visit_number", "day"], sort=True
    ):
        long_bouts = group.loc[group["duration"] > config.long_bout_min_duration_s]
        duration = group["averagestrideduration"].to_numpy(dtype=float)
        finite = duration[np.isfinite(duration)]
        mean_duration = float(finite.mean()) if len(finite) else float("nan")
        sd_duration = float(finite.std(ddof=1)) if len(finite) > 1 else float("nan")
        cv = sd_duration / mean_duration if len(finite) > 1 and mean_duration > 0 else float("nan")
        cv_se = _bootstrap_cv_se(
            finite, samples=config.bootstrap_samples,
            seed_material=f"{person}|{visit}|{day.isoformat()}|cv",
        )
        speed_se = estimate_mean_se(
            long_bouts["averagestridespeed"].to_numpy(dtype=float),
            config=PrecisionConfig(
                samples=config.bootstrap_samples, min_bouts=config.min_bouts_se,
                lag1_threshold=config.lag1_threshold,
            ),
            seed_material=f"{person}|{visit}|{day.isoformat()}|speed",
        )
        length_se = estimate_mean_se(
            long_bouts["stride_length_cm"].to_numpy(dtype=float),
            config=PrecisionConfig(
                samples=config.bootstrap_samples, min_bouts=config.min_bouts_se,
                lag1_threshold=config.lag1_threshold,
            ),
            seed_material=f"{person}|{visit}|{day.isoformat()}|length",
        )
        rows.append({
            "participantid": person,
            "visit_number": visit,
            "day": day.date().isoformat(),
            "n_bouts": int(len(group)),
            "n_long_bouts": int(len(long_bouts)),
            "stride_duration_bout_mean_s": mean_duration,
            "stride_duration_bout_sd_s": sd_duration,
            "stride_duration_bout_cv": cv,
            "se_stride_duration_bout_cv": cv_se["se"],
            "method_stride_duration_bout_cv": cv_se["method"],
            "reason_stride_duration_bout_cv": cv_se["reason"],
            "n_stride_duration_bout_cv": cv_se["n"],
            "gait_speed_bout_mean": (
                float(long_bouts["averagestridespeed"].mean()) if len(long_bouts) else float("nan")
            ),
            "se_gait_speed_mean": speed_se["se"],
            "method_gait_speed_mean": speed_se["method"],
            "reason_gait_speed_mean": speed_se["reason"],
            "n_gait_speed_mean": speed_se["n"],
            "sd_gait_speed_mean": speed_se.get("sd"),
            "stride_length_bout_mean_cm": (
                float(long_bouts["stride_length_cm"].mean()) if len(long_bouts) else float("nan")
            ),
            "se_stride_length_mean_cm": length_se["se"],
            "method_stride_length_mean_cm": length_se["method"],
            "reason_stride_length_mean_cm": length_se["reason"],
            "n_stride_length_mean_cm": length_se["n"],
            "sd_stride_length_mean_cm": length_se.get("sd"),
        })
    daily = pd.DataFrame(rows).sort_values(
        ["participantid", "day"], kind="stable"
    ).reset_index(drop=True)
    daily = apply_causal_precision_fallback(daily, config)
    return daily


def apply_causal_precision_fallback(
    frame: pd.DataFrame, config: BoutDerivedConfig = BoutDerivedConfig()
) -> pd.DataFrame:
    """按人、按日顺序执行只使用既往信息的合并标准差回退。"""
    out = frame.copy()
    for statistic, value_column, se_column, n_column, sd_column in (
        ("gait_speed_mean", "gait_speed_bout_mean", "se_gait_speed_mean",
         "n_gait_speed_mean", "sd_gait_speed_mean"),
        ("stride_length_mean_cm", "stride_length_bout_mean_cm", "se_stride_length_mean_cm",
         "n_stride_length_mean_cm", "sd_stride_length_mean_cm"),
    ):
        out[f"se_source_{statistic}"] = "direct"
        out[f"pooled_source_days_{statistic}"] = 0
        out[f"information_cutoff_day_{statistic}"] = out["day"]
        for _, group in out.groupby("participantid", sort=False):
            history: list[dict[str, float]] = []
            for index in group.index:
                n = float(out.at[index, n_column]) if pd.notna(out.at[index, n_column]) else 0.0
                se = out.at[index, se_column]
                sd = out.at[index, sd_column]
                if pd.isna(se) or (isinstance(se, float) and not math.isfinite(se)):
                    pooled, days, _ = _current_pool(history)
                    if pooled is not None and n >= 1:
                        out.at[index, se_column] = pooled / math.sqrt(n)
                        out.at[index, f"method_{statistic}"] = "pooled_prior_sd"
                        out.at[index, f"reason_{statistic}"] = None
                        out.at[index, f"se_source_{statistic}"] = "pooled_prior"
                        out.at[index, f"pooled_source_days_{statistic}"] = days
                    else:
                        out.at[index, f"se_source_{statistic}"] = "unavailable"
                        out.at[index, f"method_{statistic}"] = "unavailable"
                        out.at[index, f"reason_{statistic}"] = (
                            "no_eligible_bouts" if n <= 0 else "no_prior_pool"
                        )
                if (
                    pd.notna(sd) and float(sd) > 0
                    and n >= config.min_bouts_se
                ):
                    history.append({"n": n, "sd": float(sd)})
    out["se_source_stride_duration_bout_cv"] = np.where(
        out["se_stride_duration_bout_cv"].notna(), "direct", "unavailable"
    )
    return out


def validate_against_daily(daily: pd.DataFrame, bout_daily: pd.DataFrame) -> dict:
    """逐段派生值与发布方日表的一致性核验（单位与口径）。"""
    merged = bout_daily.merge(
        daily[["participantid", "visit_number", "day", "ws_30_avg_d", "strlen_30_avg_d",
               "strdur_all_avg_d", "wb_all_sum_d", "wb_30_sum_d"]],
        on=["participantid", "visit_number", "day"], how="inner",
    )
    for column in ("ws_30_avg_d", "strlen_30_avg_d", "strdur_all_avg_d", "wb_all_sum_d", "wb_30_sum_d"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    def compare(derived: str, release: str, tolerance: float) -> dict:
        left = pd.to_numeric(merged[derived], errors="coerce")
        right = pd.to_numeric(merged[release], errors="coerce")
        both = left.notna() & right.notna()
        diff = (left - right).abs()
        return {
            "comparable_rows": int(both.sum()),
            "both_missing_rows": int((left.isna() & right.isna()).sum()),
            "one_sided_missing_rows": int((left.isna() ^ right.isna()).sum()),
            "max_abs_diff": float(diff[both].max()) if both.any() else None,
            "within_tolerance_share": float((diff[both] <= tolerance).mean()) if both.any() else None,
            "tolerance": tolerance,
        }

    return {
        "rows": int(len(merged)),
        "speed": compare("gait_speed_bout_mean", "ws_30_avg_d", 0.0051),
        "stride_length": compare("stride_length_bout_mean_cm", "strlen_30_avg_d", 0.51),
        "stride_duration": compare("stride_duration_bout_mean_s", "strdur_all_avg_d", 0.0051),
        "bout_counts_match": float((merged["n_bouts"] == merged["wb_all_sum_d"]).mean()),
        "long_bout_counts_match": float((merged["n_long_bouts"] == merged["wb_30_sum_d"]).mean()),
        "note": "发布值保留 2 位小数，比较容差取半个最小单位；双侧缺测单独计数。",
    }


def scan_pd_bouts(
    people: Iterable[str],
    *,
    path: Path | None = None,
    config: BoutDerivedConfig = BoutDerivedConfig(),
) -> tuple[pd.DataFrame, dict]:
    """扫描 PD 受试者的逐段包并给出逐日派生表。"""
    import mobilise_d.precision as precision

    original = precision.BOUT_COLUMNS
    precision.BOUT_COLUMNS = BOUT_COLUMNS_EXTENDED
    try:
        scan = scan_bouts(path or (default_cache_dir() / BOUTS.name), people)
    finally:
        precision.BOUT_COLUMNS = original
    daily = daily_bout_features(scan.frame, config=config)
    return daily, scan.notes


def impute_scattered_missing(
    series: "PersonSeries",
    *,
    max_gap_days: int = 6,
    features: Sequence[str] | None = None,
) -> tuple["PersonSeries", dict]:
    """训练侧插补规则：只在同人相邻观测日之间线性插值，且空档不超过阈值。

    不向前/向后外推，不填补首尾缺失；被填补的格数按特征返回，便于审计。
    本规则只允许用于训练折；真实测试预测变量在主表中保持观测值。
    """
    import numpy as _np

    from .round2_scoring import PersonSeries

    names = tuple(features or series.features)
    days = [_np.datetime64(day) for day in series.days]
    filled: dict[str, int] = {}
    new_features: dict[str, tuple[float, ...]] = {}
    for name in names:
        values = _np.asarray(series.features[name], dtype=float).copy()
        count = 0
        observed = _np.flatnonzero(_np.isfinite(values))
        if len(observed) >= 2:
            for left, right in zip(observed[:-1], observed[1:]):
                gap_days = int((days[int(right)] - days[int(left)]).astype(int))
                missing = int(right - left - 1)
                if missing <= 0 or gap_days > max_gap_days + 1:
                    continue
                for position in range(int(left) + 1, int(right)):
                    weight = (days[position] - days[int(left)]) / (days[int(right)] - days[int(left)])
                    values[position] = values[int(left)] + weight * (values[int(right)] - values[int(left)])
                    count += 1
        filled[name] = count
        new_features[name] = tuple(float(value) for value in values)
    out = PersonSeries(
        person_id=series.person_id,
        days=series.days,
        features=new_features,
        feature_se=series.feature_se,
        source=series.source,
        meta={**series.meta, "imputed": filled},
    )
    return out, filled


def bootstrap_cv_se(values: Sequence[float], *, samples: int, seed_material: str) -> dict:
    """供测试与复查使用的公开入口。"""
    return _bootstrap_cv_se(np.asarray(values, dtype=float), samples=samples,
                            seed_material=seed_material)
