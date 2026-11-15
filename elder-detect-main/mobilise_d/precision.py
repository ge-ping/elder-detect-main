"""从同日步行段估计逐日特征的抽样标准误。

日聚合值在发布方数据中等于步行段级数值的简单平均（阶段 1 已验证），
因此这里估计的是“当日步行段均值”的抽样标准误：

* 相邻步行段相关性弱时用解析标准误 ``sd/sqrt(n)``；
* 滞后 1 相关明显时用块长 2 的移动块 bootstrap，保留相邻段相关；
* 段数不足或方差退化时返回不可用，并保留原因。

该标准误只覆盖“有限步行段造成的抽样不确定性”，不覆盖传感器系统偏差。
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from zipfile import ZipFile

from longitudinal.uncertainty import bootstrap_mean_se
from mobilise_d.sources import BOUTS


BOUT_COLUMNS = (
    "participantid", "datatype", "visit_number", "wbday", "visit_date", "duration",
    "averagecadence", "averagestridespeed", "averagestrideduration",
    "initialcontact_event_number",
)

FEATURE_SPECS: dict[str, dict[str, Any]] = {
    "walking_speed_m_s": {"column": "averagestridespeed", "min_duration_s": 30.0},
    "cadence_spm": {"column": "averagecadence", "min_duration_s": None},
    "stride_duration_s": {"column": "averagestrideduration", "min_duration_s": None},
}


@dataclass(frozen=True)
class BoutScan:
    frame: pd.DataFrame
    notes: dict[str, Any]


@dataclass(frozen=True)
class PrecisionConfig:
    samples: int = 200
    min_bouts: int = 5
    lag1_threshold: float = 0.1


def _members(path: Path) -> list[tuple[str, str]]:
    pairs = []
    with ZipFile(path) as outer:
        for name in outer.namelist():
            if not name.lower().endswith(".zip"):
                continue
            with ZipFile(BytesIO(outer.read(name))) as inner:
                csvs = [item for item in inner.namelist() if item.endswith(".csv")]
                if len(csvs) != 1:
                    raise ValueError(f"{name} 内 CSV 数量异常：{csvs}")
                pairs.append((name, csvs[0]))
    return pairs


def _stream(path: Path, outer_name: str, csv_name: str, columns: Sequence[str], **kwargs):
    with ZipFile(path) as outer:
        with ZipFile(BytesIO(outer.read(outer_name))) as inner:
            with inner.open(csv_name) as handle:
                for chunk in pd.read_csv(handle, usecols=list(columns), dtype=str, **kwargs):
                    yield chunk


def measurement_day(epoch: pd.Series) -> pd.Series:
    """把当地零点的 epoch 秒取整到最近 UTC 零点。"""
    seconds = pd.to_numeric(epoch, errors="coerce")
    return pd.to_datetime((seconds + 43200) // 86400, unit="D", origin="unix").dt.normalize()


def scan_bouts(
    path: Path,
    people: Iterable[str],
    *,
    chunk_size: int = 1_000_000,
) -> BoutScan:
    """分块读取逐段表，只保留目标受试者。"""
    people_set = {str(person) for person in people}
    frames = []
    for outer_name, csv_name in _members(path):
        kept = 0
        for chunk in _stream(path, outer_name, csv_name, BOUT_COLUMNS, chunksize=chunk_size):
            hit = chunk.loc[chunk["participantid"].isin(people_set)]
            if len(hit):
                kept += len(hit)
                frames.append(hit)
        print(f"扫描 {outer_name}: 保留 {kept} 行", flush=True)
    if not frames:
        raise ValueError("没有匹配到任何目标受试者的步行段")
    frame = pd.concat(frames, ignore_index=True)
    frame["visit_number"] = frame["visit_number"].astype(str).str.lower()
    frame["participantid"] = frame["participantid"].astype(str)
    frame["wbday"] = pd.to_numeric(frame["wbday"], errors="coerce")
    for spec in FEATURE_SPECS.values():
        frame[spec["column"]] = pd.to_numeric(frame[spec["column"]], errors="coerce")
    frame["duration"] = pd.to_numeric(frame["duration"], errors="coerce")
    frame["raw_day"] = measurement_day(frame["visit_date"])
    frame["origin"] = frame["raw_day"] - pd.to_timedelta(frame["wbday"] - 1, unit="D")
    modal = (
        frame.groupby(["participantid", "visit_number"])["origin"]
        .agg(lambda values: values.mode().iloc[0] if len(values.mode()) else values.iloc[0])
        .rename("modal_origin")
    )
    frame = frame.merge(modal.reset_index(), on=["participantid", "visit_number"], how="left")
    frame["day"] = frame["modal_origin"] + pd.to_timedelta(frame["wbday"] - 1, unit="D")
    frame["day_corrected"] = frame["day"].ne(frame["raw_day"])
    notes = {
        "rows": int(len(frame)),
        "people": int(frame["participantid"].nunique()),
        "person_visits": int(
            frame[["participantid", "visit_number"]].drop_duplicates().shape[0]
        ),
        "days": int(frame["day"].nunique()),
        "datatype_counts": frame["datatype"].value_counts().to_dict(),
        "rows_corrected_by_modal_origin": int(frame["day_corrected"].sum()),
        "visits_corrected_by_modal_origin": int(
            frame.loc[frame["day_corrected"], ["participantid", "visit_number"]]
            .drop_duplicates().shape[0]
        ),
    }
    return BoutScan(frame=frame, notes=notes)


def estimate_mean_se(
    values: Sequence[float],
    *,
    config: PrecisionConfig,
    seed_material: str,
) -> dict[str, Any]:
    """估计均值抽样标准误；相关明显时用块长 2 的 bootstrap。"""
    array = np.asarray([float(value) for value in values if math.isfinite(float(value))],
                       dtype=np.float64)
    n = len(array)
    if n < config.min_bouts:
        return {
            "se": None, "method": "unavailable", "n": n,
            "lag1": None, "reason": "too_few_bouts",
            "sd": float(array.std(ddof=1)) if n > 1 else None,
        }
    sd = float(array.std(ddof=1))
    lag1 = None
    if n >= 6 and sd > 1e-12:
        left, right = array[:-1], array[1:]
        if np.std(left) > 1e-12 and np.std(right) > 1e-12:
            lag1 = float(np.corrcoef(left, right)[0, 1])
    if lag1 is not None and lag1 >= config.lag1_threshold:
        result = bootstrap_mean_se(
            array, samples=config.samples, min_count=config.min_bouts,
            block_length=2, seed_material=seed_material,
        )
        return {
            "se": result["se"], "method": result["method"], "n": n,
            "lag1": lag1, "reason": result["reason"], "sd": sd,
        }
    analytic = sd / math.sqrt(n) if n > 1 else 0.0
    if not math.isfinite(analytic) or analytic <= 0:
        return {
            "se": None, "method": "unavailable", "n": n,
            "lag1": lag1, "reason": "degenerate_variance", "sd": sd,
        }
    return {
        "se": float(analytic), "method": "analytic_iid", "n": n,
        "lag1": lag1, "reason": None, "sd": sd,
    }


def daily_direct_stats(
    frame: pd.DataFrame,
    *,
    config: PrecisionConfig = PrecisionConfig(),
) -> pd.DataFrame:
    """逐日直接估计：只用当天步行段，不引入任何回退。"""
    bouts = frame.loc[frame["datatype"].astype(str).eq("wb")].copy()
    rows = []
    for (person, visit, day), group in bouts.groupby(
        ["participantid", "visit_number", "day"], sort=True
    ):
        row: dict[str, Any] = {
            "participantid": person,
            "visit_number": visit,
            "day": pd.Timestamp(day).date().isoformat(),
            "n_bouts": int(len(group)),
            "n_long_bouts": int(group["duration"].gt(30).sum()),
        }
        for feature, spec in FEATURE_SPECS.items():
            selected = group
            if spec["min_duration_s"] is not None:
                selected = group.loc[group["duration"].gt(spec["min_duration_s"])]
            estimate = estimate_mean_se(
                selected[spec["column"]].to_numpy(),
                config=config,
                seed_material=f"{person}|{visit}|{row['day']}|{feature}",
            )
            row[f"se_{feature}"] = estimate["se"]
            row[f"method_{feature}"] = estimate["method"]
            row[f"n_{feature}"] = estimate["n"]
            row[f"lag1_{feature}"] = estimate["lag1"]
            row[f"reason_{feature}"] = estimate["reason"]
            row[f"sd_{feature}"] = estimate.get("sd")
            row[f"fallback_{feature}"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def daily_precision(
    frame: pd.DataFrame,
    *,
    config: PrecisionConfig = PrecisionConfig(),
) -> pd.DataFrame:
    """历史口径：整段档案的合并回退。

    仅供复现阶段 1–4 的旧结果；它会让早期日期的回退标准误使用未来日期，
    新运行必须使用 :func:`daily_precision_causal`。
    """
    return _apply_pooled_fallback(
        daily_direct_stats(frame, config=config), config,
    )


def daily_precision_causal(
    frame: pd.DataFrame,
    *,
    config: PrecisionConfig = PrecisionConfig(),
) -> pd.DataFrame:
    """因果口径：每个观测日的标准误在当日冻结。

    段数足够时沿用当天直接估计；不足时只用同一受试者在当日之前、段数
    足够的采集日估计合并段内标准差。未来日期不会改变任何历史值。
    """
    return apply_causal_pooled_fallback(
        daily_direct_stats(frame, config=config), config,
    )


def apply_causal_pooled_fallback(
    frame: pd.DataFrame,
    config: PrecisionConfig,
) -> pd.DataFrame:
    """按人、按日顺序执行仅使用既往信息的合并标准差回退。"""
    frame = frame.copy()
    frame["day"] = frame["day"].astype(str)
    frame = frame.sort_values(
        ["participantid", "day"], kind="stable"
    ).reset_index(drop=True)
    for feature in FEATURE_SPECS:
        pooled_sd_column = f"pooled_sd_{feature}"
        source_days_column = f"pooled_source_days_{feature}"
        source_bouts_column = f"pooled_bouts_{feature}"
        source_last_column = f"pooled_source_last_day_{feature}"
        cutoff_column = f"information_cutoff_day_{feature}"
        source_column = f"se_source_{feature}"
        direct_reason_column = f"direct_reason_{feature}"
        frame[pooled_sd_column] = np.nan
        frame[source_days_column] = 0
        frame[source_bouts_column] = 0
        frame[source_last_column] = None
        frame[cutoff_column] = frame["day"]
        frame[source_column] = "direct"
        frame[direct_reason_column] = frame[f"reason_{feature}"]
        for _, group in frame.groupby("participantid", sort=False):
            weight = 0.0
            weighted_variance = 0.0
            source_days = 0
            source_bouts = 0
            source_last = None
            for index in group.index:
                n = frame.at[index, f"n_{feature}"]
                sd = frame.at[index, f"sd_{feature}"]
                se = frame.at[index, f"se_{feature}"]
                direct = pd.notna(se)
                if not direct:
                    if pd.notna(n) and float(n) >= 1 and weight > 0:
                        pooled_sd = math.sqrt(weighted_variance / weight)
                        frame.at[index, f"se_{feature}"] = pooled_sd / math.sqrt(float(n))
                        frame.at[index, f"method_{feature}"] = "pooled_prior_sd"
                        frame.at[index, f"reason_{feature}"] = None
                        frame.at[index, f"fallback_{feature}"] = True
                        frame.at[index, pooled_sd_column] = pooled_sd
                        frame.at[index, source_days_column] = source_days
                        frame.at[index, source_bouts_column] = source_bouts
                        frame.at[index, source_last_column] = source_last
                        frame.at[index, source_column] = "pooled_prior"
                    else:
                        frame.at[index, f"method_{feature}"] = "unavailable"
                        if pd.isna(n) or float(n) == 0:
                            reason = "no_eligible_bouts"
                        else:
                            reason = "no_prior_pool"
                        frame.at[index, f"reason_{feature}"] = reason
                        frame.at[index, source_column] = "unavailable"
                if (
                    pd.notna(n) and float(n) >= config.min_bouts
                    and pd.notna(sd) and float(sd) > 0
                ):
                    degrees = float(n) - 1.0
                    weight += degrees
                    weighted_variance += degrees * float(sd) ** 2
                    source_days += 1
                    source_bouts += int(n)
                    source_last = frame.at[index, "day"]
    return frame


def _apply_pooled_fallback(
    frame: pd.DataFrame, config: PrecisionConfig,
) -> pd.DataFrame:
    """段数不足的日期用同人合并段内标准差回退，并标记 ``method=pooled_sd``。

    合并标准差只用该受试者段数充足的采集日估计；回退值是可复核的
    数据估计，不是外部常数。日数仍少于 1 或没有可合并方差的日期保持不可用。
    """
    frame = frame.copy()
    for feature in FEATURE_SPECS:
        sd = pd.to_numeric(frame[f"sd_{feature}"], errors="coerce")
        n = pd.to_numeric(frame[f"n_{feature}"], errors="coerce")
        usable = n.ge(config.min_bouts) & sd.notna() & sd.gt(0)
        weights = (n - 1).where(usable, 0.0)
        variance = (sd ** 2 * weights)
        grouped = frame.assign(_variance=variance, _weight=weights).groupby(
            "participantid", sort=False
        )[["_variance", "_weight"]].sum()
        pooled = (grouped["_variance"] / grouped["_weight"]).pow(0.5)
        pooled = pooled.where(grouped["_weight"] > 0)
        needs = frame[f"se_{feature}"].isna() & n.ge(1) & pooled.reindex(
            frame["participantid"]
        ).notna().to_numpy()
        if not needs.any():
            continue
        fallback = pooled.reindex(frame.loc[needs, "participantid"]).to_numpy()
        frame.loc[needs, f"se_{feature}"] = fallback / np.sqrt(n[needs])
        frame.loc[needs, f"method_{feature}"] = "pooled_sd"
        frame.loc[needs, f"reason_{feature}"] = None
        frame.loc[needs, f"fallback_{feature}"] = True
        frame[f"pooled_sd_{feature}"] = pooled.reindex(frame["participantid"]).to_numpy()
    return frame


def summarise_precision(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"days": int(len(frame))}
    for feature in FEATURE_SPECS:
        se = pd.to_numeric(frame[f"se_{feature}"], errors="coerce")
        summary[feature] = {
            "available_days": int(se.notna().sum()),
            "unavailable_days": int(se.isna().sum()),
            "se_sources": (
                frame[f"se_source_{feature}"].value_counts().to_dict()
                if f"se_source_{feature}" in frame.columns else None
            ),
            "unavailable_reasons": (
                frame.loc[se.isna(), f"reason_{feature}"]
                .fillna("unknown").value_counts().to_dict()
            ),
            "median_se": float(se.median()) if se.notna().any() else None,
            "p90_se": float(se.quantile(0.9)) if se.notna().any() else None,
            "methods": frame[f"method_{feature}"].value_counts().to_dict(),
            "fallback_days": int(frame[f"fallback_{feature}"].sum()),
            "lag1_ge_0_1_share": float(
                pd.to_numeric(frame[f"lag1_{feature}"], errors="coerce").ge(0.1).mean()
            ),
        }
    return summary
