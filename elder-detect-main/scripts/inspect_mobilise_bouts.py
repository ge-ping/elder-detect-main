"""核查 Walking Bout DMO 包的字段、日期连接键与日聚合可重建性。

阶段 1 只做实测记录，不估计测量标准误（属于阶段 2）。逐段表约 18M 行、
解压后 4–5 GB，因此：

* 字段覆盖率对全部逐段行做一次分块扫描（``--skip-full-scan`` 可跳过）；
* 日期连接与日聚合重建在固定抽样的目标人群中完成，样本量写入产物。

实测结论（见 ``bout_support.json``）：

* ``visit_date`` 列不是临床访视日，而是该行监测日的当地零点（epoch 秒，
  当地偏移 +1..+3 小时，因此 UTC 余数为 21/22/23 点）。把 epoch 向最近的
  UTC 零点取整即得到日历日，和发布方逐日表的 ``measurement_date`` 一一对应。
* ``wbday`` 1..7 表示监测第 1..7 天，但首日可能没有步行段，不能用首条
  记录推断监测起始日。
* 计数、步数、步行时长、步频、跨步周期可逐日精确重建；步速与步长需要
  ``duration > 30`` 的步行段（不是 ``>= 30``）。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import sys
import time
from typing import Sequence
from zipfile import ZipFile

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mobilise_d import sources  # noqa: E402
from mobilise_d.daily import load_all_days  # noqa: E402

BUNDLE_DEFAULT = PROJECT_ROOT / "datasets" / "mobilise_d" / "cvs_v1.0.0"
FEATURES = (
    "duration", "numberstrides", "averagecadence", "averagestridespeed",
    "averagestridelength", "averagestrideduration", "averagestepduration_so",
    "initialcontact_event_number", "turn_number_so",
)
USECOLS = (
    "participantid", "datatype", "wb_id", "wbday", "wbstart", "wbend", "duration",
    "duration_flag", "visit_date", "visit_number", "numberstrides",
    "numberstrides_flag", "averagecadence", "averagecadence_flag",
    "turn_number_so", "turn_number_so_flag", "starttimestamputc",
    "averagestridespeed", "averagestridespeed_flag",
    "averagestridelength", "averagestridelength_flag",
    "averagestrideduration", "averagestrideduration_flag",
    "averagestepduration_so", "averagestepduration_so_flag",
    "initialcontact_event_number", "initialcontact_event_number_flag",
)


def _members(path: Path) -> list[tuple[str, str]]:
    """返回 (外层成员名, 内层 CSV 名)。"""
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


def day_from_epoch(epoch: pd.Series) -> pd.Series:
    """把当地零点的 epoch 秒取整到最近的 UTC 零点，得到日历日。"""
    seconds = pd.to_numeric(epoch, errors="coerce")
    return pd.to_datetime((seconds + 43200) // 86400, unit="D", origin="unix").dt.normalize()


def scan_all(
    path: Path,
    pairs: list[tuple[str, str]],
    chunk_size: int,
    people: set[str] | None = None,
) -> tuple[dict, pd.DataFrame | None]:
    """单次分块扫描：累计字段覆盖，并顺带抽取样本行（可选）。"""
    total = 0
    datatype: dict[str, int] = {}
    nonnull = {column: 0 for column in FEATURES}
    flag_true = {column + "_flag": 0 for column in FEATURES if column != "turn_number_so"}
    duration_bins = np.zeros(601, dtype=np.int64)
    duration_over_600 = 0
    duration_over_30 = 0
    duration_min = np.inf
    duration_max = -np.inf
    per_visit_rows: dict[str, int] = {}
    sample_frames: list[pd.DataFrame] = []
    for outer_name, csv_name in pairs:
        rows = 0
        started = time.time()
        for chunk in _stream(path, outer_name, csv_name, USECOLS, chunksize=chunk_size):
            rows += len(chunk)
            total += len(chunk)
            for value, count in chunk["datatype"].value_counts(dropna=False).items():
                datatype[str(value)] = datatype.get(str(value), 0) + int(count)
            visit_code = str(chunk["visit_number"].iloc[0])
            per_visit_rows[visit_code] = per_visit_rows.get(visit_code, 0) + len(chunk)
            for column in FEATURES:
                nonnull[column] += int(chunk[column].notna().sum())
                flag_column = column + "_flag"
                if flag_column in chunk.columns and flag_column in flag_true:
                    flag_true[flag_column] += int(chunk[flag_column].eq("true").sum())
            duration = pd.to_numeric(chunk["duration"], errors="coerce").dropna()
            if len(duration):
                values = duration.to_numpy()
                duration_min = min(duration_min, float(values.min()))
                duration_max = max(duration_max, float(values.max()))
                clipped = np.clip(np.floor(values).astype(int), 0, 600)
                counts = np.bincount(clipped, minlength=601)
                duration_bins += counts
                duration_over_600 += int((values > 600).sum())
                duration_over_30 += int((values > 30).sum())
            if people is not None:
                hit = chunk.loc[chunk["participantid"].isin(people)]
                if len(hit):
                    sample_frames.append(hit)
        print(f"扫描 {outer_name}: {rows} 行，{time.time() - started:.1f} 秒", flush=True)

    def quantile(q: float) -> float:
        cumulative = np.cumsum(duration_bins)
        total_rows = int(cumulative[-1]) + duration_over_600
        if not total_rows:
            return float("nan")
        if q >= 0.999 and duration_over_600:
            return 600.0
        target = q * (int(cumulative[-1]) - 1)
        index = int(np.searchsorted(cumulative, target, side="left"))
        return float(min(index + 0.5, 600.0))

    coverage = {
        "rows": int(total),
        "rows_per_visit": per_visit_rows,
        "datatype_counts": datatype,
        "feature_coverage": {
            column: {
                "nonnull_share": round(nonnull[column] / total, 4) if total else None,
                "flag_true_share": (
                    round(flag_true[column + "_flag"] / total, 4)
                    if column + "_flag" in flag_true and total else None
                ),
            }
            for column in FEATURES
        },
        "duration_seconds": {
            "min": float(duration_min),
            "median": quantile(0.5),
            "p90": quantile(0.9),
            "p99": quantile(0.99),
            "max": float(duration_max),
            "note": "分位数由 1 秒分箱近似",
        } if total else None,
        "row_count_duration_gt_30s": duration_over_30,
        "row_share_duration_gt_30s": (
            round(duration_over_30 / total, 4) if total else None
        ),
    }
    sample = pd.concat(sample_frames, ignore_index=True) if sample_frames else None
    return coverage, sample


def prepare_sample(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["visit_number"] = frame["visit_number"].astype(str).str.lower()
    frame["wbday_num"] = pd.to_numeric(frame["wbday"], errors="coerce")
    frame["datatype"] = frame["datatype"].astype(str)
    frame["raw_day"] = day_from_epoch(frame["visit_date"])
    # 每次采集的监测首日 = 行日期 - (wbday-1)；个别行的 epoch 用了错误时区，
    # 用该次采集的众数首日校正，保证同日聚合与发布方逐日表一致。
    frame["origin"] = frame["raw_day"] - pd.to_timedelta(frame["wbday_num"] - 1, unit="D")
    modal = (
        frame.groupby(["participantid", "visit_number"])["origin"]
        .agg(lambda values: values.mode().iloc[0] if len(values.mode()) else values.iloc[0])
        .rename("modal_origin")
    )
    frame = frame.merge(modal.reset_index(), on=["participantid", "visit_number"], how="left")
    frame["day"] = frame["modal_origin"] + pd.to_timedelta(frame["wbday_num"] - 1, unit="D")
    frame["day_corrected"] = frame["day"].ne(frame["raw_day"])
    return frame


def _weighted(frame: pd.DataFrame, column: str, weights: str) -> float | None:
    values = frame[column]
    weight = frame[weights]
    valid = values.notna() & weight.notna() & weight.gt(0)
    if not valid.any():
        return None
    return float(np.average(values[valid], weights=weight[valid]))


def reconstruct(frame: pd.DataFrame) -> pd.DataFrame:
    """按实测规则从逐段行重建日聚合值。"""
    rows = []
    wb_all = frame.loc[frame["datatype"].eq("wb")]
    for (person, visit, day), group in wb_all.groupby(
        ["participantid", "visit_number", "day"], sort=True
    ):
        long_bouts = group.loc[group["duration"].gt(30)]
        rows.append({
            "participantid": person,
            "visit_number": visit,
            "day": day.date().isoformat(),
            "n_wb": int(len(group)),
            "n_wb_gt30": int(len(long_bouts)),
            "steps_sum": float(group["initialcontact_event_number"].sum()),
            "duration_min_rounded": float(round(group["duration"].sum() / 60.0)),
            "duration_min_exact": float(group["duration"].sum() / 60.0),
            "cadence_mean_rounded": round(float(group["averagecadence"].mean())),
            "strdur_mean_2dp": round(float(group["averagestrideduration"].mean()), 2),
            "ws30_mean_2dp": (
                round(float(long_bouts["averagestridespeed"].mean()), 2)
                if len(long_bouts) else None
            ),
            "strlen30_mean_cm_rounded": (
                round(float(long_bouts["averagestridelength"].mean()) * 100.0)
                if len(long_bouts) else None
            ),
            "cadence_dur_weighted": _weighted(group, "averagecadence", "duration"),
            "strdur_dur_weighted": _weighted(group, "averagestrideduration", "duration"),
            "ws30_dur_weighted": (
                _weighted(long_bouts, "averagestridespeed", "duration")
                if len(long_bouts) else None
            ),
            "strlen30_dur_weighted_cm": (
                None if _weighted(long_bouts, "averagestridelength", "duration") is None
                else _weighted(long_bouts, "averagestridelength", "duration") * 100.0
            ),
        })
    return pd.DataFrame(rows)


def _compare(recon: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """与发布方逐日表对比：主规则精确匹配率 + 备选加权规则偏差。"""
    data = daily.loc[
        daily["participantid"].isin(recon["participantid"].unique()),
        ["participantid", "visit_number", "day", "wb_all_sum_d", "wb_30_sum_d",
         "wbsteps_all_sum_d", "walkdur_all_sum_d", "cadence_all_avg_d",
         "strdur_all_avg_d", "ws_30_avg_d", "strlen_30_avg_d"],
    ].copy()
    data["day"] = pd.to_datetime(data["day"]).dt.date.astype(str)
    merged = recon.merge(
        data, on=["participantid", "visit_number", "day"], how="inner", suffixes=("", "_release")
    )
    exact = {}
    pairs = (
        ("n_wb", "wb_all_sum_d"),
        ("n_wb_gt30", "wb_30_sum_d"),
        ("steps_sum", "wbsteps_all_sum_d"),
        ("duration_min_rounded", "walkdur_all_sum_d"),
        ("cadence_mean_rounded", "cadence_all_avg_d"),
        ("strdur_mean_2dp", "strdur_all_avg_d"),
        ("ws30_mean_2dp", "ws_30_avg_d"),
        ("strlen30_mean_cm_rounded", "strlen_30_avg_d"),
    )
    for recon_column, release_column in pairs:
        left = merged[recon_column]
        right = merged[release_column]
        valid = left.notna() & right.notna()
        exact[recon_column] = {
            "compared_days": int(valid.sum()),
            "missing_both": int((left.isna() & right.isna()).sum()),
            "exact_share": float((left[valid] == right[valid]).mean()) if valid.any() else None,
            "mismatch_days": int((left[valid] != right[valid]).sum()),
        }
    variants = {}
    for recon_column, release_column in (
        ("cadence_dur_weighted", "cadence_all_avg_d"),
        ("strdur_dur_weighted", "strdur_all_avg_d"),
        ("ws30_dur_weighted", "ws_30_avg_d"),
        ("strlen30_dur_weighted_cm", "strlen_30_avg_d"),
    ):
        left = pd.to_numeric(merged[recon_column], errors="coerce")
        right = pd.to_numeric(merged[release_column], errors="coerce")
        valid = left.notna() & right.notna()
        rel = ((left[valid] - right[valid]).abs() / right[valid].abs()).dropna()
        variants[recon_column] = {
            "compared_days": int(valid.sum()),
            "median_abs_rel_diff": float(rel.median()) if len(rel) else None,
            "within_1pct_share": float((rel <= 0.01).mean()) if len(rel) else None,
        }
    mismatch_mask = pd.Series(False, index=merged.index)
    for recon_column, release_column in pairs:
        left = merged[recon_column]
        right = merged[release_column]
        valid = left.notna() & right.notna()
        mismatch_mask |= valid & (left != right)
    mismatch_examples = merged.loc[mismatch_mask].head(5).to_dict("records")
    return {
        "sampled_days": int(len(recon)),
        "matched_days": int(len(merged)),
        "unmatched_days": int(len(recon) - len(merged)),
        "mismatch_examples": mismatch_examples,
        "exact_rule": exact,
        "duration_weighted_variants": variants,
        "rule": (
            "wb 行按 (participantid, visit_number, visit_date 取整到最近 UTC 零点) 分组："
            "wb_all_sum_d=行数；wb_30_sum_d=duration>30 的行数；"
            "wbsteps_all_sum_d=initialcontact_event_number 之和；"
            "walkdur_all_sum_d=round(秒数和/60)；"
            "cadence_all_avg_d=round(均值)；strdur_all_avg_d=均值保留 2 位；"
            "ws_30_avg_d=duration>30 的步速均值保留 2 位；"
            "strlen_30_avg_d=米均值*100 取整（厘米）；"
            "监测日按 wbday 与本次采集众数首日对齐，校正个别 epoch 时区错误"
        ),
    }


def join_check(frame: pd.DataFrame, visits: pd.DataFrame, daily_all: pd.DataFrame) -> dict:
    """日期连接键核查：visit_date 语义、wbday 与临床访视日的关系。"""
    visits = visits.copy()
    visits["day"] = pd.to_datetime(visits["day"]).dt.normalize()
    all_days = daily_all.copy()
    all_days["day"] = pd.to_datetime(all_days["day"]).dt.normalize()
    epochs = pd.to_numeric(frame["visit_date"], errors="coerce")
    remainder_hours = ((epochs % 86400) / 3600).round(0).value_counts().sort_index()
    corrected_rows = int(frame["day_corrected"].sum())
    corrected_visits = int(
        frame.loc[frame["day_corrected"], ["participantid", "visit_number"]].drop_duplicates().shape[0]
    )
    wbday_counts = frame["wbday_num"].value_counts().sort_index()

    rows = []
    for (person, visit), group in frame.groupby(["participantid", "visit_number"], sort=True):
        visit_day = visits.loc[
            visits["participantid"].eq(person) & visits["visit_number"].eq(visit), "day"
        ]
        if visit_day.empty:
            continue
        visit_day = visit_day.iloc[0]
        bout_days = sorted(pd.Timestamp(day) for day in group["day"].dropna().unique())
        daily_days = sorted(
            pd.Timestamp(day) for day in all_days.loc[
                all_days["participantid"].eq(person) & all_days["visit_number"].eq(visit), "day"
            ].unique()
        )
        wbday_days = group.groupby("wbday_num")["day"].agg(["min", "max", "nunique"])
        first_wbday = group["wbday_num"].min()
        first_wbday_day = group.loc[group["wbday_num"].eq(first_wbday), "day"].min()
        rows.append({
            "participantid": person,
            "visit_number": visit,
            "clinical_visit_day": visit_day.date().isoformat(),
            "n_bout_days": len(bout_days),
            "n_daily_days": len(daily_days),
            "bout_days_equal_daily_days": (
                [pd.Timestamp(day).date().isoformat() for day in bout_days]
                == [pd.Timestamp(day).date().isoformat() for day in daily_days]
            ),
            "min_wbday": int(first_wbday) if pd.notna(first_wbday) else None,
            "min_wbday_day": (
                pd.Timestamp(first_wbday_day).date().isoformat()
                if pd.notna(first_wbday_day) else None
            ),
            "min_wbday_offset_from_visit": (
                int((pd.Timestamp(first_wbday_day) - visit_day).days)
                if pd.notna(first_wbday_day) else None
            ),
            "wbday_maps_to_single_day": bool(
                (wbday_days["nunique"] == 1).all()
                and (wbday_days["min"] == wbday_days["max"]).all()
            ),
        })
    rows_frame = pd.DataFrame(rows)
    return {
        "person_visits_checked": int(len(rows_frame)),
        "visit_date_epoch_remainder_hours": {
            f"{hour:g}h": int(count) for hour, count in remainder_hours.items()
        },
        "visit_date_semantics": (
            "epoch 秒的当地零点（余数为 21/22/23 点，说明是 +1..+3 时区的当地午夜）；"
            "取整到最近 UTC 零点后与逐日表 measurement_date 精确对应；该列不是临床访视日。"
        ),
        "rows_corrected_by_modal_origin": corrected_rows,
        "visits_corrected_by_modal_origin": corrected_visits,
        "wbday_value_counts": {int(key): int(value) for key, value in wbday_counts.items()},
        "visits_with_wbday_gt_7": int(
            frame.loc[frame["wbday_num"].gt(7), ["participantid", "visit_number"]]
            .drop_duplicates().shape[0]
        ),
        "wbday_maps_to_single_day_share": (
            float(rows_frame["wbday_maps_to_single_day"].mean()) if len(rows_frame) else None
        ),
        "min_wbday_offset_histogram": (
            rows_frame["min_wbday_offset_from_visit"].value_counts().sort_index().to_dict()
            if len(rows_frame) else {}
        ),
        "visits_without_wbday_1": int((rows_frame["min_wbday"] > 1).sum()),
        "bout_days_equal_daily_table_days_share": (
            float(rows_frame["bout_days_equal_daily_days"].mean()) if len(rows_frame) else None
        ),
        "examples": rows_frame.head(3).to_dict("records"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="核查 Walking Bout DMO 包")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--bundle", default=str(BUNDLE_DEFAULT))
    parser.add_argument("--out", default=None)
    parser.add_argument("--sample-people", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--skip-full-scan", action="store_true")
    args = parser.parse_args(argv)
    cache = Path(args.cache_dir) if args.cache_dir else sources.default_cache_dir()
    bundle = Path(args.bundle)
    out = Path(args.out) if args.out else bundle / "bout_support.json"
    path = cache / sources.BOUTS.name
    if not sources.is_cached(path, sources.BOUTS):
        raise FileNotFoundError(f"步行段包缺失或哈希不符：{path}")

    pairs = _members(path)
    participants = pd.read_csv(bundle / "participants.csv", dtype={"participantid": str})
    people = sorted(participants["participantid"].unique())
    step = max(1, len(people) // args.sample_people)
    sample_people = set(people[::step][: args.sample_people])

    payload = {
        "stage": 1,
        "purpose": "核查逐段字段、日期连接键与日聚合可重建性；标准误估计属于阶段 2",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "archive": {
            "name": sources.BOUTS.name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "md5": sources.BOUTS.md5,
            "members": [{"outer": outer, "csv": csv} for outer, csv in pairs],
            "columns": list(USECOLS),
        },
        "sample": {
            "people": sorted(sample_people),
            "people_count": len(sample_people),
            "selection": f"按 participantid 排序后等距抽取 {args.sample_people} 人",
        },
    }
    print(f"抽样人数 {len(sample_people)}", flush=True)
    coverage = None
    sample = None
    if args.skip_full_scan:
        if out.is_file():
            previous = json.loads(out.read_text(encoding="utf-8"))
            if previous.get("archive", {}).get("md5") == sources.BOUTS.md5:
                coverage = previous.get("coverage_all_rows")
                payload["coverage_all_rows"] = coverage
                payload["coverage_reused_from"] = str(out)
        for outer_name, csv_name in pairs:
            for chunk in _stream(path, outer_name, csv_name, USECOLS, chunksize=args.chunk_size):
                hit = chunk.loc[chunk["participantid"].isin(sample_people)]
                if len(hit):
                    sample = hit if sample is None else pd.concat([sample, hit])
    else:
        coverage, sample = scan_all(path, pairs, args.chunk_size, sample_people)
        payload["coverage_all_rows"] = coverage
    if sample is None or sample.empty:
        raise ValueError("抽样没有匹配到逐段行")
    frame = prepare_sample(sample)
    payload["sample"]["rows"] = int(len(frame))
    payload["sample"]["datatype_counts"] = frame["datatype"].value_counts().to_dict()
    payload["sample"]["wbday_values"] = sorted(
        int(value) for value in frame["wbday_num"].dropna().unique()
    )

    visits = pd.read_csv(bundle / "visits.csv", dtype={"participantid": str})
    daily_all = load_all_days(cache / sources.DAILY.name).frame
    daily_all["participantid"] = daily_all["participantid"].astype(str)
    daily_all["visit_number"] = daily_all["visit_number"].astype(str).str.lower()
    payload["join_check"] = join_check(frame, visits, daily_all)

    recon = reconstruct(frame)
    payload["daily_reconstruction"] = _compare(recon, daily_all)

    sources.write_json(out, payload)
    print(json.dumps({
        "out": str(out),
        "rows_scanned": payload.get("coverage_all_rows", {}).get("rows"),
        "datatype": payload.get("coverage_all_rows", {}).get("datatype_counts"),
        "join_check": {key: value for key, value in payload["join_check"].items()
                       if key != "examples"},
        "reconstruction": payload["daily_reconstruction"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
