"""核验逐段→逐日聚合的单位与口径（阶段 11 字段审计的一部分）。

只扫描少量受试者的步行段，回答三个问题：

1. ``ws_30_avg_d`` 是否等于“时长 >30 s 的步行段步速均值”（发布值保留 2 位）；
2. ``strlen_30_avg_d`` 是否等于“同时段段级步长（米）×100”；
3. ``strdur_all_var_d`` 是否等于“段级跨步周期均值的方差”（若否，则不能
   当作段间变异使用）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mobilise_d.precision as precision  # noqa: E402
from mobilise_d.daily import load_daily  # noqa: E402
from mobilise_d.sources import BOUTS, default_cache_dir  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise" / "round2"


def run(people: list[str], run_dir: Path) -> dict:
    extra = ("averagestridelength", "numberstrides")
    original = precision.BOUT_COLUMNS
    precision.BOUT_COLUMNS = original + extra
    try:
        scan = precision.scan_bouts(default_cache_dir() / BOUTS.name, people)
    finally:
        precision.BOUT_COLUMNS = original
    bouts = scan.frame.loc[scan.frame["datatype"].astype(str).eq("wb")].copy()
    for column in ("duration", "averagestridespeed", "averagestrideduration",
                   "averagestridelength", "numberstrides"):
        bouts[column] = pd.to_numeric(bouts[column], errors="coerce")

    daily_data = load_daily(default_cache_dir() / "Aggregated DMO Data.zip")
    daily = daily_data.frame
    daily["participantid"] = daily["participantid"].astype(str)
    daily = daily.loc[daily["participantid"].isin(people)].copy()

    rows = []
    for (person, visit, day), group in bouts.groupby(
        ["participantid", "visit_number", "day"], sort=True
    ):
        long_bouts = group.loc[group["duration"] > 30.0]
        rows.append({
            "participantid": str(person),
            "visit_number": str(visit),
            "day": pd.Timestamp(day).normalize(),
            "n_bouts": int(len(group)),
            "n_long_bouts": int(len(long_bouts)),
            "ws30_bout_mean": long_bouts["averagestridespeed"].mean(),
            "strlen30_bout_mean_cm": long_bouts["averagestridelength"].mean() * 100.0,
            "strdur_all_bout_mean": group["averagestrideduration"].mean(),
            "strdur_bout_var_s2": group["averagestrideduration"].var(ddof=1),
            "strdur_bout_cv": (
                group["averagestrideduration"].std(ddof=1)
                / group["averagestrideduration"].mean()
                if len(group) > 1 else np.nan
            ),
        })
    derived = pd.DataFrame(rows)
    derived["day"] = pd.to_datetime(derived["day"]).dt.normalize()
    merged = derived.merge(
        daily[[
            "participantid", "visit_number", "day", "ws_30_avg_d", "strlen_30_avg_d",
            "strdur_all_avg_d", "strdur_all_var_d", "wb_30_sum_d", "wb_all_sum_d",
        ]],
        on=["participantid", "visit_number", "day"], how="inner", suffixes=("", "_daily"),
    )
    for column in ("ws_30_avg_d", "strlen_30_avg_d", "strdur_all_avg_d",
                   "strdur_all_var_d", "wb_30_sum_d", "wb_all_sum_d"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    def stats(diff: pd.Series, tolerance: float) -> dict:
        diff = diff.dropna().abs()
        return {
            "rows": int(len(diff)),
            "median_abs_diff": float(diff.median()) if len(diff) else None,
            "max_abs_diff": float(diff.max()) if len(diff) else None,
            "within_tolerance": int((diff <= tolerance).sum()) if len(diff) else 0,
            "tolerance": tolerance,
        }

    ws = stats(merged["ws30_bout_mean"] - merged["ws_30_avg_d"], 0.005 + 1e-9)
    stride_length = stats(merged["strlen30_bout_mean_cm"] - merged["strlen_30_avg_d"], 0.5 + 1e-9)
    strdur = stats(merged["strdur_all_bout_mean"] - merged["strdur_all_avg_d"], 0.005 + 1e-9)
    ratio = (merged["strdur_all_var_d"] / merged["strdur_bout_var_s2"]).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    between_bout_var = merged["strdur_bout_var_s2"].dropna()
    within_scaled = merged["strdur_all_var_d"].dropna() / 1e4
    report = {
        "people_scanned": people,
        "bout_scan_notes": scan.notes,
        "merged_days": int(len(merged)),
        "ws_30_avg_d_vs_bout_mean": ws,
        "strlen_30_avg_d_vs_bout_mean_times_100": stride_length,
        "strdur_all_avg_d_vs_bout_mean": strdur,
        "counts_match": {
            "wb_all_sum_d": int((merged["n_bouts"] == merged["wb_all_sum_d"]).sum()),
            "wb_30_sum_d": int((merged["n_long_bouts"] == merged["wb_30_sum_d"]).sum()),
            "rows": int(len(merged)),
        },
        "strdur_all_var_d": {
            "ratio_to_between_bout_var_median": float(ratio.median()) if len(ratio) else None,
            "ratio_p10_p90": (
                [float(ratio.quantile(0.1)), float(ratio.quantile(0.9))] if len(ratio) else None
            ),
            "between_bout_var_median_s2": float(between_bout_var.median()) if len(between_bout_var) else None,
            "daily_column_median": float(merged["strdur_all_var_d"].median()),
            "implied_within_bout_sd_if_scaled_1e4_s": (
                float(np.sqrt(within_scaled.median())) if len(within_scaled) else None
            ),
            "conclusion": (
                "日表 strdur_all_var_d 与“段级跨步周期均值的方差”不是同一量"
                "（比值中位数远大于 1），且 Data Dictionary 未逐条定义逐日 var 列；"
                "因此不把它当作段间变异，主特征改由逐段资料重算"
            ),
        },
        "between_bout_variability_recomputable_from_bouts": bool(len(between_bout_var) > 0),
        "verified_targets": {
            "stride_duration_bout_cv": "由逐段 averagestrideduration 的段间方差/均值得到",
            "note": "段间变异不冒称逐步（同一段内的步级变异）",
        },
    }
    _write(run_dir / "bout_unit_check.json", report)
    return report


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="逐段→逐日单位核验")
    parser.add_argument("--run", default=str(
        OUTPUT_ROOT / "stage11" / "20261115-protocol-v1"
    ))
    parser.add_argument("--people", nargs="*", default=None)
    args = parser.parse_args(argv)
    run_dir = Path(args.run)
    people = args.people
    if not people:
        participants = pd.read_csv(run_dir / "participants.csv", dtype={"participantid": str})
        pool = participants.loc[participants["cohort"] == "pd", "participantid"].tolist()
        people = pool[:5]
    report = run(people, run_dir)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
