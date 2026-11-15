"""阶段 2 批量回放：对冻结队列逐人重放 RMS 与 DS-PCD，输出评分表。

预测日来自阶段 1 的 ``landmarks.csv``；个人基线与 Stage 1 的
``baseline_days.csv`` 完全一致。默认同时输出两种精度模式：

* ``session_se``：使用同人同日步行段的抽样标准误；
* ``none``：不做事后精度加权（明确的敏感性版本）。

输出目录（默认 ``output/ds_pcd_mobilise/stage2``）：
``scores_<mode>.csv``、``score_windows_<mode>.csv``、``excluded_<mode>.csv``、
``bout_precision.csv``、``config.json``、``manifest.json``、``replay_summary.json``。
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
from hashlib import md5
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.deviation import compute_deviation  # noqa: E402
from longitudinal.ds_pcd import (  # noqa: E402
    BaselineUnavailable,
    DsPcdBaseline,
    fit_baseline,
    score_timeline,
)
from mobilise_d import sources  # noqa: E402
from mobilise_d.observations import (  # noqa: E402
    FEATURES,
    build_records,
    distance_history,
    ds_pcd_config,
    fit_rms_baseline,
    load_stage1,
    trend_and_persistence,
    wearable_config,
)
from mobilise_d.precision import (  # noqa: E402
    PrecisionConfig,
    daily_precision,
    daily_precision_causal,
    scan_bouts,
    summarise_precision,
)

DEFAULT_BUNDLE = PROJECT_ROOT / "datasets" / "mobilise_d" / "cvs_v1.0.0"
DEFAULT_OUT = PROJECT_ROOT.parents[1] / "output" / "ds_pcd_mobilise" / "stage2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mobilise-D 阶段 2 批量回放")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--precision-mode", choices=("both", "session_se", "none"), default="both",
    )
    parser.add_argument("--rebuild-precision", action="store_true")
    parser.add_argument(
        "--precision-policy", choices=("causal", "full_archive"), default="causal",
        help="causal：每个观测日冻结当日标准误；full_archive：旧口径（含未来日，仅复现）",
    )
    parser.add_argument("--min-bouts", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--null-samples", type=int, default=20000)
    parser.add_argument("--quantile-points", type=int, default=1000)
    parser.add_argument("--k-ref", type=float, default=0.5)
    parser.add_argument("--gap-reset-days", type=int, default=7)
    parser.add_argument(
        "--alarm-threshold-file",
        default=str(DEFAULT_OUT / "simulation" / "threshold.json"),
    )
    return parser


def _md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(repo: Path) -> dict:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return {"available": False, "error": str(exc)}
    changed = [line[3:] for line in status if line.strip()]
    return {
        "available": True,
        "commit": commit,
        "dirty": bool(changed),
        "changed_path_count": len(changed),
        "changed_paths_sample": changed[:40],
    }


def _prepare_precision(
    args: argparse.Namespace, cohort, out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache = Path(args.cache_dir) if args.cache_dir else sources.default_cache_dir()
    archive = cache / sources.BOUTS.name
    if not sources.is_cached(archive, sources.BOUTS):
        raise FileNotFoundError(f"步行段包缺失或哈希不符：{archive}")
    precision_path = out_dir / "bout_precision.csv"
    notes_path = out_dir / "bout_precision_notes.json"
    if precision_path.is_file() and not args.rebuild_precision:
        frame = pd.read_csv(precision_path, dtype={"participantid": str})
        notes = (
            json.loads(notes_path.read_text(encoding="utf-8"))
            if notes_path.is_file() else {}
        )
        stored_policy = notes.get("policy", "full_archive")
        if stored_policy != args.precision_policy:
            raise ValueError(
                f"精度表策略为 {stored_policy}，与请求的 {args.precision_policy} 不一致；"
                "请使用 --rebuild-precision 或新的输出目录"
            )
        notes["reused_from"] = str(precision_path)
        print(f"复用步行段精度表：{precision_path}（{stored_policy}）", flush=True)
        return frame, notes
    started = datetime.now(timezone.utc)
    scan = scan_bouts(archive, cohort.person_ids)
    builder = daily_precision_causal if args.precision_policy == "causal" else daily_precision
    precision = builder(
        scan.frame,
        config=PrecisionConfig(samples=args.bootstrap_samples, min_bouts=args.min_bouts),
    )
    # 17 位有效数字保证 CSV 往返不丢精度，重放逐字节一致
    precision.to_csv(precision_path, index=False, encoding="utf-8", float_format="%.17g")
    notes = {
        **scan.notes,
        "precision": summarise_precision(precision),
        "policy": args.precision_policy,
        "config": {
            "samples": args.bootstrap_samples,
            "min_bouts": args.min_bouts,
            "lag1_threshold": PrecisionConfig().lag1_threshold,
            "pooling": (
                "prior_days_only" if args.precision_policy == "causal"
                else "full_archive_legacy"
            ),
        },
        "archive": {
            "name": sources.BOUTS.name,
            "md5": sources.BOUTS.md5,
            "path": str(archive),
        },
        "built_at_utc": started.isoformat(timespec="seconds"),
        "elapsed_seconds": round(
            (datetime.now(timezone.utc) - started).total_seconds(), 1
        ),
        "md5": _md5(precision_path),
    }
    notes_path.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"步行段精度表：{precision_path}", flush=True)
    # 评分使用磁盘上的精度表，保证本次运行与后续重放读取完全相同的数值
    return pd.read_csv(precision_path, dtype={"participantid": str}), notes


def _alarm_thresholds(
    path: Path | None, modes: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """读取仿真标定的阈值；有分模式标定时按模式取用。"""
    result: dict[str, dict[str, Any]] = {}
    if path is None or not path.is_file():
        return result
    payload = json.loads(path.read_text(encoding="utf-8"))
    simulation_path = path.parent / "simulation.json"
    if simulation_path.is_file():
        simulation = json.loads(simulation_path.read_text(encoding="utf-8"))
        for mode in modes:
            entry = simulation.get("results", {}).get(mode, {}).get("calibration")
            if entry:
                result[mode] = {
                    "threshold": float(entry["threshold"]),
                    "false_alarm_rate": float(entry["false_alarm_rate"]),
                    "target_window_days": 21,
                    "source": "simulation null（仿真零假设，非真实 30 天误报率）",
                }
    for mode in modes:
        result.setdefault(mode, {
            "threshold": float(payload["threshold"]),
            "false_alarm_rate": float(payload.get("measured_false_alarm_rate", 0.0)),
            "target_window_days": payload.get("target_window_days"),
            "source": payload.get("source", "simulation"),
        })
    return result


def _score_person(
    person: str,
    landmarks: pd.DataFrame,
    records: list,
    baseline_dates: tuple[str, ...],
    rms_config,
    ds_config,
    alarm_threshold: float | None,
    precision_mode: str,
) -> list[dict[str, Any]]:
    by_date = {record.date: record for record in records}
    record_dates = [record.date for record in records]
    if max(baseline_dates) >= min(
        pd.Timestamp(value).date().isoformat() for value in landmarks["landmark_day"]
    ):
        raise ValueError(f"{person} 的基线日期没有严格早于预测日")
    rms_baseline = fit_rms_baseline(person, by_date, baseline_dates, rms_config)
    active_features = sum(1 for stats in rms_baseline.features.values() if stats.active)
    history = distance_history(records, rms_baseline, baseline_dates[-1])
    timeline = None
    ds_error: dict[str, Any] | None = None
    ds_baseline: DsPcdBaseline | None = None
    try:
        ds_baseline = fit_baseline(
            records, person, ds_config, baseline_dates=baseline_dates,
        )
    except BaselineUnavailable as exc:
        ds_error = exc.to_dict()
    else:
        timeline = score_timeline(
            records, person, ds_config, baseline=ds_baseline,
            alarm_threshold=alarm_threshold,
        )
    rows = []
    for landmark in landmarks.itertuples(index=False):
        day = pd.Timestamp(landmark.landmark_day).date().isoformat()
        record = by_date.get(day)
        if record is None:
            raise ValueError(f"{person} 在预测日 {day} 没有有效观测")
        current = compute_deviation(record.features, rms_baseline)
        trend, window, persistent = trend_and_persistence(history, day)
        row: dict[str, Any] = {
            "participantid": person,
            "landmark_day": day,
            "visit_number": str(landmark.visit_number),
            "diary_visit_number": str(landmark.diary_visit_number),
            "precision_mode": precision_mode,
            "n_valid_days": bisect_right(record_dates, day),
            "n_valid_days_before_landmark": bisect_left(record_dates, day),
            "n_valid_days_includes_landmark": True,
            "rms_baseline_days": len(baseline_dates),
            "rms_baseline_end": baseline_dates[-1],
            "rms_features_active": active_features,
            "rms_distance": current.distance,
            "rms_deviation_index": current.deviation_index,
            "trend_7d": trend,
            "trend_points": window["points"],
            "trend_span_days": window["span_days"],
            "persistent_deviation_days": persistent,
            "ds_baseline_days": (
                len(ds_baseline.baseline_dates) if ds_baseline else None
            ),
            "ds_baseline_end": ds_baseline.baseline_end if ds_baseline else None,
            "ds_outlier_excluded_days": (
                len(ds_baseline.excluded_dates) if ds_baseline else None
            ),
        }
        if precision_mode == "session_se":
            missing = [name for name in FEATURES if name not in record.feature_se]
            row["se_missing"] = "|".join(missing)
        else:
            row["se_missing"] = ""
        if ds_error is not None:
            row.update({
                "ds_status": ds_error["reason"],
                "ds_message": ds_error["message"],
                "mask": "", "T": None, "p_value": None, "q": None,
                "omnibus_D": None, "omnibus_p": None, "cusum": None,
                "cusum_resets": None, "days_since_baseline_end": None,
                "top_contributors": "", "contributions": "{}",
                "scored_days_to_landmark": 0, "alarm": None,
            })
            rows.append(row)
            continue
        ds_row = timeline.score_on(day) if timeline else None
        if ds_row is None:
            raise ValueError(f"{person} 的预测日 {day} 不在 DS-PCD 监测区间")
        scored = [
            item for item in timeline.rows
            if item.status == "scored" and item.date <= day
        ]
        row.update({
            "ds_status": ds_row.status,
            "ds_message": ds_row.note,
            "mask": "|".join(ds_row.mask),
            "T": ds_row.statistic_t,
            "p_value": ds_row.p_value,
            "q": ds_row.q,
            "omnibus_D": ds_row.omnibus_d,
            "omnibus_p": ds_row.omnibus_p,
            "cusum": ds_row.cusum,
            "cusum_resets": sum(1 for item in scored if item.cusum_reset),
            "days_since_baseline_end": ds_row.days_since_baseline_end,
            "top_contributors": "|".join(ds_row.top_contributors),
            "contributions": json.dumps(
                {key: round(value, 6) for key, value in ds_row.contributions.items()},
                ensure_ascii=False, sort_keys=True,
            ),
            "scored_days_to_landmark": len(scored),
            "alarm": ds_row.alarm,
        })
        rows.append(row)
    return rows


def run(args: argparse.Namespace) -> int:
    bundle = Path(args.bundle)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort = load_stage1(bundle)
    precision, precision_notes = _prepare_precision(args, cohort, out_dir)
    modes = ("session_se", "none") if args.precision_mode == "both" else (args.precision_mode,)
    alarm_thresholds = _alarm_thresholds(
        Path(args.alarm_threshold_file) if args.alarm_threshold_file else None, modes,
    )
    rms_config = wearable_config()
    baseline_ready = cohort.landmarks.loc[cohort.landmarks["baseline_eligible"]]
    not_ready = cohort.landmarks.loc[~cohort.landmarks["baseline_eligible"]]

    # 同一人的所有基线就绪预测日必须共享同一批基线日期
    ready_keys = baseline_ready[["participantid", "landmark_day"]]
    baseline_sets = (
        cohort.baseline_days
        .merge(ready_keys, on=["participantid", "landmark_day"], how="inner")
        .groupby(["participantid", "landmark_day"])["baseline_day"]
        .apply(lambda values: tuple(sorted(pd.Timestamp(v).date().isoformat() for v in values)))
    )
    per_person: dict[str, set[tuple[str, ...]]] = {}
    for (person, _), dates in baseline_sets.items():
        per_person.setdefault(str(person), set()).add(dates)
    inconsistent = {
        person: dates for person, dates in per_person.items() if len(dates) > 1
    }
    if inconsistent:
        raise ValueError(
            "同一受试者的基线日期不一致：" + ", ".join(sorted(inconsistent)[:5])
        )

    summary: dict[str, Any] = {
        "landmarks_total": int(len(cohort.landmarks)),
        "baseline_ready_landmarks": int(len(baseline_ready)),
        "baseline_ready_people": int(baseline_ready["participantid"].nunique()),
        "baseline_not_ready_landmarks": int(len(not_ready)),
        "precision_mode_argument": args.precision_mode,
        "alarm_thresholds": alarm_thresholds,
        "modes": {},
        "precision": precision_notes.get("precision"),
        "precision_archive": precision_notes.get("archive"),
    }

    for mode in modes:
        ds_config = ds_pcd_config(
            precision_mode=mode,
            null_samples=args.null_samples,
            quantile_points=args.quantile_points,
            k_ref=args.k_ref,
            gap_reset_days=args.gap_reset_days,
            bootstrap_samples=args.bootstrap_samples,
            min_bouts=args.min_bouts,
        )
        alarm_threshold = alarm_thresholds.get(mode, {}).get("threshold")
        records_by_person = build_records(
            cohort, precision if mode == "session_se" else None,
            precision_mode=mode,
        )
        rows: list[dict[str, Any]] = []
        for person, group in baseline_ready.groupby("participantid", sort=True):
            dates = sorted(per_person[str(person)])[0]
            records = records_by_person.get(str(person), [])
            rows.extend(_score_person(
                str(person), group.sort_values("landmark_day"), records, dates,
                rms_config, ds_config, alarm_threshold, mode,
            ))
        scores = pd.DataFrame(rows).sort_values(
            ["participantid", "landmark_day"], kind="stable"
        ).reset_index(drop=True)
        if scores.duplicated(["participantid", "landmark_day"]).any():
            raise ValueError(f"{mode} 评分表预测键不唯一")
        scores.to_csv(out_dir / f"scores_{mode}.csv", index=False, encoding="utf-8")

        not_ready_rows = not_ready[[
            "participantid", "landmark_day", "visit_number", "diary_visit_number",
            "n_prior_valid_days", "baseline_eligible",
        ]].copy()
        not_ready_rows["landmark_day"] = not_ready_rows["landmark_day"].dt.date.astype(str)
        not_ready_rows["precision_mode"] = mode
        not_ready_rows["reason"] = "insufficient_baseline_days"
        not_ready_rows.to_csv(out_dir / f"excluded_{mode}.csv", index=False, encoding="utf-8")

        windows = cohort.windows.copy()
        windows["landmark_day"] = windows["landmark_day"].dt.date.astype(str)
        joined = scores.merge(
            windows, on=["participantid", "landmark_day"], how="inner",
            suffixes=("", "_stage1"), validate="many_to_many",
        )
        joined.to_csv(out_dir / f"score_windows_{mode}.csv", index=False, encoding="utf-8")
        expected_windows = int(len(scores) * 2)
        if len(joined) != expected_windows:
            raise ValueError(
                f"{mode} 与阶段 1 时窗连接数不符：{len(joined)} != {expected_windows}"
            )
        ds_scored = scores.loc[scores["ds_status"].eq("scored")]
        summary["modes"][mode] = {
            "score_rows": int(len(scores)),
            "score_people": int(scores["participantid"].nunique()),
            "window_rows": int(len(joined)),
            "ds_scored_rows": int(len(ds_scored)),
            "ds_status_counts": scores["ds_status"].value_counts().to_dict(),
            "se_missing_rows": int(scores["se_missing"].ne("").sum()),
            "mask_counts": ds_scored["mask"].value_counts().to_dict(),
            "cusum_resets_total": int(pd.to_numeric(
                ds_scored["cusum_resets"], errors="coerce"
            ).fillna(0).sum()),
            "alarms": int(ds_scored["alarm"].fillna(False).sum()),
            "median_cusum": float(pd.to_numeric(ds_scored["cusum"]).median())
            if len(ds_scored) else None,
            "median_T": float(pd.to_numeric(ds_scored["T"]).median())
            if len(ds_scored) else None,
            "median_q": float(pd.to_numeric(ds_scored["q"]).median())
            if len(ds_scored) else None,
            "median_rms_distance": float(pd.to_numeric(ds_scored["rms_distance"]).median())
            if len(ds_scored) else None,
            "people_scored": int(scores["participantid"].nunique()),
            "config": ds_config.to_dict(),
        }

    config_payload = {
        "wearable_config": {
            name: value for name, value in vars(rms_config).items()
            if not name.startswith("_")
        },
        "ds_pcd": {
            mode: ds_pcd_config(
                precision_mode=mode,
                null_samples=args.null_samples,
                quantile_points=args.quantile_points,
                k_ref=args.k_ref,
                gap_reset_days=args.gap_reset_days,
                bootstrap_samples=args.bootstrap_samples,
                min_bouts=args.min_bouts,
            ).to_dict()
            for mode in modes
        },
        "precision": {
            "samples": args.bootstrap_samples,
            "min_bouts": args.min_bouts,
            "lag1_threshold": PrecisionConfig().lag1_threshold,
        },
        "feature_map": {
            "walking_speed_m_s": "ws_30_avg_d",
            "cadence_spm": "cadence_all_avg_d",
            "stride_duration_s": "strdur_all_avg_d",
        },
        "alarm_thresholds": alarm_thresholds,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    inputs = [
        "participants.csv", "daily_gait.csv", "landmarks.csv",
        "baseline_days.csv", "landmark_windows.csv",
    ]
    manifest = {
        "stage": 2,
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bundle": str(bundle),
        "command": "python scripts/replay_mobilise_stage2.py",
        "code": _git_state(PROJECT_ROOT.parent),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": __import__("numpy").__version__,
        },
        "inputs": [
            {"path": name, "md5": _md5(bundle / name), "rows": int(
                sum(1 for _ in (bundle / name).open("rb")) - 1
            )}
            for name in inputs
        ],
        "bout_precision": {
            "path": "bout_precision.csv",
            "md5": _md5(out_dir / "bout_precision.csv"),
            "notes": "bout_precision_notes.json",
        },
        "outputs": {
            mode: {
                "scores": f"scores_{mode}.csv",
                "score_windows": f"score_windows_{mode}.csv",
                "excluded": f"excluded_{mode}.csv",
            }
            for mode in modes
        },
        "summary": summary,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "out": str(out_dir),
        "modes": {
            mode: {
                "rows": data["score_rows"],
                "ds_scored": data["ds_scored_rows"],
                "status": data["ds_status_counts"],
            }
            for mode, data in summary["modes"].items()
        },
        "alarm_thresholds": {
            mode: value.get("threshold") for mode, value in alarm_thresholds.items()
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
