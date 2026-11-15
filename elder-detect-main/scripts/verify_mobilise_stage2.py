"""阶段 2 核验：离线/在线一致、时间截断、重放确定性与仿真行为。

读取阶段 2 产物与阶段 1 冻结队列，输出
``output/ds_pcd_mobilise/stage2/verification.json`` 与 ``VERIFICATION.md``。
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from hashlib import md5
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.baseline import build_baseline  # noqa: E402
from longitudinal.config import LongitudinalConfig  # noqa: E402
from longitudinal.ds_pcd import DsPcdBaseline, DsPcdConfig, fit_baseline  # noqa: E402
from longitudinal.schema import DailyObservation  # noqa: E402
from longitudinal.service import LongitudinalRiskService  # noqa: E402
from longitudinal.storage import ProfileStore  # noqa: E402
from mobilise_d.observations import (  # noqa: E402
    FEATURES,
    build_records,
    ds_pcd_config,
    fit_rms_baseline,
    load_stage1,
    wearable_config,
)

STAGE2 = PROJECT_ROOT.parents[1] / "output" / "ds_pcd_mobilise" / "stage2"
BUNDLE = PROJECT_ROOT / "datasets" / "mobilise_d" / "cvs_v1.0.0"
MODES = ("session_se", "none")


def _md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile_for_person(
    directory: Path,
    person: str,
    records,
    baseline_dates,
    rms_config: LongitudinalConfig,
    ds_config: DsPcdConfig,
) -> tuple[ProfileStore, Any, DsPcdBaseline]:
    store = ProfileStore(directory, person)
    for record in records:
        store.append_observation(_observation_from_record(person, record))
    rms_baseline = fit_rms_baseline(
        person, {record.date: record for record in records}, baseline_dates, rms_config,
    )
    store.save_baseline(rms_baseline)
    ds_baseline = fit_baseline(
        records, person, ds_config, baseline_dates=baseline_dates,
    )
    store.save_ds_pcd_baseline(ds_baseline)
    return store, rms_baseline, ds_baseline


def _observation_from_record(person: str, record):
    return DailyObservation.from_dict({
        "person_id": person,
        "observed_at": record.date,
        "features": dict(record.features),
        "feature_se": dict(record.feature_se),
        "quality": {
            "valid_frames": 0, "valid_steps": 0,
            "walking_duration_s": 0.0, "mean_confidence": 0.0,
        },
        "source": "stage2-verify",
    })


def offline_online_check(
    stage2: Path,
    bundle: Path,
    *,
    sample_count: int = 6,
) -> dict[str, Any]:
    cohort = load_stage1(bundle)
    precision = pd.read_csv(stage2 / "bout_precision.csv", dtype={"participantid": str})
    precision["day"] = pd.to_datetime(precision["day"]).dt.normalize()
    rms_config = wearable_config()
    summary = json.loads((stage2 / "replay_summary.json").read_text(encoding="utf-8"))
    results = []
    with __import__("tempfile").TemporaryDirectory() as directory:
        root = Path(directory)
        for mode in MODES:
            config = DsPcdConfig.from_dict(summary["modes"][mode]["config"])
            records_by_person = build_records(
                cohort, precision if mode == "session_se" else None,
                precision_mode=mode,
            )
            scores = pd.read_csv(stage2 / f"scores_{mode}.csv", dtype={"participantid": str})
            scores["landmark_day"] = pd.to_datetime(scores["landmark_day"]).dt.date.astype(str)
            step = max(1, len(scores) // sample_count)
            sampled = scores.iloc[::step].head(sample_count)
            for row in sampled.itertuples(index=False):
                person = str(row.participantid)
                baseline_rows = cohort.baseline_days.loc[
                    (cohort.baseline_days["participantid"].eq(person))
                    & (cohort.baseline_days["landmark_day"].eq(pd.Timestamp(row.landmark_day)))
                ]
                baseline_dates = tuple(sorted(
                    pd.Timestamp(value).date().isoformat()
                    for value in baseline_rows["baseline_day"]
                ))
                store, rms_baseline, ds_baseline = _profile_for_person(
                    root / mode / person, person, records_by_person[person],
                    baseline_dates, rms_config, config,
                )
                observations = {
                    item.observation_date: item for item in store.load_observations()
                }
                current = observations[row.landmark_day]
                service = LongitudinalRiskService(
                    store, rms_config, ds_pcd_config=config,
                )
                result = service.evaluate(
                    person, current, as_of=row.landmark_day, persist=False,
                )
                payload = result.to_dict()
                ds = payload["ds_pcd"] or {}
                differences = {
                    "rms_distance": abs((payload["distance"] or 0.0) - row.rms_distance),
                    "T": abs((ds.get("T") or 0.0) - (row.T or 0.0)),
                    "q": abs((ds.get("q") or 0.0) - (row.q or 0.0)),
                    "cusum": abs((ds.get("cusum") or 0.0) - (row.cusum or 0.0)),
                }
                results.append({
                    "mode": mode,
                    "participantid": person,
                    "landmark_day": row.landmark_day,
                    "service_status": payload["status"],
                    "ds_status": ds.get("status"),
                    "max_difference": max(differences.values()),
                    "differences": differences,
                    "match": max(differences.values()) < 1e-9,
                })
    return {
        "samples": results,
        "all_match": all(item["match"] for item in results),
    }


def no_future_information_check(
    stage2: Path, bundle: Path, *, samples: int = 2
) -> dict[str, Any]:
    """追加预测日之后的异常观测，预测日结果不变。"""
    cohort = load_stage1(bundle)
    precision = pd.read_csv(stage2 / "bout_precision.csv", dtype={"participantid": str})
    precision["day"] = pd.to_datetime(precision["day"]).dt.normalize()
    config = ds_pcd_config(precision_mode="none", null_samples=2000, quantile_points=200)
    rms_config = wearable_config()
    records_by_person = build_records(cohort, None, precision_mode="none")
    scores = pd.read_csv(stage2 / "scores_none.csv", dtype={"participantid": str})
    scores["landmark_day"] = pd.to_datetime(scores["landmark_day"]).dt.date.astype(str)
    step = max(1, len(scores) // samples)
    payloads = []
    with __import__("tempfile").TemporaryDirectory() as directory:
        root = Path(directory)
        for index, row in enumerate(scores.iloc[::step].head(samples).itertuples(index=False)):
            person = str(row.participantid)
            baseline_rows = cohort.baseline_days.loc[
                (cohort.baseline_days["participantid"].eq(person))
                & (cohort.baseline_days["landmark_day"].eq(pd.Timestamp(row.landmark_day)))
            ]
            baseline_dates = tuple(sorted(
                pd.Timestamp(value).date().isoformat()
                for value in baseline_rows["baseline_day"]
            ))
            store, _, _ = _profile_for_person(
                root / f"base{index}", person, records_by_person[person],
                baseline_dates, rms_config, config,
            )
            observations = {item.observation_date: item for item in store.load_observations()}
            service = LongitudinalRiskService(store, rms_config, ds_pcd_config=config)
            before = service.evaluate(
                person, observations[row.landmark_day], as_of=row.landmark_day, persist=False,
            ).to_dict()
            # 在预测日之后追加异常观测（步速骤降、步频升高、跨步周期缩短）
            later = date.fromisoformat(row.landmark_day) + timedelta(days=3)
            for offset in range(2):
                day = later + timedelta(days=offset)
                store.append_observation(DailyObservation.from_dict({
                    "person_id": person,
                    "observed_at": day.isoformat(),
                    "features": {
                        "walking_speed_m_s": 0.45,
                        "cadence_spm": 118.0,
                        "stride_duration_s": 0.8,
                    },
                    "quality": {
                        "valid_frames": 0, "valid_steps": 0,
                        "walking_duration_s": 0.0, "mean_confidence": 0.0,
                    },
                    "source": "stage2-anomaly",
                }))
            after = service.evaluate(
                person, observations[row.landmark_day], as_of=row.landmark_day, persist=False,
            ).to_dict()
            payloads.append({
                "participantid": person,
                "landmark_day": row.landmark_day,
                "identical": before == after,
                "before": {key: before[key] for key in ("distance", "ds_pcd")},
                "after": {key: after[key] for key in ("distance", "ds_pcd")},
            })
    return {"cases": payloads, "all_identical": all(item["identical"] for item in payloads)}


def determinism(stage2: Path, work: Path) -> dict[str, Any]:
    rebuild = work / "replay"
    if rebuild.exists():
        shutil.rmtree(rebuild)
    rebuild.mkdir(parents=True, exist_ok=True)
    for name in ("bout_precision.csv", "bout_precision_notes.json"):
        shutil.copy2(stage2 / name, rebuild / name)
    command = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "replay_mobilise_stage2.py"),
        "--out", str(rebuild),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    compared = {}
    for name in (
        "scores_session_se.csv", "scores_none.csv",
        "score_windows_session_se.csv", "score_windows_none.csv",
        "excluded_session_se.csv", "excluded_none.csv",
    ):
        first, second = stage2 / name, rebuild / name
        compared[name] = {
            "identical": first.is_file() and second.is_file() and _md5(first) == _md5(second),
            "md5": _md5(first) if first.is_file() else None,
        }
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stderr_tail": completed.stderr.strip().splitlines()[-5:],
        "files": compared,
        "all_identical": all(item["identical"] for item in compared.values()),
    }


def simulation_checks(simulation: Path) -> dict[str, Any]:
    payload = json.loads((simulation / "simulation.json").read_text(encoding="utf-8"))
    checks = {}
    for mode, result in payload["results"].items():
        null_c = result["null"]["mean_cusum_by_day"]
        decline_c = result["decline"]["mean_cusum_by_day"]
        checks[mode] = {
            "false_alarm_rate": result["null"]["detection_rate"],
            "false_alarm_within_0_12": result["null"]["detection_rate"] <= 0.12,
            "decline_detection": result["decline"]["detection_rate"],
            "reverse_detection": result["reverse_direction"]["detection_rate"],
            "direction_separation": (
                result["decline"]["detection_rate"]
                - result["reverse_direction"]["detection_rate"]
            ),
            "cumulative_increases": (
                sum(decline_c[-5:]) / 5 > sum(decline_c[:5]) / 5
            ),
            "null_c_near_zero": max(null_c) < 2.0,
            "reverse_c_near_zero": max(result["reverse_direction"]["mean_cusum_by_day"]) < 1.0,
        }
    threshold = json.loads((simulation / "threshold.json").read_text(encoding="utf-8"))
    return {
        "modes": checks,
        "threshold": threshold,
        "all_pass": all(
            item["false_alarm_within_0_12"]
            and item["direction_separation"] >= 0.2
            and item["cumulative_increases"]
            and item["null_c_near_zero"]
            for item in checks.values()
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 2 核验")
    parser.add_argument("--stage2", default=str(STAGE2))
    parser.add_argument("--bundle", default=str(BUNDLE))
    parser.add_argument(
        "--simulation-dir", default=None,
        help="仿真产物目录；默认 <stage2>/simulation",
    )
    parser.add_argument("--skip-rebuild", action="store_true")
    args = parser.parse_args(argv)
    stage2 = Path(args.stage2)
    bundle = Path(args.bundle)
    manifest = json.loads((stage2 / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((stage2 / "replay_summary.json").read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage2": str(stage2),
        "bundle": str(bundle),
    }
    structural = []
    baseline_days = pd.read_csv(bundle / "baseline_days.csv", dtype={"participantid": str})
    baseline_days["baseline_day"] = pd.to_datetime(baseline_days["baseline_day"])
    baseline_days["landmark_day"] = pd.to_datetime(baseline_days["landmark_day"])
    for mode in MODES:
        scores = pd.read_csv(stage2 / f"scores_{mode}.csv", dtype={"participantid": str})
        scores["landmark_day"] = pd.to_datetime(scores["landmark_day"])
        windows = pd.read_csv(
            stage2 / f"score_windows_{mode}.csv", dtype={"participantid": str}
        )
        merged_baseline = baseline_days.merge(
            scores[["participantid", "landmark_day"]],
            on=["participantid", "landmark_day"], how="inner",
        )
        baseline_before_landmark = bool(
            (merged_baseline["baseline_day"] < merged_baseline["landmark_day"]).all()
        )
        structural.append({
            "mode": mode,
            "score_rows": int(len(scores)),
            "score_people": int(scores["participantid"].nunique()),
            "score_key_unique": not scores.duplicated(["participantid", "landmark_day"]).any(),
            "window_rows": int(len(windows)),
            "window_key_unique": not windows.duplicated(
                ["participantid", "landmark_day", "horizon_days"]
            ).any(),
            "all_windows_eligible": bool(windows["status"].eq("eligible").all()),
            "event_7d": int(windows.loc[
                windows["horizon_days"].eq(7)
                & windows["event"].astype(str).str.lower().eq("true")
            ].shape[0]),
            "event_30d": int(windows.loc[
                windows["horizon_days"].eq(30)
                & windows["event"].astype(str).str.lower().eq("true")
            ].shape[0]),
            "all_ds_scored": bool(scores["ds_status"].eq("scored").all()),
            "baseline_before_landmark": baseline_before_landmark,
            "ds_baseline_days_median": float(scores["ds_baseline_days"].median()),
            "ds_outlier_excluded_landmarks": int(
                (pd.to_numeric(scores["ds_outlier_excluded_days"], errors="coerce") > 0).sum()
            ),
            "config_gap_reset_days": summary["modes"][mode]["config"]["gap_reset_days"],
            "config_k_ref": summary["modes"][mode]["config"]["k_ref"],
        })
    payload["structural"] = structural
    payload["structural_all_pass"] = all(
        item["score_key_unique"] and item["window_key_unique"]
        and item["all_windows_eligible"] and item["all_ds_scored"]
        and item["config_gap_reset_days"] == 7 and item["config_k_ref"] == 0.5
        and item["baseline_before_landmark"]
        for item in structural
    )
    payload["offline_online"] = offline_online_check(stage2, bundle)
    payload["no_future_information"] = no_future_information_check(stage2, bundle)
    simulation_dir = (
        Path(args.simulation_dir) if args.simulation_dir else stage2 / "simulation"
    )
    payload["simulation"] = simulation_checks(simulation_dir)
    if not args.skip_rebuild:
        payload["determinism"] = determinism(stage2, stage2 / "work")
    precision_notes = json.loads(
        (stage2 / "bout_precision_notes.json").read_text(encoding="utf-8")
    )
    payload["precision"] = {
        "mode": "session_se",
        "fallback_days": {
            feature: precision_notes["precision"][feature]["fallback_days"]
            for feature in precision_notes["precision"]
            if isinstance(precision_notes["precision"][feature], dict)
        },
        "available_days": {
            feature: precision_notes["precision"][feature]["available_days"]
            for feature in precision_notes["precision"]
            if isinstance(precision_notes["precision"][feature], dict)
        },
        "all_baselines_scored_with_fallback": all(
            item["all_ds_scored"] for item in structural if item["mode"] == "session_se"
        ),
    }
    (stage2 / "verification.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 阶段 2 核验记录",
        "",
        f"- 时间：{payload['verified_at_utc']}",
        f"- 产物目录：`{stage2}`",
        "",
        "## 结构检查",
        "",
        "| 模式 | 评分行 | 人数 | 键唯一 | 基线早于预测日 | 时窗行 | 7 天事件 | 30 天事件 | DS 全部可评 |",
        "|---|---:|---:|---|---|---:|---:|---:|---|",
    ]
    for item in structural:
        lines.append(
            f"| {item['mode']} | {item['score_rows']} | {item['score_people']} | "
            f"{'是' if item['score_key_unique'] else '否'} | "
            f"{'是' if item['baseline_before_landmark'] else '否'} | {item['window_rows']} | "
            f"{item['event_7d']} | {item['event_30d']} | "
            f"{'是' if item['all_ds_scored'] else '否'} |"
        )
    lines += ["", "## 离线与在线一致", ""]
    for item in payload["offline_online"]["samples"]:
        lines.append(
            f"- {item['mode']} 受试者 {item['participantid']} 预测日 {item['landmark_day']}："
            f"最大差异 {item['max_difference']:.2e}"
        )
    lines += [
        f"- 全部一致：{'是' if payload['offline_online']['all_match'] else '否'}",
        "", "## 无未来信息", "",
    ]
    for item in payload["no_future_information"]["cases"]:
        lines.append(
            f"- 受试者 {item['participantid']} 预测日 {item['landmark_day']}："
            f"{'一致' if item['identical'] else '不一致'}"
        )
    lines += [
        f"- 全部一致：{'是' if payload['no_future_information']['all_identical'] else '否'}",
        "", "## 仿真行为", "",
        "| 模式 | 零假设误报 | 方向一致检出 | 反向检出 | C 累积上升 | 零假设 C 接近 0 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for mode, item in payload["simulation"]["modes"].items():
        lines.append(
            f"| {mode} | {item['false_alarm_rate']:.3f} | {item['decline_detection']:.3f} | "
            f"{item['reverse_detection']:.3f} | "
            f"{'是' if item['cumulative_increases'] else '否'} | "
            f"{'是' if item['null_c_near_zero'] else '否'} |"
        )
    lines += [
        "",
        f"- 阈值 h={payload['simulation']['threshold']['threshold']}，"
        f"来源：{payload['simulation']['threshold']['source']}",
        "",
    ]
    if "determinism" in payload:
        det = payload["determinism"]
        lines += [
            "## 重放确定性",
            "",
            f"- 命令：`{det['command']}`，退出码 {det['returncode']}",
            f"- 逐文件一致：{'是' if det['all_identical'] else '否'}",
            "",
        ]
    (stage2 / "VERIFICATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "structural_all_pass": payload["structural_all_pass"],
        "offline_online_all_match": payload["offline_online"]["all_match"],
        "no_future_information_all_identical": payload["no_future_information"]["all_identical"],
        "simulation_all_pass": payload["simulation"]["all_pass"],
        "determinism_all_identical": payload.get("determinism", {}).get("all_identical"),
    }, ensure_ascii=False, indent=2))
    ok = (
        payload["structural_all_pass"]
        and payload["offline_online"]["all_match"]
        and payload["no_future_information"]["all_identical"]
        and payload["simulation"]["all_pass"]
        and payload.get("determinism", {}).get("all_identical", True)
    )
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
