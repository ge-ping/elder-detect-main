"""阶段 5：修正逐日标准误的因果冻结与 Cox 目标函数一致性，并给出证据。

旧版（`20261114-main`）是历史对照。本脚本不改变旧目录，只读原始包与旧产物，
在 `output/ds_pcd_mobilise/stage5/` 写入协议、修复差异、核验与中文报告。

关键检查：
1. 复现旧版“删除预测日之后的步行段后，历史标准误/CUSUM 改变”的反例；
2. 从步行段入口按日累积与完整档案按日截断的结果一致（前缀冻结）；
3. 追加未来异常段、跨访视空档、没有既往回退数据的日期；
4. 旧/新 Cox 目标函数与有限差分梯度；
5. 访视内众数日期校正在前缀规则下是否稳定。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import md5
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.cox_fit import _partial_likelihood as new_partial_likelihood  # noqa: E402
from mobilise_d import sources  # noqa: E402
from mobilise_d.observations import (  # noqa: E402
    FEATURES,
    build_records,
    ds_pcd_config,
    load_stage1,
    wearable_config,
)
from mobilise_d.precision import (  # noqa: E402
    FEATURE_SPECS,
    PrecisionConfig,
    _apply_pooled_fallback,
    apply_causal_pooled_fallback,
    daily_direct_stats,
    daily_precision,
)
from scripts.replay_mobilise_stage2 import _score_person  # noqa: E402

OLD_RUN = PROJECT_ROOT.parents[1] / "output" / "ds_pcd_mobilise" / "20261114-main"
OUT = PROJECT_ROOT.parents[1] / "output" / "ds_pcd_mobilise" / "stage5"
DEFAULT_CACHE = PROJECT_ROOT.parents[1] / "anti-fall" / "_datasets" / "Mobilise-D-CVS" / "21060991"
TOLERANCE = 1e-9
ALARM = 5.2
SCORING_VERSION = "scoring-v2-causal-precision"
COX_VERSION = "cox-v2-penalized-objective"
CODE_FILES = (
    "mobilise_d/precision.py", "mobilise_d/observations.py",
    "scripts/replay_mobilise_stage2.py", "scripts/collect_longitudinal.py",
    "scripts/build_personal_baseline.py", "scripts/predict_future_fall.py",
    "longitudinal/cox_fit.py", "longitudinal/service.py", "longitudinal/schema.py",
    "longitudinal/storage.py", "longitudinal/survival.py", "longitudinal/ds_pcd.py",
    "longitudinal/uncertainty.py", "scripts/analyse_mobilise_stage3.py",
    "scripts/seal_mobilise_stage4.py", "scripts/verify_mobilise_stage5.py",
    "tests/test_causal_precision.py", "tests/test_cox_fit.py",
    "tests/test_mobilise_stage2_adapter.py",
)


def _md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_old_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 旧版反例
# ---------------------------------------------------------------------------


def old_counterfactual(
    cohort, scores: pd.DataFrame, precision: pd.DataFrame,
) -> dict[str, Any]:
    """对每个预测日删除其后步行段日汇总，重算回退标准误并比较历史。"""
    valid = cohort.daily.loc[
        cohort.daily["has_core_gait"], ["participantid", "visit_number", "day"]
    ]
    data_config = ds_pcd_config()
    rms_config = wearable_config()
    exposed: list[str] = []
    changed_c = 0
    status_changes = 0
    n_valid_future = 0
    examples: list[dict[str, Any]] = []
    cutoff_c: dict[tuple[str, str], float] = {}
    for person, group in scores.groupby("participantid"):
        person_precision = precision.loc[precision["participantid"].eq(person)]
        person_valid = valid.loc[valid["participantid"].eq(person)]
        person_landmarks = cohort.landmarks.loc[cohort.landmarks["participantid"].eq(person)]
        person_dates = cohort.baseline_days.loc[
            cohort.baseline_days["participantid"].eq(person)
        ]
        for score in group.itertuples(index=False):
            landmark = pd.Timestamp(score.landmark_day)
            n_valid_future += int(
                score.n_valid_days != int((person_valid["day"] <= landmark).sum())
            )
            prefix = person_precision.loc[person_precision["day"] <= landmark].copy()
            future = person_precision.loc[person_precision["day"] > landmark]
            past = person_valid.loc[person_valid["day"] <= landmark].merge(
                prefix, validate="one_to_one",
            )
            features = [
                feature for feature in FEATURES
                if (past[f"method_{feature}"] == "pooled_sd").any()
                and ((future[f"n_{feature}"] >= 5) & (future[f"sd_{feature}"] > 0)).any()
            ]
            if not features:
                continue
            for feature in FEATURES:
                prefix.loc[prefix[f"method_{feature}"] == "pooled_sd", f"se_{feature}"] = np.nan
            clean = _apply_pooled_fallback(prefix, PrecisionConfig())
            after = person_valid.loc[person_valid["day"] <= landmark].merge(
                clean, validate="one_to_one",
            )
            columns = [f"se_{feature}" for feature in features]
            if np.allclose(
                after[columns], past[columns], equal_nan=True, rtol=1e-12, atol=1e-12,
            ):
                continue
            exposed.append(person)
            records = build_records(
                cohort, clean, precision_mode="session_se", person_ids=[person],
            )[person]
            dates = tuple(sorted(
                pd.Timestamp(value).date().isoformat()
                for value in person_dates.loc[
                    person_dates["landmark_day"].eq(landmark), "baseline_day"
                ]
            ))
            current = _score_person(
                person, person_landmarks.loc[person_landmarks["landmark_day"].eq(landmark)],
                [record for record in records if record.date <= score.landmark_day],
                dates, rms_config, data_config, ALARM, "session_se",
            )[0]
            status_changes += int(current["ds_status"] != score.ds_status)
            if (
                pd.notna(current["cusum"]) and pd.notna(score.cusum)
                and abs(float(current["cusum"]) - float(score.cusum)) > TOLERANCE
            ):
                cutoff_c[(person, score.landmark_day)] = float(current["cusum"])
                if not examples:
                    control = build_records(
                        cohort, person_precision, precision_mode="session_se",
                        person_ids=[person],
                    )[person]
                    same = _score_person(
                        person,
                        person_landmarks.loc[person_landmarks["landmark_day"].eq(landmark)],
                        [record for record in control if record.date <= score.landmark_day],
                        dates, rms_config, data_config, ALARM, "session_se",
                    )[0]
                    examples.append({
                        "participantid": person,
                        "landmark_day": score.landmark_day,
                        "features": features,
                        "full_input_c": float(score.cusum),
                        "cutoff_input_c": float(current["cusum"]),
                        "full_control_c": float(same["cusum"]),
                    })
                changed_c += 1
    return {
        "prediction_days": int(len(scores)),
        "prior_se_changes_if_future_removed": len(exposed),
        "people_with_se_changes": len(set(exposed)),
        "finite_c_changes_gt_tolerance": changed_c,
        "ds_status_changes": status_changes,
        "n_valid_days_counts_future": n_valid_future,
        "example": examples[0] if examples else None,
        "cutoff_c": cutoff_c,
        "exposed_people": sorted(set(exposed)),
    }


# ---------------------------------------------------------------------------
# 扫描、映射稳定性、因果表
# ---------------------------------------------------------------------------


def scan_direct_stats(
    cache: Path, people: list[str], out: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """扫描步行段，导出直接统计；同时检查众数日期校正的前缀稳定性。"""
    from mobilise_d.precision import scan_bouts

    bout_archive = cache / sources.BOUTS.name
    started = time.time()
    scan = scan_bouts(bout_archive, people)
    direct = daily_direct_stats(scan.frame, config=PrecisionConfig())
    out.mkdir(parents=True, exist_ok=True)
    direct_path = out / "direct_stats.csv"
    direct.to_csv(
        direct_path, index=False, encoding="utf-8", float_format="%.17g",
    )
    # 保存原始行，便于后续只读复算（约 1.2M 行，压缩后数十 MB）
    scan.frame.to_csv(
        out / "raw_rows_all.csv.gz", index=False, encoding="utf-8", compression="gzip",
    )
    mapping = mapping_stability(scan.frame)
    notes = {
        **scan.notes,
        "archive": {"name": sources.BOUTS.name, "md5": sources.BOUTS.md5,
                    "path": str(bout_archive)},
        "people_requested": len(people),
        "elapsed_seconds": round(time.time() - started, 1),
        "direct_stats_md5": _md5(direct_path),
        "mapping": mapping,
    }
    _write_json(out / "scan_notes.json", notes)
    return notes, direct, scan.frame


def mapping_stability(frame: pd.DataFrame) -> dict[str, Any]:
    """比较访视内全量众数与逐日前缀众数的日期映射（按 raw_day/wbday 分组）。"""
    affected = []
    visits = 0
    rows = 0
    affected_rows = 0
    for (person, visit), group in frame.groupby(["participantid", "visit_number"], sort=False):
        visits += 1
        rows += len(group)
        counts: dict[Any, int] = {}
        prefix_rows = 0
        for raw_day, block in group.groupby("raw_day", sort=True):
            for origin, count in block["origin"].value_counts().items():
                counts[origin] = counts.get(origin, 0) + int(count)
            prefix_rows += len(block)
            top = max(counts.values())
            candidates = [origin for origin, value in counts.items() if value == top]
            prefix_mode = min(candidates)
            for wbday, sub in block.groupby("wbday"):
                prefix_day = prefix_mode + pd.Timedelta(days=int(wbday) - 1)
                full_day = pd.Timestamp(sub["day"].iloc[0]).normalize()
                if pd.Timestamp(prefix_day).normalize() != full_day:
                    affected.append({
                        "participantid": str(person),
                        "visit_number": str(visit),
                        "raw_day": pd.Timestamp(raw_day).date().isoformat(),
                        "wbday": int(wbday),
                        "full_day": full_day.date().isoformat(),
                        "prefix_day": pd.Timestamp(prefix_day).date().isoformat(),
                        "rows_in_day": int(len(block)),
                        "rows_in_group": int(len(sub)),
                        "prefix_rows": prefix_rows,
                    })
                    affected_rows += len(sub)
    days = sorted({(item["participantid"], item["full_day"]) for item in affected})
    return {
        "visits_checked": visits,
        "rows_checked": rows,
        "affected_rows": affected_rows,
        "affected_groups": len(affected),
        "affected_days": len(days),
        "affected_person_visits": len({(i["participantid"], i["visit_number"]) for i in affected}),
        "examples": affected[:20],
        "affected_day_keys": sorted({
            *(f"{person}|{day}" for person, day in days),
            *(f"{item['participantid']}|{item['prefix_day']}" for item in affected),
        }),
    }


def scored_prefix_days(cohort, scores: pd.DataFrame) -> set[str]:
    valid = cohort.daily.loc[
        cohort.daily["has_core_gait"], ["participantid", "day"]
    ]
    keys = set()
    for score in scores.itertuples(index=False):
        person = score.participantid
        landmark = pd.Timestamp(score.landmark_day)
        for day in valid.loc[valid["participantid"].eq(person), "day"]:
            if day <= landmark:
                keys.add(f"{person}|{day.date().isoformat()}")
    for row in cohort.baseline_days.itertuples(index=False):
        if str(row.participantid) in set(scores["participantid"]):
            keys.add(f"{row.participantid}|{pd.Timestamp(row.baseline_day).date().isoformat()}")
    return keys


# ---------------------------------------------------------------------------
# 评分比较
# ---------------------------------------------------------------------------


def build_causal_records(cohort, causal: pd.DataFrame) -> dict[str, list]:
    return build_records(cohort, causal, precision_mode="session_se")


def _same_value(left, right, tolerance: float) -> bool:
    if left is None or right is None:
        return (left is None and right is None) or (
            left is None and isinstance(right, float) and pd.isna(right)
        ) or (right is None and isinstance(left, float) and pd.isna(left))
    if isinstance(left, (int, float, np.floating)) or isinstance(right, (int, float, np.floating)):
        if pd.isna(left) and pd.isna(right):
            return True
        if pd.isna(left) or pd.isna(right):
            return False
        return abs(float(left) - float(right)) <= tolerance
    return str(left) == str(right)


def score_prefix_check(
    cohort, records_by_person, scores_old: pd.DataFrame,
    old_cutoff: dict[tuple[str, str], float],
) -> dict[str, Any]:
    rms_config = wearable_config()
    data_config = ds_pcd_config()
    mismatches: list[dict[str, Any]] = []
    changed_vs_old = 0
    changed_vs_cutoff = 0
    status_changes = 0
    status_examples: list[dict[str, Any]] = []
    se_missing_changes = 0
    n_valid_mismatches = 0
    compared = 0
    gap_example = None
    status_counts: dict[str, int] = {}
    fields = (
        "rms_distance", "T", "q", "cusum", "mask", "ds_status",
        "n_valid_days", "n_valid_days_before_landmark",
        "ds_baseline_days", "ds_baseline_end", "rms_baseline_end",
        "rms_features_active", "trend_7d", "persistent_deviation_days",
    )
    for score in scores_old.itertuples(index=False):
        person = score.participantid
        day = score.landmark_day
        records = records_by_person[person]
        landmarks = cohort.landmarks.loc[
            (cohort.landmarks["participantid"].eq(person))
            & (cohort.landmarks["landmark_day"].eq(pd.Timestamp(day)))
        ]
        dates = tuple(sorted(
            pd.Timestamp(value).date().isoformat()
            for value in cohort.baseline_days.loc[
                (cohort.baseline_days["participantid"].eq(person))
                & (cohort.baseline_days["landmark_day"].eq(pd.Timestamp(day))),
                "baseline_day",
            ]
        ))
        full = _score_person(
            person, landmarks, records, dates, rms_config, data_config,
            ALARM, "session_se",
        )[0]
        truncated = _score_person(
            person, landmarks, [record for record in records if record.date <= day],
            dates, rms_config, data_config, ALARM, "session_se",
        )[0]
        compared += 1
        status_counts[full.get("ds_status", "unknown")] = (
            status_counts.get(full.get("ds_status", "unknown"), 0) + 1
        )
        for key in fields:
            if not _same_value(full.get(key), truncated.get(key), TOLERANCE):
                mismatches.append({
                    "participantid": person, "landmark_day": day, "field": key,
                    "full": full.get(key), "truncated": truncated.get(key),
                })
        old_missing = "" if pd.isna(score.se_missing) else str(score.se_missing)
        if old_missing != str(full.get("se_missing") or ""):
            se_missing_changes += 1
        inclusive = sum(1 for record in records if record.date <= day)
        strict = sum(1 for record in records if record.date < day)
        if full.get("n_valid_days") != inclusive:
            n_valid_mismatches += 1
        if full.get("n_valid_days_before_landmark") != strict:
            n_valid_mismatches += 1
        if pd.notna(full.get("cusum")) and pd.notna(score.cusum):
            if abs(float(full["cusum"]) - float(score.cusum)) > TOLERANCE:
                changed_vs_old += 1
        if (person, day) in old_cutoff and pd.notna(full.get("cusum")):
            if abs(float(full["cusum"]) - old_cutoff[(person, day)]) > TOLERANCE:
                changed_vs_cutoff += 1
        if full.get("ds_status") != score.ds_status:
            status_changes += 1
            if len(status_examples) < 10:
                status_examples.append({
                    "participantid": person, "landmark_day": day,
                    "old_status": score.ds_status, "new_status": full.get("ds_status"),
                    "message": full.get("ds_message"),
                })
        if gap_example is None and int(full.get("cusum_resets") or 0) > 0:
            gap_example = {
                "participantid": person, "landmark_day": day,
                "cusum_resets": int(full["cusum_resets"]),
                "cusum": float(full["cusum"]),
            }
    return {
        "landmarks_compared": compared,
        "prefix_mismatches": len(mismatches),
        "prefix_mismatch_examples": mismatches[:10],
        "causal_c_changes_vs_old_sealed": changed_vs_old,
        "causal_c_changes_vs_old_cutoff": changed_vs_cutoff,
        "status_changes_vs_old": status_changes,
        "status_change_examples": status_examples,
        "session_status_counts": status_counts,
        "se_missing_changes_vs_old": se_missing_changes,
        "n_valid_count_mismatches": n_valid_mismatches,
        "gap_example": gap_example,
    }


def se_change_counts(
    cohort, causal: pd.DataFrame, old_precision: pd.DataFrame,
    scores_old: pd.DataFrame,
) -> dict[str, Any]:
    new = causal.assign(day=causal["day"].astype(str)).drop_duplicates(
        ["participantid", "day"]
    ).set_index(["participantid", "day"])
    old = old_precision.assign(
        day=pd.to_datetime(old_precision["day"]).dt.date.astype(str)
    ).drop_duplicates(["participantid", "day"]).set_index(
        ["participantid", "day"]
    )
    changed_landmarks = 0
    changed_baseline = 0
    changed_landmark_day = 0
    per_feature = {feature: 0 for feature in FEATURE_SPECS}
    examples = []
    for score in scores_old.itertuples(index=False):
        person = str(score.participantid)
        landmark = score.landmark_day
        baseline_days = [
            pd.Timestamp(value).date().isoformat()
            for value in cohort.baseline_days.loc[
                (cohort.baseline_days["participantid"].eq(person))
                & (cohort.baseline_days["landmark_day"].eq(pd.Timestamp(landmark))),
                "baseline_day",
            ]
        ]
        days = baseline_days + [landmark]
        any_change = False
        for day in days:
            key = (person, day)
            if key not in new.index or key not in old.index:
                continue
            for feature in FEATURE_SPECS:
                left = new.loc[key, f"se_{feature}"]
                right = old.loc[key, f"se_{feature}"]
                if pd.isna(left) and pd.isna(right):
                    continue
                if pd.isna(left) or pd.isna(right) or abs(float(left) - float(right)) > TOLERANCE:
                    per_feature[feature] += 1
                    any_change = True
                    if not examples:
                        examples.append({
                            "participantid": person, "day": day, "feature": feature,
                            "old_se": None if pd.isna(right) else float(right),
                            "new_se": None if pd.isna(left) else float(left),
                            "old_method": old.loc[key, f"method_{feature}"],
                            "new_method": new.loc[key, f"se_source_{feature}"],
                        })
        if any_change:
            changed_landmarks += 1
        if any(
            (person, day) in new.index and (person, day) in old.index
            and any(
                (pd.isna(new.loc[(person, day), f"se_{f}"]) != pd.isna(old.loc[(person, day), f"se_{f}"]))
                or (
                    pd.notna(new.loc[(person, day), f"se_{f}"]) and pd.notna(old.loc[(person, day), f"se_{f}"])
                    and abs(float(new.loc[(person, day), f"se_{f}"]) - float(old.loc[(person, day), f"se_{f}"])) > TOLERANCE
                )
                for f in FEATURE_SPECS
            )
            for day in baseline_days
        ):
            changed_baseline += 1
        if (person, landmark) in new.index and (person, landmark) in old.index:
            if any(
                (pd.isna(new.loc[(person, landmark), f"se_{f}"]) != pd.isna(old.loc[(person, landmark), f"se_{f}"]))
                or (
                    pd.notna(new.loc[(person, landmark), f"se_{f}"]) and pd.notna(old.loc[(person, landmark), f"se_{f}"])
                    and abs(float(new.loc[(person, landmark), f"se_{f}"]) - float(old.loc[(person, landmark), f"se_{f}"])) > TOLERANCE
                )
                for f in FEATURE_SPECS
            ):
                changed_landmark_day += 1
    return {
        "landmarks_with_any_se_change": changed_landmarks,
        "landmarks_with_baseline_se_change": changed_baseline,
        "landmarks_with_landmark_day_se_change": changed_landmark_day,
        "feature_changes": per_feature,
        "examples": examples[:5],
    }


# ---------------------------------------------------------------------------
# 原始步行段入口的逐日累积
# ---------------------------------------------------------------------------


def raw_incremental_check(
    raw: pd.DataFrame, causal: pd.DataFrame, person: str,
) -> dict[str, Any]:
    subset = raw.loc[raw["participantid"].eq(person)].copy()
    subset["day"] = pd.to_datetime(subset["day"])
    mismatch = 0
    checked = 0
    for day in sorted(subset["day"].unique()):
        cumulative = subset.loc[subset["day"] <= day]
        table = apply_causal_pooled_fallback(
            daily_direct_stats(cumulative, config=PrecisionConfig()), PrecisionConfig(),
        )
        row = table.loc[table["day"].eq(pd.Timestamp(day).date().isoformat())]
        full = causal.loc[
            causal["participantid"].eq(person)
            & causal["day"].eq(pd.Timestamp(day).date().isoformat())
        ]
        if row.empty or full.empty:
            mismatch += 1
            continue
        checked += 1
        for feature in FEATURE_SPECS:
            left = row.iloc[0][f"se_{feature}"]
            right = full.iloc[0][f"se_{feature}"]
            if pd.isna(left) and pd.isna(right):
                continue
            if pd.isna(left) or pd.isna(right) or abs(float(left) - float(right)) > TOLERANCE:
                mismatch += 1
                break
    return {"person": person, "days_checked": checked, "mismatches": mismatch}


def future_anomaly_check(
    raw: pd.DataFrame, causal: pd.DataFrame, person: str,
) -> dict[str, Any]:
    subset = raw.loc[raw["participantid"].eq(person)].copy()
    subset["day"] = pd.to_datetime(subset["day"])
    last_day = subset["day"].max()
    future_day = last_day + pd.Timedelta(days=60)
    rows = []
    for index in range(80):
        rows.append({
            "participantid": person, "datatype": "wb", "visit_number": "t9",
            "wbday": 1, "day": future_day, "duration": 40.0,
            "averagecadence": 150.0 + index * 0.1,
            "averagestridespeed": 0.1,
            "averagestrideduration": 0.6,
            "initialcontact_event_number": 40.0,
        })
    extended = pd.concat([subset, pd.DataFrame(rows)], ignore_index=True)
    table = apply_causal_pooled_fallback(
        daily_direct_stats(extended, config=PrecisionConfig()), PrecisionConfig(),
    )
    table["day"] = pd.to_datetime(table["day"])
    compare = table.loc[table["day"] <= last_day].copy()
    compare["day"] = compare["day"].dt.date.astype(str)
    original = causal.loc[causal["participantid"].eq(person)].copy()
    original["day"] = original["day"].astype(str)
    mismatch = 0
    merged = original.merge(
        compare, on=["participantid", "day"], suffixes=("_full", "_extended"),
    )
    for feature in FEATURE_SPECS:
        left = merged[f"se_{feature}_full"]
        right = merged[f"se_{feature}_extended"]
        mismatch += int(sum(
            not (
                (pd.isna(a) and pd.isna(b))
                or (pd.notna(a) and pd.notna(b) and abs(float(a) - float(b)) <= TOLERANCE)
            )
            for a, b in zip(left, right)
        ))
    return {
        "person": person,
        "future_rows_added": len(rows),
        "compared_rows": int(len(merged)),
        "mismatches": int(mismatch),
    }


def no_prior_example(causal: pd.DataFrame) -> dict[str, Any] | None:
    for feature in FEATURE_SPECS:
        column = f"se_source_{feature}"
        if column not in causal.columns:
            continue
        rows = causal.loc[
            causal[column].eq("unavailable")
            & causal[f"reason_{feature}"].eq("no_prior_pool")
        ]
        if len(rows):
            row = rows.iloc[0]
            return {
                "participantid": str(row["participantid"]),
                "day": str(row["day"]),
                "feature": feature,
                "n": int(row[f"n_{feature}"]),
                "reason": "no_prior_pool",
                "prior_source_days": int(row[f"pooled_source_days_{feature}"]),
            }
    return None


# ---------------------------------------------------------------------------
# Cox 检查
# ---------------------------------------------------------------------------


def cox_check() -> dict[str, Any]:
    times = np.array([1.0, 2.0, 3.0, 3.0])
    events = np.array([1, 0, 1, 0])
    covariates = np.array([[-1.0], [0.0], [0.5], [1.5]])
    beta = np.array([0.7])
    ridge = 1e-4
    epsilon = 1e-5
    old_module = _load_old_module(
        OLD_RUN / "code_snapshot" / "longitudinal" / "cox_fit.py", "old_cox_fit",
    )

    def metric(function):
        value = lambda b: function(
            times, events, covariates, np.asarray(b), ridge,
        )[0]
        finite = (value(beta + epsilon) - value(beta - epsilon)) / (2 * epsilon)
        gradient = function(times, events, covariates, beta, ridge)[1][0]
        return float(finite), float(gradient), float(finite - gradient)

    old_finite, old_gradient, old_difference = metric(old_module._partial_likelihood)
    new_finite, new_gradient, new_difference = metric(new_partial_likelihood)
    old_penalized = old_module._partial_likelihood(times, events, covariates, beta, ridge)[0]
    old_unpenalized = old_module._partial_likelihood(times, events, covariates, beta, 0.0)[0]
    new_penalized = new_partial_likelihood(times, events, covariates, beta, ridge)[0]
    new_unpenalized = new_partial_likelihood(times, events, covariates, beta, 0.0)[0]
    expected_penalty = 0.5 * ridge * float(beta @ beta)
    return {
        "sample": {"times": times.tolist(), "events": events.tolist(), "x": covariates.tolist()},
        "beta": float(beta[0]),
        "ridge": ridge,
        "old": {
            "finite_difference": old_finite,
            "returned_gradient": old_gradient,
            "difference": old_difference,
            "loglik_penalty_gap": float(old_unpenalized - old_penalized),
        },
        "new": {
            "finite_difference": new_finite,
            "returned_gradient": new_gradient,
            "difference": new_difference,
            "loglik_penalty_gap": float(new_unpenalized - new_penalized),
        },
        "expected_penalty": expected_penalty,
        "old_failure_reproduced": abs(old_difference) > 1e-6,
        "new_passes": abs(new_difference) < 1e-8,
        "penalty_matches": abs((new_unpenalized - new_penalized) - expected_penalty) < 1e-12,
    }


# ---------------------------------------------------------------------------
# 差异与报告
# ---------------------------------------------------------------------------


def write_diffs(out: Path) -> dict[str, Any]:
    import difflib

    fixes = out / "fixes"
    fixes.mkdir(parents=True, exist_ok=True)
    entries = []
    scoring_chunks = []
    cox_chunks = []
    for relative in ("mobilise_d/precision.py", "scripts/replay_mobilise_stage2.py",
                     "longitudinal/cox_fit.py"):
        old_path = OLD_RUN / "code_snapshot" / relative
        new_path = PROJECT_ROOT / relative
        old_text = old_path.read_text(encoding="utf-8").splitlines()
        new_text = new_path.read_text(encoding="utf-8").splitlines()
        diff = list(difflib.unified_diff(
            old_text, new_text, fromfile=f"old/{relative}", tofile=f"new/{relative}",
            lineterm="",
        ))
        chunks = "\n".join(diff) + "\n"
        if relative == "longitudinal/cox_fit.py":
            cox_chunks.append(chunks)
        else:
            scoring_chunks.append(chunks)
        entries.append({
            "path": relative,
            "old_md5": _md5(old_path), "new_md5": _md5(new_path),
            "changed_lines": sum(1 for line in diff if line.startswith(("+", "-"))),
            "changed": _md5(old_path) != _md5(new_path),
        })
    (fixes / "fix_scoring.patch").write_text("\n".join(scoring_chunks), encoding="utf-8")
    (fixes / "fix_cox.patch").write_text("\n".join(cox_chunks), encoding="utf-8")
    versions = {
        "stage": 5,
        "scoring_version": SCORING_VERSION,
        "cox_version": COX_VERSION,
        "old_run": str(OLD_RUN),
        "old_snapshot": str(OLD_RUN / "code_snapshot"),
        "files": entries,
        "note": "旧版仅作历史对照；新结果必须使用修正版代码和新的运行编号。",
    }
    _write_json(fixes / "versions.json", versions)
    return versions


def write_code_snapshot(out: Path) -> list[dict[str, Any]]:
    snapshot = out / "code_snapshot"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    entries = []
    for relative in CODE_FILES:
        source = PROJECT_ROOT / relative
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        entries.append({"path": relative, "md5": _md5(target), "bytes": target.stat().st_size})
    return entries


def write_report(out: Path, payload: dict[str, Any]) -> None:
    old = payload["old_counterfactual"]
    prefix = payload["prefix_check"]
    se = payload["se_changes"]
    mapping = payload["mapping"]["mapping"]
    cox = payload["cox"]
    incremental = payload["incremental"]
    no_prior = payload["no_prior"]
    lines = [
        "# 阶段 5：时间截断与 Cox 拟合一致性修正",
        "",
        f"- 生成时间：{payload['generated_at_utc']}",
        f"- 旧版对照：`{OLD_RUN}`（历史结果，不修改）",
        f"- 修正版标识：`{SCORING_VERSION}` / `{COX_VERSION}`",
        f"- 冻结协议：`protocol_stage5.json`；差异：`fixes/fix_scoring.patch`、`fixes/fix_cox.patch`",
        "",
        "## 1. 修复内容",
        "",
        "### 1.1 逐日标准误在当日冻结",
        "",
        "旧版在段数不足时，用同一受试者**整段研究档案**（含预测日之后）的合格日估计合并段内标准差，再回填早期日期。"
        "修正版按人、按日顺序处理：段数足够时只用当天步行段；不足时只用**当日之前**段数足够的同人采集日估计合并标准差，"
        "记录信息截止日、来源天数、来源段数和最后来源日；没有既往数据时保留 `no_prior_pool` 缺失。"
        "日期映射仍用访视内众数，并检查了前缀规则下的稳定性。",
        "",
        "### 1.2 Cox 步长搜索与导数目标一致",
        "",
        "旧版梯度/Hessian 含岭惩罚，但步长搜索比较的 log-likelihood 漏了 `0.5·λ·||β||²`。"
        "修正版让返回值、梯度、Hessian 和步长搜索都对应同一个带岭惩罚的 Breslow 偏似然；"
        "岭系数、标准化、并列事件处理与不稳定折判据保持不变。",
        "",
        "## 2. 修复前失败证据（复现审查）",
        "",
        f"- 186 个预测日中，删除预测日之后的步行段日汇总并重算回退标准误后，"
        f"**{old['prior_se_changes_if_future_removed']} 个预测日（{old['people_with_se_changes']} 人）**"
        "至少一项历史标准误改变；"
        f"**{old['finite_c_changes_gt_tolerance']} 个预测日**的 CUSUM 改变超过 {TOLERANCE:g}；"
        f"评分状态变化 {old['ds_status_changes']} 个。",
        f"- `n_valid_days` 在 {old['n_valid_days_counts_future']} 个预测日包含未来有效天数。",
        f"- 真实例子：受试者 {old['example']['participantid']}、预测日 {old['example']['landmark_day']}，"
        f"全程标准误的 C={old['example']['full_input_c']:.6f}，预测日截止后 C={old['example']['cutoff_input_c']:.6f}；"
        f"恢复全程标准误可复现 {old['example']['full_control_c']:.6f}。",
        "",
        "## 3. 修复后核验",
        "",
        f"- 186 个预测日的完整档案评分与按日截断评分逐字段比较："
        f"**mismatch = {prefix['prefix_mismatches']}**（容差 {TOLERANCE:g}）。",
        f"- 新因果表与旧封存表相比：{se['landmarks_with_any_se_change']} 个预测日的基线或预测日标准误变化；"
        f"其中基线日变化 {se['landmarks_with_baseline_se_change']} 个、预测日当天变化 "
        f"{se['landmarks_with_landmark_day_se_change']} 个；按特征计 {se['feature_changes']}。"
        "旧版 103 个预测日的口径是“预测日之前的任意监测日”，与这里的基线/预测日统计不可直接相加。",
        f"- 新 CUSUM 与旧封存 CUSUM：{prefix['causal_c_changes_vs_old_sealed']} 个预测日不同；"
        f"与旧版“预测日截止反事实”的 CUSUM：{prefix['causal_c_changes_vs_old_cutoff']} 个预测日不同。"
        "两种修复口径并不等价：逐日冻结比预测日截止更严格，正式版采用逐日冻结。",
        f"- 访视内众数映射：检查 {mapping['visits_checked']} 次采集、{mapping['rows_checked']} 行，"
        f"前缀众数与全量众数在 {mapping['affected_rows']} 行 / {mapping['affected_days']} 个日期上不同；"
        f"是否影响已评分日期：{'是' if payload['mapping']['affects_scored_days'] else '否'}。",
        f"- 原始步行段入口逐日累积：检查 {incremental['days_checked']} 天，mismatch = {incremental['mismatches']}；"
        f"追加 {incremental['future_rows_added']} 条未来异常段后，历史行 mismatch = {incremental['future_mismatches']}；"
        f"无既往回退日期的逐日累积 mismatch = {incremental.get('no_prior_incremental', {}).get('mismatches', '未检查')}。",
        f"- 有效日计数：新评分表的 `n_valid_days` 与严格早于预测日的计数核对，mismatch = "
        f"{prefix['n_valid_count_mismatches']}；旧版有 {old['n_valid_days_counts_future']}/186 个预测日包含未来天数。",
        f"- `session_se` 可评分数：{prefix['session_status_counts']}。"
        f"其中 {prefix['status_changes_vs_old']} 个预测日因为 14 日基线中某日缺少步速标准误"
        "（无既往汇总来源或无 ≥30 秒步行段）而变为 `baseline_dates_unavailable`；"
        "`none` 模式不受会话标准误影响。阶段 6 报告该分母收缩。",
        f"- 跨访视空档示例：受试者 {prefix['gap_example']['participantid']}、预测日 "
        f"{prefix['gap_example']['landmark_day']}，CUSUM 重置 {prefix['gap_example']['cusum_resets']} 次，"
        f"C={prefix['gap_example']['cusum']:.4f}。",
        f"- 无既往回退数据的真实日期：受试者 {no_prior['participantid']}、{no_prior['day']}、"
        f"特征 {no_prior['feature']}（n={no_prior['n']}，来源天数 {no_prior['prior_source_days']}），"
        "按 `no_prior_pool` 保留缺失。",
        "",
        "## 4. Cox 目标函数核验",
        "",
        "| 版本 | 有限差分 | 返回梯度 | 差值 | loglik 与惩罚差 |",
        "|---|---:|---:|---:|---:|",
        f"| 旧版 | {cox['old']['finite_difference']:.8f} | {cox['old']['returned_gradient']:.8f} | "
        f"{cox['old']['difference']:.8f} | {cox['old']['loglik_penalty_gap']:.8f} |",
        f"| 修正版 | {cox['new']['finite_difference']:.8f} | {cox['new']['returned_gradient']:.8f} | "
        f"{cox['new']['difference']:.3e} | {cox['new']['loglik_penalty_gap']:.8f} |",
        "",
        f"期望惩罚 `0.5·λ·β² = {cox['expected_penalty']:.8f}`。旧版差值约等于 `λ·β`，"
        "与审查发现一致；修正版有限差分与梯度在 1e-8 内一致，loglik 含同一惩罚项。",
        "",
        "## 5. 冻结配置与限制",
        "",
        f"- 精度策略：`causal_prior_only`，min_bouts=5，bootstrap=200，lag1 阈值=0.1；"
        "直接估计只用当天段，回退只用当日之前段数足够的同人日期。",
        "- 有效日计数：`n_valid_days` 含预测日当天；`n_valid_days_before_landmark` 严格早于预测日。",
        "- Cox：Breslow 并列处理、训练折内标准化、固定岭 1e-4、原不稳定折判据；目标为带岭惩罚的偏似然。",
        "- 本阶段只交付修复与针对性验证，不产生新的主结果；旧版指标不能作为修复后的通过条件。",
        "- 修正版会改变部分历史标准误和 CUSUM；主结果需在阶段 6 用新运行编号重跑。",
        "",
        "## 6. 阶段 6 就绪",
        "",
        "- 新代码、冻结协议、修复差异、代码快照与核验结果已在本目录；旧运行 `20261114-main` 保持只读对照。",
        "- 阶段 6 必须以新的运行编号从原始包重放，禁止覆盖旧目录；按同一分折/种子比较修正版与旧版，"
        "并把“版本修正造成的差异”和“训练/求解造成的差异”分开报告。",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 5 修正核验")
    parser.add_argument("--old-run", default=str(OLD_RUN))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--skip-scan", action="store_true",
                        help="复用已有的 direct_stats.csv（不重新扫描原始步行段）")
    parser.add_argument("--scan-only", action="store_true",
                        help="只扫描并缓存步行段直接统计，供 --skip-scan 使用")
    args = parser.parse_args(argv)
    old_run = Path(args.old_run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    cohort = load_stage1(old_run / "data")
    scores = pd.read_csv(old_run / "scores.csv", dtype={"participantid": str})
    scores["landmark_day"] = scores["landmark_day"].astype(str)
    old_precision = pd.read_csv(
        old_run / "precision" / "bout_precision.csv", dtype={"participantid": str},
    )
    old_precision["day"] = pd.to_datetime(old_precision["day"])
    people = sorted(scores["participantid"].unique())
    if args.scan_only:
        notes, direct, _ = scan_direct_stats(Path(args.cache_dir), people, out / "bouts")
        print(json.dumps({
            "scan_only": True, "rows": notes.get("rows"), "people": notes.get("people"),
            "days": notes.get("days"), "mapping_affected_rows": notes["mapping"]["affected_rows"],
            "out": str(out),
        }, ensure_ascii=False, indent=2))
        return 0

    # 协议先写入，再运行任何修正版统计
    protocol = {
        "stage": 5,
        "created_at_utc": started.isoformat(timespec="seconds"),
        "old_run": str(old_run),
        "versions": {"scoring": SCORING_VERSION, "cox": COX_VERSION},
        "precision": {
            "policy": "causal_prior_only",
            "min_bouts": 5,
            "bootstrap_samples": 200,
            "lag1_threshold": 0.1,
            "direct": "same-day eligible bouts only",
            "fallback": "same person, prior days with n>=min_bouts only",
            "fallback_value": "pooled within-day SD / sqrt(n_day)",
            "recorded": ["information_cutoff_day", "pooled_source_days", "pooled_bouts",
                         "pooled_source_last_day", "se_source"],
            "unavailable_reasons": ["no_prior_pool", "no_eligible_bouts"],
            "date_mapping": "visit modal origin; prefix stability checked",
        },
        "valid_day_counts": {
            "n_valid_days": "inclusive of landmark",
            "n_valid_days_before_landmark": "strictly before landmark",
        },
        "cox": {
            "objective": "penalized_partial_log_likelihood",
            "ridge": 1e-4,
            "ties": "breslow",
            "standardization": "train-fold mean/sd",
            "gradient_tolerance": 1e-8,
            "unstable_fold_criteria": "unchanged",
        },
        "tolerance": TOLERANCE,
    }
    _write_json(out / "protocol_stage5.json", protocol)

    # 1) 旧版反例（无需扫描）
    old = old_counterfactual(cohort, scores, old_precision)
    cutoff = old.pop("cutoff_c")
    exposed = old.pop("exposed_people")
    _write_json(out / "old_counterfactual.json", {**old, "exposed_people": exposed})

    # 2) 扫描 / 复用直接统计
    direct_path = out / "bouts" / "direct_stats.csv"
    raw_selected_path = out / "bouts" / "selected_raw_rows.csv.gz"
    if args.skip_scan and direct_path.is_file():
        direct = pd.read_csv(direct_path, dtype={"participantid": str})
        scan_notes = _read_json(out / "bouts" / "scan_notes.json")
        mapping = scan_notes["mapping"]
        if raw_selected_path.is_file():
            raw = pd.read_csv(raw_selected_path, dtype={"participantid": str})
        else:
            all_path = out / "bouts" / "raw_rows_all.csv.gz"
            if not all_path.is_file():
                raise FileNotFoundError("缺少原始步行段缓存，请去掉 --skip-scan 重新扫描")
            raw = pd.read_csv(all_path, dtype={"participantid": str})
        raw["day"] = pd.to_datetime(raw["day"])
    else:
        scan_notes, direct, scan_frame = scan_direct_stats(
            Path(args.cache_dir), people, out / "bouts",
        )
        mapping = scan_notes["mapping"]
        raw = scan_frame
    causal = apply_causal_pooled_fallback(direct, PrecisionConfig())
    causal["day"] = causal["day"].astype(str)
    causal_path = out / "se_causal" / "main_precision.csv"
    causal_path.parent.mkdir(parents=True, exist_ok=True)
    causal.to_csv(causal_path, index=False, encoding="utf-8", float_format="%.17g")
    _write_json(out / "se_causal" / "summary.json", {
        "rows": int(len(causal)),
        "people": int(causal["participantid"].nunique()),
        "md5": _md5(causal_path),
        "methods": {
            feature: causal[f"method_{feature}"].value_counts().to_dict()
            for feature in FEATURE_SPECS
        },
        "sources": {
            feature: causal[f"se_source_{feature}"].value_counts().to_dict()
            for feature in FEATURE_SPECS
        },
        "unavailable_reasons": {
            feature: causal.loc[causal[f"se_{feature}"].isna(), f"reason_{feature}"]
            .fillna("unknown").value_counts().to_dict()
            for feature in FEATURE_SPECS
        },
    })

    # 3) 选择原始入口样例：真实反例 + 无既往回退日期
    no_prior = no_prior_example(causal)
    selected = {old["example"]["participantid"]}
    if no_prior:
        selected.add(no_prior["participantid"])
    if raw is not None and "raw_day" in raw.columns:
        raw_selected = raw.loc[raw["participantid"].isin(selected)].copy()
        raw_selected["day"] = pd.to_datetime(raw_selected["day"])
        raw_selected_path.parent.mkdir(parents=True, exist_ok=True)
        raw_selected.to_csv(raw_selected_path, index=False, encoding="utf-8")
    elif raw_selected_path.is_file():
        raw_selected = pd.read_csv(raw_selected_path, dtype={"participantid": str})
        raw_selected["day"] = pd.to_datetime(raw_selected["day"])
    else:
        raise FileNotFoundError("缺少选定人员的原始步行段缓存")

    # 4) 原始入口逐日累积与未来异常段
    incremental = raw_incremental_check(raw_selected, causal, old["example"]["participantid"])
    future = future_anomaly_check(raw_selected, causal, old["example"]["participantid"])
    incremental["future_rows_added"] = future["future_rows_added"]
    incremental["future_mismatches"] = future["mismatches"]
    if no_prior is not None:
        incremental["no_prior_incremental"] = raw_incremental_check(
            raw_selected, causal, str(no_prior["participantid"]),
        )
    _write_json(out / "incremental_check.json", {**incremental, "no_prior": no_prior})

    # 5) 评分：完整档案 vs 按日截断，新 vs 旧
    records = build_causal_records(cohort, causal)
    prefix = score_prefix_check(cohort, records, scores, cutoff)
    se = se_change_counts(cohort, causal, old_precision, scores)
    _write_json(out / "prefix_check.json", prefix)
    _write_json(out / "se_changes.json", se)

    # 6) 映射是否影响已评分日期
    scored_days = scored_prefix_days(cohort, scores)
    affected = set(mapping.get("affected_day_keys", []))
    mapping_summary = {
        "mapping": mapping,
        "affects_scored_days": bool(affected & scored_days),
        "affected_scored_days": sorted(affected & scored_days)[:20],
    }
    _write_json(out / "mapping_check.json", mapping_summary)

    # 7) Cox
    cox = cox_check()
    _write_json(out / "cox_check.json", cox)

    # 8) 差异、快照、报告
    versions = write_diffs(out)
    snapshot = write_code_snapshot(out)
    frozen = {
        "stage": 5,
        "scoring_version": SCORING_VERSION,
        "cox_version": COX_VERSION,
        "precision": protocol["precision"],
        "valid_day_counts": protocol["valid_day_counts"],
        "cox": protocol["cox"],
        "ds_pcd": {
            mode: _read_json(OLD_RUN / "stage2" / "replay_summary.json")["modes"][mode]["config"]
            for mode in ("session_se", "none")
        },
        "stage3": _read_json(OLD_RUN / "stage3" / "protocol.json"),
        "tolerance": TOLERANCE,
    }
    _write_json(out / "config_frozen.json", frozen)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "old_counterfactual": old,
        "prefix_check": prefix,
        "se_changes": se,
        "mapping": mapping_summary,
        "cox": cox,
        "incremental": incremental,
        "no_prior": no_prior,
        "versions": versions,
        "snapshot": snapshot,
    }
    checks = {
        "old_counterfactual_reproduced": (
            old["prior_se_changes_if_future_removed"] == 103
            and old["finite_c_changes_gt_tolerance"] == 44
            and old["n_valid_days_counts_future"] == 177
        ),
        "prefix_invariant": prefix["prefix_mismatches"] == 0,
        "n_valid_counts": prefix["n_valid_count_mismatches"] == 0,
        "se_missing_matches_old": prefix["se_missing_changes_vs_old"] == 0,
        "incremental_matches": incremental["mismatches"] == 0
        and incremental["future_mismatches"] == 0
        and incremental.get("no_prior_incremental", {}).get("mismatches", 0) == 0,
        "no_prior_reported": no_prior is not None,
        "mapping_stable_or_handled": (not mapping_summary["affects_scored_days"]),
        "cox_old_fails_new_passes": cox["old_failure_reproduced"] and cox["new_passes"]
        and cox["penalty_matches"],
    }
    payload["checks"] = checks
    payload["all_pass"] = all(checks.values())
    _write_json(out / "verification.json", payload)
    write_report(out, payload)
    (out / "VERIFICATION.md").write_text(
        "\n".join([
            "# 阶段 5 核验摘要",
            "",
            f"- 旧版反例：SE 改变 {old['prior_se_changes_if_future_removed']}/186，"
            f"C 改变 {old['finite_c_changes_gt_tolerance']}/186，"
            f"n_valid_days 含未来 {old['n_valid_days_counts_future']}/186。",
            f"- 修正版前缀：完整 vs 截断 mismatch = {prefix['prefix_mismatches']}。",
            f"- 原始入口逐日累积 mismatch = {incremental['mismatches']}，"
            f"追加未来异常段后 mismatch = {incremental['future_mismatches']}。",
            f"- 映射影响已评分日期：{'是' if mapping_summary['affects_scored_days'] else '否'}。",
            f"- Cox 旧差值 {cox['old']['difference']:.6f}，新差值 {cox['new']['difference']:.2e}。",
            "",
        ]),
        encoding="utf-8",
    )
    print(json.dumps({
        "out": str(out),
        "old_se_changes": old["prior_se_changes_if_future_removed"],
        "old_c_changes": old["finite_c_changes_gt_tolerance"],
        "prefix_mismatches": prefix["prefix_mismatches"],
        "causal_se_changes_vs_old": se["landmarks_with_any_se_change"],
        "causal_c_changes_vs_old": prefix["causal_c_changes_vs_old_sealed"],
        "causal_c_changes_vs_cutoff": prefix["causal_c_changes_vs_old_cutoff"],
        "mapping_affects_scored": mapping_summary["affects_scored_days"],
        "incremental_mismatches": incremental["mismatches"],
        "future_mismatches": incremental["future_mismatches"],
        "no_prior": no_prior,
        "cox_old_difference": cox["old"]["difference"],
        "cox_new_difference": cox["new"]["difference"],
        "checks": checks,
        "all_pass": payload["all_pass"],
    }, ensure_ascii=False, indent=2))
    return 0 if payload["all_pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
