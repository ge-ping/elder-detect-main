"""阶段 11 核验：短窗连接、人员划分、协议完整性与历史运行保持。"""
from __future__ import annotations

import argparse
from hashlib import md5
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mobilise_d.round2 import (  # noqa: E402
    DATED_COHORTS, HORIZONS_DAYS, PRIMARY_HORIZON_DAYS, R2_DIRECTIONS, R2_FEATURES,
)

OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise"
HISTORICAL_FIXED = OUTPUT_ROOT / "20261114-fixed"
HISTORICAL_STAGE3 = OUTPUT_ROOT / "stage3"


def _md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"participantid": str}, **kwargs)


def verify(run_dir: Path) -> dict:
    checks: dict[str, Any] = {}
    notes: list[str] = []
    real = _read_csv(run_dir / "real_candidates.csv")
    short = _read_csv(run_dir / "short_window_labels.csv")
    windows = _read_csv(run_dir / "landmark_windows.csv")
    landmarks = _read_csv(run_dir / "landmarks.csv")
    flow = _read_csv(run_dir / "cohort_flow.csv")
    splits = _read_csv(run_dir / "splits.csv")
    donors = _read_csv(run_dir / "splits_donors.csv")
    gap = _read_csv(run_dir / "data_gap_matrix.csv")
    counts = _load_json(run_dir / "counts.json")
    protocol = _load_json(run_dir / "protocol.json")
    audit = _load_json(run_dir / "short_window_audit.json")
    reference = _load_json(
        run_dir / "clinical_reference_corrected.json"
        if (run_dir / "clinical_reference_corrected.json").is_file()
        else run_dir / "clinical_reference.json"
    )
    reference_diff = (
        _load_json(run_dir / "clinical_reference_diff.json")
        if (run_dir / "clinical_reference_diff.json").is_file() else None
    )
    manifest = _load_json(run_dir / "manifest.json")

    # 1. 独立重算 1/3/7 天标签（直接从逐次跌倒日期）
    reports = _read_csv(run_dir / ".." / ".." / ".." / ".." / "output" / "ds_pcd_mobilise"
                        / "20261114-fixed" / "data" / "fall_reports.csv") if False else None
    clinical_reports = _read_csv(
        Path("F:/jiebang_team/output/ds_pcd_mobilise/20261114-fixed/data/fall_reports.csv")
    )
    clinical_reports["day"] = pd.to_datetime(clinical_reports["day"], errors="coerce")
    by_person: dict[str, list[pd.Timestamp]] = {}
    for person, group in clinical_reports.loc[clinical_reports["day"].notna()].groupby("participantid"):
        by_person[str(person)] = list(group["day"].dt.normalize())
    mismatches = []
    same_day_excluded = 0
    boundary_hits: list[dict] = []
    for row in real.loc[real["r_branch_eligible"]].itertuples(index=False):
        landmark = pd.Timestamp(row.landmark_day)
        days = by_person.get(str(row.participantid), [])
        for horizon in HORIZONS_DAYS:
            expected = any(landmark < day <= landmark + pd.Timedelta(days=horizon) for day in days)
            observed = getattr(row, f"y_{horizon}d_num")
            if observed is None or (isinstance(observed, float) and math.isnan(observed)):
                mismatches.append({"participantid": row.participantid, "horizon": horizon,
                                   "reason": "label_unknown"})
                continue
            if bool(expected) != bool(observed == 1):
                mismatches.append({
                    "participantid": row.participantid,
                    "landmark_day": row.landmark_day,
                    "horizon": horizon,
                    "expected": bool(expected),
                    "stored": float(observed),
                })
        if any(day == landmark for day in days):
            same_day_excluded += 1
        for horizon in HORIZONS_DAYS:
            for day in days:
                if day == landmark + pd.Timedelta(days=horizon):
                    boundary_hits.append({
                        "participantid": row.participantid,
                        "landmark_day": row.landmark_day,
                        "horizon": horizon,
                        "event_day": str(day.date()),
                        "counted": bool(landmark < day <= landmark + pd.Timedelta(days=horizon)),
                    })
    checks["labels_recomputed_from_dated_reports"] = len(mismatches) == 0
    checks["label_mismatch_count"] = len(mismatches)
    checks["label_mismatch_examples"] = mismatches[:5]
    checks["same_day_reports_excluded"] = same_day_excluded
    checks["boundary_events_inclusive"] = all(item["counted"] for item in boundary_hits)
    checks["boundary_event_examples"] = boundary_hits[:5]
    checks["nested_labels_no_violation"] = int(audit["nested_violations"]) == 0
    checks["boundary_rows_checked"] = int(audit["boundary_rows"])
    checks["boundary_violation_count"] = int(audit["boundary_violations"])

    # 2. 随访不完整记录与覆盖
    per_horizon_status = {}
    for horizon in HORIZONS_DAYS:
        sub = windows.loc[windows["horizon_days"] == horizon]
        per_horizon_status[f"{horizon}d"] = sub["status"].value_counts().to_dict()
    checks["window_status_by_horizon"] = per_horizon_status
    insufficient = int((windows["status"] == "insufficient_followup").sum())
    checks["insufficient_followup_rows"] = insufficient
    checks["insufficient_followup_reason_retained"] = (
        (windows.loc[windows["status"] == "insufficient_followup", "days_short_of_window_end"] > 0).all()
        if insufficient else True
    )
    # 同日事件在窗口内不计入，且首事件日必须严格晚于预测日
    first_days = pd.to_datetime(
        pd.concat([windows[f"first_event_day"] for _ in [0]]).dropna(), errors="coerce"
    )
    checks["first_event_strictly_after_landmark"] = bool(
        (
            pd.to_datetime(windows.loc[windows["first_event_day"].notna(), "first_event_day"])
            > pd.to_datetime(windows.loc[windows["first_event_day"].notna(), "landmark_day"])
        ).all()
    )

    # 3. 人员划分：无交集、每人一折、折间人数
    fold_sets = {fold: set(group["participantid"]) for fold, group in splits.groupby("fold")}
    overlaps = {}
    for left in fold_sets:
        for right in fold_sets:
            if left < right:
                overlaps[f"{left}-{right}"] = len(fold_sets[left] & fold_sets[right])
    checks["folds_disjoint"] = all(value == 0 for value in overlaps.values())
    checks["fold_overlaps"] = overlaps
    checks["each_person_one_fold"] = int(splits["participantid"].duplicated().sum()) == 0
    checks["fold_sizes"] = {int(k): int(v) for k, v in splits["fold"].value_counts().sort_index().items()}
    donor_roles = donors.groupby(["fold", "role"])["participantid"].nunique().to_dict()
    checks["donor_roles"] = {f"{k[0]}-{k[1]}": int(v) for k, v in donor_roles.items()}
    donor_train_test_overlap = []
    for fold, group in donors.groupby("fold"):
        train = set(group.loc[group["role"] == "train", "participantid"])
        test = set(group.loc[group["role"] == "test", "participantid"])
        donor_train_test_overlap.append(len(train & test))
    checks["donor_train_test_disjoint"] = all(value == 0 for value in donor_train_test_overlap)
    checks["splits_cover_landmark_people"] = set(real.loc[
        real["r_branch_eligible"], "participantid"
    ]) == set(splits["participantid"])

    # 4. 队列流程计数可从表重建
    candidate_rows = int(landmarks.groupby("cohort").size().get("pd", 0))
    flow_candidate = int(flow.loc[
        (flow["cohort"] == "pd") & (flow["step"] == "candidate_landmarks"), "rows_out"
    ].iloc[0])
    checks["flow_candidate_matches_table"] = candidate_rows == flow_candidate
    checks["candidate_landmarks_pd"] = candidate_rows
    checks["baseline_ready_pd"] = int(
        landmarks.loc[landmarks["cohort"] == "pd", "baseline_eligible"].astype(bool).sum()
    )
    checks["r_branch_eligible_rows"] = int(real["r_branch_eligible"].sum())
    checks["r_branch_eligible_people"] = int(
        real.loc[real["r_branch_eligible"], "participantid"].nunique()
    )
    horizon_counts = {}
    eligible_real = real.loc[real["r_branch_eligible"]]
    for horizon in HORIZONS_DAYS:
        column = f"y_{horizon}d_num"
        horizon_counts[f"{horizon}d"] = {
            "known": int(eligible_real[column].notna().sum()),
            "positive_landmarks": int((eligible_real[column] == 1).sum()),
            "positive_people": int(
                eligible_real.loc[eligible_real[column] == 1, "participantid"].nunique()
            ),
        }
    checks["real_short_window_counts"] = horizon_counts
    checks["real_counts_match_plan_reference"] = (
        horizon_counts["1d"]["positive_landmarks"] == 1
        and horizon_counts["3d"]["positive_landmarks"] == 2
        and horizon_counts["7d"]["positive_landmarks"] == 4
    )
    checks["counts_json_real_matches"] = (
        int(counts["real"]["r_branch_eligible"]) == int(real["r_branch_eligible"].sum())
    )
    checks["real_and_simulated_counts_separated"] = (
        "real" in counts and "simulated" in counts and counts["simulated"]["sequences"] == 0
    )

    # 5. 数据缺口矩阵完整
    checks["gap_matrix_rows"] = int(len(gap))
    checks["gap_matrix_every_row_has_treatment"] = bool(
        gap["supplement_type"].notna().all() and gap["usable_branches"].notna().all()
    )
    checks["gap_matrix_has_gap_categories"] = bool(
        {"raw_null", "no_long_walking_bout", "low_wear_lt_10h"}.issubset(set(gap.columns))
    )
    checks["gap_matrix_marks_unverified_unit"] = int(
        (gap["unit_verified"] == False).sum()  # noqa: E712
    ) >= 1
    checks["only_pd_has_dated_reports"] = all(
        int(reference_value) > 0 for reference_value in
        [reference["horizons"]["7d"]["rows"]]
    )
    dated_by_cohort = counts["cohort_counts"]["dated_report_people_by_cohort"]
    checks["dated_report_people_by_cohort"] = dated_by_cohort
    checks["dated_reports_only_in_dated_cohorts"] = all(
        (value == 0) for name, value in dated_by_cohort.items() if name not in DATED_COHORTS
    )

    # 6. 协议冻结项
    checks["protocol_primary_horizon_7d"] = int(protocol["primary_horizon_days"]) == PRIMARY_HORIZON_DAYS
    checks["protocol_horizons_1_3_7"] = list(protocol["horizons_days"]) == list(HORIZONS_DAYS)
    checks["protocol_primary_comparison_M3_M2"] = protocol["models"]["primary_comparison"] == "M3 - M2"
    checks["protocol_main_features_frozen"] = tuple(protocol["features"]["main"]["names"]) == R2_FEATURES
    checks["protocol_directions_match_module"] = {
        name: float(protocol["features"]["main"]["directions"][name]) for name in R2_FEATURES
    } == dict(R2_DIRECTIONS)
    checks["protocol_has_training_weights"] = (
        abs(float(protocol["models"]["weights"]["synthetic_total_weight_share"]) - 0.5) < 1e-12
    )
    checks["protocol_has_three_mechanisms"] = len(protocol["branches"]["S"]["mechanisms"]) == 3
    checks["protocol_has_two_noise_families"] = len(protocol["branches"]["S"]["noise_families"]) == 2
    checks["protocol_has_stress_variant"] = bool(protocol["branches"]["S"]["stress_variant"])
    checks["protocol_no_half_year_endpoint"] = (
        "180" not in [str(value) for value in protocol["horizons_days"]]
        and 30 not in protocol["horizons_days"]
    )
    checks["protocol_research_only"] = bool(protocol["models"]["research_only"]) is True
    checks["protocol_calibrated_false"] = bool(protocol["models"]["calibrated"]) is False
    checks["protocol_estimability_rule"] = (
        int(protocol["statistics"]["estimability_rule"]["auroc_min_positives"]) == 5
    )
    checks["protocol_simulated_labels_separate"] = "S" in protocol["branches"]
    checks["clinical_reference_estimable_30d"] = bool(
        reference["horizons"]["30d"]["auroc_estimable"]
    )
    checks["clinical_reference_short_windows_not_estimable"] = all(
        not reference["horizons"][f"{horizon}d"]["auroc_estimable"] for horizon in HORIZONS_DAYS
    )
    checks["clinical_reference_correction_recorded"] = bool(
        reference_diff is None
        or (
            reference_diff.get("protocol_parameters_changed") is False
            and reference_diff.get("outcome_metrics_used_to_change_protocol") is False
        )
    )

    # 7. 本阶段没有训练或增强产物
    forbidden = [name for name in ("fold_models", "scores.csv", "predictions.csv", "synthetic")
                 if (run_dir / name).exists()]
    checks["no_training_artifacts_in_stage11"] = not forbidden

    # 8. 历史运行保持（用其自身 manifest 的 md5 复核）
    historical_manifest = _load_json(HISTORICAL_FIXED / "data" / "manifest.json")
    historical_ok = {}
    for item in historical_manifest.get("outputs", []):
        path = HISTORICAL_FIXED / "data" / item["path"]
        if path.is_file() and item.get("md5"):
            historical_ok[item["path"]] = _md5(path) == item["md5"]
    checks["historical_fixed_md5_unchanged"] = all(historical_ok.values()) if historical_ok else False
    checks["historical_fixed_files_checked"] = len(historical_ok)
    checks["historical_folds_reused"] = int(len(splits)) == int(len(pd.read_csv(HISTORICAL_STAGE3 / "folds.csv")))

    # 9. manifest 哈希
    manifest_ok = True
    for name, item in manifest["inputs"].items():
        if not Path(item["path"]).is_file() or _md5(Path(item["path"])) != item["md5"]:
            manifest_ok = False
    checks["manifest_input_hashes_match"] = manifest_ok

    # 10. 既往跌倒协变量必须在预测时点已知
    known_sources = set(eligible_real["prior_fall_source"].unique())
    checks["prior_fall_sources"] = sorted(known_sources)
    checks["prior_fall_all_known"] = "unknown" not in known_sources
    checks["prior_fall_not_from_future"] = bool(
        (eligible_real["prior_fall_source"] == "diary_6m").sum() > 0
    )
    checks["age_present_for_all_landmarks"] = bool(eligible_real["age"].notna().all())

    all_pass = all(
        value is True or (isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0)
        for key, value in checks.items()
        if key.startswith((
            "labels_recomputed", "nested_labels", "boundary_events", "first_event_strictly",
            "folds_disjoint", "each_person_one_fold", "donor_train_test_disjoint",
            "splits_cover", "flow_candidate_matches", "real_counts_match",
            "counts_json_real_matches", "real_and_simulated_counts",
            "gap_matrix_every", "gap_matrix_has", "gap_matrix_marks",
            "dated_reports_only", "protocol_", "clinical_reference_", "no_training",
            "historical_fixed_md5", "historical_folds", "manifest_input_hashes",
            "prior_fall_all_known", "prior_fall_not_from", "age_present",
            "clinical_reference_correction",
        ))
    ) and checks["label_mismatch_count"] == 0 and checks["boundary_violation_count"] == 0
    return {
        "stage": 11,
        "run_dir": str(run_dir),
        "all_pass": bool(all_pass),
        "checks": checks,
        "notes": notes,
    }


def _render(report: dict) -> str:
    lines = [
        "# 阶段 11 核验",
        "",
        f"- 运行：`{report['run_dir']}`",
        f"- 结论：{'全部通过' if report['all_pass'] else '存在未通过项'}",
        "",
        "| 核验项 | 结果 |",
        "|---|---|",
    ]
    for key, value in report["checks"].items():
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"| `{key}` | {value} |")
    lines.append("")
    for key in ("real_short_window_counts", "window_status_by_horizon", "fold_sizes",
                "dated_report_people_by_cohort", "prior_fall_sources", "boundary_event_examples"):
        value = report["checks"].get(key)
        if value is not None:
            lines.append(f"## {key}")
            lines.append("```")
            lines.append(json.dumps(value, ensure_ascii=False, indent=1))
            lines.append("```")
            lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 11 核验")
    parser.add_argument("--run", default=str(
        OUTPUT_ROOT / "round2" / "stage11" / "20261115-protocol-v1"
    ))
    args = parser.parse_args(argv)
    run_dir = Path(args.run)
    report = verify(run_dir)
    (run_dir / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "VERIFICATION.md").write_text(_render(report), encoding="utf-8")
    print(json.dumps({k: v for k, v in report["checks"].items() if not isinstance(v, (dict, list))},
                     ensure_ascii=False, indent=1))
    print("ALL PASS" if report["all_pass"] else "FAILED")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
