"""阶段 1 核验：独立复算计数、抽查结局与覆盖、重放确定性。

本脚本不生成新数据，只读取派生目录与原始包并记录证据：
``output/ds_pcd_mobilise/stage1/verification.json`` 与 ``VERIFICATION.md``。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import md5
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence
from zipfile import ZipFile

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mobilise_d import sources  # noqa: E402

DEFAULT_BUNDLE = PROJECT_ROOT / "datasets" / "mobilise_d" / "cvs_v1.0.0"
DEFAULT_REPORT = PROJECT_ROOT.parents[1] / "output" / "ds_pcd_mobilise" / "stage1"
COMPARE_FILES = (
    "participants.csv", "visits.csv", "daily_gait.csv", "fall_diaries.csv",
    "fall_reports.csv", "cohort_gates.csv", "landmarks.csv", "baseline_days.csv",
    "landmark_windows.csv", "cohort_flow.csv", "unit_checks.json",
    "field_mapping.json", "sensitivity.json",
)


def _md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _raw_table(archive: ZipFile, suffix: str) -> pd.DataFrame:
    matches = [
        name for name in archive.namelist()
        if name.endswith(suffix) and "__MACOSX" not in name and not name.endswith("/")
    ]
    if len(matches) != 1:
        raise ValueError(f"原始包中 {suffix} 不唯一：{matches}")
    return pd.read_csv(BytesIO(archive.read(matches[0])), dtype=str)


def _raw_daily_gait(cache: Path, person: str, visit: str, day: str) -> list[dict]:
    """直接读增量包内的逐日表，返回同人同访视同日的原始行。"""
    rows: list[dict] = []
    with ZipFile(cache / sources.DAILY.name) as outer:
        for name in outer.namelist():
            if not name.lower().endswith(".zip") or "__MACOSX" in name:
                continue
            with ZipFile(BytesIO(outer.read(name))) as inner:
                for member in inner.namelist():
                    if not member.endswith("daily_agg_filtered.csv"):
                        continue
                    frame = pd.read_csv(BytesIO(inner.read(member)), dtype=str)
                    hit = frame.loc[
                        frame["participantid"].eq(person)
                        & frame["visit_number"].eq(visit)
                        & frame["measurement_date"].eq(day)
                    ]
                    rows.extend(hit.to_dict("records"))
    return rows


def recompute_counts(bundle: Path) -> dict:
    """从派生表独立复算主计数，不读 counts 字典。"""
    landmarks = pd.read_csv(bundle / "landmarks.csv", dtype=str)
    windows = pd.read_csv(bundle / "landmark_windows.csv", dtype=str)
    windows["event_bool"] = windows["event"].map({"True": True, "False": False})
    counts = {
        "candidate_landmarks": int(len(landmarks)),
        "candidate_people": int(landmarks["participantid"].nunique()),
        "baseline_ready_landmarks": int((landmarks["baseline_eligible"] == "True").sum()),
        "baseline_ready_people": int(
            landmarks.loc[landmarks["baseline_eligible"] == "True", "participantid"].nunique()
        ),
        "horizons": {},
    }
    for horizon in (7, 30):
        part = windows.loc[
            windows["horizon_days"].eq(str(horizon)) & windows["status"].eq("eligible")
        ]
        positive = part.loc[part["event_bool"] == True]  # noqa: E712
        ready = part.loc[part["baseline_eligible"].eq("True")]
        counts["horizons"][f"{horizon}d"] = {
            "eligible_landmarks": int(len(part)),
            "positive_landmarks": int(len(positive)),
            "positive_people": int(positive["participantid"].nunique()),
            "baseline_ready_positive_landmarks": int((ready["event_bool"] == True).sum()),  # noqa: E712
            "baseline_ready_positive_people": int(
                ready.loc[ready["event_bool"] == True, "participantid"].nunique()  # noqa: E712
            ),
        }
    return counts


def structural_checks(bundle: Path) -> list[dict]:
    checks = []
    landmarks = pd.read_csv(bundle / "landmarks.csv", dtype=str)
    windows = pd.read_csv(bundle / "landmark_windows.csv", dtype=str)
    baselines = pd.read_csv(bundle / "baseline_days.csv", dtype=str)
    daily = pd.read_csv(bundle / "daily_gait.csv", dtype=str)
    participants = pd.read_csv(bundle / "participants.csv", dtype=str)

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    add(
        "participants_unique",
        not participants["participantid"].duplicated().any(),
        f"{len(participants)} 行",
    )
    add(
        "daily_key_unique",
        not daily.duplicated(["participantid", "visit_number", "day"]).any(),
        "participantid + visit_number + day",
    )
    add(
        "landmark_key_unique",
        not landmarks.duplicated(["participantid", "landmark_day"]).any(),
        f"{len(landmarks)} 个候选预测日",
    )
    add(
        "window_key_unique",
        not windows.duplicated(["participantid", "landmark_day", "horizon_days"]).any(),
        f"{len(windows)} 个预测日 × 时窗",
    )
    merged = baselines.merge(
        landmarks[["participantid", "landmark_day"]], on=["participantid", "landmark_day"],
        how="inner",
    )
    add(
        "baseline_rows_only_for_landmarks",
        len(merged) == len(baselines),
        f"{len(baselines)} 行基线日期",
    )
    before = baselines.merge(
        landmarks[["participantid", "landmark_day"]], on=["participantid", "landmark_day"],
        how="inner", suffixes=("", "_landmark"),
    )
    add(
        "baseline_strictly_before_landmark",
        bool((pd.to_datetime(before["baseline_day"]) < pd.to_datetime(before["landmark_day"])).all()),
        "所有基线日期都早于预测日",
    )
    counts = baselines.groupby(["participantid", "landmark_day"]).size()
    merged_counts = landmarks.set_index(["participantid", "landmark_day"])["baseline_days_used"].astype(int)
    add(
        "baseline_rows_match_declared_count",
        bool((counts.reindex(merged_counts.index).fillna(0).astype(int) == merged_counts).all()),
        "baseline_days.csv 行数与 landmarks.baseline_days_used 一致",
    )
    add(
        "baseline_at_most_14",
        bool((merged_counts <= 14).all()),
        "个人基线最多 14 个最早日期",
    )
    add(
        "outcomes_separate_from_features",
        {"landmark_day", "event"}.issubset(set(windows.columns))
        and "landmark_day" in landmarks.columns and "event" not in landmarks.columns,
        "landmarks.csv 只含输入与基线，结局在 landmark_windows.csv",
    )
    eligible = windows.loc[windows["status"].eq("eligible")]
    add(
        "eligible_windows_have_boolean_event",
        bool(eligible["event"].isin(["True", "False"]).all()),
        f"{len(eligible)} 个可评估时窗",
    )
    short = windows.loc[windows["status"].eq("insufficient_followup")]
    add(
        "insufficient_windows_have_no_event",
        bool(short["event"].isna().all()),
        f"{len(short)} 个随访不足时窗不记阴性/阳性",
    )
    return checks


def trace_cross_window_event(bundle: Path, cache: Path) -> dict:
    """追溯一个 7 天阴性、30 天阳性的预测日到原始跌倒报告。"""
    windows = pd.read_csv(bundle / "landmark_windows.csv", dtype=str)
    wide = windows.pivot_table(
        index=["participantid", "landmark_day", "diary_visit_number"],
        columns="horizon_days", values="event", aggfunc="first",
    )
    hits = wide.loc[(wide["7"] == "False") & (wide["30"] == "True")]
    if hits.empty:
        raise AssertionError("没有找到跨窗事件样例")
    person, landmark, diary_visit = hits.index[0]
    windows_wide = windows.loc[
        windows["participantid"].eq(person) & windows["landmark_day"].eq(landmark)
    ]
    landmark_row = pd.read_csv(bundle / "landmarks.csv", dtype=str)
    visit_number = landmark_row.loc[
        landmark_row["participantid"].eq(person) & landmark_row["landmark_day"].eq(landmark),
        "visit_number",
    ].iloc[0]
    with ZipFile(cache / sources.CLINICAL.name) as archive:
        diaries = _raw_table(archive, "falls-diaries.csv")
        reports = _raw_table(archive, "falls-reports.csv")
    person_digit = str(diary_visit).lower()
    diary_raw = diaries.loc[
        diaries["participantid"].eq(person) & diaries["visit-number"].eq(person_digit)
    ]
    reports_raw = reports.loc[
        reports["participantid"].eq(person) & reports["visit-number"].eq(person_digit)
    ]
    report_days = [
        pd.Timestamp(value).normalize()
        for value in reports_raw["falldate"] if pd.notna(pd.Timestamp(value))
    ]
    landmark_ts = pd.Timestamp(landmark)
    in_7 = [day for day in report_days if landmark_ts < day <= landmark_ts + pd.Timedelta(days=7)]
    in_30 = [day for day in report_days if landmark_ts < day <= landmark_ts + pd.Timedelta(days=30)]
    day_7 = windows_wide.loc[windows_wide["horizon_days"].eq("7")].iloc[0]
    day_30 = windows_wide.loc[windows_wide["horizon_days"].eq("30")].iloc[0]
    return {
        "participantid": str(person),
        "monitoring_visit": str(visit_number),
        "landmark_day": str(landmark),
        "diary_visit": str(diary_visit),
        "diary_day": str(day_30["diary_day"]),
        "diary_fallstotal": int(diary_raw["fallstotal"].iloc[0]),
        "raw_report_rows": int(len(reports_raw)),
        "raw_report_days": [day.date().isoformat() for day in report_days],
        "bundle_7d": {"event": str(day_7["event"]), "window_end": str(day_7["window_end"])},
        "bundle_30d": {"event": str(day_30["event"]), "window_end": str(day_30["window_end"])},
        "recomputed_reports_in_7d": [day.date().isoformat() for day in in_7],
        "recomputed_reports_in_30d": [day.date().isoformat() for day in in_30],
        "raw_gait_rows_at_landmark": _raw_daily_gait(cache, str(person), str(visit_number), str(landmark)),
        "consistent": (
            int(diary_raw["fallstotal"].iloc[0]) == len(reports_raw)
            and str(day_7["event"]) == str(bool(in_7))
            and str(day_30["event"]) == str(bool(in_30))
            and not in_7 and bool(in_30)
        ),
    }


def trace_excluded_diary(bundle: Path, cache: Path, reason: str) -> dict:
    """追溯一条被排除的日记记录到原始访视与日记表。"""
    gates = pd.read_csv(bundle / "cohort_gates.csv", dtype=str)
    hits = gates.loc[gates["exclusion_reason"].eq(reason)]
    if hits.empty:
        raise AssertionError(f"没有 {reason} 的样例")
    row = hits.iloc[0]
    person, visit = str(row["participantid"]), str(row["visit_number"])
    with ZipFile(cache / sources.CLINICAL.name) as archive:
        diaries = _raw_table(archive, "falls-diaries.csv")
        reports = _raw_table(archive, "falls-reports.csv")
        status = _raw_table(archive, "status2-5.csv")
        baseline = _raw_table(archive, "descriptives1.csv")
    diary_raw = diaries.loc[diaries["participantid"].eq(person) & diaries["visit-number"].eq(visit)]
    reports_raw = reports.loc[reports["participantid"].eq(person) & reports["visit-number"].eq(visit)]
    visit_rows = baseline.loc[
        baseline["participantid"].eq(person) & baseline["visit-number"].eq(row["previous_visit_number"])
    ]
    if visit_rows.empty:
        visit_rows = status.loc[
            status["participantid"].eq(person) & status["visit-number"].eq(row["previous_visit_number"])
        ]
    previous_raw = visit_rows["visit-date"].iloc[0] if len(visit_rows) else None
    return {
        "participantid": person,
        "diary_visit": visit,
        "exclusion_reason": reason,
        "raw_diary_visit_date": str(diary_raw["visit-date"].iloc[0]),
        "raw_previous_visit_date": str(previous_raw) if previous_raw is not None else None,
        "raw_fallstotal": str(diary_raw["fallstotal"].iloc[0]),
        "raw_reports_in_diary": int(len(reports_raw)),
        "raw_report_days": [str(value) for value in reports_raw["falldate"]],
        "bundle": {
            key: (None if pd.isna(row[key]) else str(row[key]))
            for key in (
                "previous_visit_day", "covered_from", "block_first_day", "landmark_day",
                "diary_day", "n_report_rows", "n_report_rows_outside_window",
                "block_first_offset_days", "n_valid_days_in_previous_visit",
            )
        },
        "matches_raw_previous_visit": (
            previous_raw is not None
            and pd.Timestamp(previous_raw).normalize() == pd.Timestamp(row["previous_visit_day"])
        ),
        "matches_raw_fallstotal": int(diary_raw["fallstotal"].iloc[0]) == int(float(row["fallstotal"])),
        "matches_raw_report_rows": int(len(reports_raw)) == int(float(row["n_report_rows"])),
    }


def determinism(bundle: Path, cache: Path, work_dir: Path) -> dict:
    """在新目录按同一配置重建，逐文件比较哈希。"""
    rebuild = work_dir / "rebuild"
    if rebuild.exists():
        shutil.rmtree(rebuild)
    rebuild.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "build_mobilise_cohort.py"),
        "--cache-dir", str(cache), "--out", str(rebuild), "--skip-bouts",
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    compared = {}
    for name in COMPARE_FILES:
        first = bundle / name
        second = rebuild / name
        compared[name] = {
            "bundle_md5": _md5(first) if first.is_file() else None,
            "rebuild_md5": _md5(second) if second.is_file() else None,
            "identical": first.is_file() and second.is_file() and _md5(first) == _md5(second),
        }
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stderr_tail": completed.stderr.strip().splitlines()[-5:],
        "files": compared,
        "all_identical": all(item["identical"] for item in compared.values()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="核验 Mobilise-D 阶段 1 派生数据")
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT))
    parser.add_argument("--skip-rebuild", action="store_true")
    args = parser.parse_args(argv)
    bundle = Path(args.bundle)
    cache = Path(args.cache_dir) if args.cache_dir else sources.default_cache_dir()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))

    recomputed = recompute_counts(bundle)
    stored = manifest["counts"]
    count_checks = []
    for key in ("candidate_landmarks", "candidate_people", "baseline_ready_landmarks", "baseline_ready_people"):
        count_checks.append({
            "metric": key, "stored": stored[key], "recomputed": recomputed[key],
            "match": stored[key] == recomputed[key],
        })
    for horizon in ("7d", "30d"):
        for key in ("eligible_landmarks", "positive_landmarks", "positive_people",
                    "baseline_ready_positive_landmarks", "baseline_ready_positive_people"):
            count_checks.append({
                "metric": f"{horizon}.{key}",
                "stored": stored["horizons"][horizon][key],
                "recomputed": recomputed["horizons"][horizon][key],
                "match": stored["horizons"][horizon][key] == recomputed["horizons"][horizon][key],
            })

    payload = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bundle": str(bundle),
        "cache_dir": str(cache),
        "release": manifest["release"]["version"],
        "cohort_version": manifest["cohort_version"],
        "reference_comparison": manifest["reference_comparison"],
        "count_checks": count_checks,
        "count_checks_all_match": all(item["match"] for item in count_checks),
        "structural_checks": structural_checks(bundle),
        "trace_cross_window_event": trace_cross_window_event(bundle, cache),
        "trace_excluded_landmark_outside_window": trace_excluded_diary(
            bundle, cache, "landmark_outside_diary_window"
        ),
        "trace_excluded_report_mismatch": trace_excluded_diary(
            bundle, cache, "fall_reports_not_usable"
        ),
    }
    payload["structural_checks_all_pass"] = all(
        item["passed"] for item in payload["structural_checks"]
    )
    if not args.skip_rebuild:
        payload["determinism"] = determinism(bundle, cache, report_dir / "work")
    sources.write_json(report_dir / "verification.json", payload)

    lines = [
        "# 阶段 1 核验记录",
        "",
        f"- 时间：{payload['verified_at_utc']}",
        f"- 派生目录：`{bundle}`",
        f"- 原始包：`{cache}`（Mobilise-D CVS {payload['release']}）",
        "",
        "## 计数复算",
        "",
        "| 指标 | 派生记录 | 独立复算 | 一致 |",
        "|---|---:|---:|---|",
    ]
    for item in count_checks:
        lines.append(
            f"| {item['metric']} | {item['stored']} | {item['recomputed']} | "
            f"{'是' if item['match'] else '否'} |"
        )
    lines += ["", "## 结构检查", "", "| 检查 | 结果 | 说明 |", "|---|---|---|"]
    for item in payload["structural_checks"]:
        lines.append(f"| {item['check']} | {'通过' if item['passed'] else '失败'} | {item['detail']} |")
    trace = payload["trace_cross_window_event"]
    lines += [
        "", "## 跨窗事件追溯", "",
        f"- 受试者 {trace['participantid']}，预测日 {trace['landmark_day']}"
        f"（监测访视 {trace['monitoring_visit']}），日记访视 {trace['diary_visit']}"
        f"（{trace['diary_day']}）。",
        f"- 原始日记 fallstotal={trace['diary_fallstotal']}，原始逐次报告 {trace['raw_report_rows']} 行，"
        f"日期 {trace['raw_report_days']}。",
        f"- 7 天窗 (t, {trace['bundle_7d']['window_end']}]：派生 {trace['bundle_7d']['event']}，"
        f"复算 {trace['recomputed_reports_in_7d']}。",
        f"- 30 天窗 (t, {trace['bundle_30d']['window_end']}]：派生 {trace['bundle_30d']['event']}，"
        f"复算 {trace['recomputed_reports_in_30d']}。",
        f"- 预测日原始逐日行：{json.dumps(trace['raw_gait_rows_at_landmark'], ensure_ascii=False)[:600]}",
        f"- 结论：{'一致' if trace['consistent'] else '不一致'}。",
    ]
    excluded = payload["trace_excluded_landmark_outside_window"]
    lines += [
        "", "## 覆盖不足样例追溯", "",
        f"- 受试者 {excluded['participantid']}，日记访视 {excluded['diary_visit']}："
        f"原始访视日 {excluded['raw_diary_visit_date']}，原始上次访视日 {excluded['raw_previous_visit_date']}。",
        f"- 派生：上次访视 {excluded['bundle']['previous_visit_day']}，日记覆盖下界 "
        f"{excluded['bundle']['covered_from']}，采集首日 {excluded['bundle']['block_first_day']}，"
        f"采集末日 {excluded['bundle']['landmark_day']}。",
        f"- 与原始表一致：上次访视 {excluded['matches_raw_previous_visit']}，"
        f"日记总数 {excluded['matches_raw_fallstotal']}，报告条数 {excluded['matches_raw_report_rows']}。",
    ]
    mismatch = payload["trace_excluded_report_mismatch"]
    lines += [
        "", "## 日记/报告不一致样例追溯", "",
        f"- 受试者 {mismatch['participantid']}，日记访视 {mismatch['diary_visit']}："
        f"原始 fallstotal={mismatch['raw_fallstotal']}，原始报告 {mismatch['raw_reports_in_diary']} 行"
        f"（日期 {mismatch['raw_report_days']}）。",
        f"- 派生记录：报告 {mismatch['bundle']['n_report_rows']} 行，"
        f"区间外 {mismatch['bundle']['n_report_rows_outside_window']} 行。",
    ]
    if "determinism" in payload:
        det = payload["determinism"]
        lines += [
            "", "## 重放确定性", "",
            f"- 命令：`{det['command']}`，退出码 {det['returncode']}。",
            f"- 逐文件一致：{'是' if det['all_identical'] else '否'}。",
            "",
            "| 文件 | 一致 |",
            "|---|---|",
        ]
        for name, item in det["files"].items():
            lines.append(f"| {name} | {'是' if item['identical'] else '否'} |")
    (report_dir / "VERIFICATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "count_checks_all_match": payload["count_checks_all_match"],
        "structural_checks_all_pass": payload["structural_checks_all_pass"],
        "determinism_all_identical": payload.get("determinism", {}).get("all_identical"),
        "report_dir": str(report_dir),
    }, ensure_ascii=False, indent=2))
    ok = (
        payload["count_checks_all_match"]
        and payload["structural_checks_all_pass"]
        and payload["reference_comparison"]["matches"]
        and payload.get("determinism", {}).get("all_identical", True)
        and payload["trace_cross_window_event"]["consistent"]
    )
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
