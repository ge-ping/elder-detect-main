"""阶段 9 核验：独立集合差、明细追溯、候选/可评分标记与最小案例。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mobilise_d.expansion import analyze_expansion  # noqa: E402

OUT_ROOT = PROJECT_ROOT.parents[1] / "output" / "ds_pcd_mobilise"
DEFAULT_RUN = OUT_ROOT / "20261114-fixed"
DEFAULT_STAGE9 = OUT_ROOT / "stage9" / "20261114-expansion-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 9 核验")
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--stage9", default=str(DEFAULT_STAGE9))
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["participantid"] = frame["participantid"].astype(str).str.strip()
    frame["landmark_day"] = pd.to_datetime(frame["landmark_day"]).dt.strftime("%Y-%m-%d")
    return frame


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def independent_sets(run: Path) -> dict[str, Any]:
    """只使用原始 CSV 与集合运算重新得到期望集合。"""
    landmarks = _normalise(pd.read_csv(
        run / "data" / "landmarks.csv", dtype={"participantid": str}
    ))
    windows = _normalise(pd.read_csv(
        run / "data" / "landmark_windows.csv", dtype={"participantid": str}
    ))
    windows = windows.loc[pd.to_numeric(windows["horizon_days"]).eq(30)]
    event = {
        (str(p), str(d)): _bool(e)
        for p, d, e in zip(windows["participantid"], windows["landmark_day"], windows["event"])
    }
    s14_frame = landmarks.loc[
        landmarks["baseline_eligible"].map(_bool)
    ]
    s7_frame = landmarks.loc[
        pd.to_numeric(landmarks["n_prior_valid_days"]).ge(7)
    ]
    s14 = set(zip(s14_frame["participantid"], s14_frame["landmark_day"]))
    s7 = set(zip(s7_frame["participantid"], s7_frame["landmark_day"]))
    p14 = {key[0] for key in s14}
    p7 = {key[0] for key in s7}
    pos14 = {key for key in s14 if event[key]}
    pos7 = {key for key in s7 if event[key]}
    pp14 = {key[0] for key in pos14}
    pp7 = {key[0] for key in pos7}
    new_people = p7 - p14
    new_positive = pp7 - pp14
    from_new = new_positive & new_people
    from_existing = new_positive & p14
    extra = pos7 - pos14
    cat_new = {key for key in extra if key[0] in new_people}
    cat_existing = {key for key in extra if key[0] in from_existing}
    cat_already = {key for key in extra if key[0] in pp14}
    return {
        "s14": s14, "s7": s7, "p14": p14, "p7": p7,
        "pos14": pos14, "pos7": pos7, "pp14": pp14, "pp7": pp7,
        "new_people": new_people, "new_positive": new_positive,
        "from_new": from_new, "from_existing": from_existing,
        "extra": extra, "cat_new": cat_new, "cat_existing": cat_existing,
        "cat_already": cat_already,
        "event": event,
    }


def scorable_sets(run: Path) -> dict[str, Any]:
    predictions = _normalise(pd.read_csv(
        run / "predictions.csv", dtype={"participantid": str}
    ))
    windows = _normalise(pd.read_csv(
        run / "data" / "landmark_windows.csv", dtype={"participantid": str}
    ))
    windows = windows.loc[pd.to_numeric(windows["horizon_days"]).eq(30)]
    event = {
        (str(p), str(d)): _bool(e)
        for p, d, e in zip(windows["participantid"], windows["landmark_day"], windows["event"])
    }
    base = predictions.loc[
        predictions["variant"].eq("fixed_session") & predictions["horizon"].eq(30)
    ]
    keys = set(zip(
        base.loc[base["method"].eq("A_rms"), "participantid"],
        base.loc[base["method"].eq("A_rms"), "landmark_day"],
    ))
    bkeys = set(zip(
        base.loc[base["method"].eq("B_rms_cusum"), "participantid"],
        base.loc[base["method"].eq("B_rms_cusum"), "landmark_day"],
    ))
    people = {key[0] for key in keys}
    positive = {key for key in keys if event[key]}
    positive_people = {key[0] for key in positive}
    return {
        "keys": keys, "bkeys": bkeys, "people": people,
        "positive": positive, "positive_people": positive_people,
    }


def minimal_case() -> dict[str, Any]:
    """一人多预测日：新增者阳性、原有者新阳性、已阳性者重复事件。"""
    rows = [
        ("A", "2024-01-01", True, 20),
        ("A", "2024-01-02", False, 10),  # S7 only, existing person newly positive
        ("B", "2024-01-03", False, 10),  # S7 only new person positive
        ("C", "2024-01-04", True, 20),
        ("C", "2024-01-05", False, 10),  # extra repeated event
        ("D", "2024-01-06", True, 20),
        ("D", "2024-01-07", False, 10),
        ("E", "2024-01-08", True, 20),
        ("E", "2024-01-09", False, 10),
    ]
    landmarks = pd.DataFrame(rows, columns=[
        "participantid", "landmark_day", "baseline_eligible", "n_prior_valid_days",
    ])
    events = {
        ("A", "2024-01-01"): False,
        ("A", "2024-01-02"): True,
        ("B", "2024-01-03"): True,
        ("C", "2024-01-04"): True,
        ("C", "2024-01-05"): True,
        ("D", "2024-01-06"): False,
        ("D", "2024-01-07"): False,
        ("E", "2024-01-08"): True,
        ("E", "2024-01-09"): False,
    }
    windows = pd.DataFrame([
        {"participantid": p, "landmark_day": d, "horizon_days": 30, "event": e}
        for (p, d), e in events.items()
    ])
    result = analyze_expansion(landmarks, windows)
    summary = result["summary"]
    expected = {
        "S14": {"landmarks": 4, "people": 4, "positive_landmarks": 2, "positive_people": 2},
        "S7": {"landmarks": 9, "people": 5, "positive_landmarks": 5, "positive_people": 4},
        "new_people": 1,
        "new_positive_total": 2,
        "from_new_people": 1,
        "from_existing_people": 1,
        "extra": 3,
        "extra_new": 1,
        "extra_existing": 1,
        "extra_already": 1,
        "historical_new_positive_people": 1,
        "historical_new_positive_landmarks": 1,
    }
    actual = {
        "S14": summary["S14"],
        "S7": summary["S7"],
        "new_people": summary["new_people"],
        "new_positive_total": summary["s7_new_positive_people_total"],
        "from_new_people": summary["s7_new_positive_people_from_new_people"],
        "from_existing_people": summary["s7_existing_people_newly_positive"],
        "extra": summary["s7_extra_positive_landmarks_total"],
        "extra_new": summary["s7_extra_positive_landmarks_from_new_people"],
        "extra_existing": summary["s7_extra_positive_landmarks_from_existing_people_newly_positive"],
        "extra_already": summary["s7_extra_positive_landmarks_from_already_positive_people"],
        "historical_new_positive_people": summary["new_positive_people"],
        "historical_new_positive_landmarks": summary["new_positive_landmarks"],
    }
    return {
        "expected": expected,
        "actual": actual,
        "pass": expected == actual,
        "categories": {
            "from_new_people": sorted(result["extra_positive_landmarks"].loc[
                result["extra_positive_landmarks"]["extra_positive_category"].eq("from_new_people"),
                "participantid",
            ].tolist()),
            "from_existing_people_newly_positive": sorted(
                result["extra_positive_landmarks"].loc[
                    result["extra_positive_landmarks"]["extra_positive_category"].eq(
                        "from_existing_people_newly_positive"
                    ), "participantid",
                ].tolist()
            ),
            "from_already_positive_people": sorted(
                result["extra_positive_landmarks"].loc[
                    result["extra_positive_landmarks"]["extra_positive_category"].eq(
                        "from_already_positive_people"
                    ), "participantid",
                ].tolist()
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run = Path(args.run).resolve()
    stage9 = Path(args.stage9).resolve()
    counts = read_json(stage9 / "counts.json")
    expected = independent_sets(run)
    scorable = scorable_sets(run)
    landmarks = _normalise(pd.read_csv(run / "data" / "landmarks.csv", dtype={"participantid": str}))
    windows = _normalise(pd.read_csv(run / "data" / "landmark_windows.csv", dtype={"participantid": str}))
    windows30 = windows.loc[pd.to_numeric(windows["horizon_days"]).eq(30)]
    event = {
        (str(p), str(d)): _bool(e)
        for p, d, e in zip(windows30["participantid"], windows30["landmark_day"], windows30["event"])
    }
    checks: dict[str, bool] = {}
    required = counts["required_counts"]
    checks["required_counts_match_independent_sets"] = (
        required["s14_candidate_landmarks"] == len(expected["s14"])
        and required["s7_candidate_landmarks"] == len(expected["s7"])
        and required["s14_people"] == len(expected["p14"])
        and required["s7_people"] == len(expected["p7"])
        and required["people_intersection"] == len(expected["p14"] & expected["p7"])
        and required["people_union"] == len(expected["p14"] | expected["p7"])
        and required["s7_new_people"] == len(expected["new_people"])
        and required["s14_positive_people"] == len(expected["pp14"])
        and required["s7_positive_people"] == len(expected["pp7"])
        and required["s7_new_positive_people_total"] == len(expected["new_positive"])
        and required["s7_new_positive_people_from_new_people"] == len(expected["from_new"])
        and required["s7_existing_people_newly_positive"] == len(expected["from_existing"])
        and required["s14_positive_landmarks"] == len(expected["pos14"])
        and required["s7_positive_landmarks"] == len(expected["pos7"])
        and required["s7_extra_positive_landmarks_total"] == len(expected["extra"])
        and required["s7_extra_positive_landmarks_from_new_people"] == len(expected["cat_new"])
        and required["s7_extra_positive_landmarks_from_existing_people_newly_positive"] == len(expected["cat_existing"])
        and required["s7_extra_positive_landmarks_from_already_positive_people"] == len(expected["cat_already"])
    )
    checks["required_values"] = (
        (required["s14_candidate_landmarks"], required["s7_candidate_landmarks"],
         required["s14_people"], required["s7_people"],
         required["people_intersection"], required["people_union"],
         required["s7_new_people"], required["s14_positive_people"],
         required["s7_positive_people"], required["s7_new_positive_people_total"],
         required["s7_new_positive_people_from_new_people"],
         required["s7_existing_people_newly_positive"],
         required["s7_extra_positive_landmarks_total"],
         required["s7_extra_positive_landmarks_from_new_people"],
         required["s7_extra_positive_landmarks_from_existing_people_newly_positive"],
         required["s7_extra_positive_landmarks_from_already_positive_people"])
        == (186, 325, 144, 194, 144, 194, 50, 15, 25, 10, 6, 4, 11, 6, 4, 1)
    )
    checks["identity_6_plus_4"] = (
        required["s7_new_positive_people_from_new_people"]
        + required["s7_existing_people_newly_positive"]
        == required["s7_new_positive_people_total"]
    )
    checks["identity_11_equals_6_plus_4_plus_1"] = (
        required["s7_extra_positive_landmarks_from_new_people"]
        + required["s7_extra_positive_landmarks_from_existing_people_newly_positive"]
        + required["s7_extra_positive_landmarks_from_already_positive_people"]
        == required["s7_extra_positive_landmarks_total"]
    )
    checks["people_intersection_union"] = (
        required["people_intersection"] == 144
        and required["people_union"] == 194
        and required["people_union"] == len(expected["p14"] | expected["p7"])
    )

    details = pd.read_csv(stage9 / "landmark_sets.csv", dtype={"participantid": str})
    checks["landmark_table_463_rows"] = len(details) == 463
    checks["landmark_table_unique_keys"] = not details.duplicated(
        ["participantid", "landmark_day"]
    ).any()
    detail_flags_ok = True
    for row in details.itertuples(index=False):
        key = (row.participantid, row.landmark_day)
        if bool(row.in_s14) != (key in expected["s14"]):
            detail_flags_ok = False
        if bool(row.in_s7) != (key in expected["s7"]):
            detail_flags_ok = False
        if bool(row.event_30d) != bool(event.get(key, False)):
            detail_flags_ok = False
        if bool(row.s14_positive) != (key in expected["pos14"]):
            detail_flags_ok = False
        if bool(row.s7_positive) != (key in expected["pos7"]):
            detail_flags_ok = False
        if bool(row.extra_positive) != (key in expected["extra"]):
            detail_flags_ok = False
    checks["landmark_table_flags_match_raw"] = detail_flags_ok
    checks["landmark_table_counts"] = (
        int(details["in_s14"].sum()) == 186
        and int(details["in_s7"].sum()) == 325
        and int(details["s7_positive"].sum()) == 28
        and int(details["extra_positive"].sum()) == 11
    )

    people = pd.read_csv(stage9 / "positive_people_sets.csv", dtype={"participantid": str})
    checks["positive_people_table_194"] = len(people) == 194
    people_ok = True
    for row in people.itertuples(index=False):
        person = row.participantid
        if bool(row.in_s14) != (person in expected["p14"]):
            people_ok = False
        if bool(row.in_s7) != (person in expected["p7"]):
            people_ok = False
        if bool(row.s14_positive) != (person in expected["pp14"]):
            people_ok = False
        if bool(row.s7_positive) != (person in expected["pp7"]):
            people_ok = False
        if bool(row.is_new_person) != (person in expected["new_people"]):
            people_ok = False
        if bool(row.new_positive_total) != (person in expected["new_positive"]):
            people_ok = False
        if bool(row.new_positive_from_new_person) != (person in expected["from_new"]):
            people_ok = False
        if bool(row.new_positive_from_existing_person) != (person in expected["from_existing"]):
            people_ok = False
    checks["positive_people_flags_match_raw"] = people_ok
    checks["positive_people_counts"] = (
        int(people["s14_positive"].sum()) == 15
        and int(people["s7_positive"].sum()) == 25
        and int(people["new_positive_total"].sum()) == 10
        and int(people["new_positive_from_new_person"].sum()) == 6
        and int(people["new_positive_from_existing_person"].sum()) == 4
    )

    extra = pd.read_csv(stage9 / "extra_positive_landmarks.csv", dtype={"participantid": str})
    extra_keys = set(zip(extra["participantid"], extra["landmark_day"]))
    checks["extra_table_11_rows"] = len(extra) == 11
    checks["extra_table_keys_match"] = extra_keys == expected["extra"]
    checks["extra_table_category_counts"] = (
        int(extra["extra_positive_category"].eq("from_new_people").sum()) == 6
        and int(extra["extra_positive_category"].eq("from_existing_people_newly_positive").sum()) == 4
        and int(extra["extra_positive_category"].eq("from_already_positive_people").sum()) == 1
    )
    checks["extra_table_categories_match_people"] = all(
        (row.extra_positive_category == "from_new_people" and row.participantid in expected["from_new"])
        or (row.extra_positive_category == "from_existing_people_newly_positive" and row.participantid in expected["from_existing"])
        or (row.extra_positive_category == "from_already_positive_people" and row.participantid in expected["pp14"])
        for row in extra.itertuples(index=False)
    )

    checks["scorable_keys_172"] = len(scorable["keys"]) == 172
    checks["scorable_ab_keys_equal"] = scorable["keys"] == scorable["bkeys"]
    checks["scorable_people_134"] = len(scorable["people"]) == 134
    checks["scorable_positive_16_14"] = (
        len(scorable["positive"]) == 16 and len(scorable["positive_people"]) == 14
    )
    checks["counts_scorable_match"] = (
        counts["scorable"]["S14"]["landmarks"] == len(scorable["keys"])
        and counts["scorable"]["S14"]["people"] == len(scorable["people"])
        and counts["scorable"]["S14"]["positive_landmarks"] == len(scorable["positive"])
        and counts["scorable"]["S14"]["positive_people"] == len(scorable["positive_people"])
        and counts["scorable"]["S14"]["not_scorable_landmarks"]
        == len(expected["s14"] - scorable["keys"])
    )
    checks["s7_not_estimated_marked"] = (
        counts["scorable"]["S7"]["status"] == "not_estimated"
        and counts["scorable"]["S7"]["scorable_landmarks"] is None
        and counts["scorable"]["S7"]["scorable_people"] is None
        and counts["scorable"]["S7"]["candidate_landmarks"] == 325
        and counts["scorable"]["S7"]["candidate_people"] == 194
    )
    checks["old_field_semantics"] = (
        counts["candidate_expansion"]["new_positive_people"] == 6
        and counts["candidate_expansion"]["new_positive_landmarks"] == 6
        and counts["candidate_expansion"]["s7_new_positive_people_total"] == 10
        and counts["candidate_expansion"]["s7_extra_positive_landmarks_total"] == 11
    )

    minimal = minimal_case()
    checks["minimal_case_6_4_1_analogue"] = (
        minimal["pass"]
        and minimal["actual"]["new_positive_total"] == 2
        and minimal["actual"]["from_new_people"] == 1
        and minimal["actual"]["from_existing_people"] == 1
        and minimal["actual"]["extra_new"] == 1
        and minimal["actual"]["extra_existing"] == 1
        and minimal["actual"]["extra_already"] == 1
        and minimal["actual"]["historical_new_positive_people"] == 1
    )

    report = (stage9 / "REPORT.md").read_text(encoding="utf-8")
    checks["report_has_corrected_numbers"] = all(
        text in report for text in (
            "10 名阳性受试者", "6 名", "4 名", "11 = 6 + 4 + 1",
            "并集为 **194**", "172", "134", "16 个阳性预测日", "14 名阳性受试者",
            "S7 尚未做 7 日基线", "只新增 6 名独立阳性",
        )
    )
    checks["report_marks_s7_pending"] = "未估计" in report and "172" in report
    checks["source_hashes_match"] = (
        counts["inputs"]["landmarks.csv"]["sha256"] == sha256(run / "data" / "landmarks.csv")
        and counts["inputs"]["landmark_windows.csv"]["sha256"]
        == sha256(run / "data" / "landmark_windows.csv")
        and counts["scorable_source"]["sha256"] == sha256(run / "predictions.csv")
    )

    payload = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": str(run),
        "stage9": str(stage9),
        "checks": checks,
        "all_pass": all(bool(value) for value in checks.values()),
        "expected_sets": {
            "s14_landmarks": len(expected["s14"]), "s7_landmarks": len(expected["s7"]),
            "s14_people": len(expected["p14"]), "s7_people": len(expected["p7"]),
            "intersection": len(expected["p14"] & expected["p7"]),
            "union": len(expected["p14"] | expected["p7"]),
            "new_people": len(expected["new_people"]),
            "s14_positive_landmarks": len(expected["pos14"]),
            "s7_positive_landmarks": len(expected["pos7"]),
            "s14_positive_people": len(expected["pp14"]), "s7_positive_people": len(expected["pp7"]),
            "new_positive_total": len(expected["new_positive"]),
            "new_positive_from_new": len(expected["from_new"]),
            "new_positive_from_existing": len(expected["from_existing"]),
            "extra": len(expected["extra"]),
            "extra_from_new": len(expected["cat_new"]),
            "extra_from_existing": len(expected["cat_existing"]),
            "extra_from_already": len(expected["cat_already"]),
        },
        "minimal_case": minimal,
        "scorable": {
            "keys": len(scorable["keys"]), "people": len(scorable["people"]),
            "positive_landmarks": len(scorable["positive"]),
            "positive_people": len(scorable["positive_people"]),
        },
    }
    write_json(stage9 / "verification.json", payload)
    lines = [
        "# 阶段 9 核验记录",
        "",
        f"- 时间：{payload['verified_at_utc']}",
        f"- 源运行：`{run}`",
        f"- 阶段 9：`{stage9}`",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
    ]
    for name, value in checks.items():
        lines.append(f"| {name} | {'通过' if value else '失败'} |")
    lines += [
        "",
        f"- S14/S7 候选日：{len(expected['s14'])} / {len(expected['s7'])}；"
        f"人员：{len(expected['p14'])} / {len(expected['p7'])}；"
        f"交集/并集：{len(expected['p14'] & expected['p7'])} / {len(expected['p14'] | expected['p7'])}。",
        f"- 新增阳性人员：{len(expected['new_positive'])} = "
        f"{len(expected['from_new'])}（新增人员）+ {len(expected['from_existing'])}（原有人员新进入）；"
        f"额外阳性预测日：{len(expected['extra'])} = {len(expected['cat_new'])} + "
        f"{len(expected['cat_existing'])} + {len(expected['cat_already'])}。",
        f"- S14 session_se 可评分：{len(scorable['keys'])} / {len(scorable['people'])} / "
        f"{len(scorable['positive'])} / {len(scorable['positive_people'])}；S7 未估计。",
        f"- 最小案例：{'通过' if minimal['pass'] else '失败'}。",
        "",
        f"**all_pass = {payload['all_pass']}**",
        "",
    ]
    (stage9 / "VERIFICATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "checks": checks, "all_pass": payload["all_pass"],
        "expected_sets": payload["expected_sets"], "minimal_case": minimal["pass"],
    }, ensure_ascii=False, indent=1))
    return 0 if payload["all_pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
