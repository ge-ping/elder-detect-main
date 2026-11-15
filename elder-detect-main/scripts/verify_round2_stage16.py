"""阶段 16 核验：同日排除、分母重建、边界归属与分折一致。

独立于 ``mobilise_d.round2_time`` 的实现路径：直接用原包事件表与键表做
日期连接，重新数一遍分母和阳性。只读阶段 16 输出与上游封存文件。
"""
from __future__ import annotations

import argparse
from hashlib import md5
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mobilise_d import sources  # noqa: E402
from mobilise_d.clinical import load_clinical  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise" / "round2"
STAGE11 = OUTPUT_ROOT / "stage11" / "20261115-protocol-v1"
STAGE12 = OUTPUT_ROOT / "stage12" / "20261115-supplement-v1"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(run_dir: Path) -> dict:
    checks: dict[str, Any] = {}
    keys = pd.read_csv(run_dir / "real_keys.csv", dtype={"participantid": str})
    counts = _load(run_dir / "counts.json")
    reference = _load(run_dir / "reference_comparison.json")
    definitions = _load(run_dir / "window_definitions.json")
    exclusions = pd.read_csv(run_dir / "exclusions.csv", dtype={"participantid": str})
    synthetic_excluded = pd.read_csv(run_dir / "synthetic_exclusions.csv")
    boundary = pd.read_csv(run_dir / "boundary_samples.csv", dtype={"participantid": str})
    splits = pd.read_csv(STAGE11 / "splits.csv", dtype={"participantid": str})

    # 0. 原包事件表（独立连接）
    resolved = sources.ensure_files(sources.default_cache_dir(), sources.COHORT_FILES,
                                    offline=True)
    clinical = load_clinical(resolved["Clinical Data.zip"])
    reports = clinical.reports.copy()
    reports["participantid"] = reports["participantid"].astype(str)
    reports["fall_day"] = pd.to_datetime(reports["falldate"], errors="coerce").dt.normalize()
    reports = reports.loc[reports["fall_day"].notna(), ["participantid", "fall_day"]]
    reports = reports.drop_duplicates()

    # 1. 分母重建：从键表 + 事件表独立重算
    corrected = keys.loc[keys["corrected_key"].astype(bool)].copy()
    checks["corrected_rows"] = int(len(corrected))
    checks["corrected_people"] = int(corrected["participantid"].nunique())
    checks["matches_reference_rows"] = bool(
        len(corrected) == reference["reference_rows_after_same_day_removal"]
    )
    checks["matches_reference_people"] = bool(
        corrected["participantid"].nunique() == reference["reference_people_after_same_day_removal"]
    )
    positives = {}
    for horizon in (1, 3, 7):
        landmark = pd.to_datetime(corrected["landmark_day"])
        merged = corrected[["participantid", "landmark_day"]].copy()
        merged["landmark"] = landmark
        joined = merged.merge(reports, on="participantid", how="left")
        in_window = joined.loc[
            (joined["fall_day"] > joined["landmark"])
            & (joined["fall_day"] <= joined["landmark"] + pd.Timedelta(days=horizon))
        ]
        keys_with_event = set(zip(in_window["participantid"], in_window["landmark_day"]))
        recomputed = int(
            sum(
                1 for person, day in zip(corrected["participantid"], corrected["landmark_day"])
                if (person, day) in keys_with_event
            )
        )
        positives[f"{horizon}d"] = recomputed
        checks[f"recomputed_positives_{horizon}d"] = recomputed
        checks[f"positives_match_table_{horizon}d"] = bool(
            recomputed == int(corrected[f"y_{horizon}d"].astype(bool).sum())
        )
    checks["positives_1_3_7"] = [positives["1d"], positives["3d"], positives["7d"]]
    checks["positives_match_reference"] = bool(
        checks["positives_1_3_7"] == reference["reference_positives_1_3_7"]
    )

    # 2. 同日排除：键表、排除表与原包三处一致
    same_day = keys.loc[keys["same_day_fall"].astype(bool)]
    merged = keys[["participantid", "landmark_day"]].copy()
    merged["landmark"] = pd.to_datetime(merged["landmark_day"])
    joined = merged.merge(reports, on="participantid", how="left")
    same_day_independent = joined.loc[joined["fall_day"] == joined["landmark"]]
    independent_keys = set(zip(same_day_independent["participantid"],
                               same_day_independent["landmark_day"]))
    table_keys = set(zip(same_day["participantid"], same_day["landmark_day"]))
    checks["same_day_keys_table"] = int(len(table_keys))
    checks["same_day_keys_independent"] = int(len(independent_keys))
    checks["same_day_keys_agree"] = bool(table_keys == independent_keys)
    checks["no_same_day_key_in_corrected"] = bool(
        not (set(zip(corrected["participantid"], corrected["landmark_day"])) & independent_keys)
    )
    checks["same_day_exclusion_reason"] = bool(
        (same_day.loc[same_day["precision_scorable"].astype(bool),
                      "exclusion_reason"] == "same_day_fall").all()
    )
    checks["known_same_day_and_next_day_record"] = bool(
        ((same_day["participantid"] == "1940")
         & (same_day["landmark_day"] == "2022-05-12")).any()
    )
    next_day = reports.loc[
        (reports["participantid"] == "1940")
        & (reports["fall_day"] == pd.Timestamp("2022-05-13"))
    ]
    checks["record_has_next_day_fall"] = bool(len(next_day) == 1)
    checks["exclusions_table_matches_keys"] = bool(
        len(exclusions) == int((~keys["corrected_key"].astype(bool)).sum())
    )

    # 3. 边界归属：t / t+1 / t+3 / t+7 / t+8
    for offset in (0, 1, 3, 7, 8):
        checks[f"boundary_rows_at_t_plus_{offset}"] = int(
            boundary[f"fall_at_t_plus_{offset}"].astype(bool).sum()
        )
    checks["boundary_covers_t"] = checks["boundary_rows_at_t_plus_0"] >= 1
    checks["boundary_covers_t_plus_1"] = checks["boundary_rows_at_t_plus_1"] >= 1
    checks["boundary_covers_t_plus_3"] = checks["boundary_rows_at_t_plus_3"] >= 1
    checks["boundary_covers_t_plus_7"] = checks["boundary_rows_at_t_plus_7"] >= 1
    checks["boundary_covers_t_plus_8"] = checks["boundary_rows_at_t_plus_8"] >= 1
    corrected_boundary = boundary.loc[boundary["corrected_key"].astype(bool)]
    for horizon in (1, 3, 7):
        expected = corrected_boundary[f"expected_y_{horizon}d"].astype(bool)
        observed = corrected_boundary[f"y_{horizon}d"].astype(bool)
        checks[f"boundary_label_agrees_{horizon}d"] = bool((expected == observed).all())
    # t+8 不得进入 7 天窗口
    checks["t_plus_8_not_in_7d"] = bool(
        not (
            corrected_boundary["fall_at_t_plus_8"].astype(bool)
            & ~corrected_boundary[[f"fall_at_t_plus_{d}" for d in range(1, 8)]].any(axis=1)
            & corrected_boundary["y_7d"].astype(bool)
        ).any()
    )
    # 随访不足样例
    insufficient = keys.loc[
        ~keys[["followup_ok_1d", "followup_ok_3d", "followup_ok_7d"]].astype(bool).all(axis=1)
    ]
    checks["insufficient_followup_rows"] = int(len(insufficient))
    checks["insufficient_followup_labels_unknown"] = bool(
        insufficient[[f"y_{h}d" for h in (1, 3, 7)]].isna().all(axis=1).all()
    ) if len(insufficient) else True

    # 4. 模拟同日剔除完整
    synthetic_test = pd.read_csv(STAGE12 / "synthetic_test_scores.csv", low_memory=False)
    synthetic_train = pd.read_csv(STAGE12 / "synthetic_train_scores.csv", low_memory=False)
    for name, frame in (("test", synthetic_test), ("train", synthetic_train)):
        scored = frame.loc[frame["status"] == "scored"]
        gap = pd.to_numeric(scored["event_gap_from_snapshot"], errors="coerce")
        expected = set(scored.loc[gap == 0, "series_id"])
        listed = set(synthetic_excluded.loc[synthetic_excluded["batch"] == name, "series_id"])
        checks[f"synthetic_{name}_same_day_n"] = int(len(expected))
        checks[f"synthetic_{name}_same_day_complete"] = bool(expected == listed)
        checks[f"synthetic_{name}_same_day_no_duplicates"] = bool(
            len(listed) == len(expected)
        )
    counts_synthetic = counts["synthetic"]
    checks["synthetic_test_same_day_117"] = bool(
        counts_synthetic["test_same_day_excluded"] == 117
    )
    checks["synthetic_train_same_day_37"] = bool(
        counts_synthetic["train_same_day_excluded"] == 37
    )

    # 5. 分折一致
    fold_map = dict(zip(splits["participantid"], splits["fold"].astype(int)))
    pd_keys = keys.loc[keys["cohort"].eq("pd") & keys["baseline_eligible"].astype(bool)].copy()
    pd_keys["fold_expected"] = pd_keys["participantid"].map(fold_map)
    checks["pd_keys_missing_fold"] = int(pd_keys["fold_expected"].isna().sum())
    checks["fold_matches_splits"] = bool(
        (pd_keys["fold"].astype(float) == pd_keys["fold_expected"].astype(float)).all()
    )
    checks["people_single_fold"] = bool(
        (pd_keys.groupby("participantid")["fold"].nunique() == 1).all()
    )
    checks["baseline_ready_matches_split_people"] = bool(
        pd_keys["participantid"].nunique() == splits["participantid"].nunique()
    )
    checks["non_pd_keys_have_no_fold"] = bool(keys.loc[~keys["cohort"].eq("pd"), "fold"].isna().all())

    # 6. 标签与阶段 11 对照
    checks["label_mismatch_vs_stage11"] = counts["label_mismatch_vs_stage11"]
    checks["labels_unchanged_where_used"] = bool(
        all(value == 0 for value in counts["label_mismatch_vs_stage11"].values())
    )

    # 7. 时段端点
    checks["observation_cutoff_declared"] = bool(
        "<= t" in definitions["observation_cutoff"]["rule"]
    )
    checks["monitoring_window_30"] = bool(
        definitions["monitoring_window"]["days"] == 30
        and "第 1" in definitions["monitoring_window"]["start"]
        and "第 30" in definitions["monitoring_window"]["end"]
    )
    checks["outcome_tail_separate"] = bool(
        definitions["outcome_tail"]["days"] == 7
        and "第 31" in definitions["outcome_tail"]["start"]
    )
    checks["change_window_half_open"] = bool(
        "[u, u+H)" in definitions["change_detection_window"]["window"]
        and "变化前报警" in definitions["change_detection_window"]["pre_change_alarm"]
    )
    checks["change_window_named_separately"] = bool(
        "区别于 fall_window" in definitions["change_detection_window"]["naming"]
    )
    checks["horizons_fixed"] = bool(list(definitions["horizons_days"]) == [1, 3, 7]
                                    and int(definitions["primary_horizon_days"]) == 7)

    # 8. 上游引用与代码快照
    manifest = _load(run_dir / "manifest.json")
    checks["manifest_records_upstream"] = bool(
        len(manifest["upstream"]) == 2
        and all(len(value) >= 3 for value in manifest["upstream"].values())
    )
    checks["manifest_hashes_match"] = all(
        _md5(Path(path) / name) == digest
        for path, entries in manifest["upstream"].items()
        for name, digest in entries.items()
    )
    checks["code_snapshot_present"] = bool(
        len(list((run_dir / "code_snapshot").glob("*.py"))) >= 4
    )
    checks["release_recorded"] = bool(
        manifest["release"]["doi"] == "10.5281/zenodo.21060991"
    )
    checks["counts_reference_explained"] = bool(
        reference["explanation"]["one_key_removed"] and reference["explanation"]["synthetic_change"]
    )

    verdict = [key for key, value in checks.items()
               if isinstance(value, (bool, np.bool_)) and not key.startswith("boundary_rows_at")]
    all_pass = all(bool(checks[key]) for key in verdict)
    return {
        "stage": 16,
        "run_dir": str(run_dir),
        "all_pass": bool(all_pass),
        "failed": [key for key in verdict if not checks[key]],
        "checks": checks,
    }


def _render(report: dict) -> str:
    lines = [
        "# 阶段 16 核验",
        "",
        f"- 运行：`{report['run_dir']}`",
        f"- 结论：{'全部通过' if report['all_pass'] else '未通过：' + ', '.join(report['failed'])}",
        "",
        "| 核验项 | 结果 |",
        "|---|---|",
    ]
    for key, value in report["checks"].items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"| `{key}` | {value} |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 16 核验")
    parser.add_argument("--run", default=str(OUTPUT_ROOT / "stage16" / "20261115-keys-v2"))
    args = parser.parse_args(argv)
    run_dir = Path(args.run)
    report = verify(run_dir)
    (run_dir / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "VERIFICATION.md").write_text(_render(report), encoding="utf-8")
    print(json.dumps({key: value for key, value in report["checks"].items()
                      if not isinstance(value, (dict, list))}, ensure_ascii=False, indent=1))
    print("ALL PASS" if report["all_pass"] else "FAILED: " + ", ".join(report["failed"]))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
