"""阶段 16：修正风险集并固定三个时间边界。

只读原包与阶段 11/12 的封存输入，输出更正后的合格键、排除原因、逐阶段
分母与边界样例。本阶段不训练模型、不计算性能。

三项更正的第一项在这里落地：**预测日当天已跌倒的时点整体剔除**（训练与
测试两侧），分母从原始事件日期重建；同时冻结评分截止到 t、30 天监测窗与
已知变化检出窗三个端点，供阶段 17/18 直接引用。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import md5
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mobilise_d import sources  # noqa: E402
from mobilise_d.clinical import load_clinical  # noqa: E402
from mobilise_d.round2 import HORIZONS_DAYS, PRIMARY_HORIZON_DAYS  # noqa: E402
from mobilise_d.round2_time import (  # noqa: E402
    ALARM_WINDOW_DAYS, KEYS_VERSION, OUTCOME_TAIL_DAYS, REFERENCE_CHECK,
    SCORING_VERSION, THRESHOLD_VERSION, cohort_stage_table, dated_falls,
    diary_coverage, nested_and_boundary_audit, real_key_table, rebuild_windows,
    synthetic_key_counts, synthetic_same_day, window_definitions,
)

OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise" / "round2"
STAGE11 = OUTPUT_ROOT / "stage11" / "20261115-protocol-v1"
STAGE12 = OUTPUT_ROOT / "stage12" / "20261115-supplement-v1"

CODE_SNAPSHOT_FILES = (
    "mobilise_d/round2_time.py",
    "mobilise_d/round2.py",
    "mobilise_d/clinical.py",
    "scripts/round2_stage16_keys.py",
    "scripts/verify_round2_stage16.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 16：时间边界与合格键")
    parser.add_argument("--run-id", default="20261115-keys-v2")
    parser.add_argument("--out-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--stage11", default=str(STAGE11))
    parser.add_argument("--stage12", default=str(STAGE12))
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")


def _md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(repo: Path) -> dict:
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                                capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": commit, "dirty": bool(status),
                "status_lines": len(status.splitlines())}
    except Exception as error:  # noqa: BLE001
        return {"commit": None, "error": f"{type(error).__name__}: {error}"}


def _snapshot_code(run_dir: Path) -> list[str]:
    target = run_dir / "code_snapshot"
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for relative in CODE_SNAPSHOT_FILES:
        source = PROJECT_ROOT / relative
        if not source.is_file():
            continue
        destination = target / Path(relative).name
        shutil.copy2(source, destination)
        written.append(str(destination))
    return written


def run(args: argparse.Namespace) -> int:
    run_dir = Path(args.out_root) / f"stage16/{args.run_id}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"运行目录已存在且非空，拒绝覆盖：{run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    stage11 = Path(args.stage11)
    stage12 = Path(args.stage12)

    print("[stage16] 读取原包事件日期与阶段 11/12 输入", flush=True)
    resolved = sources.ensure_files(sources.default_cache_dir(), sources.COHORT_FILES,
                                    offline=True)
    clinical = load_clinical(resolved["Clinical Data.zip"])
    falls = dated_falls(clinical)
    coverage = diary_coverage(clinical)

    candidates = pd.read_csv(stage11 / "real_candidates.csv", dtype={"participantid": str})
    participants = pd.read_csv(stage11 / "participants.csv", dtype={"participantid": str})
    splits = pd.read_csv(stage11 / "splits.csv", dtype={"participantid": str})
    scores = pd.read_csv(stage12 / "real_scores.csv", dtype={"participantid": str})
    synthetic_test = pd.read_csv(stage12 / "synthetic_test_scores.csv", low_memory=False)
    synthetic_train = pd.read_csv(stage12 / "synthetic_train_scores.csv", low_memory=False)

    print("[stage16] 从原始日期重建 1/3/7 天窗口", flush=True)
    windows = rebuild_windows(candidates, falls, coverage)
    keys = real_key_table(candidates, windows, scores, splits)

    # 与阶段 11 的旧标签对照（只定位差异，不参与更正）
    mismatch: dict[str, int] = {}
    for horizon in HORIZONS_DAYS:
        old = f"y_old_{horizon}d"
        new = f"y_{horizon}d"
        if old in keys.columns:
            comparable = keys[[old, new]].dropna()
            mismatch[f"{horizon}d"] = int(
                (comparable[old].astype(bool) != comparable[new].astype(bool)).sum()
            )
    audit = nested_and_boundary_audit(keys, falls)
    boundary_table = audit.pop("boundary_table")
    boundary_table.to_csv(run_dir / "boundary_samples.csv", index=False, encoding="utf-8")

    # 阶段 16 主表
    keys.to_csv(run_dir / "real_keys.csv", index=False, encoding="utf-8")
    exclusions = keys.loc[~keys["corrected_key"]].copy()
    exclusions[[
        "participantid", "cohort", "visit_number", "landmark_day", "fold",
        "baseline_eligible", "precision_status", "same_day_fall",
        "same_day_fall_dates", "exclusion_reason",
    ]].to_csv(run_dir / "exclusions.csv", index=False, encoding="utf-8")

    stage_table = cohort_stage_table(keys, participants.loc[participants["cohort"] == "pd"])
    stage_table.to_csv(run_dir / "cohort_flow_corrected.csv", index=False,
                       encoding="utf-8")

    same_day_all = keys.loc[keys["same_day_fall"].astype(bool)].copy()
    same_day_all[[
        "participantid", "cohort", "visit_number", "landmark_day", "fold",
        "baseline_eligible", "precision_scorable", "exclusion_reason",
        "same_day_fall_dates", "y_1d", "y_3d", "y_7d", "first_event_1d",
        "first_event_3d", "first_event_7d",
    ]].to_csv(run_dir / "same_day_records.csv", index=False, encoding="utf-8")

    # 模拟分支：同日事件剔除计数
    test_excluded = synthetic_same_day(synthetic_test.assign(batch="test"))
    train_excluded = synthetic_same_day(synthetic_train.assign(batch="train"))
    excluded_frame = pd.concat([train_excluded, test_excluded], ignore_index=True)
    excluded_frame.to_csv(run_dir / "synthetic_exclusions.csv", index=False,
                          encoding="utf-8")
    synthetic_counts = {
        "train": synthetic_key_counts(synthetic_train.assign(batch="train"), batch="train"),
        "test": synthetic_key_counts(synthetic_test.assign(batch="test"), batch="test"),
    }
    _write_json(run_dir / "synthetic_key_counts.json", synthetic_counts)

    # 三个时段的口径
    definitions = window_definitions()
    definitions["reference_check"] = REFERENCE_CHECK
    definitions["frozen_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_json(run_dir / "window_definitions.json", definitions)

    # 分母与差异解释
    corrected = keys.loc[keys["corrected_key"]]
    old_test = keys.loc[keys["precision_scorable"]].copy()
    counts: dict[str, Any] = {
        "keys_version": KEYS_VERSION,
        "scoring_version": SCORING_VERSION,
        "threshold_version": THRESHOLD_VERSION,
        "real": {
            "candidate_rows_all_cohorts": int(len(keys)),
            "candidate_rows_pd": int((keys["cohort"] == "pd").sum()),
            "r_branch_eligible": int(keys["r_branch_eligible"].astype(bool).sum()),
            "precision_scorable_before_same_day": int(len(old_test)),
            "precision_scorable_people_before_same_day": int(old_test["participantid"].nunique()),
            "same_day_excluded": int((~keys["corrected_key"] & keys["precision_scorable"]).sum()),
            "corrected_keys": int(len(corrected)),
            "corrected_people": int(corrected["participantid"].nunique()),
            "positives_before_same_day": {
                f"{h}d": int(old_test[f"y_{h}d"].sum()) for h in HORIZONS_DAYS
            },
            "positives_corrected": {
                f"{h}d": int(corrected[f"y_{h}d"].sum()) for h in HORIZONS_DAYS
            },
            "positive_people_corrected": {
                f"{h}d": int(corrected.loc[corrected[f"y_{h}d"].eq(True), "participantid"].nunique())
                for h in HORIZONS_DAYS
            },
            "corrected_by_fold": {
                str(fold): int((corrected["fold"] == fold).sum()) for fold in (0, 1, 2)
            },
        },
        "synthetic": {
            "train_rows": synthetic_counts["train"]["rows"],
            "train_same_day_excluded": synthetic_counts["train"]["same_day_excluded"],
            "test_rows": synthetic_counts["test"]["rows"],
            "test_same_day_excluded": synthetic_counts["test"]["same_day_excluded"],
            "test_same_day_by_fold": {
                str(fold): int((test_excluded["fold"] == fold).sum()) for fold in (0, 1, 2)
            },
            "train_same_day_by_fold": {
                str(fold): int((train_excluded["fold"] == fold).sum()) for fold in (0, 1, 2)
            },
        },
        "label_mismatch_vs_stage11": mismatch,
        "audit": {key: value for key, value in audit.items() if key != "boundary_table"},
    }
    reference = {
        "reference_rows_after_same_day_removal": REFERENCE_CHECK[
            "reference_rows_after_same_day_removal"],
        "actual_rows_after_same_day_removal": counts["real"]["corrected_keys"],
        "reference_people_after_same_day_removal": REFERENCE_CHECK[
            "reference_people_after_same_day_removal"],
        "actual_people_after_same_day_removal": counts["real"]["corrected_people"],
        "reference_positives_1_3_7": REFERENCE_CHECK["reference_positives_1_3_7"],
        "actual_positives_1_3_7": [
            counts["real"]["positives_corrected"][f"{h}d"] for h in HORIZONS_DAYS
        ],
        "reference_old_test_rows": REFERENCE_CHECK["old_real_test_rows"],
        "actual_old_test_rows": counts["real"]["precision_scorable_before_same_day"],
        "reference_old_test_people": REFERENCE_CHECK["old_real_test_people"],
        "actual_old_test_people": counts["real"]["precision_scorable_people_before_same_day"],
        "reference_synthetic_same_day": REFERENCE_CHECK["old_synthetic_test_same_day_rows"],
        "actual_synthetic_same_day": counts["synthetic"]["test_same_day_excluded"],
        "explanation": {
            "one_key_removed": (
                "PD 1940 的 2022-05-12 预测日当天有跌倒（2022-05-13 再次跌倒），"
                "按新规则整日剔除；该键在旧评估中同时是 1/3/7 天阳性"
            ),
            "people_change": "该人另一个候选预测日不具备 14 天基线资格，因此人数同步减少 1",
            "synthetic_change": (
                f"旧共同测试的 {REFERENCE_CHECK['old_synthetic_test_same_day_rows']} 条同日事件"
                f"记录与重建值 {counts['synthetic']['test_same_day_excluded']} 条一致"
            ),
        },
    }
    _write_json(run_dir / "counts.json", counts)
    _write_json(run_dir / "reference_comparison.json", reference)

    code_files = _snapshot_code(run_dir)
    manifest = {
        "stage": 16,
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "upstream": {
            str(stage11): {
                "real_candidates.csv": _md5(stage11 / "real_candidates.csv"),
                "splits.csv": _md5(stage11 / "splits.csv"),
                "participants.csv": _md5(stage11 / "participants.csv"),
                "protocol.json": _md5(stage11 / "protocol.json"),
            },
            str(stage12): {
                "real_scores.csv": _md5(stage12 / "real_scores.csv"),
                "synthetic_test_scores.csv": _md5(stage12 / "synthetic_test_scores.csv"),
                "synthetic_train_scores.csv": _md5(stage12 / "synthetic_train_scores.csv"),
            },
        },
        "release": {
            "doi": "10.5281/zenodo.21060991",
            "version": "V1.0.0",
            "license": "CC BY-NC-ND 4.0",
            "files_used": ["Clinical Data.zip", "Aggregated DMO Data.zip"],
        },
        "environment": {
            "python_version": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "executable": sys.executable,
        },
        "git": _git_state(PROJECT_ROOT),
        "code_snapshot": code_files,
        "outputs": {
            "real_keys": str(run_dir / "real_keys.csv"),
            "exclusions": str(run_dir / "exclusions.csv"),
            "cohort_flow_corrected": str(run_dir / "cohort_flow_corrected.csv"),
            "same_day_records": str(run_dir / "same_day_records.csv"),
            "synthetic_exclusions": str(run_dir / "synthetic_exclusions.csv"),
            "synthetic_key_counts": str(run_dir / "synthetic_key_counts.json"),
            "window_definitions": str(run_dir / "window_definitions.json"),
            "boundary_samples": str(run_dir / "boundary_samples.csv"),
            "counts": str(run_dir / "counts.json"),
            "reference_comparison": str(run_dir / "reference_comparison.json"),
        },
    }
    _write_json(run_dir / "manifest.json", manifest)
    print(json.dumps(counts, ensure_ascii=False, indent=1)[:4000], flush=True)
    print(f"[stage16] 完成：{run_dir}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
