"""重建 Mobilise-D 阶段 1 队列：日表、事件与覆盖表、候选预测日、流程计数。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import md5
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mobilise_d import sources  # noqa: E402
from mobilise_d.clinical import load_clinical  # noqa: E402
from mobilise_d.cohort import CohortConfig, build_cohort  # noqa: E402
from mobilise_d.daily import (  # noqa: E402
    field_mapping,
    load_all_days,
    load_daily,
    unit_consistency_checks,
)


DEFAULT_OUT = PROJECT_ROOT / "datasets" / "mobilise_d" / "cvs_v1.0.0"

CODE_SNAPSHOT_FILES = (
    "mobilise_d/__init__.py",
    "mobilise_d/sources.py",
    "mobilise_d/clinical.py",
    "mobilise_d/daily.py",
    "mobilise_d/cohort.py",
    "scripts/fetch_mobilise_d.py",
    "scripts/build_mobilise_cohort.py",
    "scripts/inspect_mobilise_bouts.py",
    "scripts/verify_mobilise_stage1.py",
    "tests/test_mobilise_cohort.py",
)

# 上次核验（output/ds_pcd_review_20261113）的实际计数，作为导入参照。
REFERENCE = {
    "wide": {"people": 231, "landmarks": 463, "positive_7d": 13, "positive_30d": 40,
             "positive_people_7d": 13, "positive_people_30d": 37},
    "baseline_ready": {"people": 144, "landmarks": 186, "positive_7d": 4, "positive_30d": 17,
                       "positive_people_7d": 4, "positive_people_30d": 15},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建 Mobilise-D 阶段 1 可评估队列")
    parser.add_argument("--cache-dir", default=None, help="原始包目录")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="派生数据目录")
    parser.add_argument("--skip-bouts", action="store_true", help="不校验步行段包（仅用于包未下载时）")
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
        "repo": str(repo),
        "commit": commit,
        "dirty": bool(changed),
        "changed_path_count": len(changed),
        "changed_paths_sample": changed[:40],
    }


def _write_table(frame: pd.DataFrame, path: Path) -> dict:
    frame.to_csv(path, index=False, encoding="utf-8")
    return {"path": path.name, "rows": int(len(frame)), "columns": list(frame.columns), "md5": _md5(path)}


def _code_snapshot(out_dir: Path) -> list[dict]:
    """把本次研究代码复制进派生目录，避免未提交改动无法追溯。"""
    snapshot_dir = out_dir / "code_snapshot"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    entries = []
    for relative in CODE_SNAPSHOT_FILES:
        source = PROJECT_ROOT / relative
        target = snapshot_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        entries.append({"path": relative, "md5": _md5(target), "bytes": target.stat().st_size})
    return entries


def _reference_check(counts: dict) -> dict:
    wide = counts["horizons"]
    actual = {
        "wide": {
            "people": counts["candidate_people"],
            "landmarks": counts["candidate_landmarks"],
            "positive_7d": wide["7d"]["positive_landmarks"],
            "positive_30d": wide["30d"]["positive_landmarks"],
            "positive_people_7d": wide["7d"]["positive_people"],
            "positive_people_30d": wide["30d"]["positive_people"],
        },
        "baseline_ready": {
            "people": counts["baseline_ready_people"],
            "landmarks": counts["baseline_ready_landmarks"],
            "positive_7d": wide["7d"]["baseline_ready_positive_landmarks"],
            "positive_30d": wide["30d"]["baseline_ready_positive_landmarks"],
            "positive_people_7d": wide["7d"]["baseline_ready_positive_people"],
            "positive_people_30d": wide["30d"]["baseline_ready_positive_people"],
        },
    }
    diffs = []
    for group, expected in REFERENCE.items():
        for key, value in expected.items():
            if actual[group][key] != value:
                diffs.append({
                    "group": group, "metric": key,
                    "expected": value, "actual": actual[group][key],
                })
    return {"expected": REFERENCE, "actual": actual, "mismatches": diffs, "matches": not diffs}


def _sensitivity(
    clinical, daily, config: CohortConfig, all_days: pd.DataFrame | None
) -> dict:
    """佩戴质控与发布方过滤规则造成的收缩，单独列出。"""
    main = build_cohort(clinical, daily, config).counts
    report = {
        "main": {
            "candidate_landmarks": main["candidate_landmarks"],
            "candidate_people": main["candidate_people"],
            "baseline_ready_landmarks": main["baseline_ready_landmarks"],
            "positive_7d": main["horizons"]["7d"]["positive_landmarks"],
            "positive_30d": main["horizons"]["30d"]["positive_landmarks"],
        },
        "wear_thresholds": {},
    }
    for hours in config.wear_sensitivity_hours:
        scenario = CohortConfig(**{
            **config.to_dict(),
            "horizons": tuple(config.horizons),
            "feature_columns": tuple(config.feature_columns),
            "require_wear_hours": hours,
        })
        counts = build_cohort(clinical, daily, scenario).counts
        report["wear_thresholds"][f"{hours:g}h"] = {
            "valid_day_rows_before": int(daily["has_core_gait"].sum()),
            "valid_day_rows_after": int(
                (daily["has_core_gait"] & daily["total_worn_during_waking_h_d"].ge(hours)).sum()
            ),
            "candidate_landmarks": counts["candidate_landmarks"],
            "candidate_people": counts["candidate_people"],
            "baseline_ready_landmarks": counts["baseline_ready_landmarks"],
            "positive_7d": counts["horizons"]["7d"]["positive_landmarks"],
            "positive_30d": counts["horizons"]["30d"]["positive_landmarks"],
        }
    report["wear_note"] = (
        "发布方 daily_agg_filtered 表已把清醒佩戴时长限制在 12 小时以上"
        "（最小值 12.001），因此 8/10/12 小时阈值不再产生收缩；"
        "佩戴质控在本数据上不是额外的选择来源。"
    )
    if all_days is not None:
        counts = build_cohort(clinical, all_days, config).counts
        report["release_all_days"] = {
            "candidate_landmarks": counts["candidate_landmarks"],
            "candidate_people": counts["candidate_people"],
            "baseline_ready_landmarks": counts["baseline_ready_landmarks"],
            "positive_7d": counts["horizons"]["7d"]["positive_landmarks"],
            "positive_30d": counts["horizons"]["30d"]["positive_landmarks"],
            "note": "不套用发布方 daily_agg_filtered 的过滤规则，仅要求三项步态特征可用；"
                    "用于说明发布方过滤本身造成的收缩。",
        }
    return report


def _coverage_decomposition(gates: pd.DataFrame) -> dict:
    """解释窗口排除：采集结束日在日记区间之前，或距日记结束不足 30 天。"""
    if gates.empty:
        return {}
    excluded = gates.loc[~gates["candidate"]].copy()
    window_rows = excluded.loc[excluded["exclusion_reason"] == "landmark_outside_diary_window"].copy()
    window_rows["landmark_days_to_diary_end"] = (
        pd.to_datetime(window_rows["diary_day"]) - pd.to_datetime(window_rows["block_first_day"])
    ).dt.days
    before = window_rows.loc[
        pd.to_datetime(window_rows["block_first_day"]) < pd.to_datetime(window_rows["covered_from"])
    ]
    insufficient = window_rows.loc[
        (pd.to_datetime(window_rows["block_first_day"]) >= pd.to_datetime(window_rows["covered_from"]))
        & (window_rows["landmark_days_to_diary_end"] < 30)
    ]
    return {
        "excluded_landmark_before_diary_coverage": int(len(before)),
        "excluded_landmark_too_close_to_diary_end_for_30d": int(len(insufficient)),
        "excluded_window_other": int(len(window_rows) - len(before) - len(insufficient)),
    }


def run(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir) if args.cache_dir else sources.default_cache_dir()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    required = [sources.CLINICAL, sources.DAILY, sources.DICTIONARY]
    resolved = sources.ensure_files(cache_dir, required, offline=True)
    bout_status = "skipped"
    bout_info = None
    if not args.skip_bouts:
        try:
            bout_path = sources.ensure_files(cache_dir, [sources.BOUTS], offline=True)
            bout_status = "verified"
            bout_info = {
                "path": str(bout_path[sources.BOUTS.name]),
                "bytes": bout_path[sources.BOUTS.name].stat().st_size,
                "md5": sources.BOUTS.md5,
            }
        except FileNotFoundError as exc:
            bout_status = "missing"
            bout_info = {"error": str(exc)}

    clinical = load_clinical(resolved[sources.CLINICAL.name])
    daily = load_daily(resolved[sources.DAILY.name])
    config = CohortConfig()
    result = build_cohort(clinical, daily.frame, config)

    outputs = []
    outputs.append(_write_table(result.participants, out_dir / "participants.csv"))
    outputs.append(_write_table(result.visits, out_dir / "visits.csv"))
    outputs.append(_write_table(result.daily, out_dir / "daily_gait.csv"))
    outputs.append(_write_table(result.diaries, out_dir / "fall_diaries.csv"))
    outputs.append(_write_table(result.reports, out_dir / "fall_reports.csv"))
    outputs.append(_write_table(result.gates, out_dir / "cohort_gates.csv"))
    outputs.append(_write_table(result.landmarks, out_dir / "landmarks.csv"))
    outputs.append(_write_table(result.baseline_days, out_dir / "baseline_days.csv"))
    outputs.append(_write_table(result.windows, out_dir / "landmark_windows.csv"))
    outputs.append(_write_table(result.flow, out_dir / "cohort_flow.csv"))

    unit_checks = unit_consistency_checks(daily.frame)
    sources.write_json(out_dir / "unit_checks.json", unit_checks)
    sources.write_json(out_dir / "field_mapping.json", field_mapping())
    all_days = None
    try:
        all_days = load_all_days(resolved[sources.DAILY.name]).frame
    except Exception as exc:  # 场景失败不阻断主产物，记录原因
        all_days_note = sources.describe_error(exc)
    else:
        all_days_note = None
    sensitivity = _sensitivity(clinical, daily.frame, config, all_days)
    sensitivity["all_days_scenario_error"] = all_days_note
    sources.write_json(out_dir / "sensitivity.json", sensitivity)
    for extra in ("field_mapping.json", "unit_checks.json", "sensitivity.json"):
        outputs.append({"path": extra, "rows": None, "md5": _md5(out_dir / extra)})
    if (out_dir / "bout_support.json").is_file():
        outputs.append({
            "path": "bout_support.json",
            "rows": None,
            "md5": _md5(out_dir / "bout_support.json"),
        })

    reference = _reference_check(result.counts)
    coverage = _coverage_decomposition(result.gates)
    bout_support = None
    bout_support_path = out_dir / "bout_support.json"
    if bout_support_path.is_file():
        support = json.loads(bout_support_path.read_text(encoding="utf-8"))
        reconstruction = support.get("daily_reconstruction", {})
        bout_support = {
            "path": bout_support_path.name,
            "md5": _md5(bout_support_path),
            "rows_scanned": (support.get("coverage_all_rows") or {}).get("rows"),
            "reconstructed_days": reconstruction.get("matched_days"),
            "exact_match_min_share": min(
                (item["exact_share"] or 0) for item in reconstruction.get("exact_rule", {}).values()
            ) if reconstruction.get("exact_rule") else None,
        }

    manifest = {
        "stage": 1,
        "plan": "planning-files/ds-pcd-mobilise-implementation-plan.md",
        "cohort_version": "cohort_cvs_v1.0.0",
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "release": sources.release_manifest(),
        "cache": {
            "dir": str(cache_dir),
            "files": [
                {
                    "name": item.name,
                    "path": str(cache_dir / item.name),
                    "bytes": (cache_dir / item.name).stat().st_size,
                    "md5": item.md5,
                    "status": "verified",
                }
                for item in required
            ],
            "walking_bouts": {"status": bout_status, **(bout_info or {})},
        },
        "code": _git_state(PROJECT_ROOT.parent),
        "code_snapshot": _code_snapshot(out_dir),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": __import__("numpy").__version__,
        },
        "config": config.to_dict(),
        "population": {
            "description": "Mobilise-D CVS 帕金森队列，基线年龄 >= 65 岁",
            "cohort_flag": "cohort pd == T",
            "age_source": "descriptives1.age（T1）",
        },
        "outcome_definition": {
            "window": "(t, t+H], H in {7, 30}; 同日跌倒不计入",
            "coverage": "日记总数与逐次报告条数一致，逐次报告日期都在 [covered_from, diary_day]",
            "negative_requires": "日记覆盖到 t+H 且上述一致性成立",
            "diary_window": "max(上一次访视日, 本次日记日 - 6 个月)",
        },
        "archive_inventory": {
            item.name: sources.zip_inventory(cache_dir / item.name)
            for item in required
        },
        "field_mapping": "field_mapping.json",
        "unit_checks": unit_checks,
        "daily_notes": daily.notes,
        "clinical_notes": clinical.notes(),
        "cohort_notes": result.notes,
        "coverage_decomposition": coverage,
        "counts": result.counts,
        "reference_comparison": reference,
        "sensitivity": "sensitivity.json",
        "bout_support": bout_support,
        "outputs": outputs,
        "limitations": [
            "Mobilise-D CVS 是每半年一次的临床访视，每次有效观测日通常 5–9 天，"
            "不能当作连续日常监测；跨访视拼出的 14 天基线跨度可达数月。",
            "候选预测日要求日记与逐次跌倒报告一致且覆盖完整窗口；"
            "因记录不一致或随访缺失被排除的访视带有选择性。",
            "逐日 strlen_*_avg_d 为厘米，Data Dictionary 的米是步行段级定义。",
            "逐日表只有聚合值，测量标准误需在阶段 2 从步行段包估计；"
            "本阶段只核查逐段字段支持度。",
            "研究人群为 65 岁及以上帕金森患者，结果不外推到其他疾病或社区老人。",
        ],
        "rerun": "python scripts/build_mobilise_cohort.py",
    }
    sources.write_json(out_dir / "manifest.json", manifest)

    report = [
        "# Mobilise-D 阶段 1 派生数据说明",
        "",
        "本目录由 `scripts/build_mobilise_cohort.py` 从 Mobilise-D CVS V1.0.0 原始包重建。",
        "",
        "## 重建命令",
        "",
        "```bash",
        "python scripts/fetch_mobilise_d.py --verify-only",
        "python scripts/build_mobilise_cohort.py",
        "```",
        "",
        "## 数据版本",
        "",
        f"- 发布：{sources.ATTRIBUTION}",
        f"- 原始包缓存：`{cache_dir}`（MD5 见 `manifest.json`）",
        "",
        "## 主要计数",
        "",
        f"- 65 岁及以上帕金森受试者：{result.counts['older_pd_participants']} 人，"
        f"其中 {result.counts['older_pd_participants_with_valid_gait']} 人有有效步态日",
        f"- 候选预测时点：{result.counts['candidate_landmarks']} 个，"
        f"来自 {result.counts['candidate_people']} 人",
        f"- 有 14 个既往有效日：{result.counts['baseline_ready_landmarks']} 个时点，"
        f"{result.counts['baseline_ready_people']} 人",
        f"- 7 天阳性：{result.counts['horizons']['7d']['positive_landmarks']} 个时点"
        f"（{result.counts['horizons']['7d']['positive_people']} 人）",
        f"- 30 天阳性：{result.counts['horizons']['30d']['positive_landmarks']} 个时点"
        f"（{result.counts['horizons']['30d']['positive_people']} 人）",
        "",
        "## 文件",
        "",
        "| 文件 | 内容 |",
        "|---|---|",
        "| `participants.csv` | 目标人群与每人有效日/访视覆盖 |",
        "| `visits.csv` | 访视日期与状态 |",
        "| `daily_gait.csv` | 同人同日步态聚合、佩戴时长、可用性标记 |",
        "| `fall_diaries.csv` / `fall_reports.csv` | 跌倒日记与逐次报告 |",
        "| `cohort_gates.csv` | 每条日记记录的结局与覆盖门槛及原因 |",
        "| `landmarks.csv` | 候选预测日、基线覆盖、间隔与佩戴质量 |",
        "| `baseline_days.csv` | 每个预测日实际使用的基线日期 |",
        "| `landmark_windows.csv` | 预测日 × 时窗的结局与覆盖证据 |",
        "| `cohort_flow.csv` | 各步筛选人数/时点数 |",
        "| `sensitivity.json` | 佩戴质控与发布方过滤的影响 |",
        "| `field_mapping.json` / `unit_checks.json` | 字段单位与实测一致性 |",
        "| `bout_support.json` | 逐段字段覆盖、日期连接键与日聚合重建核验 |",
        "| `code_snapshot/` | 本次运行的源码快照（含未提交改动） |",
        "",
        "## 已知限制",
        "",
        *[f"- {item}" for item in manifest["limitations"]],
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps({
        "out_dir": str(out_dir),
        "counts": result.counts,
        "reference_matches": reference["matches"],
        "reference_mismatches": reference["mismatches"],
        "bouts": bout_status,
        "daily_notes": daily.notes,
        "coverage_decomposition": coverage,
        "sensitivity": sensitivity,
    }, ensure_ascii=False, indent=2))
    return 0 if reference["matches"] else 3


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
