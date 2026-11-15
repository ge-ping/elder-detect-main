"""阶段 6：从原包重放修正版并封存；已有成功目录拒绝覆盖。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import md5
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_ROOT = PROJECT_ROOT.parents[1] / "output" / "ds_pcd_mobilise"
OLD = OUT_ROOT / "20261114-main"
STAGE5_PRECISION = OUT_ROOT / "stage5" / "se_causal" / "main_precision.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 6 修正版重放与封存")
    parser.add_argument("--run-id", default="20261114-fixed")
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--old-run", default=str(OLD))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--precision-source", default=str(STAGE5_PRECISION))
    parser.add_argument("--skip-precision-reuse", action="store_true",
                        help="不复用阶段 5 精度表，改由 Stage 2 重新扫描")
    parser.add_argument("--force", action="store_true",
                        help="仅用于未完成目录；已完成目录始终拒绝覆盖")
    return parser


def _md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, cwd=str(PROJECT_ROOT), stdout=handle,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        )
    if completed.returncode != 0:
        raise RuntimeError(f"命令失败（{completed.returncode}）：{' '.join(command)}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_root = Path(args.out_root)
    run_dir = out_root / args.run_id
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "completed":
            raise FileExistsError(
                f"已存在成功运行目录：{run_dir}；拒绝覆盖。请使用新的运行编号。"
            )
        if not args.force:
            raise FileExistsError(
                f"运行目录存在但未完成：{run_dir}；使用 --force 明确重试。"
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    old = Path(args.old_run)
    data = run_dir / "data"
    if not (data / "manifest.json").is_file():
        _run([
            sys.executable, str(PROJECT_ROOT / "scripts" / "build_mobilise_cohort.py"),
            "--out", str(data),
            *(["--cache-dir", args.cache_dir] if args.cache_dir else []),
        ], logs / "step1_data.log")
    if (run_dir / "simulation").exists():
        shutil.rmtree(run_dir / "simulation")
    shutil.copytree(old / "simulation", run_dir / "simulation")
    stage2 = run_dir / "stage2"
    stage2.mkdir(parents=True, exist_ok=True)
    precision_source = Path(args.precision_source)
    reused_precision = not args.skip_precision_reuse and precision_source.is_file()
    if reused_precision:
        shutil.copy2(precision_source, stage2 / "bout_precision.csv")
        notes = {
            "policy": "causal",
            "source": str(precision_source),
            "source_md5": _md5(precision_source),
            "md5": _md5(stage2 / "bout_precision.csv"),
            "config": {"samples": 200, "min_bouts": 5,
                       "lag1_threshold": 0.1, "pooling": "prior_days_only"},
            "archive": {"name": "Walking Bout DMO Data.zip",
                        "md5": "4e9b7952cf2015ee3cc6615c0e626323"},
        }
        (stage2 / "bout_precision_notes.json").write_text(
            json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    _run([
        sys.executable, str(PROJECT_ROOT / "scripts" / "replay_mobilise_stage2.py"),
        "--bundle", str(data), "--out", str(stage2),
        "--precision-policy", "causal",
        "--alarm-threshold-file", str(run_dir / "simulation" / "threshold.json"),
        *(["--rebuild-precision"] if not reused_precision else []),
    ], logs / "step2_scores.log")
    _run([
        sys.executable, str(PROJECT_ROOT / "scripts" / "analyse_mobilise_stage6.py"),
        "--fixed-run", str(run_dir), "--old-run", str(old),
        "--out", str(run_dir / "stage3"), "--folds", str(old / "splits.csv"),
    ], logs / "step3_analysis.log")
    for name in ("splits.csv",):
        shutil.copy2(old / name, run_dir / name)
    shutil.copy2(data / "cohort_flow.csv", run_dir / "cohort_flow.csv")
    shutil.copy2(stage2 / "scores_session_se.csv", run_dir / "scores.csv")
    shutil.copy2(stage2 / "scores_none.csv", run_dir / "scores_none.csv")
    for name in ("predictions.csv", "metrics.csv", "bootstrap.json",
                 "fold_status.csv", "analysis_rules.json", "comparison.json",
                 "selection.json", "report.md"):
        shutil.copy2(run_dir / "stage3" / name, run_dir / name)
    if (run_dir / "fold_models").exists():
        shutil.rmtree(run_dir / "fold_models")
    shutil.copytree(run_dir / "stage3" / "fold_models", run_dir / "fold_models")
    # 二次重放：复用数据与已核验精度表，重算评分与分析，供核验脚本比对
    replay = run_dir / "replay2"
    if replay.exists():
        shutil.rmtree(replay)
    (replay / "stage2").mkdir(parents=True)
    shutil.copytree(data, replay / "data")
    shutil.copy2(stage2 / "bout_precision.csv", replay / "stage2" / "bout_precision.csv")
    shutil.copy2(stage2 / "bout_precision_notes.json", replay / "stage2" / "bout_precision_notes.json")
    shutil.copytree(run_dir / "simulation", replay / "simulation")
    _run([
        sys.executable, str(PROJECT_ROOT / "scripts" / "replay_mobilise_stage2.py"),
        "--bundle", str(replay / "data"), "--out", str(replay / "stage2"),
        "--precision-policy", "causal",
        "--alarm-threshold-file", str(replay / "simulation" / "threshold.json"),
    ], logs / "step4_replay2_scores.log")
    _run([
        sys.executable, str(PROJECT_ROOT / "scripts" / "analyse_mobilise_stage6.py"),
        "--fixed-run", str(replay), "--old-run", str(old),
        "--out", str(replay / "stage3"), "--folds", str(old / "splits.csv"),
    ], logs / "step5_replay2_analysis.log")
    manifest = {
        "run_id": args.run_id, "stage": 6, "status": "running",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "old_run": str(old), "folds": "splits.csv",
        "data": str(data), "stage2": str(stage2), "stage3": str(run_dir / "stage3"),
        "precision": {
            "reused_from_stage5": reused_precision,
            "source": str(precision_source) if reused_precision else "rescan",
        },
        "simulation": {"source": str(old / "simulation"),
                       "threshold_md5": _md5(run_dir / "simulation" / "threshold.json")},
        "report": "report.md", "verification": "verification.json",
        "reproduce": "reproduce.md",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _run([
        sys.executable, str(PROJECT_ROOT / "scripts" / "verify_mobilise_stage6.py"),
        "--run", str(run_dir),
    ], logs / "step6_verify.log")
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": args.run_id, "path": str(run_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
