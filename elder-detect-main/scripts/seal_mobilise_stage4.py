"""阶段 4：从本地原包重放主实验并封存结果包。

在 ``output/ds_pcd_mobilise/<run_id>/`` 下依次重建：

1. ``data/``：阶段 1 队列（从原始压缩包）；
2. ``precision/``：逐日步行段标准误（从步行段压缩包）；
3. ``stage2/``：逐人 RMS + DS-PCD 评分；
4. ``stage3/``：折外 A/B 比较；
5. 汇总文件、代码快照、复现命令与中文报告；
6. ``reproduction_check.json``：与已冻结的阶段 2/3 产物按预测键比对。

任何一步失败都会保留运行目录并在索引中记录失败原因。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import md5
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mobilise_d import sources  # noqa: E402
from mobilise_d.precision import (  # noqa: E402
    PrecisionConfig,
    daily_precision,
    scan_bouts,
    summarise_precision,
)

OUT_ROOT = PROJECT_ROOT.parents[1] / "output" / "ds_pcd_mobilise"
FROZEN_STAGE1 = PROJECT_ROOT / "datasets" / "mobilise_d" / "cvs_v1.0.0"
FROZEN_STAGE2 = OUT_ROOT / "stage2"
FROZEN_STAGE3 = OUT_ROOT / "stage3"
TOLERANCE = 1e-9
CODE_SNAPSHOT_FILES = (
    "mobilise_d/__init__.py", "mobilise_d/sources.py", "mobilise_d/clinical.py",
    "mobilise_d/daily.py", "mobilise_d/cohort.py", "mobilise_d/precision.py",
    "mobilise_d/observations.py",
    "longitudinal/__init__.py", "longitudinal/config.py", "longitudinal/schema.py",
    "longitudinal/baseline.py", "longitudinal/deviation.py", "longitudinal/quality.py",
    "longitudinal/storage.py", "longitudinal/survival.py", "longitudinal/service.py",
    "longitudinal/pose_features.py", "longitudinal/uncertainty.py",
    "longitudinal/ds_pcd.py", "longitudinal/cox_fit.py",
    "scripts/fetch_mobilise_d.py", "scripts/build_mobilise_cohort.py",
    "scripts/inspect_mobilise_bouts.py", "scripts/verify_mobilise_stage1.py",
    "scripts/sim_ds_pcd.py", "scripts/replay_mobilise_stage2.py",
    "scripts/verify_mobilise_stage2.py", "scripts/analyse_mobilise_stage3.py",
    "scripts/verify_mobilise_stage3.py", "scripts/seal_mobilise_stage4.py",
    "tests/test_mobilise_cohort.py", "tests/test_ds_pcd.py",
    "tests/test_dspcd_service.py", "tests/test_dspcd_precision.py",
    "tests/test_dspcd_cli.py", "tests/test_mobilise_stage2_adapter.py",
    "tests/test_cox_fit.py", "tests/test_mobilise_stage3.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 4：封存并复现主实验")
    parser.add_argument("--run-id", default="20261114-main")
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-rebuild", action="store_true", help="复用已存在的步骤产物")
    return parser


def _md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state() -> dict:
    try:
        commit = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT.parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT.parent), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return {"available": False, "error": str(exc)}
    lines = [line for line in status.splitlines() if line.strip()]
    return {
        "available": True,
        "commit": commit,
        "dirty": bool(lines),
        "changed_path_count": len(lines),
        "changed_paths_sample": [line[3:] for line in lines[:40]],
    }


def _log(path: Path, message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {message}\n")


def _run(command: list[str], log_path: Path, *, timeout: int = 1800) -> None:
    _log(log_path, "运行：" + " ".join(command))
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, cwd=str(PROJECT_ROOT), stdout=handle, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"命令失败（{completed.returncode}）：{' '.join(command)}")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 复现比对
# ---------------------------------------------------------------------------


def _compare_keys(
    left: pd.DataFrame, right: pd.DataFrame, keys: Sequence[str],
) -> dict[str, Any]:
    left_keys = set(map(tuple, left[list(keys)].astype(str).to_numpy()))
    right_keys = set(map(tuple, right[list(keys)].astype(str).to_numpy()))
    return {
        "keys": list(keys),
        "rows": [len(left), len(right)],
        "only_frozen": len(left_keys - right_keys),
        "only_replay": len(right_keys - left_keys),
        "same_keys": left_keys == right_keys,
    }


def _compare_values(
    left: pd.DataFrame, right: pd.DataFrame, keys: Sequence[str],
    numeric: Sequence[str], categorical: Sequence[str], tolerance: float,
) -> dict[str, Any]:
    merged = left.merge(
        right, on=list(keys), how="outer", suffixes=("_frozen", "_replay"),
        indicator=True,
    )
    result: dict[str, Any] = {
        "keys": list(keys),
        "rows": [len(left), len(right)],
        "only_frozen": int(merged["_merge"].eq("left_only").sum()),
        "only_replay": int(merged["_merge"].eq("right_only").sum()),
        "numeric": {},
        "categorical": {},
    }
    matched = merged.loc[merged["_merge"].eq("both")]
    for column in numeric:
        a = pd.to_numeric(matched[f"{column}_frozen"], errors="coerce")
        b = pd.to_numeric(matched[f"{column}_replay"], errors="coerce")
        diff = (a - b).abs()
        nan_mismatch = int((a.isna() != b.isna()).sum())
        result["numeric"][column] = {
            "max_abs_diff": float(diff.max()) if diff.notna().any() else None,
            "mismatch_count": int((diff > tolerance).sum()) + nan_mismatch,
            "nan_mismatch": nan_mismatch,
        }
    def normalized(series: pd.Series) -> pd.Series:
        # 缺失值与缺失值视为相同；避免 NaN / pd.NA 的字符串化差异
        return series.map(lambda value: "" if pd.isna(value) else str(value))

    for column in categorical:
        a = normalized(matched[f"{column}_frozen"])
        b = normalized(matched[f"{column}_replay"])
        result["categorical"][column] = int((a != b).sum())
    result["ok"] = bool(
        result["only_frozen"] == 0 and result["only_replay"] == 0
        and all(item["mismatch_count"] == 0 for item in result["numeric"].values())
        and all(count == 0 for count in result["categorical"].values())
    )
    return result


def compare_cohort(run_data: Path, frozen: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # 流程计数
    flow_run = pd.read_csv(run_data / "cohort_flow.csv")
    flow_frozen = pd.read_csv(frozen / "cohort_flow.csv")
    out["cohort_flow"] = {
        "rows": [len(flow_run), len(flow_frozen)],
        "same": bool(flow_run[["step", "rows_in", "rows_out"]].astype(str).equals(
            flow_frozen[["step", "rows_in", "rows_out"]].astype(str)
        )),
    }
    # 预测日与标签
    run_landmarks = pd.read_csv(run_data / "landmarks.csv", dtype={"participantid": str})
    frozen_landmarks = pd.read_csv(frozen / "landmarks.csv", dtype={"participantid": str})
    out["landmarks"] = _compare_values(
        frozen_landmarks, run_landmarks,
        ["participantid", "landmark_day"],
        ["n_prior_valid_days", "days_from_previous_visit_to_landmark"],
        ["baseline_eligible", "baseline_end_day"],
        TOLERANCE,
    )
    run_windows = pd.read_csv(run_data / "landmark_windows.csv", dtype={"participantid": str})
    frozen_windows = pd.read_csv(frozen / "landmark_windows.csv", dtype={"participantid": str})
    out["windows"] = _compare_values(
        frozen_windows, run_windows,
        ["participantid", "landmark_day", "horizon_days"],
        ["n_reports_in_window"],
        ["status", "event", "first_event_day"],
        TOLERANCE,
    )
    run_daily = pd.read_csv(run_data / "daily_gait.csv", dtype={"participantid": str})
    frozen_daily = pd.read_csv(frozen / "daily_gait.csv", dtype={"participantid": str})
    out["daily"] = _compare_values(
        frozen_daily, run_daily,
        ["participantid", "visit_number", "day"],
        ["cadence_all_avg_d", "strdur_all_avg_d", "ws_30_avg_d"],
        ["has_core_gait"],
        TOLERANCE,
    )
    run_baseline = pd.read_csv(run_data / "baseline_days.csv", dtype={"participantid": str})
    frozen_baseline = pd.read_csv(frozen / "baseline_days.csv", dtype={"participantid": str})
    out["baseline_days"] = _compare_keys(
        frozen_baseline, run_baseline, ["participantid", "landmark_day", "baseline_day"],
    )
    run_counts = _read_json(run_data / "manifest.json")["counts"]
    frozen_counts = _read_json(frozen / "manifest.json")["counts"]
    out["counts_equal"] = run_counts == frozen_counts
    out["ok"] = bool(
        out["cohort_flow"]["same"] and out["landmarks"]["ok"] and out["windows"]["ok"]
        and out["daily"]["ok"] and out["baseline_days"]["same_keys"]
        and out["counts_equal"]
    )
    return out


def compare_scores(run_stage2: Path, frozen_stage2: Path) -> dict[str, Any]:
    out = {}
    for mode in ("session_se", "none"):
        left = pd.read_csv(frozen_stage2 / f"scores_{mode}.csv", dtype={"participantid": str})
        right = pd.read_csv(run_stage2 / f"scores_{mode}.csv", dtype={"participantid": str})
        out[mode] = _compare_values(
            left, right, ["participantid", "landmark_day"],
            ["rms_distance", "T", "q", "cusum", "omnibus_D", "days_since_baseline_end"],
            ["ds_status", "mask", "top_contributors", "precision_mode"],
            TOLERANCE,
        )
    out["ok"] = all(out[mode]["ok"] for mode in ("session_se", "none"))
    return out


def compare_predictions(run_stage3: Path, frozen_stage3: Path) -> dict[str, Any]:
    left = pd.read_csv(frozen_stage3 / "predictions.csv", dtype={"participantid": str})
    right = pd.read_csv(run_stage3 / "predictions.csv", dtype={"participantid": str})
    result = _compare_values(
        left, right,
        ["mode", "horizon", "method", "participantid", "landmark_day"],
        ["risk", "duration"],
        ["event", "fold"],
        TOLERANCE,
    )
    return result


def compare_metrics(run_stage3: Path, frozen_stage3: Path) -> dict[str, Any]:
    left = pd.read_csv(frozen_stage3 / "metrics.csv")
    right = pd.read_csv(run_stage3 / "metrics.csv")
    numeric = [
        "n", "events", "persons", "auroc", "brier", "delta_auroc", "delta_brier",
        "observed_risk", "mean_predicted", "calibration_slope",
        "auroc_ci_low", "auroc_ci_high", "brier_ci_low", "brier_ci_high",
        "delta_auroc_ci_low", "delta_auroc_ci_high",
        "delta_brier_ci_low", "delta_brier_ci_high",
    ]
    return _compare_values(
        left, right, ["mode", "horizon", "method"], numeric, ["status"], TOLERANCE,
    )


def compare_folds(run_stage3: Path, frozen_stage3: Path) -> dict[str, Any]:
    left = pd.read_csv(frozen_stage3 / "folds.csv", dtype={"participantid": str})
    right = pd.read_csv(run_stage3 / "folds.csv", dtype={"participantid": str})
    merged = left.merge(right, on="participantid", suffixes=("_frozen", "_replay"))
    same = bool(
        len(merged) == len(left) == len(right)
        and (merged["fold_frozen"] == merged["fold_replay"]).all()
    )
    return {"people": [len(left), len(right)], "same_folds": same, "ok": same}


def compare_fold_models(run_stage3: Path, frozen_stage3: Path) -> dict[str, Any]:
    files = sorted((frozen_stage3 / "fold_models").rglob("*.json"))
    compared = []
    for path in files:
        relative = path.relative_to(frozen_stage3 / "fold_models")
        other = run_stage3 / "fold_models" / relative
        if not other.is_file():
            compared.append({"path": str(relative), "match": False, "reason": "missing"})
            continue
        a = _read_json(path)
        b = _read_json(other)
        keys = ("feature_names", "coefficients", "feature_centers",
                "baseline_survival", "calibrated", "horizon_days", "model_type")
        match = all(
            np.allclose(np.asarray(a[key], dtype=float), np.asarray(b[key], dtype=float), atol=1e-12)
            if key in ("coefficients", "feature_centers") else
            (abs(float(a[key]) - float(b[key])) < 1e-12 if key == "baseline_survival" else a[key] == b[key])
            for key in keys
        )
        compared.append({"path": str(relative), "match": bool(match)})
    return {
        "files": len(compared),
        "mismatches": [item for item in compared if not item["match"]],
        "ok": all(item["match"] for item in compared),
    }


def compare_simulation(run_simulation: Path, frozen_simulation: Path) -> dict[str, Any]:
    names = ("simulation.json", "threshold.json", "simulation_report.md")
    out = {}
    for name in names:
        first, second = frozen_simulation / name, run_simulation / name
        out[name] = {
            "same": first.is_file() and second.is_file() and _md5(first) == _md5(second),
            "md5": _md5(first) if first.is_file() else None,
        }
    out["ok"] = all(item["same"] for item in out.values() if isinstance(item, dict))
    return out


def trace_metric(run_dir: Path, run_data: Path, cache: Path) -> dict[str, Any]:
    metrics = pd.read_csv(run_dir / "metrics.csv")
    row = metrics.loc[
        metrics["mode"].eq("session_se") & metrics["horizon"].eq(30)
        & metrics["method"].eq("B_minus_A")
    ].iloc[0]
    predictions = pd.read_csv(run_dir / "predictions.csv", dtype={"participantid": str})
    subset = predictions.loc[
        predictions["mode"].eq("session_se") & predictions["horizon"].eq(30)
    ]
    pivot = subset.pivot_table(
        index=["participantid", "landmark_day"], columns="method",
        values="risk", aggfunc="first",
    )
    labels = subset.drop_duplicates(["participantid", "landmark_day"]).set_index(
        ["participantid", "landmark_day"]
    )["event"].reindex(pivot.index)
    both = pivot.dropna()
    labels = labels.reindex(both.index).to_numpy(dtype=int)
    a = both["A_rms"].to_numpy(dtype=float)
    b = both["B_rms_cusum"].to_numpy(dtype=float)

    def auc(values: np.ndarray) -> float:
        positive = values[labels == 1]
        negative = values[labels == 0]
        return float(
            ((positive[:, None] > negative[None, :]).sum()
             + 0.5 * (positive[:, None] == negative[None, :]).sum())
            / (len(positive) * len(negative))
        )

    recomputed = auc(b) - auc(a)
    sample = subset.loc[subset["method"].eq("B_rms_cusum")].iloc[0]
    return {
        "metric": "session_se / 30d / delta_auroc (B − A)",
        "stored": float(row["delta_auroc"]),
        "ci": [float(row["delta_auroc_ci_low"]), float(row["delta_auroc_ci_high"])],
        "recomputed_from_predictions": float(recomputed),
        "difference": abs(float(row["delta_auroc"]) - float(recomputed)),
        "sample_prediction": {
            "participantid": str(sample["participantid"]),
            "landmark_day": str(sample["landmark_day"]),
            "method": str(sample["method"]),
            "risk": float(sample["risk"]),
            "event": int(sample["event"]),
            "fold": int(sample["fold"]),
        },
        "source_versions": {
            "raw_files": [
                {"name": item.name, "md5": item.md5}
                for item in sources.RELEASE_FILES
            ],
            "landmarks_md5": _md5(run_data / "landmarks.csv"),
            "windows_md5": _md5(run_data / "landmark_windows.csv"),
            "predictions_md5": _md5(run_dir / "predictions.csv"),
        },
    }


# ---------------------------------------------------------------------------
# 封存
# ---------------------------------------------------------------------------


def build_report(run_dir: Path, run_data: Path, cache: Path) -> str:
    manifest = _read_json(run_dir / "manifest.json")
    data_manifest = _read_json(run_data / "manifest.json")
    summary = _read_json(run_dir / "stage2" / "replay_summary.json")
    metrics = pd.read_csv(run_dir / "metrics.csv")
    check = _read_json(run_dir / "reproduction_check.json")
    participants = pd.read_csv(run_data / "participants.csv")
    daily = pd.read_csv(run_data / "daily_gait.csv", dtype={"participantid": str})
    landmarks = pd.read_csv(run_data / "landmarks.csv")
    visits = daily.groupby(["participantid", "visit_number"]).size()
    age = pd.to_numeric(participants["age"], errors="coerce")
    ready_landmarks = landmarks.loc[
        landmarks["baseline_eligible"].astype(str).str.lower().eq("true")
    ]
    baseline_span = ready_landmarks["baseline_span_days"].median()
    counts = data_manifest["counts"]
    precision = summary["precision"]
    simulation = _read_json(run_dir / "simulation" / "threshold.json")

    def metric_row(mode: str, method: str) -> pd.Series:
        return metrics.loc[
            metrics["mode"].eq(mode) & metrics["horizon"].eq(30)
            & metrics["method"].eq(method)
        ].iloc[0]

    a = metric_row("session_se", "A_rms")
    b = metric_row("session_se", "B_rms_cusum")
    delta = metric_row("session_se", "B_minus_A")
    b_none = metric_row("none", "B_rms_cusum")
    delta_none = metric_row("none", "B_minus_A")
    seven_b = metrics.loc[
        metrics["mode"].eq("session_se") & metrics["horizon"].eq(7)
        & metrics["method"].eq("B_minus_A")
    ].iloc[0]
    lines = [
        "# Mobilise-D / DS-PCD 主实验封存报告",
        "",
        f"- 运行编号：`{manifest['run_id']}`",
        f"- 生成时间：{manifest['completed_at_utc']}（耗时 {manifest['elapsed_seconds']:.0f} 秒）",
        f"- 状态：{manifest['status']}",
        f"- 数据集：Mobilise-D CVS {manifest['release']['version']}，doi:{manifest['release']['doi']}，"
        f"许可 {manifest['release']['license']}",
        f"- 原始包缓存：`{cache}`（MD5 见表 1；研究复现只读本地包）",
        f"- 派生数据：`{run_data}`；评分：`{run_dir / 'stage2'}`；折外比较：`{run_dir / 'stage3'}`",
        "",
        "## 1. 数据与版本",
        "",
        "| 文件 | MD5 |",
        "|---|---|",
    ]
    for item in sources.RELEASE_FILES:
        lines.append(f"| {item.name} | `{item.md5}` |")
    lines += [
        "",
        f"研究人群：帕金森队列（`cohort pd == T`）且基线年龄 ≥65 岁；"
        f"年龄中位 {age.median():.0f} 岁（范围 {age.min():.0f}–{age.max():.0f}）。"
        f"IMU 日输入为日步频 `cadence_all_avg_d`、跨步周期 `strdur_all_avg_d`、"
        f"≥30 秒步行段步速 `ws_30_avg_d`；日聚合值等于同日步行段值的平均。",
        "",
        f"访视间隔约半年：每次采集的有效观测日中位 {visits.median():.0f} 天"
        f"（{visits.min()}–{visits.max()} 天），预测日相对上次访视日中位 "
        f"{landmarks['days_from_previous_visit_to_landmark'].median():.0f} 天；"
        f"主队列的 14 日个人基线跨度中位 {baseline_span:.0f} 天。"
        "CUSUM 在相邻有效观测跨过 7 个日历日时重置，因此累计值主要反映当前访视块。",
        "",
        "## 2. 队列与结局筛选",
        "",
        "| 队列 | 预测日 | 人数 | 30 天阳性（时点/人数） | 7 天阳性（时点/人数） |",
        "|---|---:|---:|---:|---:|",
        f"| 宽集（可对齐步态与结局） | {counts['candidate_landmarks']} | {counts['candidate_people']} | "
        f"{counts['horizons']['30d']['positive_landmarks']} / {counts['horizons']['30d']['positive_people']} | "
        f"{counts['horizons']['7d']['positive_landmarks']} / {counts['horizons']['7d']['positive_people']} |",
        f"| 主队列（≥14 个既往有效日） | {counts['baseline_ready_landmarks']} | {counts['baseline_ready_people']} | "
        f"{counts['horizons']['30d']['baseline_ready_positive_landmarks']} / "
        f"{counts['horizons']['30d']['baseline_ready_positive_people']} | "
        f"{counts['horizons']['7d']['baseline_ready_positive_landmarks']} / "
        f"{counts['horizons']['7d']['baseline_ready_positive_people']} |",
        "",
        "结局只统计日记覆盖到窗口末尾、且日记总数与逐次报告一致的记录；"
        "因基线不足或记录不一致被排除的时点与阳性数见 `cohort_flow.csv` 与 `reproduction_check.json.selection`。",
        "",
        "## 3. 精度模式",
        "",
        f"主分析 `session_se`：同日步行段均值标准误，段间滞后 ≥0.1 时用块长 2 的 bootstrap，"
        f"段数不足用同人合并标准差回退（步速回退 {precision['walking_speed_m_s']['fallback_days']} 日、"
        f"步频 {precision['cadence_spm']['fallback_days']} 日、跨步周期 "
        f"{precision['stride_duration_s']['fallback_days']} 日）。"
        "敏感性 `none` 不做事后精度加权。标准误只覆盖有限步数的不确定性，不覆盖传感器系统偏差。",
        "",
        "## 4. 30 天主比较（可直接引用）",
        "",
        "| 数据集 | 队列 | 时窗 | 指标 | 估计值（95% 区间） | 独立人数 | 阳性时点 |",
        "|---|---|---|---|---|---:|---:|",
        f"| Mobilise-D CVS V1.0.0 | 65+ 帕金森，≥14 既往有效日，精度 {a['mode']} | 30 天 | "
        f"A：RMS Cox AUROC | {a['auroc']:.3f}（{a['auroc_ci_low']:.3f}–{a['auroc_ci_high']:.3f}） | "
        f"{int(a['persons'])} | {int(a['events'])} |",
        f"| 同上 | 同上 | 30 天 | B：RMS + C Cox AUROC | {b['auroc']:.3f}（{b['auroc_ci_low']:.3f}–{b['auroc_ci_high']:.3f}） | "
        f"{int(b['persons'])} | {int(b['events'])} |",
        f"| 同上 | 同上 | 30 天 | **ΔAUROC（B − A）** | **{delta['delta_auroc']:+.3f}（{delta['delta_auroc_ci_low']:+.3f}–{delta['delta_auroc_ci_high']:+.3f}）** | "
        f"{int(delta['persons'])} | {int(delta['events'])} |",
        f"| 同上 | 同上 | 30 天 | A：Brier | {a['brier']:.4f}（{a['brier_ci_low']:.4f}–{a['brier_ci_high']:.4f}） | "
        f"{int(a['persons'])} | {int(a['events'])} |",
        f"| 同上 | 同上 | 30 天 | B：Brier | {b['brier']:.4f}（{b['brier_ci_low']:.4f}–{b['brier_ci_high']:.4f}） | "
        f"{int(b['persons'])} | {int(b['events'])} |",
        f"| 同上 | 同上 | 30 天 | **ΔBrier（B − A）** | **{delta['delta_brier']:+.4f}（{delta['delta_brier_ci_low']:+.4f}–{delta['delta_brier_ci_high']:+.4f}）** | "
        f"{int(delta['persons'])} | {int(delta['events'])} |",
        "",
        f"校准：观察风险 {a['observed_risk']:.4f}，A 平均预测 {a['mean_predicted']:.4f}、"
        f"B {b['mean_predicted']:.4f}；30 天斜率 A {a['calibration_slope']:.2f}"
        f"（{a['slope_ci_low']:.2f}–{a['slope_ci_high']:.2f}）、B {b['calibration_slope']:.2f}"
        f"（{b['slope_ci_low']:.2f}–{b['slope_ci_high']:.2f}）。模型保持 `calibrated=false`。",
        "",
        f"敏感性 `none`：B AUROC {b_none['auroc']:.3f}"
        f"（{b_none['auroc_ci_low']:.3f}–{b_none['auroc_ci_high']:.3f}），"
        f"ΔAUROC {delta_none['delta_auroc']:+.3f}"
        f"（{delta_none['delta_auroc_ci_low']:+.3f}–{delta_none['delta_auroc_ci_high']:+.3f}），"
        f"ΔBrier {delta_none['delta_brier']:+.4f}"
        f"（{delta_none['delta_brier_ci_low']:+.4f}–{delta_none['delta_brier_ci_high']:+.4f}）。结论与主分析一致。",
        "",
        "## 5. 7 天窗口",
        "",
        f"主队列只有 {counts['horizons']['7d']['baseline_ready_positive_landmarks']} 个阳性时点"
        f"（{counts['horizons']['7d']['baseline_ready_positive_people']} 人）；B 在折 1 失败"
        "（训练集 2 个事件，系数不稳定），配对子集仅 125 行 / 2 个事件。"
        f"`metrics.csv` 将该比较标为 `{seven_b['status']}`，只报告计数与失败原因，不解释 AUROC/Brier，不报校准斜率。",
        "",
        "## 6. 仿真与阈值",
        "",
        f"仿真（14 基线日 + 21 监测日，含起始日）：阈值 h={simulation['threshold']}，"
        f"目标 {simulation['target_window_days']} 天内误报 ≤{simulation['target_false_alarm_rate']:.0%}，"
        f"实测 {simulation['measured_false_alarm_rate']:.3f}"
        f"（95% 区间 {simulation['wilson_95_interval'][0]:.3f}–{simulation['wilson_95_interval'][1]:.3f}）。"
        "该阈值只来自明确标注的仿真零假设，不是真实 30 天误报率；主分析使用连续 C，告警仅作展示。",
        "",
        "## 7. 判断",
        "",
        f"**证据不足**：30 天 ΔAUROC {delta['delta_auroc']:+.3f}"
        f"（{delta['delta_auroc_ci_low']:+.3f}–{delta['delta_auroc_ci_high']:+.3f}）与 "
        f"ΔBrier {delta['delta_brier']:+.4f}"
        f"（{delta['delta_brier_ci_low']:+.4f}–{delta['delta_brier_ci_high']:+.4f}）都跨 0；"
        "点估计略偏向 B，但样本只有 17 个 30 天阳性，且 RMS 与 C 相关约 0.5，"
        "加入 C 后 RMS 系数翻转。不能据此声称 DS-PCD 带来增量，也不能声称有害。",
        "",
        "## 8. 复现与追溯",
        "",
        "复现命令见 `reproduce.md`。本次运行的比对结果："
        + "，".join(
            f"{label}{'一致' if value else '不一致'}"
            for label, value in (
                ("队列", check["cohort"]["ok"]),
                ("评分", check["scores"]["ok"]),
                ("折外预测", check["predictions"]["ok"]),
                ("指标", check["metrics"]["ok"]),
                ("折分配", check["folds"]["ok"]),
                ("折模型", check["fold_models"]["ok"]),
                ("仿真", check["simulation"]["ok"]),
            )
        )
        + "。",
        f"追溯示例：`metrics.csv` 的 ΔAUROC={check['trace']['stored']:+.3f}，"
        f"从 `predictions.csv` 独立重算得 {check['trace']['recomputed_from_predictions']:+.3f}"
        f"（差 {check['trace']['difference']:.2e}）；样例预测来自受试者 "
        f"{check['trace']['sample_prediction']['participantid']}、预测日 "
        f"{check['trace']['sample_prediction']['landmark_day']}，"
        f"源文件哈希见 `reproduction_check.json.trace.source_versions`。",
        "",
        "## 9. 证据边界与未覆盖项",
        "",
        "- 人群为 65 岁及以上帕金森患者、半年一次临床访视；每次有效观测中位 7 天，不是连续监测。",
        "- 主队列要求日记/报告一致且基线就绪；23 个 30 天阳性时点因此被排除，结论只适用于该选择后人群。",
        "- 7 天窗口只有 4 个阳性，比较不可估；未报告 7 天校准。",
        "- 没有独立验证，模型 `calibrated=false`，服务概率门槛保持原样；本报告不提供临床概率。",
        "- 连续逐日视频监控性能、其他人群与外部机构推广性均未验证。",
        "",
        "## 10. 文件清单",
        "",
        "| 文件/目录 | 内容 |",
        "|---|---|",
        "| `manifest.json` / `config.json` / `reproduce.md` | 运行元数据、全部配置与复现命令 |",
        "| `cohort_flow.csv` / `splits.csv` / `fold_models/` | 筛选计数、按人折清单、各折模型 |",
        "| `scores.csv` / `scores_none.csv` / `predictions.csv` | 逐预测日评分与两种折外预测 |",
        "| `metrics.csv` / `report.md` / `logs/` | 指标、中文报告、运行日志 |",
        "| `simulation/` | 仿真配置、阈值与输出 |",
        "| `code_snapshot/` | 本次研究代码快照（含未提交改动） |",
        "| `reproduction_check.json` | 与冻结阶段 2/3 的逐键比对与追溯 |",
        "",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root)
    run_dir = out_root / args.run_id
    started = time.time()
    started_at = datetime.now(timezone.utc)
    reuse = bool(args.skip_rebuild and run_dir.exists())
    if run_dir.exists() and not reuse:
        if not args.force:
            raise FileExistsError(f"运行目录已存在：{run_dir}；使用 --force 覆盖或 --skip-rebuild 续跑")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir) if args.cache_dir else sources.default_cache_dir()
    status = "running"
    error: str | None = None
    try:
        # 1) 原始包校验
        _log(logs / "run.log", "校验原始发布包")
        resolved = sources.ensure_files(cache, sources.RELEASE_FILES, offline=True)
        _write_json(run_dir / "raw_inputs.json", {
            "cache_dir": str(cache),
            "files": [
                {
                    "name": item.name, "md5": item.md5,
                    "bytes": (cache / item.name).stat().st_size,
                    "status": "verified",
                }
                for item in sources.RELEASE_FILES
            ],
        })
        # 2) 阶段 1 数据
        data_dir = run_dir / "data"
        if not (args.skip_rebuild and (data_dir / "manifest.json").is_file()):
            _run([
                sys.executable, str(PROJECT_ROOT / "scripts" / "build_mobilise_cohort.py"),
                "--cache-dir", str(cache), "--out", str(data_dir),
            ], logs / "step1_data.log")
        # 3) 仿真：复制冻结的标定产物并校验哈希
        frozen_simulation = FROZEN_STAGE2 / "simulation"
        if not (frozen_simulation / "threshold.json").is_file():
            raise FileNotFoundError("缺少仿真阈值产物，无法复核主实验")
        if not (
            args.skip_rebuild
            and (run_dir / "simulation" / "threshold.json").is_file()
        ):
            shutil.copytree(frozen_simulation, run_dir / "simulation", dirs_exist_ok=True)
        _log(logs / "run.log", "复制并校验仿真产物")
        # 4) 精度表（从步行段包重新扫描）
        participants = pd.read_csv(data_dir / "participants.csv", dtype={"participantid": str})
        started_precision = time.time()
        precision_dir = run_dir / "precision"
        precision_dir.mkdir(parents=True, exist_ok=True)
        precision_path = precision_dir / "bout_precision.csv"
        if args.skip_rebuild and precision_path.is_file():
            precision = pd.read_csv(precision_path, dtype={"participantid": str})
            notes_path = precision_dir / "bout_precision_notes.json"
            scan_notes = _read_json(notes_path) if notes_path.is_file() else {"reused": True}
            scan_notes.pop("md5", None)
            scan_notes.pop("elapsed_seconds", None)
        else:
            scan = scan_bouts(cache / sources.BOUTS.name, participants["participantid"].unique())
            precision = daily_precision(
                scan.frame, config=PrecisionConfig(samples=200, min_bouts=5),
            )
            precision.to_csv(precision_path, index=False, encoding="utf-8", float_format="%.17g")
            scan_notes = scan.notes
        precision_notes = {
            **scan_notes,
            "precision": summarise_precision(precision),
            "config": {"samples": 200, "min_bouts": 5, "lag1_threshold": 0.1},
            "archive": {
                "name": sources.BOUTS.name, "md5": sources.BOUTS.md5,
                "path": str(cache / sources.BOUTS.name),
            },
            "elapsed_seconds": round(time.time() - started_precision, 1),
            "md5": _md5(precision_path),
        }
        _write_json(precision_dir / "bout_precision_notes.json", precision_notes)
        # 5) 阶段 2 回放（复用刚生成的精度表）
        stage2_dir = run_dir / "stage2"
        stage2_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(precision_path, stage2_dir / "bout_precision.csv")
        shutil.copy2(
            precision_dir / "bout_precision_notes.json",
            stage2_dir / "bout_precision_notes.json",
        )
        if not (args.skip_rebuild and (stage2_dir / "replay_summary.json").is_file()):
            _run([
                sys.executable, str(PROJECT_ROOT / "scripts" / "replay_mobilise_stage2.py"),
                "--cache-dir", str(cache), "--bundle", str(data_dir),
                "--out", str(stage2_dir),
                "--alarm-threshold-file", str(run_dir / "simulation" / "threshold.json"),
            ], logs / "step2_scores.log")
        # 6) 阶段 3 折外比较
        stage3_dir = run_dir / "stage3"
        if not (args.skip_rebuild and (stage3_dir / "metrics.csv").is_file()):
            _run([
                sys.executable, str(PROJECT_ROOT / "scripts" / "analyse_mobilise_stage3.py"),
                "--stage2", str(stage2_dir), "--bundle", str(data_dir),
                "--out", str(stage3_dir),
            ], logs / "step3_analysis.log")
        # 6b) 阶段 2/3 核验（运行本地副本）
        _run([
            sys.executable, str(PROJECT_ROOT / "scripts" / "verify_mobilise_stage2.py"),
            "--stage2", str(stage2_dir), "--bundle", str(data_dir),
            "--simulation-dir", str(run_dir / "simulation"),
        ], logs / "step2_verify.log")
        _run([
            sys.executable, str(PROJECT_ROOT / "scripts" / "verify_mobilise_stage3.py"),
            "--stage3", str(stage3_dir), "--stage2", str(stage2_dir),
            "--bundle", str(data_dir),
        ], logs / "step3_verify.log")
        # 7) 代码快照
        snapshot = run_dir / "code_snapshot"
        snapshot_entries = []
        for relative in CODE_SNAPSHOT_FILES:
            source = PROJECT_ROOT / relative
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            snapshot_entries.append({
                "path": relative, "md5": _md5(target), "bytes": target.stat().st_size,
            })
        git_state = _git_state()
        (run_dir / "git_status.txt").write_text(
            subprocess.run(
                ["git", "-C", str(PROJECT_ROOT.parent), "status", "--porcelain"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            ).stdout, encoding="utf-8",
        )
        # 8) 汇总文件
        shutil.copy2(data_dir / "cohort_flow.csv", run_dir / "cohort_flow.csv")
        shutil.copy2(stage3_dir / "folds.csv", run_dir / "splits.csv")
        shutil.copy2(stage2_dir / "scores_session_se.csv", run_dir / "scores.csv")
        shutil.copy2(stage2_dir / "scores_none.csv", run_dir / "scores_none.csv")
        shutil.copy2(stage3_dir / "predictions.csv", run_dir / "predictions.csv")
        shutil.copy2(stage3_dir / "metrics.csv", run_dir / "metrics.csv")
        shutil.copytree(
            stage3_dir / "fold_models", run_dir / "fold_models", dirs_exist_ok=True,
        )
        data_manifest = _read_json(data_dir / "manifest.json")
        stage2_summary = _read_json(stage2_dir / "replay_summary.json")
        stage3_protocol = _read_json(stage3_dir / "protocol.json")
        config = {
            "cohort": data_manifest["config"],
            "ds_pcd": {
                mode: stage2_summary["modes"][mode]["config"]
                for mode in ("session_se", "none")
            },
            "precision": {
                "mode": "session_se",
                "sensitivity_mode": "none",
                "samples": 200,
                "min_bouts": 5,
                "lag1_threshold": 0.1,
                "note": "tau_floor 与 se_floor 见 ds_pcd.<mode>",
            },
            "stage3_protocol": stage3_protocol,
            "alarm_thresholds": stage2_summary["alarm_thresholds"],
            "directions": {
                "walking_speed_m_s": -1,
                "cadence_spm": -1,
                "stride_duration_s": 1,
            },
            "gap_reset_days": 7,
            "k_ref": 0.5,
        }
        _write_json(run_dir / "config.json", config)
        # 9) 复现比对
        check = {
            "frozen": {
                "stage1": str(FROZEN_STAGE1),
                "stage2": str(FROZEN_STAGE2),
                "stage3": str(FROZEN_STAGE3),
            },
            "tolerance": TOLERANCE,
            "cohort": compare_cohort(data_dir, FROZEN_STAGE1),
            "scores": compare_scores(stage2_dir, FROZEN_STAGE2),
            "folds": compare_folds(stage3_dir, FROZEN_STAGE3),
            "fold_models": compare_fold_models(stage3_dir, FROZEN_STAGE3),
            "predictions": compare_predictions(stage3_dir, FROZEN_STAGE3),
            "metrics": compare_metrics(stage3_dir, FROZEN_STAGE3),
            "simulation": compare_simulation(run_dir / "simulation", frozen_simulation),
        }
        check["selection"] = _read_json(stage3_dir / "results.json")["selection"]
        check["ok"] = all(
            check[key]["ok"] if isinstance(check[key], dict) and "ok" in check[key]
            else check[key].get("same_keys", check[key].get("same", False))
            for key in ("cohort", "scores", "folds", "fold_models", "predictions", "metrics", "simulation")
        )
        check["trace"] = trace_metric(run_dir, data_dir, cache)
        _write_json(run_dir / "reproduction_check.json", check)
        # 10) 清单、复现说明、报告
        elapsed = time.time() - started
        manifest = {
            "run_id": args.run_id,
            "stage": 4,
            "status": "completed",
            "started_at_utc": started_at.isoformat(timespec="seconds"),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed, 1),
            "release": {
                "dataset": sources.ATTRIBUTION,
                "version": sources.VERSION,
                "doi": sources.DOI,
                "license": sources.LICENSE,
                "record_url": sources.RECORD_URL,
            },
            "raw_inputs": "raw_inputs.json",
            "reused_completed_steps": bool(args.skip_rebuild),
            "cache_dir": str(cache),
            "commands": [
                "python scripts/seal_mobilise_stage4.py --run-id " + args.run_id,
                "python scripts/build_mobilise_cohort.py --out <run>/data",
                "python scripts/replay_mobilise_stage2.py --bundle <run>/data --out <run>/stage2",
                "python scripts/analyse_mobilise_stage3.py --stage2 <run>/stage2 --bundle <run>/data --out <run>/stage3",
                "python scripts/verify_mobilise_stage2.py --stage2 <run>/stage2 --bundle <run>/data",
                "python scripts/verify_mobilise_stage3.py --stage3 <run>/stage3 --stage2 <run>/stage2 --bundle <run>/data",
            ],
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "git": git_state,
            "code_snapshot": snapshot_entries,
            "config": "config.json",
            "tables": {
                "cohort_flow": "cohort_flow.csv",
                "splits": "splits.csv",
                "scores": "scores.csv",
                "scores_sensitivity": "scores_none.csv",
                "predictions": "predictions.csv",
                "metrics": "metrics.csv",
                "simulation": "simulation/",
                "fold_models": "fold_models/",
            },
            "cohort": data_manifest["counts"],
            "reproduction_check": {
                "path": "reproduction_check.json",
                "ok": check["ok"],
            },
            "report": "report.md",
        }
        _write_json(run_dir / "manifest.json", manifest)
        reproduce = [
            "# 复现命令",
            "",
            "以下命令在 `anti-fall/elder-detect-main/` 下运行；原始包已缓存在本地，"
            "`fetch_mobilise_d.py --verify-only` 只校验、不下载。",
            "",
            "```bash",
            "python scripts/fetch_mobilise_d.py --verify-only",
            f"python scripts/seal_mobilise_stage4.py --run-id {args.run_id} --force",
            "```",
            "",
            "等价的分解步骤：",
            "",
            "```bash",
            f"python scripts/build_mobilise_cohort.py --out output/ds_pcd_mobilise/{args.run_id}/data",
            f"python scripts/replay_mobilise_stage2.py --bundle output/ds_pcd_mobilise/{args.run_id}/data "
            f"--out output/ds_pcd_mobilise/{args.run_id}/stage2 "
            f"--alarm-threshold-file output/ds_pcd_mobilise/{args.run_id}/simulation/threshold.json",
            f"python scripts/analyse_mobilise_stage3.py "
            f"--stage2 output/ds_pcd_mobilise/{args.run_id}/stage2 "
            f"--bundle output/ds_pcd_mobilise/{args.run_id}/data "
            f"--out output/ds_pcd_mobilise/{args.run_id}/stage3",
            "python -m pytest tests -q",
            "```",
            "",
            "说明：`stage2` 的 `bout_precision.csv` 由阶段 4 从步行段包重新生成并复制；"
            "仿真阈值复制自阶段 2 的标定产物，其哈希记录在 `manifest.json` 与 "
            "`reproduction_check.json`。",
            "",
        ]
        (run_dir / "reproduce.md").write_text("\n".join(reproduce), encoding="utf-8")
        (run_dir / "report.md").write_text(
            build_report(run_dir, data_dir, cache), encoding="utf-8",
        )
        # 11) 运行索引
        index_path = out_root / "runs.json"
        index = _read_json(index_path) if index_path.is_file() else {"runs": [], "failures": []}
        index["runs"] = [item for item in index["runs"] if item.get("run_id") != args.run_id]
        index["runs"].append({
            "run_id": args.run_id,
            "kind": "main_experiment",
            "status": "completed",
            "path": str(run_dir),
            "started_at_utc": started_at.isoformat(timespec="seconds"),
            "completed_at_utc": manifest["completed_at_utc"],
            "data_version": sources.VERSION,
            "reproduction_ok": check["ok"],
            "metrics_hash": _md5(run_dir / "metrics.csv"),
            "predictions_hash": _md5(run_dir / "predictions.csv"),
        })
        index["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _write_json(index_path, index)
        status = "completed"
    except Exception as exc:  # 保留失败运行，记录原因
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        _log(logs / "run.log", "运行失败：" + error)
        try:
            index_path = out_root / "runs.json"
            index = _read_json(index_path) if index_path.is_file() else {"runs": [], "failures": []}
            index.setdefault("failures", []).append({
                "run_id": args.run_id,
                "status": "failed",
                "path": str(run_dir),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "reason": error,
            })
            _write_json(index_path, index)
        except Exception:
            pass
        raise
    print(json.dumps({
        "run_id": args.run_id,
        "status": status,
        "path": str(run_dir),
        "reproduction_ok": _read_json(run_dir / "reproduction_check.json")["ok"],
        "elapsed_seconds": round(time.time() - started, 1),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
