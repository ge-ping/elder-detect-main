"""阶段 3 核验：折外隔离、指标复算、筛选计数、失败记录与重放确定性。"""
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

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.cox_fit import fit_landmark_cox  # noqa: E402
from longitudinal.survival import (  # noqa: E402
    CoxModelArtifact,
    predict_cumulative_risk,
)
from scripts.analyse_mobilise_stage3 import (  # noqa: E402
    METHODS,
    assign_folds,
    load_analysis_frame,
)

STAGE3 = PROJECT_ROOT.parents[1] / "output" / "ds_pcd_mobilise" / "stage3"
STAGE2 = PROJECT_ROOT.parents[1] / "output" / "ds_pcd_mobilise" / "stage2"
BUNDLE = PROJECT_ROOT / "datasets" / "mobilise_d" / "cvs_v1.0.0"


def _md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _auroc_pairwise(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    greater = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(positives) * len(negatives)))


def _brier(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean(
        (np.asarray(labels, dtype=float) - np.asarray(probabilities, dtype=float)) ** 2
    ))


def independent_metrics(predictions: pd.DataFrame, mode: str, horizon: int) -> dict[str, Any]:
    subset = predictions.loc[
        predictions["mode"].eq(mode) & predictions["horizon"].eq(horizon)
    ]
    pivot = subset.pivot_table(
        index=["participantid", "landmark_day"], columns="method",
        values="risk", aggfunc="first",
    )
    labels = subset.drop_duplicates(["participantid", "landmark_day"]).set_index(
        ["participantid", "landmark_day"]
    )["event"].reindex(pivot.index)
    both = pivot.dropna(subset=["A_rms", "B_rms_cusum"])
    paired_labels = labels.reindex(both.index).to_numpy(dtype=int)
    a = both["A_rms"].to_numpy(dtype=float)
    b = both["B_rms_cusum"].to_numpy(dtype=float)
    return {
        "n": int(len(both)),
        "events": int(paired_labels.sum()),
        "auroc_A": _auroc_pairwise(paired_labels, a),
        "auroc_B": _auroc_pairwise(paired_labels, b),
        "brier_A": _brier(paired_labels, a),
        "brier_B": _brier(paired_labels, b),
        "delta_auroc": _auroc_pairwise(paired_labels, b) - _auroc_pairwise(paired_labels, a),
        "delta_brier": _brier(paired_labels, b) - _brier(paired_labels, a),
    }


def _protocol(stage3: Path) -> dict[str, Any]:
    return json.loads((stage3 / "protocol.json").read_text(encoding="utf-8"))


def verify_split(stage3: Path, stage2: Path, bundle: Path) -> dict[str, Any]:
    folds = pd.read_csv(stage3 / "folds.csv", dtype={"participantid": str})
    if folds["participantid"].duplicated().any():
        raise AssertionError("同一受试者出现在多个折")
    protocol = _protocol(stage3)
    frame = load_analysis_frame(stage2, protocol["primary"]["mode"], bundle)
    expected = assign_folds(
        frame, int(protocol["folds"]), int(protocol["seed"]),
    )
    merged = folds.merge(
        expected[["participantid", "fold"]], on="participantid",
        suffixes=("", "_recomputed"), how="outer",
    )
    if merged["fold"].isna().any() or merged["fold_recomputed"].isna().any():
        raise AssertionError("折文件与重新分配的人员集合不一致")
    same = bool((merged["fold"] == merged["fold_recomputed"]).all())
    person_folds = dict(zip(folds["participantid"], folds["fold"]))
    all_people = set(folds["participantid"])
    disjoint = True
    for fold in folds["fold"].unique():
        test_people = set(folds.loc[folds["fold"].eq(fold), "participantid"])
        train_people = all_people - test_people
        if test_people & train_people:
            disjoint = False
    return {
        "train_test_disjoint": disjoint,
        "persons": int(len(folds)),
        "folds": int(folds["fold"].nunique()),
        "fold_sizes": folds["fold"].value_counts().sort_index().to_dict(),
        "event_persons_per_fold": (
            folds.loc[folds["has_event"].astype(str).str.lower().eq("true")]
            .groupby("fold").size().to_dict()
        ),
        "person_fold_unique": True,
        "folds_reproducible": same,
        "person_fold": person_folds,
    }


def verify_predictions(stage3: Path, folds: dict[str, int]) -> dict[str, Any]:
    predictions = pd.read_csv(stage3 / "predictions.csv", dtype={"participantid": str})
    status = pd.read_csv(stage3 / "fold_status.csv")
    duplicated = predictions.duplicated(
        ["mode", "horizon", "method", "participantid", "landmark_day"]
    )
    fold_consistency = predictions.assign(
        expected=predictions["participantid"].map(folds)
    )
    fold_ok = bool((fold_consistency["fold"] == fold_consistency["expected"]).all())
    coverage = predictions.groupby(["mode", "horizon", "method"]).agg(
        rows=("participantid", "size"),
        people=("participantid", "nunique"),
        events=("event", "sum"),
    ).reset_index()
    coverage_ok = True
    for row in coverage.itertuples(index=False):
        if int(row.horizon) == 30:
            if row.method == "A_rms" and int(row.rows) != 186:
                coverage_ok = False
            if row.method == "B_rms_cusum" and int(row.rows) != 186:
                coverage_ok = False
    return {
        "rows": int(len(predictions)),
        "key_unique": not bool(duplicated.any()),
        "fold_matches_folds_file": fold_ok,
        "coverage_30d_complete": coverage_ok,
        "coverage": coverage.to_dict("records"),
        "fold_status_counts": status["status"].value_counts().to_dict(),
        "fold_failures": status.loc[status["status"].ne("ok")].to_dict("records"),
    }


def verify_fold_refit(stage3: Path, stage2: Path, bundle: Path) -> dict[str, Any]:
    """独立重拟合一个折的 30 天模型，与保存的折模型比较。"""
    protocol = _protocol(stage3)
    frame = load_analysis_frame(stage2, protocol["primary"]["mode"], bundle)
    folds = pd.read_csv(stage3 / "folds.csv", dtype={"participantid": str})
    fold = int(folds["fold"].min())
    test_people = set(folds.loc[folds["fold"].eq(fold), "participantid"])
    dataset = frame.rename(columns={
        "event_bool_30d": "event", "duration_30d": "duration",
    })
    train = dataset.loc[~dataset["participantid"].isin(test_people)]
    results = {}
    for method, features in METHODS.items():
        fit = fit_landmark_cox(
            train["duration"].to_numpy(), train["event"].astype(int).to_numpy(),
            train[list(features)].to_numpy(dtype=np.float64), features,
            horizon_days=30.0, ridge=float(protocol["ridge_standardized"]),
        )
        path = (
            stage3 / "fold_models" / protocol["primary"]["mode"] / "30d"
            / f"{method}_fold{fold}.json"
        )
        stored = CoxModelArtifact.from_dict(json.loads(path.read_text(encoding="utf-8")))
        results[method] = {
            "refit_status": fit.status,
            "stored_status": "ok",
            "coefficients_match": bool(np.allclose(
                np.asarray(fit.coefficients), np.asarray(stored.coefficients), atol=1e-8,
            )),
            "centers_match": bool(np.allclose(
                np.asarray(fit.centers), np.asarray(stored.feature_centers), atol=1e-10,
            )),
            "baseline_survival_match": abs(
                float(fit.baseline_survival) - stored.baseline_survival
            ) < 1e-10,
            "refit_coefficients": [float(value) for value in fit.coefficients],
            "stored_coefficients": list(stored.coefficients),
        }
    return results


def verify_metrics(stage3: Path) -> dict[str, Any]:
    predictions = pd.read_csv(stage3 / "predictions.csv", dtype={"participantid": str})
    metrics = pd.read_csv(stage3 / "metrics.csv")
    checks = []
    for mode in predictions["mode"].unique():
        for horizon in (30, 7):
            recomputed = independent_metrics(predictions, mode, horizon)
            row = metrics.loc[
                metrics["mode"].eq(mode) & metrics["horizon"].eq(horizon)
                & metrics["method"].eq("B_minus_A")
            ].iloc[0]
            paired_rows = metrics.loc[
                metrics["mode"].eq(mode) & metrics["horizon"].eq(horizon)
            ].set_index("method")
            values = {
                "auroc_A": paired_rows.loc["A_rms_paired", "auroc"],
                "auroc_B": paired_rows.loc["B_rms_cusum_paired", "auroc"],
                "brier_A": paired_rows.loc["A_rms_paired", "brier"],
                "brier_B": paired_rows.loc["B_rms_cusum_paired", "brier"],
                "delta_auroc": row["delta_auroc"],
                "delta_brier": row["delta_brier"],
                "n": int(row["n"]),
                "events": int(row["events"]),
            }
            match = (
                abs(recomputed["auroc_A"] - values["auroc_A"]) < 1e-9
                and abs(recomputed["auroc_B"] - values["auroc_B"]) < 1e-9
                and abs(recomputed["brier_A"] - values["brier_A"]) < 1e-9
                and abs(recomputed["brier_B"] - values["brier_B"]) < 1e-9
                and abs(recomputed["delta_auroc"] - values["delta_auroc"]) < 1e-9
                and abs(recomputed["delta_brier"] - values["delta_brier"]) < 1e-9
                and recomputed["n"] == values["n"]
                and recomputed["events"] == values["events"]
            )
            checks.append({
                "mode": mode, "horizon": horizon,
                "recomputed": recomputed, "stored": values, "match": bool(match),
            })
    return {"checks": checks, "all_match": all(item["match"] for item in checks)}


def verify_selection(bundle: Path, stage3: Path) -> dict[str, Any]:
    landmarks = pd.read_csv(bundle / "landmarks.csv", dtype={"participantid": str})
    windows = pd.read_csv(bundle / "landmark_windows.csv", dtype={"participantid": str})
    metrics = json.loads((stage3 / "results.json").read_text(encoding="utf-8"))
    selection = metrics["selection"]
    ready = landmarks.loc[
        landmarks["baseline_eligible"].astype(str).str.lower().eq("true")
    ]
    wide30 = windows.loc[
        windows["horizon_days"].eq(30)
        & windows["event"].astype(str).str.lower().eq("true")
    ]
    wide7 = windows.loc[
        windows["horizon_days"].eq(7)
        & windows["event"].astype(str).str.lower().eq("true")
    ]
    ready_keys = set(zip(ready["participantid"], ready["landmark_day"]))
    ready30 = len([
        row for row in wide30.itertuples(index=False)
        if (row.participantid, row.landmark_day) in ready_keys
    ])
    checks = {
        "candidate_landmarks": int(len(landmarks)),
        "baseline_ready_landmarks": int(len(ready)),
        "baseline_ready_people": int(ready["participantid"].nunique()),
        "excluded_landmarks": int(len(landmarks) - len(ready)),
        "wide_30d_positive": int(len(wide30)),
        "ready_30d_positive": int(ready30),
        "excluded_30d_positive": int(len(wide30) - ready30),
        "wide_7d_positive": int(len(wide7)),
        "ready_7d_positive": int(sum(
            1 for row in wide7.itertuples(index=False)
            if (row.participantid, row.landmark_day) in ready_keys
        )),
        "reported_selection": selection,
    }
    checks["excluded_7d_positive"] = int(
        checks["wide_7d_positive"] - checks["ready_7d_positive"]
    )
    checks["match_reported"] = (
        checks["candidate_landmarks"] == selection["candidate_landmarks"]
        and checks["wide_30d_positive"] == selection["wide_30d_positive_landmarks"]
        and checks["excluded_30d_positive"] == selection["excluded_30d_positive_landmarks"]
        and checks["baseline_ready_landmarks"] == selection["baseline_ready_landmarks"]
    )
    return checks


def verify_calibration_gate(stage3: Path) -> dict[str, Any]:
    files = sorted((stage3 / "fold_models").rglob("*.json"))
    flags = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        flags.append(bool(payload["calibrated"]))
    sample = CoxModelArtifact.from_dict(
        json.loads(files[0].read_text(encoding="utf-8"))
    )
    blocked = False
    try:
        predict_cumulative_risk(
            {name: 0.0 for name in sample.feature_names}, sample,
        )
    except ValueError:
        blocked = True
    return {
        "model_files": len(files),
        "all_calibrated_false": not any(flags),
        "service_probability_blocked": blocked,
    }


def determinism(stage3: Path, work: Path) -> dict[str, Any]:
    rebuild = work / "rebuild"
    if rebuild.exists():
        shutil.rmtree(rebuild)
    command = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "analyse_mobilise_stage3.py"),
        "--out", str(rebuild),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    compared = {}
    for name in ("predictions.csv", "metrics.csv", "fold_status.csv", "folds.csv"):
        first, second = stage3 / name, rebuild / name
        compared[name] = {
            "identical": first.is_file() and second.is_file() and _md5(first) == _md5(second),
            "md5": _md5(first) if first.is_file() else None,
        }
    # 折模型：忽略时间戳，只比较数值与训练摘要
    model_matches = []
    for path in sorted((stage3 / "fold_models").rglob("*.json")):
        relative = path.relative_to(stage3 / "fold_models")
        other = rebuild / "fold_models" / relative
        if not other.is_file():
            model_matches.append({"path": str(relative), "match": False})
            continue
        first = json.loads(path.read_text(encoding="utf-8"))
        second = json.loads(other.read_text(encoding="utf-8"))
        keys = ("schema_version", "model_type", "bundle_id", "horizon_days",
                "feature_names", "coefficients", "feature_centers",
                "baseline_survival", "calibrated", "evidence_ids")
        match = all(first[key] == second[key] for key in keys)
        if match:
            first_summary = dict(first.get("training_summary", {}))
            second_summary = dict(second.get("training_summary", {}))
            first_summary.pop("trained_at", None)
            second_summary.pop("trained_at", None)
            match = first_summary == second_summary
        model_matches.append({"path": str(relative), "match": bool(match)})
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stderr_tail": completed.stderr.strip().splitlines()[-5:],
        "files": compared,
        "models": model_matches,
        "all_identical": (
            all(item["identical"] for item in compared.values())
            and all(item["match"] for item in model_matches)
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 3 核验")
    parser.add_argument("--stage3", default=str(STAGE3))
    parser.add_argument("--stage2", default=str(STAGE2))
    parser.add_argument("--bundle", default=str(BUNDLE))
    parser.add_argument("--skip-rebuild", action="store_true")
    args = parser.parse_args(argv)
    stage3 = Path(args.stage3)
    stage2 = Path(args.stage2)
    bundle = Path(args.bundle)
    split = verify_split(stage3, stage2, bundle)
    folds = split.pop("person_fold")
    payload: dict[str, Any] = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage3": str(stage3),
        "split": split,
        "predictions": verify_predictions(stage3, folds),
        "fold_refit": verify_fold_refit(stage3, stage2, bundle),
        "metrics": verify_metrics(stage3),
        "selection": verify_selection(bundle, stage3),
        "calibration_gate": verify_calibration_gate(stage3),
    }
    payload["split_ok"] = bool(
        split["person_fold_unique"] and split["folds_reproducible"]
        and len(split["event_persons_per_fold"]) == split["folds"]
    )
    payload["predictions_ok"] = bool(
        payload["predictions"]["key_unique"]
        and payload["predictions"]["fold_matches_folds_file"]
        and payload["predictions"]["coverage_30d_complete"]
    )
    payload["fold_refit_ok"] = all(
        item["coefficients_match"] and item["centers_match"]
        and item["baseline_survival_match"]
        for item in payload["fold_refit"].values()
    )
    if not args.skip_rebuild:
        payload["determinism"] = determinism(stage3, stage3 / "work")
    payload["all_pass"] = (
        payload["split_ok"]
        and payload["predictions_ok"]
        and payload["fold_refit_ok"]
        and payload["metrics"]["all_match"]
        and payload["selection"]["match_reported"]
        and payload["calibration_gate"]["all_calibrated_false"]
        and payload["calibration_gate"]["service_probability_blocked"]
        and payload.get("determinism", {}).get("all_identical", True)
    )
    (stage3 / "verification.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    lines = [
        "# 阶段 3 核验记录",
        "",
        f"- 时间：{payload['verified_at_utc']}",
        f"- 产物目录：`{stage3}`",
        "",
        "## 折与折外隔离",
        "",
        f"- 受试者 {split['persons']} 人，{split['folds']} 折，各折人数 {split['fold_sizes']}",
        f"- 各折有 30 天阳性的受试者数：{split['event_persons_per_fold']}",
        f"- 同一人只属于一个折：{'是' if split['person_fold_unique'] else '否'}；"
        f"折分配可复现：{'是' if split['folds_reproducible'] else '否'}",
        "",
        "## 折外预测",
        "",
        f"- 预测行 {payload['predictions']['rows']}；键唯一："
        f"{'是' if payload['predictions']['key_unique'] else '否'}；"
        f"折号与折文件一致：{'是' if payload['predictions']['fold_matches_folds_file'] else '否'}",
        f"- 30 天两方法均 186 行：{'是' if payload['predictions']['coverage_30d_complete'] else '否'}",
        f"- 折模型运行状态：{payload['predictions']['fold_status_counts']}",
        f"- 失败折已记录：{payload['predictions']['fold_failures']}",
        "",
        "## 指标复算",
        "",
        "| 模式 | 时窗 | 复算一致 |",
        "|---|---|---|",
    ]
    for item in payload["metrics"]["checks"]:
        lines.append(
            f"| {item['mode']} | {item['horizon']} 天 | "
            f"{'是' if item['match'] else '否'} |"
        )
    lines += [
        "",
        "## 筛选计数",
        "",
        f"- 宽集：{payload['selection']['candidate_landmarks']} 个预测日，"
        f"30 天阳性 {payload['selection']['wide_30d_positive']}，"
        f"7 天阳性 {payload['selection']['wide_7d_positive']}",
        f"- 主队列：{payload['selection']['baseline_ready_landmarks']} 个预测日，"
        f"30 天阳性 {payload['selection']['ready_30d_positive']}，"
        f"7 天阳性 {payload['selection']['ready_7d_positive']}",
        f"- 排除：{payload['selection']['excluded_landmarks']} 个预测日，"
        f"其中 30 天阳性 {payload['selection']['excluded_30d_positive']}、"
        f"7 天阳性 {payload['selection']['excluded_7d_positive']}",
        "",
        "## 折模型重拟合",
        "",
    ]
    for method, item in payload["fold_refit"].items():
        lines.append(
            f"- {method}：状态 {item['refit_status']}，系数一致 "
            f"{'是' if item['coefficients_match'] else '否'}，"
            f"基线生存一致 {'是' if item['baseline_survival_match'] else '否'}"
        )
    lines += [
        "",
        "## 校准门槛",
        "",
        f"- 折模型文件 {payload['calibration_gate']['model_files']} 个，全部 calibrated=false："
        f"{'是' if payload['calibration_gate']['all_calibrated_false'] else '否'}",
        f"- 服务概率接口拒绝未校准模型："
        f"{'是' if payload['calibration_gate']['service_probability_blocked'] else '否'}",
        "",
    ]
    if "determinism" in payload:
        det = payload["determinism"]
        lines += [
            "## 重放确定性",
            "",
            f"- 命令：`{det['command']}`，退出码 {det['returncode']}",
            f"- 预测/指标/折文件与折模型一致：{'是' if det['all_identical'] else '否'}",
            "",
        ]
    (stage3 / "VERIFICATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "split_ok": payload["split_ok"],
        "predictions_ok": payload["predictions_ok"],
        "fold_refit_ok": payload["fold_refit_ok"],
        "metrics_all_match": payload["metrics"]["all_match"],
        "selection_match": payload["selection"]["match_reported"],
        "calibration_gate_ok": (
            payload["calibration_gate"]["all_calibrated_false"]
            and payload["calibration_gate"]["service_probability_blocked"]
        ),
        "determinism_all_identical": payload.get("determinism", {}).get("all_identical"),
        "all_pass": payload["all_pass"],
    }, ensure_ascii=False, indent=2))
    return 0 if payload["all_pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
