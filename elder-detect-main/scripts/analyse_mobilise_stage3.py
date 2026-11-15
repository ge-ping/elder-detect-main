"""阶段 3：折外比较「RMS」与「RMS + DS-PCD 累计分数」的 30 天跌倒预测。

协议在第一次拟合前写入 ``protocol.json``：

* 主队列：阶段 1 的 186 个基线就绪预测日（144 人），30 天随访完整；
* A：Landmark Cox + ``rms_distance``；B：同一 Cox + ``cusum``；
* 3 折按人分组，按“是否有 30 天阳性”平衡；两种方法、两个时窗、两种
  精度模式共用同一折清单；
* 岭正则化固定 1e-4（标准化尺度，仅数值稳定）；
* 指标：折外 AUROC（越大越好）、Brier（越小越好）、配对差 B−A；
  按人聚类 bootstrap 2000 次给 95% 区间；
* 研究模型全部 ``calibrated=false``，用离线研究预测函数输出风险。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import md5
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.cox_fit import CoxFit, fit_landmark_cox  # noqa: E402
from longitudinal.survival import (  # noqa: E402
    CoxModelArtifact,
    predict_cumulative_risk_research,
)

DEFAULT_STAGE2 = PROJECT_ROOT.parents[1] / "output" / "ds_pcd_mobilise" / "stage2"
DEFAULT_OUT = PROJECT_ROOT.parents[1] / "output" / "ds_pcd_mobilise" / "stage3"
DEFAULT_BUNDLE = PROJECT_ROOT / "datasets" / "mobilise_d" / "cvs_v1.0.0"
HORIZONS = (30, 7)
FOLDS = 3
SEED = 20261113
RIDGE = 1e-4
BOOTSTRAP_REPLICATES = 2000
METHODS = {"A_rms": ("rms_distance",), "B_rms_cusum": ("rms_distance", "cusum")}
PRIMARY_MODE = "session_se"
SENSITIVITY_MODE = "none"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mobilise-D 阶段 3 折外比较")
    parser.add_argument("--stage2", default=str(DEFAULT_STAGE2))
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--ridge", type=float, default=RIDGE)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--primary-mode", default=PRIMARY_MODE)
    parser.add_argument("--sensitivity-mode", default=SENSITIVITY_MODE)
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
        "available": True, "commit": commit, "dirty": bool(changed),
        "changed_path_count": len(changed), "changed_paths_sample": changed[:40],
    }


# ---------------------------------------------------------------------------
# 数据与折
# ---------------------------------------------------------------------------


def load_analysis_frame(stage2: Path, mode: str, bundle: Path) -> pd.DataFrame:
    scores = pd.read_csv(stage2 / f"scores_{mode}.csv", dtype={"participantid": str})
    windows = pd.read_csv(stage2 / f"score_windows_{mode}.csv", dtype={"participantid": str})
    landmarks = pd.read_csv(bundle / "landmarks.csv", dtype={"participantid": str})
    landmarks["landmark_day"] = pd.to_datetime(landmarks["landmark_day"]).dt.normalize()
    scores["landmark_day"] = pd.to_datetime(scores["landmark_day"]).dt.normalize()
    windows["landmark_day"] = pd.to_datetime(windows["landmark_day"]).dt.normalize()
    for column in ("cusum", "rms_distance", "T", "q"):
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    frame = scores[[
        "participantid", "landmark_day", "visit_number", "rms_distance", "cusum",
        "T", "q", "trend_7d", "persistent_deviation_days", "precision_mode",
    ]].copy()
    for horizon in HORIZONS:
        subset = windows.loc[windows["horizon_days"].eq(horizon), [
            "participantid", "landmark_day", "status", "event", "first_event_day",
            "window_end", "diary_day",
        ]].rename(columns={
            "status": f"status_{horizon}d",
            "event": f"event_{horizon}d",
            "first_event_day": f"first_event_{horizon}d",
            "window_end": f"window_end_{horizon}d",
        })
        frame = frame.merge(
            subset, on=["participantid", "landmark_day"], how="left", validate="one_to_one",
        )
    frame = frame.merge(
        landmarks[["participantid", "landmark_day", "diary_day", "baseline_eligible"]],
        on=["participantid", "landmark_day"], how="left", validate="one_to_one",
    )
    frame["landmark_day"] = pd.to_datetime(frame["landmark_day"]).dt.normalize()
    for horizon in HORIZONS:
        event = frame[f"event_{horizon}d"].astype(str).str.lower().eq("true")
        first = pd.to_datetime(frame[f"first_event_{horizon}d"], errors="coerce")
        duration = np.where(
            event,
            (first - frame["landmark_day"]).dt.days.astype(float),
            float(horizon),
        )
        frame[f"event_bool_{horizon}d"] = event
        frame[f"duration_{horizon}d"] = np.clip(duration, 1e-6, float(horizon))
        if not frame.loc[frame[f"status_{horizon}d"].eq("eligible")].shape[0] == len(frame):
            raise ValueError(f"{horizon} 天窗口不全部可评估，主队列口径被破坏")
        if event.any() and first[event].isna().any():
            raise ValueError(f"{horizon} 天事件缺少首次跌倒日期")
    frame["person_has_event_30d"] = frame.groupby("participantid")["event_bool_30d"].transform("max")
    return frame.sort_values(["participantid", "landmark_day"]).reset_index(drop=True)


def assign_folds(
    frame: pd.DataFrame, folds: int, seed: int,
) -> pd.DataFrame:
    """按人分组，按是否有 30 天阳性平衡分配。"""
    people = frame.groupby("participantid", as_index=False).agg(
        has_event=("person_has_event_30d", "max"),
        n_landmarks=("landmark_day", "size"),
        n_events=("event_bool_30d", "sum"),
    )
    rng = np.random.default_rng(seed)
    assignments: dict[str, int] = {}
    for has_event in (True, False):
        group = sorted(people.loc[people["has_event"].eq(has_event), "participantid"])
        rng.shuffle(group)
        for index, person in enumerate(group):
            assignments[str(person)] = index % folds
    people["fold"] = people["participantid"].map(assignments).astype(int)
    # 校验每折两类都来自同一分配，且同一人只属于一个折
    counts = people.groupby("fold")["has_event"].agg(["sum", "size"])
    if counts["sum"].min() < 1:
        raise ValueError("某些折没有 30 天阳性受试者，无法按人平衡")
    return people


# ---------------------------------------------------------------------------
# 拟合与预测
# ---------------------------------------------------------------------------


def _artifact(
    fit: CoxFit,
    fold: int,
    horizon: int,
    method: str,
    *,
    mode: str,
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    ridge: float,
    seed: int,
) -> CoxModelArtifact:
    return CoxModelArtifact.from_dict({
        "schema_version": 2,
        "model_type": "landmark_cox",
        "bundle_id": f"mobilise-stage3-{mode}-{horizon}d-{method}-f{fold}-s{seed}",
        "horizon_days": horizon,
        "feature_names": list(fit.feature_names),
        "coefficients": list(fit.coefficients),
        "feature_centers": list(fit.centers),
        "baseline_survival": fit.baseline_survival,
        "calibrated": False,
        "trained_at": fit.notes.get("fitted_at_utc", ""),
        "evidence_ids": ["mobilise_d_cvs_v1.0.0", "cox_1972", "landmarking_2007"],
        "training_summary": {
            "fold": fold,
            "mode": mode,
            "method": method,
            "persons": int(train_rows["participantid"].nunique()),
            "landmarks": int(len(train_rows)),
            "events": int(train_rows["event"].sum()),
            "test_persons": int(test_rows["participantid"].nunique()),
            "test_landmarks": int(len(test_rows)),
            "test_events": int(test_rows["event"].sum()),
            "ridge": ridge,
            "seed": seed,
        },
    })


def fit_predict(
    frame: pd.DataFrame,
    people: pd.DataFrame,
    horizon: int,
    method: str,
    features: Sequence[str],
    *,
    mode: str,
    ridge: float,
    seed: int,
    model_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    dataset = frame.rename(columns={
        f"event_bool_{horizon}d": "event",
        f"duration_{horizon}d": "duration",
    })
    for fold in sorted(people["fold"].unique()):
        test_people = set(people.loc[people["fold"].eq(fold), "participantid"])
        train = dataset.loc[~dataset["participantid"].isin(test_people)]
        test = dataset.loc[dataset["participantid"].isin(test_people)]
        fit = fit_landmark_cox(
            train["duration"].to_numpy(),
            train["event"].astype(int).to_numpy(),
            train[list(features)].to_numpy(dtype=np.float64),
            features,
            horizon_days=float(horizon),
            ridge=ridge,
        )
        status = {
            "mode": mode, "horizon": horizon, "fold": int(fold), "method": method,
            "status": fit.status, "converged": fit.converged,
            "train_rows": int(len(train)), "train_events": int(train["event"].sum()),
            "train_persons": int(train["participantid"].nunique()),
            "test_rows": int(len(test)), "test_events": int(test["event"].sum()),
            "iterations": fit.iterations, "log_likelihood": fit.log_likelihood,
            "baseline_survival": fit.baseline_survival,
            "coefficients": "|".join(f"{value:.6g}" for value in fit.coefficients),
            "feature_names": "|".join(fit.feature_names),
            "note": fit.notes.get("reason", ""),
        }
        statuses.append(status)
        if fit.status != "ok":
            continue
        artifact = _artifact(
            fit, int(fold), horizon, method, mode=mode,
            train_rows=train, test_rows=test, ridge=ridge, seed=seed,
        )
        target = model_dir / mode / f"{horizon}d"
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{method}_fold{fold}.json").write_text(
            json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8",
        )
        for row in test.itertuples(index=False):
            features_payload = {name: float(getattr(row, name)) for name in features}
            risk = predict_cumulative_risk_research(features_payload, artifact)
            predictions.append({
                "participantid": str(row.participantid),
                "landmark_day": row.landmark_day.date().isoformat(),
                "fold": int(fold),
                "mode": mode,
                "horizon": horizon,
                "method": method,
                "risk": float(risk),
                "event": int(row.event),
                "duration": float(row.duration),
                "rms_distance": float(row.rms_distance),
                "cusum": float(row.cusum),
            })
    return predictions, statuses


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


def auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = labels == 1
    negatives = ~positives
    if positives.sum() == 0 or negatives.sum() == 0:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # 平均秩处理并列
    frame = pd.Series(scores)
    for value, group in frame.groupby(frame, sort=False):
        if len(group) > 1:
            ranks[group.index.to_numpy()] = ranks[group.index.to_numpy()].mean()
    positive_ranks = ranks[positives].sum()
    return float(
        (positive_ranks - positives.sum() * (positives.sum() + 1) / 2.0)
        / (positives.sum() * negatives.sum())
    )


def brier(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((np.asarray(labels, dtype=float) - np.asarray(probabilities, dtype=float)) ** 2))


def calibration_slope(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    """logit(p) 对 y 的逻辑回归斜率；事件过少或不可估时返回 None。"""
    labels = np.asarray(labels, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(probabilities / (1 - probabilities))
    if labels.sum() < 5 or (1 - labels).sum() < 5:
        return None
    design = np.column_stack([np.ones_like(logits), logits])
    beta = np.zeros(2)
    for _ in range(100):
        eta = design @ beta
        mu = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        weights = np.clip(mu * (1 - mu), 1e-8, None)
        gradient = design.T @ (labels - mu) - 1e-6 * beta
        hessian = -(design.T * weights) @ design - 1e-6 * np.eye(2)
        try:
            step = np.linalg.solve(-hessian, gradient)
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    if not np.all(np.isfinite(beta)) or abs(float(beta[1])) > 20:
        return None
    return float(beta[1])


def metric_block(
    labels: np.ndarray, probabilities: np.ndarray,
) -> dict[str, Any]:
    return {
        "n": int(len(labels)),
        "events": int(np.sum(labels)),
        "persons": None,
        "auroc": auroc(labels, probabilities),
        "brier": brier(labels, probabilities),
        "observed_risk": float(np.mean(labels)) if len(labels) else None,
        "mean_predicted": float(np.mean(probabilities)) if len(probabilities) else None,
        "calibration_slope": calibration_slope(labels, probabilities),
    }


def paired_bootstrap(
    predictions: pd.DataFrame,
    *,
    horizon: int,
    mode: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """按人聚类重采样，计算 A/B 指标与配对差。"""
    subset = predictions.loc[
        predictions["horizon"].eq(horizon) & predictions["mode"].eq(mode)
    ]
    pivot = subset.pivot_table(
        index=["participantid", "landmark_day", "fold"],
        columns="method", values="risk", aggfunc="first",
    )
    meta = subset.drop_duplicates(["participantid", "landmark_day"]).set_index(
        ["participantid", "landmark_day"]
    )[["event", "fold"]]
    pivot = pivot.join(meta, how="inner")
    methods = ["A_rms", "B_rms_cusum"]
    missing = [method for method in methods if method not in pivot.columns]
    if missing:
        return {"status": "not_estimable", "reason": f"缺少预测列：{missing}"}
    rng = np.random.default_rng(seed)
    people = np.asarray(sorted(pivot.index.get_level_values("participantid").unique()))
    person_rows = {person: pivot.xs(person, level="participantid") for person in people}
    delta_auc, delta_brier = [], []
    auc_values = {method: [] for method in methods}
    brier_values = {method: [] for method in methods}
    slope_values = {method: [] for method in methods}
    failures = 0
    for _ in range(replicates):
        sample = rng.choice(people, size=len(people), replace=True)
        frames = [person_rows[person] for person in sample]
        block = pd.concat(frames, ignore_index=False)
        both = np.isfinite(block[methods].to_numpy(dtype=float)).all(axis=1)
        if both.sum() == 0:
            failures += 1
            continue
        paired = block.loc[both]
        paired_labels = paired["event"].to_numpy(dtype=int)
        if paired_labels.sum() == 0 or paired_labels.sum() == len(paired_labels):
            failures += 1
            continue
        auc_pair = {}
        brier_pair = {}
        for method in methods:
            probabilities = paired[method].to_numpy(dtype=float)
            value = auroc(paired_labels, probabilities)
            if value is None:
                failures += 1
                auc_pair = {}
                break
            auc_pair[method] = value
            brier_pair[method] = brier(paired_labels, probabilities)
            auc_values[method].append(value)
            brier_values[method].append(brier_pair[method])
            slope = calibration_slope(paired_labels, probabilities)
            if slope is not None:
                slope_values[method].append(slope)
        if len(auc_pair) != 2:
            continue
        delta_auc.append(auc_pair["B_rms_cusum"] - auc_pair["A_rms"])
        delta_brier.append(brier_pair["B_rms_cusum"] - brier_pair["A_rms"])
    def interval(values: Sequence[float]) -> list[float] | None:
        if len(values) < 50:
            return None
        return [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]
    return {
        "status": "ok",
        "methods": methods,
        "replicates_requested": replicates,
        "replicates_invalid": failures,
        "replicates_valid": replicates - failures,
        "delta_auroc_B_minus_A": interval(delta_auc),
        "delta_brier_B_minus_A": interval(delta_brier),
        "auroc": {method: interval(values) for method, values in auc_values.items()},
        "brier": {method: interval(values) for method, values in brier_values.items()},
        "calibration_slope": {method: interval(values) for method, values in slope_values.items()},
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    stage2 = Path(args.stage2)
    bundle = Path(args.bundle)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "fold_models"
    modes = [args.primary_mode, args.sensitivity_mode]
    if len(set(modes)) != 2:
        modes = [args.primary_mode]
    frames = {mode: load_analysis_frame(stage2, mode, bundle) for mode in modes}
    primary_frame = frames[args.primary_mode]
    if len(primary_frame) != 186 or primary_frame["participantid"].nunique() != 144:
        raise ValueError("主队列应为 186 个预测日 / 144 人")
    people = assign_folds(primary_frame, args.folds, args.seed)
    folds = people[["participantid", "fold", "has_event", "n_landmarks", "n_events"]].copy()
    folds.to_csv(out_dir / "folds.csv", index=False, encoding="utf-8")

    protocol = {
        "stage": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {
            "stage2_dir": str(stage2),
            "bundle": str(bundle),
            "inputs": {
                name: {"md5": _md5(stage2 / name)}
                for name in (
                    "scores_session_se.csv", "scores_none.csv",
                    "score_windows_session_se.csv", "score_windows_none.csv",
                )
            },
            "bundle_inputs": {
                name: {"md5": _md5(bundle / name)}
                for name in ("landmarks.csv", "landmark_windows.csv", "baseline_days.csv")
            },
        },
        "cohort": {
            "landmarks": int(len(primary_frame)),
            "persons": int(primary_frame["participantid"].nunique()),
            "events_30d": int(primary_frame["event_bool_30d"].sum()),
            "events_7d": int(primary_frame["event_bool_7d"].sum()),
            "complete_followup": True,
        },
        "models": {
            "A_rms": {"features": ["rms_distance"]},
            "B_rms_cusum": {"features": ["rms_distance", "cusum"]},
        },
        "folds": int(args.folds),
        "fold_assignment": "按人分组；按是否有 30 天阳性分层后随机分配；所有方法/时窗/模式共用",
        "seed": int(args.seed),
        "ridge_standardized": float(args.ridge),
        "metrics": {
            "auroc": "越大越好",
            "brier": "越小越好",
            "delta": "B − A；ΔAUROC>0 与 ΔBrier<0 都表示 B 更好",
            "bootstrap": f"按人聚类 {int(args.bootstrap)} 次，百分位 95% 区间",
            "calibration": "观察风险、平均预测风险；30 天附斜率与区间；7 天事件过少不报斜率",
        },
        "primary": {"mode": args.primary_mode, "horizon": 30},
        "sensitivity": {"mode": args.sensitivity_mode, "horizon": 30},
        "calibrated": False,
        "row_tolerance": 1e-9,
    }
    (out_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    all_predictions: list[dict[str, Any]] = []
    all_statuses: list[dict[str, Any]] = []
    for mode in modes:
        frame = frames[mode]
        if len(frame) != len(primary_frame):
            raise ValueError(f"{mode} 的可评行数与主队列不一致")
        for horizon in HORIZONS:
            for method, features in METHODS.items():
                predictions, statuses = fit_predict(
                    frame, people, horizon, method, features, mode=mode,
                    ridge=args.ridge, seed=args.seed, model_dir=model_dir,
                )
                all_predictions.extend(predictions)
                all_statuses.extend(statuses)
    predictions = pd.DataFrame(all_predictions)
    if predictions.empty:
        raise ValueError("没有任何折外预测")
    predictions = predictions.sort_values(
        ["mode", "horizon", "method", "participantid", "landmark_day"]
    ).reset_index(drop=True)
    predictions.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8")
    statuses = pd.DataFrame(all_statuses)
    statuses.to_csv(out_dir / "fold_status.csv", index=False, encoding="utf-8")
    if predictions.duplicated(["mode", "horizon", "method", "participantid", "landmark_day"]).any():
        raise ValueError("折外预测键不唯一")

    metrics: list[dict[str, Any]] = []
    bootstrap_results: dict[str, Any] = {}
    for mode in modes:
        for horizon in HORIZONS:
            subset = predictions.loc[
                predictions["mode"].eq(mode) & predictions["horizon"].eq(horizon)
            ]
            for method in METHODS:
                block = subset.loc[subset["method"].eq(method)]
                labels = block["event"].to_numpy(dtype=int)
                probabilities = block["risk"].to_numpy(dtype=float)
                entry = {
                    "mode": mode, "horizon": horizon, "method": method,
                    **metric_block(labels, probabilities),
                    "persons": int(block["participantid"].nunique()),
                    "folds_available": int(block["fold"].nunique()),
                    "status": "ok" if len(block) else "not_estimable",
                }
                entry["auroc_ci"] = None
                metrics.append(entry)
            # 配对差
            pivot = subset.pivot_table(
                index=["participantid", "landmark_day"], columns="method",
                values="risk", aggfunc="first",
            )
            meta = subset.drop_duplicates(["participantid", "landmark_day"]).set_index(
                ["participantid", "landmark_day"]
            )["event"]
            pivot = pivot.join(meta, how="inner")
            both = pivot.dropna(subset=list(METHODS))
            if len(both):
                labels = both["event"].to_numpy(dtype=int)
                a = both["A_rms"].to_numpy(dtype=float)
                b = both["B_rms_cusum"].to_numpy(dtype=float)
                # 配对子集上的 A/B 指标：两组使用同一批有效行
                for method, values in (("A_rms", a), ("B_rms_cusum", b)):
                    metrics.append({
                        "mode": mode, "horizon": horizon,
                        "method": f"{method}_paired",
                        "n": int(len(both)), "events": int(labels.sum()),
                        "persons": int(
                            both.index.get_level_values("participantid").nunique()
                        ),
                        "auroc": auroc(labels, values),
                        "brier": brier(labels, values),
                        "observed_risk": float(np.mean(labels)),
                        "mean_predicted": float(np.mean(values)),
                        "calibration_slope": calibration_slope(labels, values),
                        "folds_available": int(
                            subset.loc[
                                subset["method"].eq(method), "fold"
                            ].nunique()
                        ),
                        "status": "ok",
                    })
                metrics.append({
                    "mode": mode, "horizon": horizon, "method": "B_minus_A",
                    "n": int(len(both)), "events": int(labels.sum()),
                    "persons": int(both.index.get_level_values("participantid").nunique()),
                    "delta_auroc": (auroc(labels, b) or 0.0) - (auroc(labels, a) or 0.0)
                    if auroc(labels, a) is not None and auroc(labels, b) is not None else None,
                    "delta_brier": brier(labels, b) - brier(labels, a),
                    "status": "ok",
                })
            bootstrap_results[f"{mode}_{horizon}d"] = paired_bootstrap(
                predictions, horizon=horizon, mode=mode,
                replicates=args.bootstrap, seed=args.seed,
            )
    paired_n = {
        (row["mode"], int(row["horizon"])): row["n"]
        for row in metrics if row["method"] == "B_minus_A"
    }
    enriched = []
    for row in metrics:
        key = f"{row['mode']}_{int(row['horizon'])}d"
        bootstrap = bootstrap_results.get(key, {})
        item = dict(row)
        if row["method"] in ("A_rms", "B_rms_cusum", "A_rms_paired", "B_rms_cusum_paired"):
            base = row["method"].replace("_paired", "")
            attach = row["method"].endswith("_paired") or (
                row["n"] == paired_n.get((row["mode"], int(row["horizon"])))
            )
            interval = bootstrap.get("auroc", {}).get(base) if attach else None
            item["auroc_ci_low"] = interval[0] if interval else None
            item["auroc_ci_high"] = interval[1] if interval else None
            interval = bootstrap.get("brier", {}).get(base) if attach else None
            item["brier_ci_low"] = interval[0] if interval else None
            item["brier_ci_high"] = interval[1] if interval else None
            interval = bootstrap.get("calibration_slope", {}).get(base) if attach else None
            item["slope_ci_low"] = interval[0] if interval else None
            item["slope_ci_high"] = interval[1] if interval else None
        if row["method"] == "B_minus_A":
            interval = bootstrap.get("delta_auroc_B_minus_A")
            item["delta_auroc_ci_low"] = interval[0] if interval else None
            item["delta_auroc_ci_high"] = interval[1] if interval else None
            interval = bootstrap.get("delta_brier_B_minus_A")
            item["delta_brier_ci_low"] = interval[0] if interval else None
            item["delta_brier_ci_high"] = interval[1] if interval else None
        enriched.append(item)
    failure_counts = {
        (mode, horizon): statuses.loc[
            statuses["mode"].eq(mode) & statuses["horizon"].eq(horizon)
            & statuses["status"].ne("ok")
        ][["fold", "method", "status"]].to_dict("records")
        for mode in modes for horizon in HORIZONS
    }
    for item in enriched:
        key = (item["mode"], int(item["horizon"]))
        failures = failure_counts.get(key, [])
        item["fold_failures"] = len(failures)
        if int(item["horizon"]) == 7:
            item["status"] = (
                "not_estimable" if item["method"] == "B_minus_A" else "descriptive_unstable"
            )
            item["note"] = (
                f"{int(item['events'])} 个阳性；7 天仅作描述"
                + (f"；失败折：{failures}" if failures else "")
            )
        elif failures:
            item["status"] = "partial"
            item["note"] = f"失败折：{failures}"
        else:
            item["status"] = "ok"
            item["note"] = ""
    metrics_frame = pd.DataFrame(enriched)
    metrics_frame.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8")
    (out_dir / "bootstrap.json").write_text(
        json.dumps(bootstrap_results, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # 宽集筛选影响（描述性）
    landmarks = pd.read_csv(bundle / "landmarks.csv", dtype={"participantid": str})
    windows = pd.read_csv(bundle / "landmark_windows.csv", dtype={"participantid": str})
    wide_positive = windows.loc[
        windows["horizon_days"].eq(30) & windows["event"].astype(str).str.lower().eq("true")
    ]
    wide_positive_7 = windows.loc[
        windows["horizon_days"].eq(7) & windows["event"].astype(str).str.lower().eq("true")
    ]
    selection = {
        "candidate_landmarks": int(len(landmarks)),
        "candidate_people": int(landmarks["participantid"].nunique()),
        "baseline_ready_landmarks": int(landmarks["baseline_eligible"].astype(str).str.lower().eq("true").sum()),
        "baseline_ready_people": int(primary_frame["participantid"].nunique()),
        "excluded_landmarks": int(len(landmarks) - len(primary_frame)),
        "excluded_30d_positive_landmarks": int(len(wide_positive) - int(primary_frame["event_bool_30d"].sum())),
        "wide_30d_positive_landmarks": int(len(wide_positive)),
        "wide_30d_positive_people": int(wide_positive["participantid"].nunique()),
        "wide_7d_positive_landmarks": int(len(wide_positive_7)),
        "wide_7d_positive_people": int(wide_positive_7["participantid"].nunique()),
        "note": "宽集只用于描述筛选影响；模型只在基线就绪的 186 行上拟合与评估。",
    }

    manifest = {
        "stage": 3,
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "python scripts/analyse_mobilise_stage3.py",
        "protocol": {"path": "protocol.json", "md5": _md5(out_dir / "protocol.json")},
        "code": _git_state(PROJECT_ROOT.parent),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "inputs": protocol["data"],
        "cohort": protocol["cohort"],
        "selection": selection,
        "predictions": {
            "rows": int(len(predictions)),
            "by_mode": {
                mode: int((predictions["mode"] == mode).sum()) for mode in modes
            },
        },
        "fold_status": statuses["status"].value_counts().to_dict(),
        "bootstrap": {
            key: {
                "status": value.get("status"),
                "valid": value.get("replicates_valid"),
                "invalid": value.get("replicates_invalid"),
            }
            for key, value in bootstrap_results.items()
        },
        "outputs": {
            "folds": "folds.csv",
            "predictions": "predictions.csv",
            "fold_status": "fold_status.csv",
            "fold_models": "fold_models/",
            "metrics": "metrics.csv",
            "bootstrap": "bootstrap.json",
        },
        "calibrated": False,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    summary = {
        "rows": int(len(predictions)),
        "modes": modes,
        "fold_status": statuses[["mode", "horizon", "fold", "method", "status"]]
        .to_dict("records"),
        "metrics": metrics_frame.to_dict("records"),
        "bootstrap": bootstrap_results,
        "selection": selection,
    }
    (out_dir / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        "out": str(out_dir),
        "predictions": int(len(predictions)),
        "fold_status": statuses["status"].value_counts().to_dict(),
        "metrics": [
            {
                "mode": row["mode"], "horizon": int(row["horizon"]),
                "method": row["method"],
                **{key: (round(row[key], 4) if isinstance(row.get(key), float) else row.get(key))
                   for key in ("auroc", "brier", "delta_auroc", "delta_brier", "n", "events")},
            }
            for row in metrics
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
