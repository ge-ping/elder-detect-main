"""阶段 14：运行 1/3/7 天风险对照、训练增强贡献与组件消融。

输出：
* ``paired_predictions.csv``：每个测试键的 M0–M3、三条件、四消融变体配对预测；
* ``metrics_by_cell.csv``：按 branch/horizon/scenario/condition/scoring_version 的指标；
* ``primary_7d.csv``、``secondary_1_3d.csv``：主表与预设次要窗口表；
* ``training_augmentation.csv``：同一 M 模型的 R+S − R-only；
* ``ablation.csv``：完整 / 无精度 / 仅方向 T / 不分方向但同样累积；
* ``detection_table.csv``：误报对齐下的检出率、延迟与未检出比例；
* ``joint_tests.json``：人员聚类稳健的 T/C 联合检验；
* ``conclusions.json``：按预设规则给出的三种结论状态。

只使用冻结模型与固定测试键；不重新调参、不新增窗口或机制。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import md5
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.cox_horizon import (  # noqa: E402
    DEFAULT_HORIZONS, fit_horizon_cox, joint_test_coefficients, predict_horizon_risk,
)
from mobilise_d.round2 import R2_FEATURES  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise" / "round2"
STAGE11 = OUTPUT_ROOT / "stage11" / "20261115-protocol-v1"
STAGE12 = OUTPUT_ROOT / "stage12" / "20261115-supplement-v1"
STAGE13 = OUTPUT_ROOT / "stage13" / "20261115-freeze-v2"
SEED = 20261115
BOOTSTRAP = 2000

LEVEL_COLUMNS = tuple(f"level_{name}" for name in R2_FEATURES)
CHANGE_COLUMNS = tuple(f"change_{name}" for name in R2_FEATURES)
BASE = ("age_z", "prior_fall_any")
LADDERS = {
    "M0": BASE,
    "M1": BASE + LEVEL_COLUMNS,
    "M2": BASE + LEVEL_COLUMNS + CHANGE_COLUMNS,
    "M3": BASE + LEVEL_COLUMNS + CHANGE_COLUMNS + ("T", "C"),
}
ABLATIONS = {
    "full": ("T", "C"),
    "no_precision": ("T_none", "C_none"),
    "direction_only": ("T",),
    "no_direction_cumulative": ("D_snapshot", "C_omnibus"),
}
CONDITIONS = ("R-only", "R+S", "S-only")
MECHANISMS = ("level_dominant", "sustained_change", "change_no_effect")
NOISE_FAMILIES = ("gaussian_ar1", "heavy_tail_spike")

CODE_SNAPSHOT_FILES = (
    "scripts/round2_stage14_evaluate.py",
    "scripts/verify_round2_stage14.py",
    "longitudinal/cox_horizon.py",
    "mobilise_d/round2.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 14：评价与消融")
    parser.add_argument("--run-id", default="20261115-eval-v1")
    parser.add_argument("--out-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--stage11", default=str(STAGE11))
    parser.add_argument("--stage12", default=str(STAGE12))
    parser.add_argument("--stage13", default=str(STAGE13))
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")


def _norm_person(values: pd.Series) -> pd.Series:
    """统一人员编号：去掉浮点写法尾部的 ``.0``。"""
    return values.astype(str).str.replace(r"\.0$", "", regex=True)


def _md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


def _fast_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    from scipy.stats import rankdata

    ranks = rankdata(scores, method="average")
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _auprc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = labels == 1
    n_pos = int(positives.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    hits = np.cumsum(sorted_labels == 1)
    precision = hits / np.arange(1, len(sorted_labels) + 1)
    return float(precision[sorted_labels == 1].mean())


def _calibration(labels: np.ndarray, scores: np.ndarray) -> tuple[float | None, float | None]:
    if len(np.unique(labels)) < 2:
        return None, None
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped))
    if np.std(logit) < 1e-12:
        return None, None
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    model.fit(logit.reshape(-1, 1), labels.astype(int))
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(labels) & np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    positives = int(labels.sum()) if len(labels) else 0
    slope, intercept = _calibration(labels, scores) if len(labels) else (None, None)
    return {
        "n": int(len(labels)),
        "positives": positives,
        "prevalence": float(labels.mean()) if len(labels) else None,
        "auroc": _fast_auroc(labels, scores) if positives >= 5 else None,
        "auroc_not_estimable_reason": None if positives >= 5 else "fewer_than_5_positives",
        "auprc": _auprc(labels, scores) if positives >= 5 else None,
        "brier": float(np.mean((scores - labels) ** 2)) if len(labels) else None,
        "brier_all_predictions_zero": float(labels.mean()) if len(labels) else None,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "mean_prediction": float(scores.mean()) if len(labels) else None,
    }


def _bootstrap_delta(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    clusters: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict:
    """配对 bootstrap（按簇）：ΔBrier 与 ΔAUROC（b − a）。"""
    labels = np.asarray(labels, dtype=float)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    unique = np.unique(clusters)
    index_by_cluster = {key: np.flatnonzero(clusters == key) for key in unique}
    rng = np.random.default_rng(seed)
    brier_diffs = []
    auroc_diffs = []
    for _ in range(samples):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([index_by_cluster[key] for key in chosen])
        y = labels[index]
        if len(np.unique(y)) < 2:
            continue
        a, b = scores_a[index], scores_b[index]
        brier_diffs.append(float(np.mean((b - y) ** 2) - np.mean((a - y) ** 2)))
        auc_a, auc_b = _fast_auroc(y, a), _fast_auroc(y, b)
        if auc_a is not None and auc_b is not None:
            auroc_diffs.append(auc_b - auc_a)

    def interval(values: Sequence[float]) -> list[float] | None:
        if len(values) < 50:
            return None
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]

    return {
        "delta_brier": float(np.mean((scores_b - labels) ** 2) - np.mean((scores_a - labels) ** 2)),
        "delta_brier_ci": interval(brier_diffs),
        "delta_auroc": (
            (_fast_auroc(labels, scores_b) - _fast_auroc(labels, scores_a))
            if (_fast_auroc(labels, scores_b) is not None and _fast_auroc(labels, scores_a) is not None)
            else None
        ),
        "delta_auroc_ci": interval(auroc_diffs),
        "bootstrap_samples_usable": len(brier_diffs),
        "clusters": int(len(unique)),
        "note": "按人/序列簇重抽；未重现模型选择与重拟合的不确定性",
    }


def conclusion_state(entry: Mapping[str, Any], *, min_positives: int = 5) -> dict:
    """按预设规则给出三种结论状态。"""
    positives = int(entry.get("positives") or 0)
    ci = entry.get("delta_brier_ci")
    delta = entry.get("delta_brier")
    if positives < min_positives:
        return {
            "state": "not_estimable",
            "reason": f"阳性 {positives} < {min_positives}，区分度与增量均不可估",
        }
    if ci is None or delta is None:
        return {"state": "not_estimable", "reason": "bootstrap 区间不可用"}
    half_width = (ci[1] - ci[0]) / 2.0
    if ci[1] < 0:
        return {"state": "incremental_evidence", "note": "ΔBrier 95% 区间整体低于 0"}
    if ci[0] > 0:
        return {"state": "no_practical_increment_or_decline", "note": "ΔBrier 95% 区间整体高于 0"}
    return {
        "state": "no_clear_increment_wide_interval",
        "note": f"ΔBrier 区间跨 0，半宽 {half_width:.5f}",
        "half_width": half_width,
    }


# ---------------------------------------------------------------------------
# 数据装配
# ---------------------------------------------------------------------------


def _training_age_params(diagnostics: Mapping[str, Any]) -> dict[int, tuple[float, float]]:
    return {
        int(fold): (float(value["real"]["age_mean"]), float(value["real"]["age_sd"]))
        for fold, value in diagnostics["folds"].items()
    }


def _real_rows(stage12: Path, stage11: Path, variants: pd.DataFrame) -> pd.DataFrame:
    scores = pd.read_csv(stage12 / "real_scores.csv", dtype={"participantid": str})
    candidates = pd.read_csv(stage11 / "real_candidates.csv", dtype={"participantid": str})
    splits = pd.read_csv(stage11 / "splits.csv", dtype={"participantid": str})
    frame = scores.loc[scores["status"] == "scored"].copy()
    frame = frame.merge(
        candidates[["participantid", "landmark_day", "first_event_7d"]],
        on=["participantid", "landmark_day"], how="left", validate="one_to_one",
    )
    frame = frame.merge(splits[["participantid", "fold"]], on="participantid", how="left")
    gap = (pd.to_datetime(frame["first_event_7d"]) - pd.to_datetime(frame["landmark_day"])).dt.days
    frame["event_gap"] = np.where(frame["y_7d"].astype(float) == 1, gap, np.nan)
    frame["duration"] = np.where(frame["y_7d"].astype(float) == 1, frame["event_gap"], 7.0)
    frame["event"] = frame["y_7d"].astype(float)
    frame["participantid"] = _norm_person(frame["participantid"])
    frame["key"] = frame["participantid"] + "|" + frame["landmark_day"].astype(str)
    variant = variants.loc[variants["test_set"] == "real_test"].copy()
    variant["key"] = variant["participantid"] + "|" + variant["landmark_day"].astype(str)
    frame = frame.merge(
        variant[["key", "T", "C", "T_none", "C_none", "D_snapshot", "C_omnibus",
                 "status_full", "status_none"]],
        on="key", how="left", suffixes=("", "_variant"), validate="one_to_one",
    )
    frame["branch"] = "R"
    frame["test_set"] = "real_test"
    frame["cohort"] = "pd"
    frame["scenario"] = "pd_real"
    frame["scoring_version"] = "round2-scoring-v1"
    return frame


def _synthetic_rows(stage12: Path, variants: pd.DataFrame) -> pd.DataFrame:
    test = pd.read_csv(stage12 / "synthetic_test_scores.csv")
    train = pd.read_csv(stage12 / "synthetic_train_scores.csv")
    frame = pd.concat([train.assign(batch="train"), test.assign(batch="test")],
                      ignore_index=True, sort=False)
    frame = frame.loc[frame["status"] == "scored"].copy()
    gap = pd.to_numeric(frame["event_gap_from_snapshot"], errors="coerce")
    frame["event_gap"] = gap
    frame["duration"] = np.where((gap > 0) & (gap <= 7), gap, 7.0)
    frame["event"] = ((gap > 0) & (gap <= 7)).astype(float)
    frame["key"] = frame["series_id"]
    variant = variants.loc[variants["test_set"].str.startswith("synthetic")].copy()
    rename = {"series_id": "key"}
    variant = variant.rename(columns=rename)
    frame = frame.merge(
        variant[["key", "T", "C", "T_none", "C_none", "D_snapshot", "C_omnibus",
                 "status_full", "status_none"]],
        on="key", how="left", suffixes=("", "_variant"), validate="one_to_one",
    )
    frame["branch"] = "S"
    frame["test_set"] = np.where(frame["batch"] == "test", "synthetic_test", "synthetic_train")
    frame["cohort"] = "synthetic_pd_like"
    frame["scenario"] = frame["mechanism"] + "|" + frame["noise_family"]
    frame["scoring_version"] = "round2-scoring-v1"
    return frame


def _fit_and_predict(
    *,
    train_real: pd.DataFrame,
    train_synthetic: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    condition: str,
    fold: int,
    bundle: str,
    age_mean: float,
    age_sd: float,
    synthetic_weight_share: float = 0.5,
) -> tuple[Any, np.ndarray, dict]:
    real = train_real.copy()
    synthetic = train_synthetic.copy()
    for part in (real, synthetic):
        part["age_z"] = (part["age"].astype(float) - age_mean) / age_sd
    if condition == "R-only":
        train = real
        train["weight"] = 1.0
        weights_info = {"real_weight": float(len(real)), "synthetic_weight": 0.0}
    elif condition == "S-only":
        train = synthetic
        train["weight"] = 1.0
        weights_info = {"real_weight": 0.0, "synthetic_weight": float(len(synthetic))}
    else:
        real["weight"] = 1.0
        per_person = synthetic_weight_share * float(len(real)) / max(len(synthetic), 1)
        synthetic["weight"] = per_person
        train = pd.concat([real, synthetic], ignore_index=True, sort=False)
        weights_info = {
            "real_weight": float(len(real)),
            "synthetic_weight": float(per_person * len(synthetic)),
        }
    train = train.loc[np.isfinite(train[list(features)].to_numpy(dtype=float)).all(axis=1)]
    test = test.copy()
    test["age_z"] = (test["age"].astype(float) - age_mean) / age_sd
    test = test.loc[np.isfinite(test[list(features)].to_numpy(dtype=float)).all(axis=1)]
    fit = fit_horizon_cox(
        train["duration"].to_numpy(dtype=float),
        train["event"].to_numpy(dtype=int),
        train[list(features)].to_numpy(dtype=float),
        list(features),
        horizon_days=7.0, ridge=1e-4,
        weights=train["weight"].to_numpy(dtype=float),
        horizons_days=DEFAULT_HORIZONS,
    )
    fit = type(fit)(**{**fit.__dict__, "notes": {**fit.notes, "bundle_id": bundle}})
    matrix = test[list(features)].to_numpy(dtype=float)
    predictions = {
        horizon: predict_horizon_risk(fit, matrix, horizon) for horizon in DEFAULT_HORIZONS
    }
    return fit, predictions, {
        "train_rows": int(len(train)),
        "train_events": int(train["event"].sum()),
        "test_rows": int(len(test)),
        **weights_info,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    run_dir = Path(args.out_root) / f"stage14/{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stage11, stage12, stage13 = Path(args.stage11), Path(args.stage12), Path(args.stage13)
    stage13_run = pd.read_csv(stage13 / "oof_predictions.csv", low_memory=False)
    variants = pd.read_csv(run_dir / "variant_scores.csv", dtype={"participantid": str})
    diagnostics = json.loads((stage13 / "training_diagnostics.json").read_text(encoding="utf-8"))
    age_params = _training_age_params(diagnostics)
    thresholds = json.loads((stage12 / "threshold_calibration.json").read_text(encoding="utf-8"))

    real = _real_rows(stage12, stage11, variants)
    synthetic = _synthetic_rows(stage12, variants)
    folds = sorted(real["fold"].dropna().unique())

    # 组装已冻结的 M0–M3 预测（Stage 13）
    frozen = stage13_run.copy()
    frozen["age_z"] = np.nan
    for fold in folds:
        mask = frozen["fold"] == int(fold)
        mean, sd = age_params[int(fold)]
        frozen.loc[mask, "age_z"] = (frozen.loc[mask, "age"].astype(float) - mean) / sd
    paired_rows: list[pd.DataFrame] = []

    def add_predictions(source: pd.DataFrame, prefix: str, variant: str) -> None:
        for horizon in (1, 3, 7):
            column = f"p_{horizon}d"
            if column not in source:
                continue
            part = pd.DataFrame({
                "branch": source["branch"] if "branch" in source else prefix,
                "test_set": source["test_set"] if "test_set" in source else "",
                "key": source["key"] if "key" in source else "",
                "fold": source["fold"],
                "model": source["model"],
                "condition": source["condition"],
                "ablation": variant,
                "horizon_days": horizon,
                "scenario": source["scenario"] if "scenario" in source else "",
                "mechanism": source.get("mechanism", ""),
                "noise_family": source.get("noise_family", ""),
                "trajectory": source.get("trajectory", ""),
                "cohort": source.get("cohort", ""),
                "scoring_version": source.get("scoring_version", "round2-scoring-v1"),
                "prediction": source[column],
                "label": source[f"y_{horizon}d"],
            })
            paired_rows.append(part)

    frozen["participantid"] = _norm_person(frozen["participantid"])
    frozen["key"] = np.where(
        frozen["test_set"] == "real_test",
        frozen["participantid"] + "|" + frozen["landmark_day"].astype(str),
        frozen["series_id"].astype(str),
    )
    frozen["scenario"] = np.where(
        frozen["test_set"] == "real_test", "pd_real",
        frozen["mechanism"].astype(str) + "|" + frozen["noise_family"].astype(str),
    )
    frozen["cohort"] = np.where(frozen["test_set"] == "real_test", "pd", "synthetic_pd_like")
    frozen["branch"] = np.where(frozen["test_set"] == "real_test", "R", "S")
    add_predictions(frozen, prefix="frozen", variant="full")

    # 消融变体：同一折、同一条件、同一训练行，只换 DS-PCD 组件
    ablation_records: list[dict] = []
    for fold in folds:
        fold = int(fold)
        mean, sd = age_params[fold]
        train_real = real.loc[real["fold"] != fold]
        train_synthetic = synthetic.loc[(synthetic["fold"] == fold) & (synthetic["batch"] == "train")]
        test_real = real.loc[real["fold"] == fold]
        test_synthetic = synthetic.loc[(synthetic["fold"] == fold) & (synthetic["batch"] == "test")]
        for condition in CONDITIONS:
            for variant, extra in ABLATIONS.items():
                if variant == "full":
                    continue
                features = LADDERS["M2"] + extra
                for name, train_real_part, test_part, branch, test_set in (
                    ("real", train_real, test_real, "R", "real_test"),
                    ("synthetic", train_synthetic, test_synthetic, "S", "synthetic_test"),
                ):
                    if not len(test_part):
                        continue
                    bundle = f"round2-abl-fold{fold}-{condition}-{variant}-{name}"
                    fit, predictions, info = _fit_and_predict(
                        train_real=train_real_part,
                        train_synthetic=train_synthetic,
                        test=test_part,
                        features=features,
                        condition=condition,
                        fold=fold,
                        bundle=bundle,
                        age_mean=mean,
                        age_sd=sd,
                    )
                    ablation_records.append({
                        "bundle": bundle,
                        "fold": fold,
                        "condition": condition,
                        "variant": variant,
                        "branch": branch,
                        "status": fit.status,
                        "features": list(features),
                        **info,
                    })
                    for horizon in (1, 3, 7):
                        paired_rows.append(pd.DataFrame({
                            "branch": branch,
                            "test_set": test_set,
                            "key": test_part["key"],
                            "fold": fold,
                            "model": f"M3_{variant}",
                            "condition": condition,
                            "ablation": variant,
                            "horizon_days": horizon,
                            "scenario": test_part["scenario"],
                            "mechanism": test_part.get("mechanism", ""),
                            "noise_family": test_part.get("noise_family", ""),
                            "trajectory": test_part.get("trajectory", ""),
                            "cohort": test_part["cohort"],
                            "scoring_version": "round2-scoring-v1",
                            "prediction": predictions[horizon],
                            "label": test_part[f"y_{horizon}d"],
                        }))
    paired = pd.concat(paired_rows, ignore_index=True, sort=False)
    paired.to_csv(run_dir / "paired_predictions.csv", index=False, encoding="utf-8")

    # 指标表
    metric_rows: list[dict] = []
    for keys, group in paired.groupby(
        ["branch", "test_set", "scenario", "condition", "ablation", "model", "horizon_days"]
    ):
        branch, test_set, scenario, condition, ablation, model, horizon = keys
        metrics = _metrics(group["label"].to_numpy(), group["prediction"].to_numpy())
        metric_rows.append({
            "branch": branch, "test_set": test_set, "scenario": scenario,
            "condition": condition, "ablation": ablation, "model": model,
            "horizon_days": horizon, **metrics,
        })
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(run_dir / "metrics_by_cell.csv", index=False, encoding="utf-8")

    # 主比较：ΔBrier(M3−M2) 与 ΔAUROC，配对 bootstrap
    def paired_delta(
        frame: pd.DataFrame, model_a: str, model_b: str, *, cluster: str, seed: int
    ) -> dict:
        index = ["key", "label"] if cluster == "key" else ["key", "label", cluster]
        wide = frame.pivot_table(
            index=index, columns="model", values="prediction",
            aggfunc="first",
        ).reset_index()
        if model_a not in wide.columns or model_b not in wide.columns:
            return {"status": "not_paired"}
        wide = wide.dropna(subset=[model_a, model_b, "label"])
        if not len(wide):
            return {"status": "no_rows"}
        return _bootstrap_delta(
            wide["label"].to_numpy(), wide[model_a].to_numpy(), wide[model_b].to_numpy(),
            wide[cluster].astype(str).to_numpy(), samples=args.bootstrap, seed=seed,
        )

    comparison_path = run_dir / "comparison.csv"
    comparison_rows: list[dict] = []
    for (branch, condition, horizon), group in ([] if comparison_path.is_file() else paired.loc[
        paired["ablation"] == "full"
    ].groupby(["branch", "condition", "horizon_days"])):
        cluster = "participantid" if branch == "R" else "key"
        if cluster == "participantid":
            group = group.assign(participantid=group["key"].str.split("|").str[0])
        for scenario, subset in group.groupby("scenario"):
            delta = paired_delta(subset, "M2", "M3", cluster=cluster, seed=SEED + int(horizon))
            positives = int(subset.loc[subset["model"] == "M2", "label"].sum())
            row = {
                "branch": branch, "condition": condition, "horizon_days": int(horizon),
                "scenario": scenario,
                "n": int(subset.loc[subset["model"] == "M2", "key"].nunique()),
                "positives": positives,
                **delta,
            }
            row.update(conclusion_state(row))
            comparison_rows.append(row)
    comparison = (
        pd.read_csv(comparison_path) if comparison_path.is_file()
        else pd.DataFrame(comparison_rows)
    )
    if not comparison_path.is_file():
        comparison.to_csv(comparison_path, index=False, encoding="utf-8")
    comparison.loc[(comparison["branch"] == "S") & (comparison["horizon_days"] == 7)].to_csv(
        run_dir / "primary_7d.csv", index=False, encoding="utf-8"
    )
    comparison.loc[(comparison["branch"] == "S") & (comparison["horizon_days"] != 7)].to_csv(
        run_dir / "secondary_1_3d.csv", index=False, encoding="utf-8"
    )

    # 等权综合（三种机制等权）
    composite_rows = []
    for (branch, condition, horizon), group in comparison.groupby(
        ["branch", "condition", "horizon_days"]
    ):
        if branch != "S":
            continue
        by_mechanism = group.loc[
            group["scenario"].str.startswith(tuple(MECHANISMS)), :
        ].copy()
        by_mechanism["mechanism"] = by_mechanism["scenario"].str.split("|").str[0]
        means = by_mechanism.groupby("mechanism")[
            ["delta_brier", "delta_auroc", "positives", "n"]
        ].mean(numeric_only=True)
        composite_rows.append({
            "branch": branch, "condition": condition, "horizon_days": int(horizon),
            "mechanisms": int(len(means)),
            "delta_brier_equal_weight": float(means["delta_brier"].mean()),
            "delta_auroc_equal_weight": float(means["delta_auroc"].mean()),
            "positives_mean": float(means["positives"].mean()),
            "note": "三种机制等权平均；压力测试与噪声族不进入综合指标",
        })
    composite = pd.DataFrame(composite_rows)
    composite.to_csv(run_dir / "composite_equal_weight.csv", index=False, encoding="utf-8")

    # 训练增强贡献
    augmentation_rows = []
    augmentation_path = run_dir / "training_augmentation.csv"
    full = paired.loc[paired["ablation"] == "full"]
    for (branch, horizon, model, scenario), group in ([] if augmentation_path.is_file() else full.groupby(
        ["branch", "horizon_days", "model", "scenario"]
    )):
        pivot = group.pivot_table(index=["key", "label"], columns="condition",
                                  values="prediction", aggfunc="first").reset_index()
        if not {"R-only", "R+S"} <= set(pivot.columns):
            continue
        pivot = pivot.dropna(subset=["R-only", "R+S", "label"])
        cluster = pivot["key"].str.split("|").str[0] if branch == "R" else pivot["key"]
        delta = _bootstrap_delta(
            pivot["label"].to_numpy(), pivot["R-only"].to_numpy(), pivot["R+S"].to_numpy(),
            np.asarray(cluster, dtype=str), samples=max(200, args.bootstrap // 4),
            seed=SEED + int(horizon),
        )
        augmentation_rows.append({
            "branch": branch, "horizon_days": int(horizon), "model": model,
            "scenario": scenario, "n": int(len(pivot)),
            "positives": int(pivot["label"].sum()), **delta,
        })
    augmentation = (
        pd.read_csv(augmentation_path) if augmentation_path.is_file()
        else pd.DataFrame(augmentation_rows)
    )
    if not augmentation_path.is_file():
        augmentation.to_csv(augmentation_path, index=False, encoding="utf-8")

    # 组件消融（相对 M2 的增量）
    ablation_rows = []
    ablation_path = run_dir / "ablation.csv"
    base_rows = paired.loc[(paired["ablation"] == "full") & (paired["model"] == "M2")].copy()
    if ablation_path.is_file():
        ablation = pd.read_csv(ablation_path)
        if set(ablation["variant"].astype(str)) != set(ABLATIONS):
            ablation = pd.DataFrame()
    else:
        ablation = pd.DataFrame()
    if not len(ablation):
        for (branch, condition, horizon, scenario), variant_group in paired.loc[
            paired["ablation"] != "full"
        ].groupby(["branch", "condition", "horizon_days", "scenario"]):
            base = base_rows.loc[
                (base_rows["branch"] == branch)
                & (base_rows["condition"] == condition)
                & (base_rows["horizon_days"] == horizon)
                & (base_rows["scenario"] == scenario)
            ]
            for variant, group in variant_group.groupby("ablation"):
                model = f"M3_{variant}"
                model_rows = group.loc[group["model"] == model]
                if not len(model_rows) or not len(base):
                    continue
                subset = pd.concat([base, model_rows], ignore_index=True, sort=False)
                cluster = "participantid" if branch == "R" else "key"
                if cluster == "participantid":
                    subset = subset.assign(participantid=subset["key"].str.split("|").str[0])
                delta = paired_delta(subset, "M2", model, cluster=cluster, seed=SEED + int(horizon))
                ablation_rows.append({
                    "branch": branch, "condition": condition, "horizon_days": int(horizon),
                    "variant": variant, "scenario": scenario,
                    "n": int(subset.loc[subset["model"] == "M2", "key"].nunique()),
                    "positives": int(subset.loc[subset["model"] == "M2", "label"].sum()),
                    **delta,
                })
            # 完整 DS 也按同一路径重算，保证与其它变体同口径
            full_rows = paired.loc[
                (paired["ablation"] == "full") & (paired["model"] == "M3")
                & (paired["branch"] == branch) & (paired["condition"] == condition)
                & (paired["horizon_days"] == horizon) & (paired["scenario"] == scenario)
            ]
            if len(full_rows) and len(base):
                subset = pd.concat([base, full_rows], ignore_index=True, sort=False)
                cluster = "participantid" if branch == "R" else "key"
                if cluster == "participantid":
                    subset = subset.assign(participantid=subset["key"].str.split("|").str[0])
                delta = paired_delta(subset, "M2", "M3", cluster=cluster, seed=SEED + int(horizon))
                ablation_rows.append({
                    "branch": branch, "condition": condition, "horizon_days": int(horizon),
                    "variant": "full", "scenario": scenario,
                    "n": int(subset.loc[subset["model"] == "M2", "key"].nunique()),
                    "positives": int(subset.loc[subset["model"] == "M2", "label"].sum()),
                    **delta,
                })
        ablation = pd.DataFrame(ablation_rows)
        ablation.to_csv(ablation_path, index=False, encoding="utf-8")

    # 误报 / 检出 / 提前量（S 分支，误报率对齐）
    detection_path = stage12 / "synthetic_detection_daily.csv"
    detection_rows = []
    joint_tests: dict[str, Any] = {}
    if detection_path.is_file():
        det = pd.read_csv(detection_path, low_memory=False)
        det["fold"] = det["fold"].astype(int)
        det["ds_threshold"] = det["fold"].map(
            lambda fold: thresholds["per_fold"][str(int(fold))]["methods"]["ds_pcd"]["threshold"]
        )
        det["rms_threshold"] = det["fold"].map(
            lambda fold: thresholds["per_fold"][str(int(fold))]["methods"]["rms"]["threshold"]
        )
        det["ds_alarm"] = det["C"] >= det["ds_threshold"]
        det["rms_alarm"] = det["rms"] >= det["rms_threshold"]
        det = det.sort_values(["series_id", "day"], kind="stable").reset_index(drop=True)
        det["day_offset"] = det.groupby("series_id").cumcount() + 1
        stable = det.loc[det["trajectory"] == "stable"]
        for method, column in (("ds_pcd", "ds_alarm"), ("rms", "rms_alarm")):
            per_series = stable.groupby("series_id")[column].any()
            detection_rows.append({
                "method": method, "subset": "stable_null", "n": int(per_series.size),
                "false_alarm_30d": float(per_series.mean()),
                "false_alarm_ci": [
                    float(per_series.mean() - 1.96 * math.sqrt(
                        per_series.mean() * (1 - per_series.mean()) / max(per_series.size, 1))),
                    float(per_series.mean() + 1.96 * math.sqrt(
                        per_series.mean() * (1 - per_series.mean()) / max(per_series.size, 1))),
                ],
            })
        changed = det.loc[det["trajectory"].isin(("slow_linear", "rapid_step"))]
        onset = changed.groupby("series_id")["onset_offset"].first()
        for method, column in (("ds_pcd", "ds_alarm"), ("rms", "rms_alarm")):
            first = changed.loc[changed[column]].groupby("series_id")["day_offset"].min()
            for window in (1, 3, 7):
                within, late, never, latencies = 0, 0, 0, []
                for series_id, start in onset.items():
                    if start is None:
                        continue
                    if series_id not in first.index:
                        never += 1
                        continue
                    day = int(first.loc[series_id])
                    if day < start:
                        never += 1  # 只在变化前报警，不计为检出
                        continue
                    latencies.append(day - start)
                    if day - start <= window:
                        within += 1
                    else:
                        late += 1
                total = max(len(onset), 1)
                detection_rows.append({
                    "method": method, "subset": "known_change",
                    "window_days": window,
                    "n_change_series": int(len(onset)),
                    "detected_within_window": within / total,
                    "detected_late": late / total,
                    "undetected_share": never / total,
                    "median_latency_days": float(np.median(latencies)) if latencies else None,
                })
    detection_table = pd.DataFrame(detection_rows)
    detection_table.to_csv(run_dir / "detection_table.csv", index=False, encoding="utf-8")

    # T/C 联合检验（真实与半合成，人员/序列聚类）
    for branch, condition in (("R", "R-only"), ("R", "R+S"), ("S", "S-only")):
        subset = synthetic if branch == "S" else real
        if condition == "S-only":
            subset = synthetic.loc[synthetic["batch"] == "test"]
        fold = int(folds[0])
        mean, sd = age_params[fold]
        train_real = real.loc[real["fold"] != fold]
        train_synthetic = synthetic.loc[(synthetic["fold"] == fold) & (synthetic["batch"] == "train")]
        test = synthetic.loc[(synthetic["fold"] == fold) & (synthetic["batch"] == "test")] \
            if branch == "S" else real.loc[real["fold"] == fold]
        features = LADDERS["M3"]
        fit, _, _ = _fit_and_predict(
            train_real=train_real, train_synthetic=train_synthetic, test=test,
            features=features, condition=condition, fold=fold,
            bundle=f"joint-{branch}-{condition}", age_mean=mean, age_sd=sd,
        )
        test = test.copy()
        test["age_z"] = (test["age"].astype(float) - mean) / sd
        test = test.loc[np.isfinite(test[list(features)].to_numpy(dtype=float)).all(axis=1)]
        clusters = (
            test["participantid"].astype(str) if branch == "R" else test["series_id"].astype(str)
        )
        joint_tests[f"{branch}|{condition}|fold{fold}"] = joint_test_coefficients(
            fit,
            test["duration"].to_numpy(dtype=float),
            test["event"].to_numpy(dtype=int),
            test[list(features)].to_numpy(dtype=float),
            clusters.to_numpy(),
            ["T", "C"],
        )
    _write_json(run_dir / "joint_tests.json", joint_tests)

    # 结论
    conclusions = {"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for branch in ("S", "R"):
        subset = comparison.loc[(comparison["branch"] == branch) & (comparison["horizon_days"] == 7)]
        if not len(subset):
            continue
        conclusions[branch] = {
            "by_condition_scenario": subset[[
                "condition", "scenario", "n", "positives", "delta_brier",
                "delta_brier_ci", "delta_auroc", "delta_auroc_ci", "state",
            ]].to_dict("records"),
        }
    composite_7d = composite.loc[composite["horizon_days"] == 7] if len(composite) else pd.DataFrame()
    conclusions["composite_7d"] = composite_7d.to_dict("records")
    _write_json(run_dir / "conclusions.json", conclusions)
    _write_json(run_dir / "ablation_fit_status.json", {"models": ablation_records})
    _write_json(run_dir / "counts.json", {
        "paired_rows": int(len(paired)),
        "keys": {
            "real": int(real["key"].nunique()),
            "synthetic_test": int(synthetic.loc[synthetic["batch"] == "test", "key"].nunique()),
            "synthetic_train": int(synthetic.loc[synthetic["batch"] == "train", "key"].nunique()),
        },
        "comparisons": int(len(comparison)),
        "bootstrap": args.bootstrap,
    })

    snapshot = run_dir / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for relative in CODE_SNAPSHOT_FILES:
        source = PROJECT_ROOT / relative
        if source.is_file():
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    manifest = {
        "stage": 14,
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "upstream": {
            "protocol": str(stage11 / "protocol.json"),
            "protocol_md5": _md5(stage11 / "protocol.json"),
            "stage12": str(stage12),
            "stage13": str(stage13),
            "freeze_manifest": str(stage13 / "freeze_manifest.json"),
            "freeze_md5": _md5(stage13 / "freeze_manifest.json"),
            "variant_scores": str(run_dir / "variant_scores.csv"),
        },
        "outputs": sorted(str(path.relative_to(run_dir)) for path in run_dir.rglob("*") if path.is_file()),
        "seed": SEED,
        "bootstrap": args.bootstrap,
    }
    _write_json(run_dir / "manifest.json", manifest)
    print(f"[stage14] 完成：{run_dir}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
