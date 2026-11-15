"""阶段 6 核验：独立复算指标/区间、折与标签、缺失规则、二次重放。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import md5
import json
from pathlib import Path
import sys
from typing import Any, Sequence
from zipfile import ZipFile

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.survival import (  # noqa: E402
    CoxModelArtifact,
    predict_cumulative_risk,
)

OUT_ROOT = PROJECT_ROOT.parents[1] / "output" / "ds_pcd_mobilise"
RUN = OUT_ROOT / "20261114-fixed"
OLD = OUT_ROOT / "20261114-main"
CACHE = PROJECT_ROOT.parents[1] / "anti-fall" / "_datasets" / "Mobilise-D-CVS" / "21060991"
TOLERANCE = 1e-9


def _md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8",
    )


def pairwise_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return None
    greater = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(positive) * len(negative)))


def brier(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((np.asarray(labels, float) - np.asarray(probabilities, float)) ** 2))


def logistic_slope(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    y = np.asarray(labels, float)
    p = np.clip(np.asarray(probabilities, float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    if y.sum() < 5 or (1 - y).sum() < 5:
        return None
    design = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(100):
        eta = design @ beta
        mu = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        w = np.clip(mu * (1 - mu), 1e-8, None)
        gradient = design.T @ (y - mu) - 1e-6 * beta
        hessian = -(design.T * w) @ design - 1e-6 * np.eye(2)
        step = np.linalg.solve(-hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(beta[1]) if np.all(np.isfinite(beta)) and abs(beta[1]) <= 20 else None


def paired_rows(predictions: pd.DataFrame, variant: str, horizon: int) -> pd.DataFrame:
    subset = predictions.loc[
        predictions["variant"].eq(variant) & predictions["horizon"].eq(horizon)
    ]
    pivot = subset.pivot_table(
        index=["participantid", "landmark_day"], columns="method",
        values="risk", aggfunc="first",
    )
    meta = subset.drop_duplicates(["participantid", "landmark_day"]).set_index(
        ["participantid", "landmark_day"]
    )[["event"]]
    pivot = pivot.join(meta, how="inner").dropna(subset=["A_rms", "B_rms_cusum"])
    return pivot


def independent_metrics(pivot: pd.DataFrame) -> dict[str, Any]:
    labels = pivot["event"].to_numpy(dtype=int)
    a = pivot["A_rms"].to_numpy(dtype=float)
    b = pivot["B_rms_cusum"].to_numpy(dtype=float)
    return {
        "n": int(len(pivot)), "events": int(labels.sum()),
        "persons": int(pivot.index.get_level_values("participantid").nunique()),
        "auroc_A": pairwise_auc(labels, a), "auroc_B": pairwise_auc(labels, b),
        "brier_A": brier(labels, a), "brier_B": brier(labels, b),
        "observed": float(labels.mean()),
        "mean_A": float(a.mean()), "mean_B": float(b.mean()),
        "slope_A": logistic_slope(labels, a), "slope_B": logistic_slope(labels, b),
        "delta_auroc": pairwise_auc(labels, b) - pairwise_auc(labels, a),
        "delta_brier": brier(labels, b) - brier(labels, a),
    }


def independent_bootstrap(pivot: pd.DataFrame, replicates: int, seed: int) -> dict[str, Any]:
    people = np.asarray(sorted(pivot.index.get_level_values("participantid").unique()))
    person_rows = {person: pivot.xs(person, level="participantid") for person in people}
    rng = np.random.default_rng(seed)
    auc = {"A_rms": [], "B_rms_cusum": []}
    br = {"A_rms": [], "B_rms_cusum": []}
    slope = {"A_rms": [], "B_rms_cusum": []}
    dal, dbr = [], []
    invalid = 0
    for _ in range(replicates):
        sample = rng.choice(people, size=len(people), replace=True)
        block = pd.concat([person_rows[person] for person in sample], ignore_index=False)
        labels = block["event"].to_numpy(dtype=int)
        if labels.sum() == 0 or labels.sum() == len(labels):
            invalid += 1
            continue
        values = {}
        for method in ("A_rms", "B_rms_cusum"):
            probabilities = block[method].to_numpy(dtype=float)
            value = pairwise_auc(labels, probabilities)
            if value is None:
                values = {}
                break
            values[method] = (value, brier(labels, probabilities))
            auc[method].append(value)
            br[method].append(brier(labels, probabilities))
            current = logistic_slope(labels, probabilities)
            if current is not None:
                slope[method].append(current)
        if len(values) == 2:
            dal.append(values["B_rms_cusum"][0] - values["A_rms"][0])
            dbr.append(values["B_rms_cusum"][1] - values["A_rms"][1])

    def interval(values):
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if len(values) >= 50 else None

    return {
        "replicates_requested": replicates,
        "replicates_invalid": invalid,
        "replicates_valid": replicates - invalid,
        "ci": {
            "auroc": {method: interval(values) for method, values in auc.items()},
            "brier": {method: interval(values) for method, values in br.items()},
            "slope": {method: interval(values) for method, values in slope.items()},
            "delta_auroc": interval(dal),
            "delta_brier": interval(dbr),
        },
    }


def check_labels(predictions: pd.DataFrame, windows: pd.DataFrame) -> dict[str, Any]:
    with ZipFile(CACHE / "Clinical Data.zip") as archive:
        name = next(n for n in archive.namelist() if n.endswith("-falls-reports.csv"))
        with archive.open(name) as handle:
            reports = pd.read_csv(handle, dtype={"participantid": str})
    reports["day"] = pd.to_datetime(reports["falldate"], errors="coerce").dt.normalize()
    report_days = reports.groupby(["participantid", "visit-number"]).day.agg(list)
    labels = predictions[["participantid", "landmark_day", "horizon", "event", "duration"]].drop_duplicates()
    labels["landmark_day"] = pd.to_datetime(labels["landmark_day"])
    window_subset = windows[[
        "participantid", "landmark_day", "horizon", "covered_from",
        "diary_day", "diary_visit_number",
    ]]
    matched = labels.merge(
        window_subset, on=["participantid", "landmark_day", "horizon"],
        how="left", validate="one_to_one",
    )
    if matched["covered_from"].isna().any():
        raise AssertionError("存在无法匹配日记窗口的预测日")
    mismatches = 0
    for row in matched.itertuples():
        t = pd.Timestamp(row.landmark_day)
        end = t + pd.Timedelta(days=int(row.horizon))
        if not (pd.Timestamp(row.covered_from) <= t and end <= pd.Timestamp(row.diary_day)):
            mismatches += 1
            continue
        days = report_days.get((row.participantid, row.diary_visit_number), [])
        events = [day for day in days if pd.notna(day) and t < day <= end]
        event = int(bool(events))
        duration = (min(events) - t).days if events else int(row.horizon)
        if event != int(row.event) or duration != int(row.duration):
            mismatches += 1
    return {
        "labels_checked": int(len(matched)),
        "mismatches": mismatches,
        "horizons": {str(h): int((matched["horizon"] == h).sum()) for h in (7, 30)},
        "ok": mismatches == 0,
    }


def check_scores(run: Path) -> dict[str, Any]:
    fixed = pd.read_csv(run / "stage2" / "scores_session_se.csv", dtype={"participantid": str})
    old = pd.read_csv(OLD / "scores.csv", dtype={"participantid": str})
    merged = old.merge(
        fixed, on=["participantid", "landmark_day"], suffixes=("_old", "_fixed"),
    )
    status_changes = int((merged["ds_status_old"] != merged["ds_status_fixed"]).sum())
    rms_diff = float((merged["rms_distance_old"] - merged["rms_distance_fixed"]).abs().max())
    c_diff = int(
        ((merged["cusum_old"] - merged["cusum_fixed"]).abs() > TOLERANCE).sum()
    )
    return {
        "rows": int(len(merged)),
        "rms_max_abs_diff": rms_diff,
        "status_changes": status_changes,
        "c_changes": c_diff,
        "ok": rms_diff <= TOLERANCE,
    }


def check_replay(run: Path) -> dict[str, Any]:
    replay = run / "replay2"
    keys = ["variant", "horizon", "method", "participantid", "landmark_day"]
    main = pd.read_csv(run / "predictions.csv")
    second = pd.read_csv(replay / "stage3" / "predictions.csv")
    merged = main.merge(second, on=keys, suffixes=("_1", "_2"))
    risk_diff = float((merged["risk_1"] - merged["risk_2"]).abs().max())
    duration_diff = float((merged["duration_1"] - merged["duration_2"]).abs().max())
    return {
        "main_rows": int(len(main)), "replay_rows": int(len(second)),
        "matched_rows": int(len(merged)),
        "risk_max_abs_diff": risk_diff,
        "duration_max_abs_diff": duration_diff,
        "predictions_md5_same": _md5(run / "predictions.csv") == _md5(replay / "stage3" / "predictions.csv"),
        "metrics_md5_same": _md5(run / "metrics.csv") == _md5(replay / "stage3" / "metrics.csv"),
        "stage2_scores_md5_same": _md5(run / "stage2" / "scores_session_se.csv")
        == _md5(replay / "stage2" / "scores_session_se.csv"),
        "ok": len(main) == len(second) == len(merged)
        and risk_diff <= TOLERANCE and duration_diff <= TOLERANCE
        and _md5(run / "metrics.csv") == _md5(replay / "stage3" / "metrics.csv"),
    }


def check_calibration(pivot: pd.DataFrame, comparison: dict[str, Any]) -> dict[str, Any]:
    metrics = independent_metrics(pivot)
    main = comparison["main_metrics"]
    fields = {
        "auroc": abs(metrics["auroc_A"] - main["auroc_A"]) <= TOLERANCE
        and abs(metrics["auroc_B"] - main["auroc_B"]) <= TOLERANCE,
        "brier": abs(metrics["brier_A"] - main["brier_A"]) <= TOLERANCE
        and abs(metrics["brier_B"] - main["brier_B"]) <= TOLERANCE,
        "delta": abs(metrics["delta_auroc"] - main["delta_auroc"]) <= TOLERANCE
        and abs(metrics["delta_brier"] - main["delta_brier"]) <= TOLERANCE,
        "calibration": abs(metrics["observed"] - main["observed_risk"]) <= TOLERANCE
        and abs(metrics["mean_A"] - main["mean_predicted_A"]) <= TOLERANCE
        and abs(metrics["mean_B"] - main["mean_predicted_B"]) <= TOLERANCE
        and abs(metrics["slope_A"] - main["slope_A"]) <= TOLERANCE
        and abs(metrics["slope_B"] - main["slope_B"]) <= TOLERANCE,
    }
    return {"metrics": metrics, "fields": fields, "ok": all(fields.values())}


def check_models(run: Path) -> dict[str, Any]:
    files = sorted((run / "fold_models").rglob("*.json"))
    flags = []
    for path in files:
        payload = _read_json(path)
        flags.append(bool(payload.get("calibrated")))
    sample = CoxModelArtifact.from_dict(_read_json(files[0]))
    blocked = False
    try:
        predict_cumulative_risk({name: 0.0 for name in sample.feature_names}, sample)
    except ValueError:
        blocked = True
    return {"files": len(files), "all_calibrated_false": not any(flags),
            "service_probability_blocked": blocked,
            "ok": bool(files) and not any(flags) and blocked}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 6 核验")
    parser.add_argument("--run", default=str(RUN))
    parser.add_argument("--old-run", default=str(OLD))
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args(argv)
    run = Path(args.run)
    comparison = _read_json(run / "comparison.json")
    rules = _read_json(run / "analysis_rules.json")
    predictions = pd.read_csv(run / "predictions.csv", dtype={"participantid": str})
    folds = pd.read_csv(run / "splits.csv", dtype={"participantid": str})
    windows = pd.read_csv(run / "data" / "landmark_windows.csv", dtype={"participantid": str})
    windows["landmark_day"] = pd.to_datetime(windows["landmark_day"]).dt.normalize()
    windows = windows.rename(columns={"horizon_days": "horizon"})

    split = {
        "people": int(len(folds)),
        "folds": {int(k): int(v) for k, v in folds["fold"].value_counts().sort_index().items()},
        "same_as_old": _md5(run / "splits.csv") == _md5(Path(args.old_run) / "splits.csv"),
        "person_unique": not folds["participantid"].duplicated().any(),
        "person_fold_unique": folds.groupby("participantid")["fold"].nunique().max() == 1
        if len(folds) else False,
    }
    split["ok"] = split["same_as_old"] and split["person_unique"] and split["person_fold_unique"]

    fold_map = dict(zip(folds["participantid"], folds["fold"]))
    pred_fold_ok = bool(
        predictions.assign(expected=predictions["participantid"].map(fold_map))["fold"]
        .eq(predictions.assign(expected=predictions["participantid"].map(fold_map))["expected"])
        .all()
    )
    predictions_frame = predictions.copy()
    predictions_frame["landmark_day"] = predictions_frame["landmark_day"].astype(str)
    labels = check_labels(
        predictions_frame[["participantid", "landmark_day", "horizon", "event", "duration"]],
        windows,
    )

    primary = paired_rows(predictions, "fixed_session", 30)
    none = paired_rows(predictions, "fixed_none", 30)
    old_common = paired_rows(predictions, "old_on_common", 30)
    old_all = paired_rows(predictions, "old_all", 30)
    calibration = check_calibration(primary, comparison)
    bootstrap = independent_bootstrap(primary, args.replicates, 20261113)
    ci_match = True
    for method in ("A_rms", "B_rms_cusum"):
        for metric in ("auroc", "brier", "calibration_slope"):
            stored = comparison["main_metrics"]
            key = {"auroc": "ci_auroc", "brier": "ci_brier", "calibration_slope": "ci_slope"}[metric]
            stored_ci = stored[f"{key}_{'A' if method == 'A_rms' else 'B'}"]
            computed = bootstrap["ci"]["auroc" if metric == "auroc" else "brier" if metric == "brier" else "slope"][method]
            if computed is None or stored_ci is None:
                ci_match = False
                continue
            if abs(computed[0] - stored_ci[0]) > TOLERANCE or abs(computed[1] - stored_ci[1]) > TOLERANCE:
                ci_match = False
    delta_ci_match = True
    for key, stored in (("delta_auroc", "ci_delta_auroc"), ("delta_brier", "ci_delta_brier")):
        computed = bootstrap["ci"][key]
        expected = comparison["main_metrics"][stored]
        if computed is None or expected is None:
            delta_ci_match = False
        elif abs(computed[0] - expected[0]) > TOLERANCE or abs(computed[1] - expected[1]) > TOLERANCE:
            delta_ci_match = False

    old_metrics_ok = True
    for pivot, key in ((old_common, "old_on_common"), (old_all, "old_all")):
        values = independent_metrics(pivot)
        stored = comparison["old_comparison"][key]
        old_metrics_ok &= (
            abs(values["auroc_A"] - stored["auroc_A"]) <= TOLERANCE
            and abs(values["auroc_B"] - stored["auroc_B"]) <= TOLERANCE
            and abs(values["brier_A"] - stored["brier_A"]) <= TOLERANCE
            and abs(values["delta_auroc"] - stored["delta_auroc"]) <= TOLERANCE
            and abs(values["delta_brier"] - stored["delta_brier"]) <= TOLERANCE
        )

    selection = _read_json(run / "selection.json")
    missing_rules = {
        "session_rows": int(len(primary)), "session_events": int(primary["event"].sum()),
        "session_people": int(primary.index.get_level_values("participantid").nunique()),
        "none_rows": int(len(none)), "none_events": int(none["event"].sum()),
        "unavailable_rows": int(selection["session_unavailable_landmarks"]),
        "unavailable_people": int(selection["session_unavailable_people"]),
        "unavailable_events": int(selection["session_unavailable_30d_positives"]),
        "ok": (
            len(primary) == 172 and int(primary["event"].sum()) == 16
            and len(none) == 186 and int(none["event"].sum()) == 17
            and selection["session_unavailable_landmarks"] == 14
        ),
    }
    none_check = check_calibration(none, {"main_metrics": comparison["sensitivity_metrics"]})
    none_bootstrap = independent_bootstrap(none, args.replicates, 20261113)
    none_ci = True
    for method in ("A_rms", "B_rms_cusum"):
        key = {"A_rms": "A", "B_rms_cusum": "B"}[method]
        for metric, stored_key, computed_key in (
            ("auroc", f"ci_auroc_{key}", "auroc"),
            ("brier", f"ci_brier_{key}", "brier"),
            ("slope", f"ci_slope_{key}", "slope"),
        ):
            computed = none_bootstrap["ci"][computed_key][method]
            stored = comparison["sensitivity_metrics"][stored_key]
            if computed is None or stored is None:
                none_ci = False
            elif abs(computed[0] - stored[0]) > TOLERANCE or abs(computed[1] - stored[1]) > TOLERANCE:
                none_ci = False
    for key, stored in (("delta_auroc", "ci_delta_auroc"), ("delta_brier", "ci_delta_brier")):
        computed = none_bootstrap["ci"][key]
        expected = comparison["sensitivity_metrics"][stored]
        if computed is None or expected is None:
            none_ci = False
        elif abs(computed[0] - expected[0]) > TOLERANCE or abs(computed[1] - expected[1]) > TOLERANCE:
            none_ci = False
    sensitivity_check = {"ok": none_check["ok"] and none_ci,
                         "metrics": none_check["metrics"], "ci_match": none_ci}
    scores = check_scores(run)
    replay = check_replay(run)
    models = check_models(run)
    checks = {
        "split": split["ok"],
        "prediction_folds": pred_fold_ok,
        "labels": labels["ok"],
        "metrics_recomputed": calibration["ok"] and ci_match and delta_ci_match,
        "sensitivity_recomputed": sensitivity_check["ok"],
        "old_comparison": old_metrics_ok,
        "missing_rules": missing_rules["ok"],
        "scores_control": scores["ok"],
        "second_replay": replay["ok"],
        "calibration_gate": models["ok"],
    }
    payload = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": str(run),
        "split": split,
        "prediction_folds_ok": pred_fold_ok,
        "labels": labels,
        "independent_metrics": calibration,
        "sensitivity": sensitivity_check,
        "bootstrap": bootstrap,
        "ci_match": ci_match and delta_ci_match,
        "old_comparison_ok": old_metrics_ok,
        "missing_rules": missing_rules,
        "scores": scores,
        "second_replay": replay,
        "models": models,
        "checks": checks,
        "all_pass": all(checks.values()),
    }
    _write_json(run / "verification.json", payload)
    lines = [
        "# 阶段 6 核验记录",
        "",
        f"- 时间：{payload['verified_at_utc']}",
        f"- 运行：`{run}`",
        "",
        "## 结构",
        "",
        f"- 人员分折与旧版一致：{'是' if split['same_as_old'] else '否'}；每折人数 {split['folds']}；"
        f"同一人只属一折：{'是' if split['person_fold_unique'] else '否'}。",
        f"- 预测折号与折文件一致：{'是' if pred_fold_ok else '否'}。",
        f"- 首次跌倒日期与窗口核对：{labels['labels_checked']} 行，mismatch {labels['mismatches']}。",
        "",
        "## 指标复算",
        "",
        f"- 主表 AUROC/Brier/配对差与校准：{'一致' if calibration['ok'] else '不一致'}。",
        f"- 敏感性 none 表与区间：{'一致' if sensitivity_check['ok'] else '不一致'}。",
        f"- 按人聚类区间：{'一致' if ci_match and delta_ci_match else '不一致'}。",
        f"- 旧版共同 172 与全部 186 的对照：{'一致' if old_metrics_ok else '不一致'}。",
        "",
        "## 缺失规则与二次重放",
        "",
        f"- 主分析 {missing_rules['session_rows']} 行 / {missing_rules['session_events']} 个阳性 / "
        f"{missing_rules['session_people']} 人；none {missing_rules['none_rows']} 行 / "
        f"{missing_rules['none_events']} 个阳性；不可评分 {missing_rules['unavailable_rows']} 行 / "
        f"{missing_rules['unavailable_people']} 人（阳性 {missing_rules['unavailable_events']}）。",
        f"- 二次重放：预测 {replay['matched_rows']} 行匹配，风险最大差 {replay['risk_max_abs_diff']:.2e}，"
        f"metrics 哈希一致：{'是' if replay['metrics_md5_same'] else '否'}。",
        f"- 折模型 {models['files']} 个全部 calibrated=false："
        f"{'是' if models['all_calibrated_false'] else '否'}；服务概率门槛拒绝："
        f"{'是' if models['service_probability_blocked'] else '否'}。",
        "",
        f"**all_pass = {payload['all_pass']}**",
        "",
    ]
    (run / "VERIFICATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(_jsonable({k: payload[k] for k in ("checks", "all_pass")}), ensure_ascii=False, indent=1))
    return 0 if payload["all_pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
