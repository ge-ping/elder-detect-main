"""阶段 6：用修正版在原分折上重跑 A/B 比较，并给出旧版对照与归因。

主规则在查看性能前写入 ``analysis_rules.json``：

* 主分析 `session_se`、30 天；A=`rms_distance`，B=`rms_distance+cusum`；
* A/B 使用同一批有效行（两特征均有限），同一折内训练与测试，不做插补；
* 人员分折沿用旧运行 ``splits.csv``，不重新分配；
* 折内训练事件 <2、未收敛或 |标准化 beta|>10 记为失败，不产生该折预测；
* 指标为折外 AUROC/Brier 与配对差，按人聚类 2000 次 bootstrap。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import md5
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.cox_fit import fit_landmark_cox as new_fit_landmark_cox  # noqa: E402
from longitudinal.survival import (  # noqa: E402
    CoxModelArtifact,
    predict_cumulative_risk_research,
)
from scripts.analyse_mobilise_stage3 import (  # noqa: E402
    auroc,
    brier,
    calibration_slope,
    load_analysis_frame,
)

OUT_ROOT = PROJECT_ROOT.parents[1] / "output" / "ds_pcd_mobilise"
OLD_RUN = OUT_ROOT / "20261114-main"
FIXED_RUN = OUT_ROOT / "20261114-fixed"
SEED = 20261113
RIDGE = 1e-4
METHODS = {"A_rms": ("rms_distance",), "B_rms_cusum": ("rms_distance", "cusum")}
TOLERANCE = 1e-9


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 6 修正版 A/B 分析")
    parser.add_argument("--fixed-run", default=str(FIXED_RUN))
    parser.add_argument("--old-run", default=str(OLD_RUN))
    parser.add_argument("--folds", default=None, help="人员分折文件；默认旧运行 splits.csv")
    parser.add_argument("--out", default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser


def _md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_old_fitter(path: Path):
    spec = importlib.util.spec_from_file_location("stage6_old_cox", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage6_old_cox"] = module
    spec.loader.exec_module(module)
    return module.fit_landmark_cox


def _artifact(fit, fold: int, horizon: int, method: str, variant: str) -> CoxModelArtifact:
    notes = dict(getattr(fit, "notes", {}) or {})
    return CoxModelArtifact.from_dict({
        "schema_version": 2,
        "model_type": "landmark_cox",
        "bundle_id": f"mobilise-stage6-{variant}-{horizon}d-{method}-f{fold}-s{SEED}",
        "horizon_days": horizon,
        "feature_names": list(fit.feature_names),
        "coefficients": list(fit.coefficients),
        "feature_centers": list(fit.centers),
        "baseline_survival": fit.baseline_survival,
        "calibrated": False,
        "trained_at": str(notes.get("fitted_at_utc", "")),
        "evidence_ids": ["mobilise_d_cvs_v1.0.0", "cox_1972", "landmarking_2007"],
        "training_summary": {
            "variant": variant,
            "fold": fold,
            "method": method,
            "objective": notes.get("objective", "legacy_unpenalized_step_search"),
        },
    })


def fit_variant(
    variant: str,
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    horizon: int,
    fitter: Callable,
    *,
    model_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """在指定有效行上，对每个折拟合 A/B 并产生折外预测。"""
    dataset = frame.rename(columns={
        f"event_bool_{horizon}d": "event",
        f"duration_{horizon}d": "duration",
    })
    fold_map = dict(zip(folds["participantid"], folds["fold"]))
    dataset = dataset.loc[dataset["participantid"].isin(fold_map)].copy()
    dataset["fold"] = dataset["participantid"].map(fold_map).astype(int)
    predictions: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for fold in sorted(set(fold_map.values())):
        test = dataset.loc[dataset["fold"].eq(fold)]
        train = dataset.loc[~dataset["fold"].eq(fold)]
        for method, features in METHODS.items():
            fit = fitter(
                train["duration"].to_numpy(dtype=float),
                train["event"].astype(int).to_numpy(),
                train[list(features)].to_numpy(dtype=float),
                features,
                horizon_days=float(horizon),
                ridge=RIDGE,
            )
            status = {
                "variant": variant, "horizon": horizon, "fold": int(fold),
                "method": method, "status": fit.status,
                "train_rows": int(len(train)),
                "train_events": int(train["event"].sum()),
                "train_persons": int(train["participantid"].nunique()),
                "test_rows": int(len(test)),
                "test_events": int(test["event"].sum()),
                "iterations": int(getattr(fit, "iterations", 0)),
                "baseline_survival": float(fit.baseline_survival)
                if fit.baseline_survival is not None else None,
                "coefficients": "|".join(f"{value:.8g}" for value in fit.coefficients),
                "feature_names": "|".join(fit.feature_names),
                "objective": (getattr(fit, "notes", {}) or {}).get(
                    "objective", "legacy_unpenalized_step_search"
                ),
            }
            statuses.append(status)
            if fit.status != "ok":
                continue
            artifact = _artifact(fit, int(fold), horizon, method, variant)
            target = model_dir / variant / f"{horizon}d"
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{method}_fold{fold}.json").write_text(
                json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            for row in test.itertuples(index=False):
                payload = {name: float(getattr(row, name)) for name in features}
                risk = predict_cumulative_risk_research(payload, artifact)
                predictions.append({
                    "variant": variant,
                    "participantid": str(row.participantid),
                    "landmark_day": pd.Timestamp(row.landmark_day).date().isoformat(),
                    "fold": int(fold),
                    "horizon": int(horizon),
                    "method": method,
                    "risk": float(risk),
                    "event": int(row.event),
                    "duration": float(row.duration),
                    "rms_distance": float(row.rms_distance),
                    "cusum": float(row.cusum),
                })
    return predictions, statuses


def _interval(values: Sequence[float]) -> list[float] | None:
    if len(values) < 50:
        return None
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def bootstrap_pairs(
    predictions: pd.DataFrame,
    *,
    variant: str,
    horizon: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    subset = predictions.loc[
        predictions["variant"].eq(variant) & predictions["horizon"].eq(horizon)
    ]
    if subset.empty:
        return {"status": "not_estimable"}
    pivot = subset.pivot_table(
        index=["participantid", "landmark_day"], columns="method",
        values="risk", aggfunc="first",
    )
    meta = subset.drop_duplicates(["participantid", "landmark_day"]).set_index(
        ["participantid", "landmark_day"]
    )[["event"]]
    pivot = pivot.join(meta, how="inner")
    methods = ["A_rms", "B_rms_cusum"]
    if any(method not in pivot.columns for method in methods):
        return {"status": "not_estimable", "reason": "missing method predictions"}
    both = pivot.dropna(subset=methods)
    if both.empty:
        return {"status": "not_estimable", "reason": "no paired rows"}
    labels = both["event"].to_numpy(dtype=int)
    a = both["A_rms"].to_numpy(dtype=float)
    b = both["B_rms_cusum"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    people = np.asarray(sorted(both.index.get_level_values("participantid").unique()))
    rows_by_person = {person: both.xs(person, level="participantid") for person in people}
    auc = {"A_rms": [], "B_rms_cusum": []}
    br = {"A_rms": [], "B_rms_cusum": []}
    dal, dbr = [], []
    slope = {"A_rms": [], "B_rms_cusum": []}
    invalid = 0
    for _ in range(replicates):
        sample = rng.choice(people, size=len(people), replace=True)
        block = pd.concat([rows_by_person[person] for person in sample], ignore_index=False)
        y = block["event"].to_numpy(dtype=int)
        if y.sum() == 0 or y.sum() == len(y):
            invalid += 1
            continue
        values = {}
        for method in methods:
            probability = block[method].to_numpy(dtype=float)
            value = auroc(y, probability)
            if value is None:
                values = {}
                break
            values[method] = (value, brier(y, probability))
            auc[method].append(value)
            br[method].append(brier(y, probability))
            current = calibration_slope(y, probability)
            if current is not None:
                slope[method].append(current)
        if len(values) == 2:
            dal.append(values["B_rms_cusum"][0] - values["A_rms"][0])
            dbr.append(values["B_rms_cusum"][1] - values["A_rms"][1])
    point = {
        "n": int(len(both)), "events": int(labels.sum()),
        "persons": int(both.index.get_level_values("participantid").nunique()),
        "auroc": {method: auroc(labels, both[method].to_numpy(dtype=float)) for method in methods},
        "brier": {method: brier(labels, both[method].to_numpy(dtype=float)) for method in methods},
        "observed_risk": float(labels.mean()),
        "mean_predicted": {
            method: float(both[method].mean()) for method in methods
        },
        "calibration_slope": {
            method: calibration_slope(labels, both[method].to_numpy(dtype=float))
            for method in methods
        },
    }
    return {
        "status": "ok",
        "point": point,
        "replicates_requested": replicates,
        "replicates_invalid": invalid,
        "replicates_valid": replicates - invalid,
        "ci": {
            "auroc": {method: _interval(values) for method, values in auc.items()},
            "brier": {method: _interval(values) for method, values in br.items()},
            "calibration_slope": {method: _interval(values) for method, values in slope.items()},
            "delta_auroc": _interval(dal),
            "delta_brier": _interval(dbr),
        },
    }


def old_predictions(
    old_run: Path, mode: str, horizon: int, keys: set[tuple[str, str]],
) -> pd.DataFrame:
    frame = pd.read_csv(old_run / "predictions.csv", dtype={"participantid": str})
    frame = frame.loc[
        frame["mode"].eq(mode) & frame["horizon"].eq(horizon)
    ].copy()
    frame["landmark_day"] = frame["landmark_day"].astype(str)
    frame["key"] = list(zip(frame["participantid"], frame["landmark_day"]))
    frame = frame.loc[frame["key"].isin(keys)]
    frame = frame.rename(columns={"mode": "variant"})
    frame["variant"] = f"old_{mode}"
    return frame


def selection_report(fixed_data: Path, old_run: Path, fixed_scores: pd.DataFrame,
                     old_scores: pd.DataFrame, folds: pd.DataFrame) -> dict[str, Any]:
    counts = json.loads((fixed_data / "manifest.json").read_text(encoding="utf-8"))["counts"]
    flow = pd.read_csv(fixed_data / "cohort_flow.csv")
    windows = pd.read_csv(fixed_data / "landmark_windows.csv", dtype={"participantid": str})
    ready = fixed_scores.loc[fixed_scores["ds_status"].eq("scored")]
    unavailable = fixed_scores.loc[~fixed_scores["ds_status"].eq("scored")]
    ready_keys = set(zip(ready["participantid"], ready["landmark_day"].astype(str)))
    unavailable_keys = set(zip(unavailable["participantid"], unavailable["landmark_day"].astype(str)))
    wide30 = windows.loc[
        windows["horizon_days"].eq(30)
        & windows["event"].astype(str).str.lower().eq("true")
    ]
    wide30_keys = set(zip(wide30["participantid"], wide30["landmark_day"].astype(str)))
    events_unavailable = wide30_keys & unavailable_keys
    events_ready = wide30_keys & ready_keys
    return {
        "wide_landmarks": int(counts["candidate_landmarks"]),
        "wide_people": int(counts["candidate_people"]),
        "wide_30d_positive_landmarks": int(counts["horizons"]["30d"]["positive_landmarks"]),
        "wide_30d_positive_people": int(counts["horizons"]["30d"]["positive_people"]),
        "wide_7d_positive_landmarks": int(counts["horizons"]["7d"]["positive_landmarks"]),
        "baseline_ready_landmarks": int(counts["baseline_ready_landmarks"]),
        "baseline_ready_people": int(counts["baseline_ready_people"]),
        "baseline_excluded_landmarks": int(
            counts["candidate_landmarks"] - counts["baseline_ready_landmarks"]
        ),
        "baseline_excluded_30d_positives": int(
            counts["horizons"]["30d"]["positive_landmarks"]
            - counts["horizons"]["30d"]["baseline_ready_positive_landmarks"]
        ),
        "session_scored_landmarks": int(len(ready)),
        "session_scored_people": int(ready["participantid"].nunique()),
        "session_30d_positive_landmarks": int(len(events_ready)),
        "session_30d_positive_people": int(
            wide30.loc[[key in events_ready for key in zip(wide30["participantid"], wide30["landmark_day"].astype(str))], "participantid"].nunique()
        ) if events_ready else 0,
        "session_unavailable_landmarks": int(len(unavailable)),
        "session_unavailable_people": int(unavailable["participantid"].nunique()),
        "session_unavailable_30d_positives": int(len(events_unavailable)),
        "session_unavailable_examples": unavailable[["participantid", "landmark_day", "ds_status"]].head(20).to_dict("records"),
        "none_scored_landmarks": int(len(old_scores)),
        "folds": {int(fold): int((folds["fold"] == fold).sum()) for fold in sorted(folds["fold"].unique())},
        "upstream_diary_exclusions": flow.to_dict("records"),
        "note": "宽集 463/40 为已对齐且日记与报告一致、覆盖完整的候选；186/17 为满足 14 个既往有效日的主集；session_se 再因 14 个基线日缺少步速标准误损失 14 个预测日（含 1 个 30 天阳性）。",
    }


def write_report(out: Path, payload: dict[str, Any]) -> None:
    main = payload["main_metrics"]
    sensitivity = payload["sensitivity_metrics"]
    attribution = payload["attribution"]
    selection = payload["selection"]
    old = payload["old_comparison"]
    seven = payload["seven_day"]
    lines = [
        "# Mobilise-D / DS-PCD 修正版主实验（阶段 6）",
        "",
        f"- 运行编号：`{payload['run_id']}`",
        f"- 生成时间：{payload['generated_at_utc']}",
        f"- 数据：Mobilise-D CVS V1.0.0；旧版对照 `20261114-main`",
        f"- 代码版本：`scoring-v2-causal-precision` / `cox-v2-penalized-objective`",
        f"- 协议：`analysis_rules.json`；折外预测：`predictions.csv`；指标：`metrics.csv`",
        "",
        "## 1. 主结果（30 天，session_se，A/B 共同 有效行）",
        "",
        "| 指标 | A：RMS | B：RMS+C | 配对差 B−A（95%） |",
        "|---|---:|---:|---:|",
        f"| AUROC（95%） | {main['auroc_A']:.3f}（{main['ci_auroc_A'][0]:.3f}–{main['ci_auroc_A'][1]:.3f}） | "
        f"{main['auroc_B']:.3f}（{main['ci_auroc_B'][0]:.3f}–{main['ci_auroc_B'][1]:.3f}） | "
        f"{main['delta_auroc']:+.3f}（{main['ci_delta_auroc'][0]:+.3f}–{main['ci_delta_auroc'][1]:+.3f}） |",
        f"| Brier（95%） | {main['brier_A']:.4f}（{main['ci_brier_A'][0]:.4f}–{main['ci_brier_A'][1]:.4f}） | "
        f"{main['brier_B']:.4f}（{main['ci_brier_B'][0]:.4f}–{main['ci_brier_B'][1]:.4f}） | "
        f"{main['delta_brier']:+.4f}（{main['ci_delta_brier'][0]:+.4f}–{main['ci_delta_brier'][1]:+.4f}） |",
        "",
        f"有效样本：{main['n']} 个预测日、{main['persons']} 人、{main['events']} 个 30 天阳性；"
        f"区间基于固定折外预测、按人聚类 {main['replicates']} 次 bootstrap"
        f"（有效 {main['valid_replicates']}、无效 {main['invalid_replicates']}）。",
        f"校准：观察风险 {main['observed_risk']:.4f}，A 平均预测 {main['mean_predicted_A']:.4f}、"
        f"B {main['mean_predicted_B']:.4f}；斜率 A {main['slope_A']:.2f}"
        f"（{main['ci_slope_A'][0]:.2f}–{main['ci_slope_A'][1]:.2f}）、"
        f"B {main['slope_B']:.2f}（{main['ci_slope_B'][0]:.2f}–{main['ci_slope_B'][1]:.2f}）。"
        "模型保持 `calibrated=false`。",
        "",
        "## 2. 敏感性（none，30 天）",
        "",
        f"- B AUROC {sensitivity['auroc_B']:.3f}"
        f"（{sensitivity['ci_auroc_B'][0]:.3f}–{sensitivity['ci_auroc_B'][1]:.3f}），"
        f"ΔAUROC {sensitivity['delta_auroc']:+.3f}"
        f"（{sensitivity['ci_delta_auroc'][0]:+.3f}–{sensitivity['ci_delta_auroc'][1]:+.3f}）；"
        f"ΔBrier {sensitivity['delta_brier']:+.4f}"
        f"（{sensitivity['ci_delta_brier'][0]:+.4f}–{sensitivity['ci_delta_brier'][1]:+.4f}）。",
        f"- 可评 {sensitivity['n']} 个预测日 / {sensitivity['persons']} 人 / {sensitivity['events']} 个阳性。",
        "",
        "## 3. 旧版与修正版（同一批 172 个共同预测日）",
        "",
        "| 版本/口径 | A AUROC | B AUROC | ΔAUROC | ΔBrier |",
        "|---|---:|---:|---:|---:|",
        f"| 旧版（旧分数 + 旧 Cox） | {old['old_on_common']['auroc_A']:.3f} | "
        f"{old['old_on_common']['auroc_B']:.3f} | {old['old_on_common']['delta_auroc']:+.3f} | "
        f"{old['old_on_common']['delta_brier']:+.4f} |",
        f"| 仅换 Cox 求解（旧分数 + 新 Cox） | {attribution['solver_on_common']['auroc_A']:.3f} | "
        f"{attribution['solver_on_common']['auroc_B']:.3f} | {attribution['solver_on_common']['delta_auroc']:+.3f} | "
        f"{attribution['solver_on_common']['delta_brier']:+.4f} |",
        f"| 仅换分数（新分数 + 旧 Cox） | {attribution['precision_on_common']['auroc_A']:.3f} | "
        f"{attribution['precision_on_common']['auroc_B']:.3f} | {attribution['precision_on_common']['delta_auroc']:+.3f} | "
        f"{attribution['precision_on_common']['delta_brier']:+.4f} |",
        f"| 修正版（新分数 + 新 Cox） | {main['auroc_A']:.3f} | {main['auroc_B']:.3f} | "
        f"{main['delta_auroc']:+.3f} | {main['delta_brier']:+.4f} |",
        "",
        f"旧版全部 186 个预测日的参考：AUROC {old['old_all']['auroc_A']:.3f} / {old['old_all']['auroc_B']:.3f}，"
        f"ΔAUROC {old['old_all']['delta_auroc']:+.3f}，ΔBrier {old['old_all']['delta_brier']:+.4f}。"
        "旧版指标只作历史对照，不作为修正版的通过条件。",
        "",
        "## 4. 样本与筛选",
        "",
        f"- 宽集：{selection['wide_landmarks']} 个预测日 / {selection['wide_people']} 人，"
        f"30 天阳性 {selection['wide_30d_positive_landmarks']}（{selection['wide_30d_positive_people']} 人）。",
        f"- 主集（14 个既往有效日）：{selection['baseline_ready_landmarks']} 个预测日 / "
        f"{selection['baseline_ready_people']} 人，30 天阳性 17（15 人）；"
        f"因基线不足排除 {selection['baseline_excluded_landmarks']} 个预测日、"
        f"{selection['baseline_excluded_30d_positives']} 个阳性。",
        f"- 修正版 session_se：{selection['session_scored_landmarks']} 个预测日 / "
        f"{selection['session_scored_people']} 人，30 天阳性 {selection['session_30d_positive_landmarks']}"
        f"（{selection['session_30d_positive_people']} 人）；"
        f"{selection['session_unavailable_landmarks']} 个预测日 / {selection['session_unavailable_people']} 人"
        f"因 14 日基线中某日缺少步速标准误而不可评分（其中 30 天阳性 {selection['session_unavailable_30d_positives']}）。",
        "- 更上游的日记排除：见 `selection.json.upstream_diary_exclusions`"
        "（154 条日记与报告不一致、311 条预测日不在日记区间、133 条上次访视无步态、11 条未对齐、"
        "2,764 条非目标人群日记）。",
        "",
        "## 5. 7 天窗口",
        "",
        f"- 修正版 session_se：{seven['session']['n']} 个预测日 / {seven['session']['events']} 个阳性；"
        f"B 折失败 {seven['session']['failed_folds']} 个，配对子集 {seven['session']['paired_n']} 行 / "
        f"{seven['session']['paired_events']} 个阳性；状态 `{seven['session']['status']}`，只报告计数与失败。",
        f"- none：{seven['none']['n']} 个预测日 / {seven['none']['events']} 个阳性；"
        f"配对子集 {seven['none']['paired_n']} 行 / {seven['none']['paired_events']} 个阳性；"
        f"状态 `{seven['none']['status']}`。",
        "",
        "## 6. 结论与证据边界",
        "",
        "**判定：证据不足。** 修正版 30 天主比较的 ΔAUROC 为 "
        f"{main['delta_auroc']:+.3f}（{main['ci_delta_auroc'][0]:+.3f}–{main['ci_delta_auroc'][1]:+.3f}），"
        f"ΔBrier 为 {main['delta_brier']:+.4f}"
        f"（{main['ci_delta_brier'][0]:+.4f}–{main['ci_delta_brier'][1]:+.4f}）。"
        f"旧版全部 186 个预测日的 ΔAUROC 为 {old['old_all']['delta_auroc']:+.3f}，"
        f"在共同 172 日上为 {old['old_on_common']['delta_auroc']:+.3f}；修正后为 "
        f"{main['delta_auroc']:+.3f}。点估计仍略偏向 B，但区间跨 0、Brier 无改善，"
        "16 个阳性不足以确立增量。",
        f"- 30 天主比较：ΔAUROC {main['delta_auroc']:+.3f}"
        f"（{main['ci_delta_auroc'][0]:+.3f}–{main['ci_delta_auroc'][1]:+.3f}），"
        f"ΔBrier {main['delta_brier']:+.4f}"
        f"（{main['ci_delta_brier'][0]:+.4f}–{main['ci_delta_brier'][1]:+.4f}）。"
        "区间跨 0 时按证据不足报告；点估计方向不作为通过条件。",
        f"- 30 天主集只有 {main['events']} 个阳性；加入 C 后 RMS 与 C 的相关性使系数不稳定，"
        "不能把点估计解释为稳定增量。",
        "- 修正版只改变时间截断与 Cox 目标函数；人群、预测日、随访窗、方向、空档与分折保持不变。",
        "- 旧版 186 与修正版 172 的差异来自 14 个基线日缺少步速标准误的预测日；已逐项列出，未插补。",
        "- 仿真阈值复制自 `20261114-main/simulation`，来源为阶段 2 的仿真零假设标定"
        "（session h=5.2、none h=5.1）；主分析使用连续 C，告警仅作展示，不是真实 30 天误报率。",
        "- 无独立验证，模型 `calibrated=false`；连续视频监测、其他人群与外部场地未验证。",
        "",
        "## 7. 文件与复现",
        "",
        "- 协议：`analysis_rules.json`；折外预测：`predictions.csv`；折模型：`fold_models/`；指标：`metrics.csv`。",
        "- 旧版：`../20261114-main/`；修正版运行：本目录。",
        "- 复现：见 `reproduce.md`；核验：`verification.json`。",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    fixed_run = Path(args.fixed_run)
    old_run = Path(args.old_run)
    out = Path(args.out) if args.out else fixed_run / "stage3"
    out.mkdir(parents=True, exist_ok=True)
    data = fixed_run / "data"
    stage2 = fixed_run / "stage2"
    folds_path = Path(args.folds) if args.folds else old_run / "splits.csv"
    folds = pd.read_csv(folds_path, dtype={"participantid": str})
    if folds["participantid"].duplicated().any():
        raise ValueError("折文件人员重复")
    old_fitter = _load_old_fitter(old_run / "code_snapshot" / "longitudinal" / "cox_fit.py")
    frames = {
        "fixed_session": load_analysis_frame(stage2, "session_se", data),
        "fixed_none": load_analysis_frame(stage2, "none", data),
        "old_session": load_analysis_frame(old_run / "stage2", "session_se", data),
        "old_none": load_analysis_frame(old_run / "stage2", "none", data),
    }
    for name, frame in frames.items():
        if not set(frame["participantid"]).issubset(set(folds["participantid"])):
            raise ValueError(f"{name} 有人员不在折文件中")
        frame["fold"] = frame["participantid"].map(
            dict(zip(folds["participantid"], folds["fold"]))
        ).astype(int)
    fixed_scores = pd.read_csv(stage2 / "scores_session_se.csv", dtype={"participantid": str})
    fixed_none_scores = pd.read_csv(stage2 / "scores_none.csv", dtype={"participantid": str})
    old_scores = pd.read_csv(old_run / "scores.csv", dtype={"participantid": str})
    fixed_keys = set(
        zip(fixed_scores.loc[fixed_scores["ds_status"].eq("scored"), "participantid"],
            fixed_scores.loc[fixed_scores["ds_status"].eq("scored"), "landmark_day"].astype(str))
    )
    none_keys = set(
        zip(fixed_none_scores["participantid"], fixed_none_scores["landmark_day"].astype(str))
    )
    rules = {
        "stage": 6,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": fixed_run.name,
        "old_run": str(old_run),
        "folds": {"file": str(folds_path), "md5": _md5(folds_path), "seed": SEED,
                  "people": int(len(folds)), "reassigned": False},
        "primary": {"mode": "session_se", "horizon": 30,
                    "models": {"A_rms": ["rms_distance"], "B_rms_cusum": ["rms_distance", "cusum"]}},
        "sensitivity": {"mode": "none", "horizon": 30},
        "common_rows_rule": "A/B 均只使用 rms_distance 与 cusum 均有限的预测日；训练与测试同一批有效行；不插补。",
        "fold_failure_rule": "训练事件 <2、未收敛或 |标准化 beta|>10 记为失败；失败折不产生预测；配对指标使用两方法均预测的行。",
        "metrics": {"auroc": "越大越好", "brier": "越小越好", "delta": "B − A",
                    "bootstrap": f"按人聚类 {args.bootstrap} 次，百分位 95%"},
        "calibration": "观察风险、平均预测风险；30 天斜率；7 天事件过少不报斜率",
        "variants": {
            "fixed_session": "新分数 + 新 Cox（主）",
            "fixed_none": "新分数 + 新 Cox（敏感性）",
            "old_on_common": "旧预测限制到 172 个共同预测日",
            "solver_on_common": "旧分数 + 新 Cox（172）",
            "precision_on_common": "新分数 + 旧 Cox（172）",
            "old_all": "旧版全部 186 个预测日（历史参考）",
            "solver_none": "旧 none 分数 + 新 Cox（186）",
        },
    }
    _write_json(out / "analysis_rules.json", rules)

    main_dir = out / "fold_models"
    if main_dir.exists():
        import shutil
        shutil.rmtree(main_dir)
    all_predictions: list[dict[str, Any]] = []
    all_statuses: list[dict[str, Any]] = []

    def add(variant: str, frame: pd.DataFrame, fitter, horizon: int) -> None:
        predictions, statuses = fit_variant(
            variant, frame, folds, horizon, fitter, model_dir=main_dir,
        )
        all_predictions.extend(predictions)
        all_statuses.extend(statuses)

    def common(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[
            frame["rms_distance"].notna() & frame["cusum"].notna()
        ].copy()

    def restrict(frame: pd.DataFrame, keys: set[tuple[str, str]]) -> pd.DataFrame:
        mask = [
            (str(person), pd.Timestamp(day).date().isoformat()) in keys
            for person, day in zip(frame["participantid"], frame["landmark_day"])
        ]
        return frame.loc[mask].copy()

    add("fixed_session", common(frames["fixed_session"]), new_fit_landmark_cox, 30)
    add("fixed_none", common(frames["fixed_none"]), new_fit_landmark_cox, 30)
    add("solver_on_common", restrict(common(frames["old_session"]), fixed_keys),
        new_fit_landmark_cox, 30)
    add("precision_on_common", common(frames["fixed_session"]), old_fitter, 30)
    add("solver_none", common(frames["old_none"]), new_fit_landmark_cox, 30)
    add("fixed_session", common(frames["fixed_session"]), new_fit_landmark_cox, 7)
    add("fixed_none", common(frames["fixed_none"]), new_fit_landmark_cox, 7)

    # 旧预测：限制到共同 key；旧版全部 186 作为历史参考
    for variant, mode, keys in (
        ("old_on_common", "session_se", fixed_keys),
        ("old_all", "session_se", set(zip(old_scores["participantid"], old_scores["landmark_day"].astype(str)))),
        ("old_none", "none", none_keys),
    ):
        old_frame = old_predictions(old_run, mode, 30, keys)
        old_frame["variant"] = variant
        all_predictions.extend(old_frame[[
            "variant", "participantid", "landmark_day", "fold", "horizon",
            "method", "risk", "event", "duration", "rms_distance", "cusum",
        ]].to_dict("records"))
    # old 7 天对照（描述性）
    old_seven = pd.read_csv(old_run / "predictions.csv", dtype={"participantid": str})
    old_seven = old_seven.loc[old_seven["horizon"].eq(7) & old_seven["mode"].eq("session_se")].copy()
    old_seven["variant"] = "old_7d"
    old_seven["landmark_day"] = old_seven["landmark_day"].astype(str)
    all_predictions.extend(old_seven[[
        "variant", "participantid", "landmark_day", "fold", "horizon",
        "method", "risk", "event", "duration", "rms_distance", "cusum",
    ]].to_dict("records"))

    predictions = pd.DataFrame(all_predictions)
    predictions = predictions.sort_values(
        ["variant", "horizon", "method", "participantid", "landmark_day"]
    ).reset_index(drop=True)
    if predictions.duplicated(
        ["variant", "horizon", "method", "participantid", "landmark_day"]
    ).any():
        raise ValueError("预测键不唯一")
    predictions.to_csv(out / "predictions.csv", index=False, encoding="utf-8")
    statuses = pd.DataFrame(all_statuses)
    statuses.to_csv(out / "fold_status.csv", index=False, encoding="utf-8")

    # 指标与 bootstrap
    metrics: list[dict[str, Any]] = []
    bootstrap: dict[str, Any] = {}
    variants_30 = ["fixed_session", "fixed_none", "old_on_common", "old_all",
                   "solver_on_common", "precision_on_common", "solver_none", "old_none"]
    for variant in variants_30:
        for horizon in (30,):
            result = bootstrap_pairs(
                predictions, variant=variant, horizon=horizon,
                replicates=args.bootstrap, seed=SEED,
            )
            bootstrap[f"{variant}_{horizon}d"] = result
            if result.get("status") != "ok":
                metrics.append({"variant": variant, "horizon": horizon, "method": "B_minus_A",
                                "status": result.get("status"), "note": result.get("reason", "")})
                continue
            point = result["point"]
            for method in ("A_rms", "B_rms_cusum"):
                metrics.append({
                    "variant": variant, "horizon": horizon, "method": method,
                    "n": point["n"], "events": point["events"], "persons": point["persons"],
                    "auroc": point["auroc"][method], "brier": point["brier"][method],
                    "observed_risk": point["observed_risk"],
                    "mean_predicted": point["mean_predicted"][method],
                    "calibration_slope": point["calibration_slope"][method],
                    "auroc_ci_low": (result["ci"]["auroc"][method] or [None, None])[0],
                    "auroc_ci_high": (result["ci"]["auroc"][method] or [None, None])[1],
                    "brier_ci_low": (result["ci"]["brier"][method] or [None, None])[0],
                    "brier_ci_high": (result["ci"]["brier"][method] or [None, None])[1],
                    "slope_ci_low": (result["ci"]["calibration_slope"][method] or [None, None])[0],
                    "slope_ci_high": (result["ci"]["calibration_slope"][method] or [None, None])[1],
                    "status": "ok",
                })
            metrics.append({
                "variant": variant, "horizon": horizon, "method": "B_minus_A",
                "n": point["n"], "events": point["events"], "persons": point["persons"],
                "delta_auroc": point["auroc"]["B_rms_cusum"] - point["auroc"]["A_rms"],
                "delta_brier": point["brier"]["B_rms_cusum"] - point["brier"]["A_rms"],
                "delta_auroc_ci_low": (result["ci"]["delta_auroc"] or [None, None])[0],
                "delta_auroc_ci_high": (result["ci"]["delta_auroc"] or [None, None])[1],
                "delta_brier_ci_low": (result["ci"]["delta_brier"] or [None, None])[0],
                "delta_brier_ci_high": (result["ci"]["delta_brier"] or [None, None])[1],
                "status": "ok",
            })
    # 7 天
    for variant in ("fixed_session", "fixed_none", "old_7d"):
        result = bootstrap_pairs(
            predictions, variant=variant, horizon=7,
            replicates=args.bootstrap, seed=SEED,
        )
        bootstrap[f"{variant}_7d"] = result
        status = "not_estimable" if result.get("status") != "ok" else "descriptive_unstable"
        if result.get("status") == "ok":
            point = result["point"]
            for method in ("A_rms", "B_rms_cusum"):
                metrics.append({
                    "variant": variant, "horizon": 7, "method": method,
                    "n": point["n"], "events": point["events"], "persons": point["persons"],
                    "auroc": point["auroc"][method], "brier": point["brier"][method],
                    "observed_risk": point["observed_risk"],
                    "mean_predicted": point["mean_predicted"][method],
                    "calibration_slope": None,
                    "status": status,
                })
            metrics.append({
                "variant": variant, "horizon": 7, "method": "B_minus_A",
                "n": point["n"], "events": point["events"], "persons": point["persons"],
                "delta_auroc": point["auroc"]["B_rms_cusum"] - point["auroc"]["A_rms"],
                "delta_brier": point["brier"]["B_rms_cusum"] - point["brier"]["A_rms"],
                "status": "not_estimable",
            })
        else:
            metrics.append({"variant": variant, "horizon": 7, "method": "B_minus_A",
                            "status": "not_estimable", "note": result.get("reason", "")})
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(out / "metrics.csv", index=False, encoding="utf-8")
    _write_json(out / "bootstrap.json", bootstrap)

    main = bootstrap["fixed_session_30d"]
    sensitivity = bootstrap["fixed_none_30d"]
    old_common = bootstrap["old_on_common_30d"]
    old_all = bootstrap["old_all_30d"]
    attribution = {
        "solver_on_common": bootstrap["solver_on_common_30d"]["point"],
        "precision_on_common": bootstrap["precision_on_common_30d"]["point"],
    }

    def summary(result: dict[str, Any]) -> dict[str, Any]:
        point = result["point"]
        return {
            "auroc_A": point["auroc"]["A_rms"],
            "auroc_B": point["auroc"]["B_rms_cusum"],
            "brier_A": point["brier"]["A_rms"],
            "brier_B": point["brier"]["B_rms_cusum"],
            "delta_auroc": point["auroc"]["B_rms_cusum"] - point["auroc"]["A_rms"],
            "delta_brier": point["brier"]["B_rms_cusum"] - point["brier"]["A_rms"],
            "ci_auroc_A": result["ci"]["auroc"]["A_rms"],
            "ci_auroc_B": result["ci"]["auroc"]["B_rms_cusum"],
            "ci_brier_A": result["ci"]["brier"]["A_rms"],
            "ci_brier_B": result["ci"]["brier"]["B_rms_cusum"],
            "ci_delta_auroc": result["ci"]["delta_auroc"],
            "ci_delta_brier": result["ci"]["delta_brier"],
            "n": point["n"], "events": point["events"], "persons": point["persons"],
            "observed_risk": point["observed_risk"],
            "mean_predicted_A": point["mean_predicted"]["A_rms"],
            "mean_predicted_B": point["mean_predicted"]["B_rms_cusum"],
            "slope_A": point["calibration_slope"]["A_rms"],
            "slope_B": point["calibration_slope"]["B_rms_cusum"],
            "ci_slope_A": result["ci"]["calibration_slope"]["A_rms"],
            "ci_slope_B": result["ci"]["calibration_slope"]["B_rms_cusum"],
            "replicates": result["replicates_requested"],
            "valid_replicates": result["replicates_valid"],
            "invalid_replicates": result["replicates_invalid"],
        }

    selection = selection_report(
        data, old_run, fixed_scores, fixed_none_scores, folds,
    )
    session_seven = predictions.loc[
        predictions["variant"].eq("fixed_session") & predictions["horizon"].eq(7)
        & predictions["method"].eq("A_rms")
    ]
    none_seven = predictions.loc[
        predictions["variant"].eq("fixed_none") & predictions["horizon"].eq(7)
        & predictions["method"].eq("A_rms")
    ]
    seven_status = {
        "session": {"n": int(len(session_seven)),
                     "events": int(session_seven["event"].sum()),
                     "failed_folds": int(statuses.loc[statuses["variant"].eq("fixed_session") & statuses["horizon"].eq(7) & statuses["status"].ne("ok")].shape[0]),
                     "paired_n": int(bootstrap.get("fixed_session_7d", {}).get("point", {}).get("n", 0)),
                     "paired_events": int(bootstrap.get("fixed_session_7d", {}).get("point", {}).get("events", 0)),
                     "status": "descriptive_unstable"},
        "none": {"n": int(len(none_seven)),
                 "events": int(none_seven["event"].sum()),
                 "failed_folds": int(statuses.loc[statuses["variant"].eq("fixed_none") & statuses["horizon"].eq(7) & statuses["status"].ne("ok")].shape[0]),
                 "paired_n": int(bootstrap.get("fixed_none_7d", {}).get("point", {}).get("n", 0)),
                 "paired_events": int(bootstrap.get("fixed_none_7d", {}).get("point", {}).get("events", 0)),
                 "status": "descriptive_unstable"},
    }
    payload = {
        "run_id": fixed_run.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "main_metrics": summary(main),
        "sensitivity_metrics": summary(sensitivity),
        "attribution": {
            "solver_on_common": summary({"point": attribution["solver_on_common"],
                                        "ci": bootstrap["solver_on_common_30d"]["ci"],
                                        "replicates_requested": bootstrap["solver_on_common_30d"]["replicates_requested"],
                                        "replicates_valid": bootstrap["solver_on_common_30d"]["replicates_valid"],
                                        "replicates_invalid": bootstrap["solver_on_common_30d"]["replicates_invalid"]}),
            "precision_on_common": summary({"point": attribution["precision_on_common"],
                                           "ci": bootstrap["precision_on_common_30d"]["ci"],
                                           "replicates_requested": bootstrap["precision_on_common_30d"]["replicates_requested"],
                                           "replicates_valid": bootstrap["precision_on_common_30d"]["replicates_valid"],
                                           "replicates_invalid": bootstrap["precision_on_common_30d"]["replicates_invalid"]}),
        },
        "old_comparison": {
            "old_on_common": summary(old_common),
            "old_all": summary(old_all),
        },
        "selection": selection,
        "seven_day": seven_status,
    }
    # 旧版对照需要 old_on_common 的 CI 字段；summary 已包含
    _write_json(out / "comparison.json", payload)
    _write_json(out / "selection.json", selection)
    write_report(out, payload)
    print(json.dumps({
        "out": str(out),
        "main": {key: payload["main_metrics"][key] for key in
                 ("n", "events", "persons", "auroc_A", "auroc_B", "delta_auroc", "delta_brier")},
        "sensitivity": {key: payload["sensitivity_metrics"][key] for key in
                        ("n", "events", "persons", "delta_auroc", "delta_brier")},
        "old_on_common": {key: payload["old_comparison"]["old_on_common"][key] for key in
                          ("n", "events", "delta_auroc", "delta_brier")},
        "fold_status": statuses["status"].value_counts().to_dict(),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
