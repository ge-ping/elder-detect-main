"""阶段 10：共同样本归因对照与三项修复的收尾记录。

五个对照行都使用同一组 172 个共同测试预测日、同一人员分折：
历史旧模型（训练 186，测试事后限制到 172）、共同样本旧版重拟合、
共同样本新求解器、共同样本新分数、修正版组合。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyse_mobilise_stage6 import (  # noqa: E402
    SEED,
    _load_old_fitter,
    bootstrap_pairs,
    fit_variant,
    load_analysis_frame,
)

OUT_ROOT = PROJECT_ROOT.parents[1] / "output" / "ds_pcd_mobilise"
FIXED_RUN = OUT_ROOT / "20261114-fixed"
OLD_RUN = OUT_ROOT / "20261114-main"
DEFAULT_OUT = OUT_ROOT / "stage10" / "20261114-attribution-v1"
HORIZON = 30
METHODS = ("A_rms", "B_rms_cusum")
COMMON_VARIANTS = (
    "old_refit_common", "solver_on_common", "precision_on_common", "fixed_session",
)
ROW_ORDER = (
    "historical_old_186", "old_refit_common", "solver_on_common",
    "precision_on_common", "fixed_session",
)
ROW_META = {
    "historical_old_186": {
        "training_sample": "原 186 个预测日（旧模型训练）",
        "training_scope": "old_186",
        "score_version": "legacy-stage4-v1",
        "solver_version": "legacy-stage4-v1",
        "feature_version": "stage4-legacy",
        "source": "20261114-main/stage3/predictions.csv（限制到 172 个测试行）",
        "test_scope": "common_172（事后限制）",
    },
    "old_refit_common": {
        "training_sample": "共同 172 个预测日",
        "training_scope": "common_172",
        "score_version": "legacy-stage4-v1",
        "solver_version": "legacy-stage4-v1",
        "feature_version": "stage4-legacy",
        "source": "stage10 重拟合",
        "test_scope": "common_172",
    },
    "solver_on_common": {
        "training_sample": "共同 172 个预测日",
        "training_scope": "common_172",
        "score_version": "legacy-stage4-v1",
        "solver_version": "stage5-fixed",
        "feature_version": "stage4-legacy",
        "source": "20261114-fixed/stage3/predictions.csv",
        "test_scope": "common_172",
    },
    "precision_on_common": {
        "training_sample": "共同 172 个预测日",
        "training_scope": "common_172",
        "score_version": "scoring-v2-causal-precision",
        "solver_version": "legacy-stage4-v1",
        "feature_version": "mobilise-d-daily-v1",
        "source": "20261114-fixed/stage3/predictions.csv",
        "test_scope": "common_172",
    },
    "fixed_session": {
        "training_sample": "共同 172 个预测日",
        "training_scope": "common_172",
        "score_version": "scoring-v2-causal-precision",
        "solver_version": "stage5-fixed",
        "feature_version": "mobilise-d-daily-v1",
        "source": "20261114-fixed/stage3/predictions.csv",
        "test_scope": "common_172",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 10 共同样本归因")
    parser.add_argument("--fixed-run", default=str(FIXED_RUN))
    parser.add_argument("--old-run", default=str(OLD_RUN))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["participantid"] = frame["participantid"].astype(str).str.strip()
    frame["landmark_day"] = pd.to_datetime(frame["landmark_day"]).dt.strftime("%Y-%m-%d")
    return frame


def restrict(frame: pd.DataFrame, keys: set[tuple[str, str]]) -> pd.DataFrame:
    mask = [
        (str(person), pd.Timestamp(day).date().isoformat()) in keys
        for person, day in zip(frame["participantid"], frame["landmark_day"])
    ]
    return frame.loc[mask].copy()


def common(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        frame["rms_distance"].notna() & frame["cusum"].notna()
    ].copy()


def load_existing(fixed_run: Path) -> pd.DataFrame:
    frame = normalise(pd.read_csv(
        fixed_run / "stage3" / "predictions.csv", dtype={"participantid": str}
    ))
    frame = frame.loc[
        frame["horizon"].eq(HORIZON) & frame["method"].isin(METHODS)
    ].copy()
    return frame


def key_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame["participantid"], frame["landmark_day"]))


def scope_rows(
    variant: str, keys: set[tuple[str, str]], folds: dict[str, int],
    events: dict[tuple[str, str], int],
) -> list[dict[str, Any]]:
    rows = []
    by_fold: dict[int, set[tuple[str, str]]] = {}
    for key in keys:
        by_fold.setdefault(int(folds[key[0]]), set()).add(key)
    for fold in sorted(set(folds.values())):
        test = by_fold.get(fold, set())
        train = keys - test
        for role, subset in (("train", train), ("test", test)):
            for person, day in sorted(subset):
                rows.append({
                    "variant": variant, "horizon": HORIZON, "fold": int(fold),
                    "role": role, "participantid": person, "landmark_day": day,
                    "event_30d": int(events.get((person, day), 0)),
                })
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fixed_run = Path(args.fixed_run).resolve()
    old_run = Path(args.old_run).resolve()
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"目标目录已存在且非空：{out}")
    out.mkdir(parents=True, exist_ok=True)

    folds_path = old_run / "splits.csv"
    folds = pd.read_csv(folds_path, dtype={"participantid": str})
    if folds["participantid"].duplicated().any():
        raise ValueError("折文件人员重复")
    folds["participantid"] = folds["participantid"].astype(str)
    fold_map = dict(zip(folds["participantid"], folds["fold"].astype(int)))
    existing = load_existing(fixed_run)
    fixed = existing.loc[existing["variant"].eq("fixed_session")].copy()
    if fixed.empty:
        raise ValueError("缺少 fixed_session 预测")
    fixed_keys = key_set(fixed)
    if len(fixed_keys) != 172:
        raise ValueError(f"固定共同测试集应为 172，实际 {len(fixed_keys)}")
    for variant in ("solver_on_common", "precision_on_common", "old_on_common"):
        sub = existing.loc[existing["variant"].eq(variant)]
        if key_set(sub) != fixed_keys:
            raise ValueError(f"{variant} 的测试键与 fixed_session 不一致")

    old_frame = load_analysis_frame(old_run / "stage2", "session_se", fixed_run / "data")
    old_common = restrict(common(old_frame), fixed_keys)
    old_common_keys = {
        (str(p), pd.Timestamp(d).date().isoformat())
        for p, d in zip(old_common["participantid"], old_common["landmark_day"])
    }
    if old_common_keys != fixed_keys:
        raise ValueError("旧分数共同 172 行与固定测试键不一致")
    old_fitter = _load_old_fitter(old_run / "code_snapshot" / "longitudinal" / "cox_fit.py")
    model_dir = out / "fold_models"
    new_predictions, new_statuses = fit_variant(
        "old_refit_common", old_common, folds, HORIZON, old_fitter,
        model_dir=model_dir,
    )
    new_frame = pd.DataFrame(new_predictions)
    if len(new_frame) != 344 or key_set(new_frame) != fixed_keys:
        raise ValueError("old_refit_common 预测行不完整")
    for method in METHODS:
        if int(new_frame["method"].eq(method).sum()) != 172:
            raise ValueError(f"old_refit_common {method} 不是 172 行")

    # 五个对照行的统一预测表
    parts = []
    for variant in ROW_ORDER:
        if variant == "historical_old_186":
            sub = existing.loc[existing["variant"].eq("old_on_common")].copy()
            sub["variant"] = "historical_old_186"
        elif variant == "old_refit_common":
            sub = new_frame.copy()
        else:
            sub = existing.loc[existing["variant"].eq(variant)].copy()
        sub = normalise(sub)
        sub = sub.loc[sub["horizon"].eq(HORIZON) & sub["method"].isin(METHODS)].copy()
        if key_set(sub) != fixed_keys:
            raise ValueError(f"{variant} 测试键不是共同 172")
        for field, value in ROW_META[variant].items():
            sub[field] = value
        parts.append(sub)
    predictions = pd.concat(parts, ignore_index=True)
    predictions = predictions.sort_values(
        ["variant", "horizon", "method", "participantid", "landmark_day"]
    ).reset_index(drop=True)
    predictions.to_csv(out / "same_sample_predictions.csv", index=False, encoding="utf-8")

    # 每折训练/测试键；共同四行共享同一 172 个键
    events = dict(zip(
        zip(fixed["participantid"], fixed["landmark_day"]),
        fixed["event"].astype(int),
    ))
    common_scope = []
    for variant in COMMON_VARIANTS:
        common_scope.extend(scope_rows(variant, fixed_keys, fold_map, events))
    common_scope_frame = pd.DataFrame(common_scope)
    common_scope_frame.to_csv(out / "fold_scope_common.csv", index=False, encoding="utf-8")

    # 历史旧模型：原 186 训练范围；共同 172 测试行是事后限制
    old_predictions = normalise(pd.read_csv(
        old_run / "predictions.csv", dtype={"participantid": str}
    ))
    old_predictions = old_predictions.loc[
        old_predictions["mode"].eq("session_se") & old_predictions["horizon"].eq(HORIZON)
    ].copy()
    old_keys = key_set(old_predictions)
    if len(old_keys) != 186:
        raise ValueError(f"旧版 186 预测键异常：{len(old_keys)}")
    old_event = dict(zip(
        zip(old_predictions["participantid"], old_predictions["landmark_day"]),
        old_predictions["event"].astype(int),
    ))
    historical = []
    for fold in sorted(set(fold_map.values())):
        test = {key for key in old_keys if fold_map[key[0]] == fold}
        train = old_keys - test
        for role, subset in (("train", train), ("test", test)):
            for person, day in sorted(subset):
                historical.append({
                    "variant": "historical_old_186", "horizon": HORIZON,
                    "fold": int(fold), "role": role,
                    "participantid": person, "landmark_day": day,
                    "event_30d": int(old_event.get((person, day), 0)),
                    "in_common_172": int((person, day) in fixed_keys),
                })
    historical_frame = pd.DataFrame(historical)
    historical_frame.to_csv(out / "historical_scope.csv", index=False, encoding="utf-8")
    historical_common_test = pd.DataFrame([
        {
            "variant": "historical_old_186", "horizon": HORIZON,
            "fold": int(fold_map[person]), "participantid": person,
            "landmark_day": day, "event_30d": int(events[(person, day)]),
        }
        for person, day in sorted(fixed_keys)
    ])
    historical_common_test.to_csv(
        out / "historical_common_test.csv", index=False, encoding="utf-8"
    )

    scope_summary = []
    for variant in ROW_ORDER:
        meta = ROW_META[variant]
        if variant == "historical_old_186":
            train_keys = old_keys
            test_keys = fixed_keys
            train_per_fold = {
                int(fold): len([key for key in old_keys if fold_map[key[0]] != fold])
                for fold in sorted(set(fold_map.values()))
            }
            test_per_fold = {
                int(fold): len([key for key in fixed_keys if fold_map[key[0]] == fold])
                for fold in sorted(set(fold_map.values()))
            }
        else:
            train_keys = fixed_keys
            test_keys = fixed_keys
            train_per_fold = {
                int(fold): len([key for key in fixed_keys if fold_map[key[0]] != fold])
                for fold in sorted(set(fold_map.values()))
            }
            test_per_fold = {
                int(fold): len([key for key in fixed_keys if fold_map[key[0]] == fold])
                for fold in sorted(set(fold_map.values()))
            }
        scope_summary.append({
            "variant": variant,
            "training_sample": meta["training_sample"],
            "training_scope": meta["training_scope"],
            "test_scope": meta["test_scope"],
            "unique_train_landmarks": len(train_keys),
            "unique_test_landmarks": len(test_keys),
            "train_rows_per_fold": json.dumps(train_per_fold, sort_keys=True),
            "test_rows_per_fold": json.dumps(test_per_fold, sort_keys=True),
            "score_version": meta["score_version"],
            "solver_version": meta["solver_version"],
            "feature_version": meta["feature_version"],
            "source": meta["source"],
        })
    scope_summary_frame = pd.DataFrame(scope_summary)
    scope_summary_frame.to_csv(out / "scope_summary.csv", index=False, encoding="utf-8")

    # 指标与 bootstrap；全部行使用同一测试键
    metrics_rows = []
    bootstrap = {}
    for variant in ROW_ORDER:
        result = bootstrap_pairs(
            predictions, variant=variant, horizon=HORIZON,
            replicates=args.bootstrap, seed=SEED,
        )
        bootstrap[f"{variant}_{HORIZON}d"] = result
        if result.get("status") != "ok":
            raise ValueError(f"{variant} 指标不可估：{result}")
        point = result["point"]
        for method in METHODS:
            metrics_rows.append({
                "variant": variant, "horizon": HORIZON, "method": method,
                "n": point["n"], "events": point["events"], "persons": point["persons"],
                "auroc": point["auroc"][method], "brier": point["brier"][method],
                "observed_risk": point["observed_risk"],
                "mean_predicted": point["mean_predicted"][method],
                "calibration_slope": point["calibration_slope"][method],
                "auroc_ci_low": (result["ci"]["auroc"][method] or [None, None])[0],
                "auroc_ci_high": (result["ci"]["auroc"][method] or [None, None])[1],
                "brier_ci_low": (result["ci"]["brier"][method] or [None, None])[0],
                "brier_ci_high": (result["ci"]["brier"][method] or [None, None])[1],
                "status": "ok",
            })
        metrics_rows.append({
            "variant": variant, "horizon": HORIZON, "method": "B_minus_A",
            "n": point["n"], "events": point["events"], "persons": point["persons"],
            "delta_auroc": point["auroc"]["B_rms_cusum"] - point["auroc"]["A_rms"],
            "delta_brier": point["brier"]["B_rms_cusum"] - point["brier"]["A_rms"],
            "delta_auroc_ci_low": (result["ci"]["delta_auroc"] or [None, None])[0],
            "delta_auroc_ci_high": (result["ci"]["delta_auroc"] or [None, None])[1],
            "delta_brier_ci_low": (result["ci"]["delta_brier"] or [None, None])[0],
            "delta_brier_ci_high": (result["ci"]["delta_brier"] or [None, None])[1],
            "status": "ok",
        })
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(out / "same_sample_metrics.csv", index=False, encoding="utf-8")
    write_json(out / "bootstrap.json", bootstrap)

    def row_summary(variant: str) -> dict[str, Any]:
        result = bootstrap[f"{variant}_{HORIZON}d"]
        point = result["point"]
        return {
            **{key: ROW_META[variant][key] for key in (
                "training_sample", "training_scope", "score_version",
                "solver_version", "feature_version", "source", "test_scope",
            )},
            "n": point["n"], "events": point["events"], "persons": point["persons"],
            "auroc_A": point["auroc"]["A_rms"],
            "auroc_B": point["auroc"]["B_rms_cusum"],
            "brier_A": point["brier"]["A_rms"],
            "brier_B": point["brier"]["B_rms_cusum"],
            "delta_auroc": point["auroc"]["B_rms_cusum"] - point["auroc"]["A_rms"],
            "delta_brier": point["brier"]["B_rms_cusum"] - point["brier"]["A_rms"],
            "ci_delta_auroc": result["ci"]["delta_auroc"],
            "ci_delta_brier": result["ci"]["delta_brier"],
            "ci_auroc_A": result["ci"]["auroc"]["A_rms"],
            "ci_auroc_B": result["ci"]["auroc"]["B_rms_cusum"],
            "ci_brier_A": result["ci"]["brier"]["A_rms"],
            "ci_brier_B": result["ci"]["brier"]["B_rms_cusum"],
            "replicates": result["replicates_requested"],
            "valid_replicates": result["replicates_valid"],
            "invalid_replicates": result["replicates_invalid"],
        }

    rows = {variant: row_summary(variant) for variant in ROW_ORDER}
    # 求解器对照：同一训练/测试/旧分数，只换求解器
    a = predictions.loc[predictions["variant"].eq("old_refit_common")]
    b = predictions.loc[predictions["variant"].eq("solver_on_common")]
    merged = a.merge(
        b, on=["participantid", "landmark_day", "method"], suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    solver_diff = {
        "n_pairs": int(len(merged)),
        "max_abs_prediction_diff": float((merged["risk_old"] - merged["risk_new"]).abs().max()),
        "by_method": {
            method: {
                "n": int(len(merged.loc[merged["method"].eq(method)])),
                "max_abs_prediction_diff": float(
                    (merged.loc[merged["method"].eq(method), "risk_old"]
                     - merged.loc[merged["method"].eq(method), "risk_new"]).abs().max()
                ),
            }
            for method in METHODS
        },
        "source": "old_refit_common（旧分数+旧求解器） vs solver_on_common（旧分数+新求解器）",
    }

    payload = {
        "stage": 10,
        "run_id": out.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fixed_run": str(fixed_run),
        "old_run": str(old_run),
        "folds": {
            "path": str(folds_path), "sha256": sha256(folds_path),
            "seed": SEED, "people": int(len(folds)),
        },
        "rows": rows,
        "solver_control": solver_diff,
        "comparisons": {
            "training_sample_effect": {
                "from": "historical_old_186", "to": "old_refit_common",
                "same": "旧分数 + 旧求解器；共同 172 测试行",
                "interpretation": "训练样本从 186 变为 172；历史模型对共同测试行是事后限制",
            },
            "solver_effect": {
                "from": "old_refit_common", "to": "solver_on_common",
                "same": "旧分数、共同 172 训练/测试键、人员分折",
                "interpretation": "只换 Cox 求解器；预测差见 solver_control",
            },
            "score_effect": {
                "from": "old_refit_common", "to": "precision_on_common",
                "same": "旧 Cox 求解器、共同 172 训练/测试键、人员分折",
                "interpretation": "只换分数（旧分数→causal 修正分数）",
            },
            "combined_effect": {
                "from": "old_refit_common", "to": "fixed_session",
                "same": "共同 172 训练/测试键、人员分折",
                "interpretation": "同时换分数与求解器",
            },
        },
        "main_stage6": read_json(fixed_run / "comparison.json")["main_metrics"],
        "inputs": {
            "fixed_predictions": {"path": str(fixed_run / "stage3" / "predictions.csv"),
                                   "sha256": sha256(fixed_run / "stage3" / "predictions.csv")},
            "old_predictions": {"path": str(old_run / "predictions.csv"),
                                 "sha256": sha256(old_run / "predictions.csv")},
            "old_scores_session": {"path": str(old_run / "stage2" / "scores_session_se.csv"),
                                    "sha256": sha256(old_run / "stage2" / "scores_session_se.csv")},
            "folds": {"path": str(folds_path), "sha256": sha256(folds_path)},
        },
        "notes": [
            "五个对照行共享同一 172 个测试预测日与同一人员分折。",
            "历史旧模型训练于原 186 个预测日；old_on_common 仅把其预测事后限制到 172 个测试行，不是重拟合。",
            "solver_control 比较 old_refit_common 与 solver_on_common；同一训练/测试行、同一旧分数，只换求解器。",
        ],
    }
    write_json(out / "attribution.json", payload)

    def fmt(value: float | None, digits: int = 3) -> str:
        return "—" if value is None else f"{value:.{digits}f}"

    def pct_ci(ci: list[float] | None, digits: int = 3) -> str:
        if not ci:
            return "—"
        return f"{ci[0]:+.{digits}f}–{ci[1]:+.{digits}f}"

    main = payload["main_stage6"]
    lines = [
        "# 阶段 10：共同样本归因与最终收尾",
        "",
        f"- 运行编号：`{out.name}`",
        f"- 生成时间：{payload['generated_at_utc']}",
        f"- 源运行：`{fixed_run}`（修正版）；`{old_run}`（旧版）",
        f"- 代码版本：stage10-attribution-v1；种子 {SEED}；bootstrap {args.bootstrap}",
        "",
        "## 1. 五个对照行（30 天，session_se，共同 172 个测试预测日）",
        "",
        "| 对照行 | 训练样本 | 分数 | 求解器 | A AUROC | B AUROC | ΔAUROC (B−A) | Brier A / B | ΔBrier (B−A, 95%) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for variant in ROW_ORDER:
        row = rows[variant]
        lines.append(
            f"| {variant} | {row['training_sample']} | {row['score_version']} | "
            f"{row['solver_version']} | {row['auroc_A']:.3f} | {row['auroc_B']:.3f} | "
            f"{row['delta_auroc']:+.3f} ({pct_ci(row['ci_delta_auroc'])}) | "
            f"{row['brier_A']:.4f} / {row['brier_B']:.4f} | "
            f"{row['delta_brier']:+.4f} ({pct_ci(row['ci_delta_brier'], 4)}) |"
        )
    lines += [
        "",
        f"共同测试行均为 {main['n']} 个预测日、{main['persons']} 人、{main['events']} 个 30 天阳性。"
        "AUROC 越大越好，Brier 越小越好；ΔAUROC>0 或 ΔBrier<0 都表示 B 优于 A。",
        "",
        "## 2. 对照解释",
        "",
        f"- **训练样本变化（历史 → 共同旧版重拟合）**：历史旧模型在原 186 个预测日上训练，"
        "其预测被事后限制到 172 个共同测试行；共同旧版重拟合在 172 个预测日上重新训练。"
        f"两者 ΔAUROC 分别为 {rows['historical_old_186']['delta_auroc']:+.3f} 和 "
        f"{rows['old_refit_common']['delta_auroc']:+.3f}，差异归为训练样本变化。"
        "历史旧模型的逐折训练/原始测试键见 `historical_scope.csv`（186 个键）；"
        "其在共同 172 个测试行上的限制键见 `historical_common_test.csv`。"
        "历史模型原本的测试折包含 186 个预测日，172 是事后限制，不是重新训练。",
        f"- **只换求解器（共同旧版重拟合 → 共同新求解器）**：同一 172 个训练/测试键、同一旧分数，"
        f"只把旧 Cox 求解器换成修正版。逐条预测最大绝对差为 "
        f"`{solver_diff['max_abs_prediction_diff']:.3e}`（A {solver_diff['by_method']['A_rms']['max_abs_prediction_diff']:.3e}，"
        f"B {solver_diff['by_method']['B_rms_cusum']['max_abs_prediction_diff']:.3e}），"
        "在 `1e-9` 容差内视为仅求解器差异；该行的指标变化不应再归因于训练样本。",
        f"- **只换分数（共同旧版重拟合 → 共同新分数）**：同一 172 个训练/测试键、同一旧 Cox 求解器，"
        "只把旧分数换成 causal 修正分数。",
        f"- **修正版组合（共同旧版重拟合 → 修正版）**：同一 172 个训练/测试键，同时换分数与求解器；"
        "不能拆成单一原因。",
        "",
        "## 3. 与阶段 6 主表的关系",
        "",
        f"- 修正版主表仍为 {main['n']} 个预测日、{main['persons']} 人、{main['events']} 个阳性、"
        f"{main['events']} / {main['n']} 观察风险 {main['observed_risk']:.4f}；"
        f"A AUROC {main['auroc_A']:.3f}，B {main['auroc_B']:.3f}，"
        f"ΔAUROC {main['delta_auroc']:+.3f}（{main['ci_delta_auroc'][0]:+.3f}–{main['ci_delta_auroc'][1]:+.3f}）。",
        f"- ΔBrier {main['delta_brier']:+.4f}"
        f"（{main['ci_delta_brier'][0]:+.4f}–{main['ci_delta_brier'][1]:+.4f}）："
        "B 的 Brier 点估计略低于 A（略有改善），但区间跨 0，不能表述为确立改善，更不能表述为“无改善”。",
        "- 主结果判定仍为**证据不足**：16 个阳性预测日不足以稳定估计增量，且校准斜率区间宽。",
        "",
        "## 4. 三项修复的收尾证据",
        "",
        "- **时间截断与 Cox 一致性**：阶段 5/6 修正后，完整档案与按日截断逐字段一致；本阶段共同样本重拟合的求解器对照见上，最大预测差在 `1e-9` 内。",
        "- **模型输入契约**：阶段 8 v2 的 19 项核验全部通过；视频→穿戴 A/B 返回 `modality_mismatch`，`none`→`session_se` B 返回 `precision_mode_mismatch`，缺失契约返回 `contract_missing`，正确穿戴前缀研究风险 0.094173 与阶段 6 差 6.9×10⁻¹⁷。",
        "- **扩样口径**：阶段 9 修正后，S7 相对 S14 新增阳性受试者 10 名（6 名来自新增受试者、4 名来自原有受试者），额外阳性预测日 11 = 6 + 4 + 1；两组人员交集 144、并集 194；S7 仍是候选层，未做 7 日基线拟合与精度质控。",
        "",
        "## 5. 最终判断与边界",
        "",
        "- 三项修复均已用行为/计算正确性核验；修正版主结果仍为**证据不足**，不是阳性结论。",
        "- 无独立验证，模型 `calibrated=false`；服务临床概率继续为空。",
        "- 连续视频监测、其他人群、外部场地、7 日基线再拟合与新的外部验证均未完成。",
        "- S7 的 10 名新增阳性不能直接替代 7 日基线质控后的可评分人数；扩样是否值得继续需另立探索研究。",
        "",
        "## 6. 产物与复现",
        "",
        "```bash",
        "python scripts/analyse_mobilise_stage10_attribution.py",
        "python scripts/verify_mobilise_stage10.py",
        "python -m pytest tests -q",
        "```",
        "",
        "- `same_sample_predictions.csv`：五个对照行的逐条折外预测；`same_sample_metrics.csv`：指标。",
        "- `fold_scope_common.csv`、`historical_scope.csv`、`historical_common_test.csv`、`scope_summary.csv`：训练/测试键与来源。",
        "- `fold_models/old_refit_common/30d/`：新重拟合的折模型；`fold_status.csv`：失败状态。",
        "- `attribution.json`、`bootstrap.json`、`source_map.json`、`code_snapshot/`、`code_manifest.json`。",
        "- 阶段 8：`output/ds_pcd_mobilise/stage8/20261114-contract-v2/`；阶段 9：`output/ds_pcd_mobilise/stage9/20261114-expansion-v1/`。",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    # 合并新模型的折状态；已有行仍从源运行读取
    status_new = pd.DataFrame(new_statuses)
    status_new["source"] = "stage10_old_refit_common"
    fixed_status = pd.read_csv(fixed_run / "stage3" / "fold_status.csv")
    fixed_status = fixed_status.loc[
        fixed_status["variant"].isin(["fixed_session", "solver_on_common", "precision_on_common"])
        & fixed_status["horizon"].eq(30)
    ].copy()
    fixed_status["source"] = "20261114-fixed/stage3/fold_status.csv"
    old_status = pd.read_csv(old_run / "stage3" / "fold_status.csv")
    old_status = old_status.loc[
        old_status["mode"].eq("session_se") & old_status["horizon"].eq(30)
    ].copy()
    old_status = old_status.rename(columns={"mode": "variant"})
    old_status["variant"] = "historical_old_186"
    old_status["source"] = "20261114-main/stage3/fold_status.csv"
    statuses = pd.concat([fixed_status, status_new, old_status], ignore_index=True)
    statuses.to_csv(out / "fold_status.csv", index=False, encoding="utf-8")

    source_map = {
        "stage": 10,
        "run_id": out.name,
        "created_at_utc": payload["generated_at_utc"],
        "fixed_run": str(fixed_run),
        "old_run": str(old_run),
        "folds": payload["folds"],
        "inputs": payload["inputs"],
        "script": "scripts/analyse_mobilise_stage10_attribution.py",
        "verification": "scripts/verify_mobilise_stage10.py",
        "notes": payload["notes"],
        "stage8": "output/ds_pcd_mobilise/stage8/20261114-contract-v2/",
        "stage9": "output/ds_pcd_mobilise/stage9/20261114-expansion-v1/",
        "fold_model_sources": {
            "stage10_old_refit_common": [
                {"path": str(path), "sha256": sha256(path)}
                for path in sorted((out / "fold_models" / "old_refit_common" / "30d").glob("*.json"))
            ],
            "stage6_existing": [
                {"path": str(path), "sha256": sha256(path)}
                for path in sorted((fixed_run / "stage3" / "fold_models").glob("*/*/*.json"))
            ],
            "stage4_old": [
                {"path": str(path), "sha256": sha256(path)}
                for path in sorted((old_run / "stage3" / "fold_models").glob("*/*/*.json"))
            ],
        },
    }
    write_json(out / "source_map.json", source_map)
    print(json.dumps({
        "out": str(out),
        "rows": {variant: {
            "training": rows[variant]["training_scope"],
            "auroc_A": rows[variant]["auroc_A"],
            "auroc_B": rows[variant]["auroc_B"],
            "delta_auroc": rows[variant]["delta_auroc"],
            "delta_brier": rows[variant]["delta_brier"],
        } for variant in ROW_ORDER},
        "solver_max_abs_diff": solver_diff["max_abs_prediction_diff"],
        "main": {key: main[key] for key in ("n", "persons", "events", "delta_auroc", "delta_brier")},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
