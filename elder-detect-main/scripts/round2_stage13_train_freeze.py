"""阶段 13：完成公平训练并冻结待评模型。

* 对照阶梯 M0–M3 固定（C7），主比较 M3−M2；
* 训练条件：R-only（仅真实训练行）、R+S（真实＋半合成，合成总权重 0.5×真实）、
  S-only（仅半合成，只在 S 分支独立报告）；
* 所有模型同一家族：加权 Breslow Landmark Cox、固定岭、7 天截断资料拟合，
  1/3/7 天由同一基线生存阶梯查询；
* 只使用各折训练/开发部分选择与拟合，不计算任何外层测试性能；
* 冻结后保存清单、系数、基线阶梯、权重与代码/配置快照。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import md5
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.cox_horizon import (  # noqa: E402
    DEFAULT_HORIZONS, fit_horizon_cox, predict_horizon_risk,
)
from mobilise_d.round2 import R2_FEATURES  # noqa: E402
from mobilise_d.round2_features import impute_scattered_missing  # noqa: E402
from mobilise_d.round2_scoring import PersonSeries  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise" / "round2"
STAGE11 = OUTPUT_ROOT / "stage11" / "20261115-protocol-v1"
STAGE12 = OUTPUT_ROOT / "stage12" / "20261115-supplement-v1"
SEED = 20261115
SYNTHETIC_WEIGHT_SHARE = 0.5
RIDGE = 1e-4

LEVEL_COLUMNS = tuple(f"level_{name}" for name in R2_FEATURES)
CHANGE_COLUMNS = tuple(f"change_{name}" for name in R2_FEATURES)
LADDERS: dict[str, tuple[str, ...]] = {
    "M0": ("age_z", "prior_fall_any"),
    "M1": ("age_z", "prior_fall_any") + LEVEL_COLUMNS,
    "M2": ("age_z", "prior_fall_any") + LEVEL_COLUMNS + CHANGE_COLUMNS,
    "M3": ("age_z", "prior_fall_any") + LEVEL_COLUMNS + CHANGE_COLUMNS + ("T", "C"),
}
CONDITIONS = ("R-only", "R+S", "S-only")

CODE_SNAPSHOT_FILES = (
    "longitudinal/cox_horizon.py",
    "mobilise_d/round2.py",
    "mobilise_d/round2_scoring.py",
    "mobilise_d/round2_features.py",
    "mobilise_d/round2_synth.py",
    "longitudinal/ds_pcd.py",
    "scripts/round2_stage13_train_freeze.py",
    "scripts/verify_round2_stage13.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 13：训练并冻结模型")
    parser.add_argument("--run-id", default="20261115-freeze-v1")
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


def _truncated_outcome(frame: pd.DataFrame, *, event_gap: str, horizon: int = 7) -> pd.DataFrame:
    """把事件间隔转成 7 天截断的 duration/event。"""
    gap = pd.to_numeric(frame[event_gap], errors="coerce")
    event = ((gap > 0) & (gap <= horizon)).astype(int)
    duration = np.where(event == 1, gap.fillna(horizon), float(horizon))
    out = frame.copy()
    out["duration"] = duration.astype(float)
    out["event"] = event
    return out


def build_real_matrix(
    real_scores: pd.DataFrame,
    real_candidates: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    fold: int,
) -> tuple[pd.DataFrame, dict]:
    """真实候选的模型输入（含折内 age 标准化与 7 天截断结局）。"""
    frame = real_scores.loc[real_scores["status"] == "scored"].copy()
    frame = frame.merge(
        real_candidates[[
            "participantid", "landmark_day", "first_event_7d", "y_7d_num",
        ]].rename(columns={"y_7d_num": "y_7d_checked"}),
        on=["participantid", "landmark_day"], how="left", validate="one_to_one",
    )
    frame = frame.merge(splits[["participantid", "fold"]], on="participantid",
                        how="left", validate="many_to_one")
    landmark = pd.to_datetime(frame["landmark_day"])
    gap = (pd.to_datetime(frame["first_event_7d"]) - landmark).dt.days
    frame["event_gap"] = np.where(frame["y_7d"].astype(float) == 1, gap, np.nan)
    # 用真实首次事件日重算，而不是相信标签列
    frame["event_gap"] = np.where(
        pd.notna(frame["first_event_7d"]), gap,
        np.where(frame["y_7d"].astype(float) == 1, 1.0, np.nan),
    )
    frame = _truncated_outcome(frame, event_gap="event_gap")
    train = frame.loc[frame["fold"] != fold]
    age_mean = float(train["age"].mean())
    age_sd = float(train["age"].std(ddof=0)) or 1.0
    frame["age_z"] = (frame["age"].astype(float) - age_mean) / age_sd
    frame["source"] = "real"
    frame["weight"] = 1.0
    diagnostics = {
        "rows": int(len(frame)),
        "train_rows": int((frame["fold"] != fold).sum()),
        "test_rows": int((frame["fold"] == fold).sum()),
        "events_train": int(frame.loc[frame["fold"] != fold, "event"].sum()),
        "events_test": int(frame.loc[frame["fold"] == fold, "event"].sum()),
        "age_mean": age_mean,
        "age_sd": age_sd,
        "label_mismatch": int(
            (frame["y_7d"].astype(float).fillna(-1) != frame["y_7d_checked"].astype(float).fillna(-2)).sum()
        ),
    }
    return frame, diagnostics


def build_synthetic_matrix(
    synthetic: pd.DataFrame, *, fold: int, age_mean: float, age_sd: float
) -> pd.DataFrame:
    """半合成行的模型输入与 7 天截断结局。"""
    frame = synthetic.loc[synthetic["fold"] == fold].copy()
    frame = _truncated_outcome(frame, event_gap="event_gap_from_snapshot")
    frame["age_z"] = (frame["age"].astype(float) - age_mean) / age_sd
    frame["source"] = "synthetic"
    frame["weight"] = 1.0
    return frame


def _weights_for_condition(
    real_train: pd.DataFrame, synthetic_train: pd.DataFrame, condition: str
) -> tuple[pd.DataFrame, dict]:
    """按训练条件给行赋权；合成总权重 = 0.5×真实总权重。"""
    if condition == "R-only":
        return real_train.copy(), {
            "real_rows": int(len(real_train)),
            "real_weight": float(len(real_train)),
            "synthetic_rows": 0,
            "synthetic_weight": 0.0,
            "weight_per_synthetic": None,
        }
    if condition == "S-only":
        frame = synthetic_train.copy()
        frame["weight"] = 1.0
        return frame, {
            "real_rows": 0,
            "real_weight": 0.0,
            "synthetic_rows": int(len(frame)),
            "synthetic_weight": float(len(frame)),
            "weight_per_synthetic": 1.0,
        }
    if condition != "R+S":
        raise ValueError(f"未知训练条件：{condition}")
    real = real_train.copy()
    synthetic = synthetic_train.copy()
    real_weight = float(real["weight"].sum())
    total_synthetic_weight = SYNTHETIC_WEIGHT_SHARE * real_weight
    per_person = total_synthetic_weight / max(len(synthetic), 1)
    synthetic["weight"] = per_person
    frame = pd.concat([real, synthetic], ignore_index=True, sort=False)
    return frame, {
        "real_rows": int(len(real)),
        "real_weight": real_weight,
        "synthetic_rows": int(len(synthetic)),
        "synthetic_weight": float(per_person * len(synthetic)),
        "weight_per_synthetic": float(per_person),
        "family_normalisation": (
            "每条虚拟个体都是独立抽样，没有复制行；因此每条权重相同，"
            "总合成权重固定为真实训练总权重的 0.5"
        ),
    }


def _complete(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """保留协变量全部有限的行使；缺协变量的键单独计数，不插补。"""
    if not len(frame):
        return frame.copy()
    mask = np.isfinite(frame[list(features)].to_numpy(dtype=float)).all(axis=1)
    return frame.loc[mask].copy()


def _model_matrix(frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(f"输入缺少模型特征：{missing}")
    matrix = frame[list(features)].to_numpy(dtype=float)
    if np.any(~np.isfinite(matrix)):
        raise ValueError("模型输入含非有限数值")
    return matrix


def _fit_one(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    fold: int,
    model: str,
    condition: str,
    bundle_id: str,
) -> tuple[Any, dict]:
    matrix = _model_matrix(frame, features)
    fit = fit_horizon_cox(
        frame["duration"].to_numpy(dtype=float),
        frame["event"].to_numpy(dtype=int),
        matrix,
        features,
        horizon_days=7.0,
        ridge=RIDGE,
        weights=frame["weight"].to_numpy(dtype=float),
        horizons_days=DEFAULT_HORIZONS,
    )
    metadata = {
        "bundle_id": bundle_id,
        "fold": int(fold),
        "model": model,
        "condition": condition,
        "features": list(features),
        "rows": int(len(frame)),
        "events": int(frame["event"].sum()),
        "total_weight": float(frame["weight"].sum()),
        "synthetic_rows": int((frame["source"] == "synthetic").sum()),
        "real_rows": int((frame["source"] == "real").sum()),
        "ridge": RIDGE,
        "horizon_days": list(DEFAULT_HORIZONS),
        "fit_horizon": 7.0,
        "outcome": "(t, t+h] 内至少一次跌倒；7 天截断资料拟合，1/3/7 天同曲线查询",
    }
    fit = type(fit)(**{**fit.__dict__, "notes": {**fit.notes, **metadata}})
    return fit, metadata


def run(args: argparse.Namespace) -> int:
    run_dir = Path(args.out_root) / f"stage13/{args.run_id}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"运行目录已存在且非空，拒绝覆盖：{run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    stage11 = Path(args.stage11)
    stage12 = Path(args.stage12)
    real_scores = pd.read_csv(stage12 / "real_scores.csv", dtype={"participantid": str})
    synthetic_train = pd.read_csv(stage12 / "synthetic_train_scores.csv")
    synthetic_test = pd.read_csv(stage12 / "synthetic_test_scores.csv")
    synthetic_stress = pd.read_csv(stage12 / "synthetic_stress_scores.csv")
    real_candidates = pd.read_csv(stage11 / "real_candidates.csv", dtype={"participantid": str})
    splits = pd.read_csv(stage11 / "splits.csv", dtype={"participantid": str})
    protocol = json.loads((stage11 / "protocol.json").read_text(encoding="utf-8"))
    folds = sorted(splits["fold"].unique())

    model_records: list[dict] = []
    predictions: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {"folds": {}, "imputation": {}}
    for fold in folds:
        fold = int(fold)
        real_frame, real_diag = build_real_matrix(
            real_scores, real_candidates, splits, fold=fold
        )
        real_train_all = real_frame.loc[real_frame["fold"] != fold].copy()
        real_test_all = real_frame.loc[real_frame["fold"] == fold].copy()
        # 主比较的共同测试键：M3 协变量齐全的预测日；所有 M 模型都用同一批键
        real_test_common = _complete(real_test_all, LADDERS["M3"])
        real_train = real_train_all
        real_test = real_test_common
        # 训练侧插补：只用于训练行的块均值/变化量（不进入真实主表）
        imputed_rows = 0
        for row in real_train.itertuples(index=False):
            if any(not math.isfinite(float(getattr(row, column))) for column in
                   LEVEL_COLUMNS + CHANGE_COLUMNS):
                imputed_rows += 1
        age_mean, age_sd = real_diag["age_mean"], real_diag["age_sd"]
        synthetic_fold = build_synthetic_matrix(
            synthetic_train, fold=fold, age_mean=age_mean, age_sd=age_sd
        )
        synthetic_test_fold = build_synthetic_matrix(
            synthetic_test, fold=fold, age_mean=age_mean, age_sd=age_sd
        )
        synthetic_test_common = _complete(synthetic_test_fold, LADDERS["M3"])
        diagnostics["folds"][str(fold)] = {
            "real": real_diag,
            "synthetic_train_rows": int(len(synthetic_fold)),
            "real_train_rows_with_missing_block_covariate": imputed_rows,
            "real_test_rows_all": int(len(real_test_all)),
            "real_test_rows_common": int(len(real_test_common)),
            "real_test_rows_dropped_missing_covariate": int(len(real_test_all) - len(real_test_common)),
            "real_train_rows_dropped_missing_covariate_M3": int(
                len(real_train_all) - len(_complete(real_train_all, LADDERS["M3"]))
            ),
            "synthetic_test_rows_all": int(len(synthetic_test_fold)),
            "synthetic_test_rows_common": int(len(synthetic_test_common)),
            "synthetic_train_rows_complete_M3": int(
                len(_complete(synthetic_fold, LADDERS["M3"]))
            ),
        }
        for model, features in LADDERS.items():
            for condition in CONDITIONS:
                train_complete = _complete(real_train, features)
                synthetic_train_complete = _complete(synthetic_fold, features)
                frame, weight_info = _weights_for_condition(
                    train_complete, synthetic_train_complete, condition
                )
                bundle_id = f"round2-fold{fold}-{model}-{condition}"
                fit, metadata = _fit_one(
                    frame, features, fold=fold, model=model, condition=condition,
                    bundle_id=bundle_id,
                )
                model_dir = run_dir / "fold_models" / f"fold{fold}" / condition / model
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / "model.json").write_text(
                    json.dumps(fit.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
                )
                record = {
                    "bundle_id": bundle_id,
                    "fold": fold,
                    "model": model,
                    "condition": condition,
                    "status": fit.status,
                    "converged": bool(fit.converged),
                    "rows": int(len(frame)),
                    "events": int(frame["event"].sum()),
                    "effective_events": fit.effective_events,
                    "total_weight": fit.total_weight,
                    "active_features": list(fit.active_features),
                    "coefficients": dict(zip(fit.feature_names, fit.coefficients)),
                    "baseline_times": list(fit.baseline_times),
                    "baseline_survival_tail": (
                        fit.baseline_survival[-1] if fit.baseline_survival else None
                    ),
                    "degenerate_horizons": fit.notes.get("degenerate_horizons_days"),
                    "weights": weight_info,
                    "model_path": str((model_dir / "model.json").relative_to(run_dir)),
                }
                model_records.append(record)
                # 保存折外预测，但本阶段不计算任何指标
                for label, subset in (("real_test", real_test),
                                      ("synthetic_test", synthetic_test_common)):
                    if not len(subset):
                        continue
                    usable = subset.copy()
                    missing = [
                        column for column in features if usable[column].isna().any()
                    ]
                    if missing:
                        usable = usable.loc[
                            ~usable[list(features)].isna().any(axis=1)
                        ].copy()
                    if not len(usable):
                        continue
                    matrix = _model_matrix(usable, features)
                    preds = {
                        f"p_{horizon}d": predict_horizon_risk(fit, matrix, horizon)
                        for horizon in DEFAULT_HORIZONS
                    }
                    out = pd.DataFrame({
                        "bundle_id": bundle_id,
                        "fold": fold,
                        "model": model,
                        "condition": condition,
                        "branch": "R" if label == "real_test" else "S",
                        "test_set": label,
                        **{
                            key: usable[key].to_numpy()
                            for key in (
                                "participantid", "landmark_day", "series_id", "age",
                                "prior_fall_any", "T", "C", "y_1d", "y_3d", "y_7d",
                                "event_gap", "event_gap_from_snapshot", "mechanism",
                                "noise_family", "trajectory",
                            ) if key in usable.columns
                        },
                        **preds,
                        "score_rows": int(len(usable)),
                        "dropped_missing_covariate": int(len(subset) - len(usable)),
                    })
                    predictions.append(out)
    prediction_frame = pd.concat(predictions, ignore_index=True, sort=False)
    prediction_frame.to_csv(run_dir / "oof_predictions.csv", index=False, encoding="utf-8")
    _write_json(run_dir / "model_registry.json", {
        "models": model_records,
        "ladders": {model: list(features) for model, features in LADDERS.items()},
        "conditions": list(CONDITIONS),
        "primary_comparison": "M3-M2",
        "ridge": RIDGE,
        "weight_share": SYNTHETIC_WEIGHT_SHARE,
        "horizons_days": list(DEFAULT_HORIZONS),
    })
    _write_json(run_dir / "training_diagnostics.json", diagnostics)

    # 冻结清单（记录冻结时未查看外层测试性能）
    freeze = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "outcome_metrics_computed": False,
        "note": (
            "阶段 13 只拟合并保存折外预测，未计算 AUROC/Brier/校准等任何外层测试指标；"
            "阶段 14 才在固定模型与固定测试范围上评价"
        ),
        "folds": folds,
        "models": sorted({record["model"] for record in model_records}),
        "conditions": list(CONDITIONS),
        "primary_comparison": "M3-M2",
        "primary_horizon_days": 7,
        "secondary_horizons_days": [1, 3],
        "status_counts": (
            pd.Series([record["status"] for record in model_records]).value_counts().to_dict()
        ),
        "model_fingerprints": {
            record["bundle_id"]: _model_fingerprint(run_dir / record["model_path"])
            for record in model_records
        },
        "input_files": {
            name: {"path": str(path), "md5": _md5(path)}
            for name, path in {
                "protocol": stage11 / "protocol.json",
                "real_scores": stage12 / "real_scores.csv",
                "synthetic_train": stage12 / "synthetic_train_scores.csv",
                "synthetic_test": stage12 / "synthetic_test_scores.csv",
                "synthetic_stress": stage12 / "synthetic_stress_scores.csv",
                "thresholds": stage12 / "threshold_calibration.json",
                "splits": stage11 / "splits.csv",
            }.items()
        },
        "test_scope": {
            "real": {
                "keys": int(prediction_frame.query(
                    "test_set == 'real_test' and model == 'M3' and condition == 'R-only'"
                ).shape[0]),
                "note": "固定为可评分且 M3 协变量齐全的真实预测日；所有 M 模型同一分母",
            },
            "synthetic": {
                "keys": int(prediction_frame.query(
                    "test_set == 'synthetic_test' and model == 'M3' and condition == 'R-only'"
                ).shape[0]),
                "note": "固定为可评分且 M3 协变量齐全的虚拟序列；所有 M 模型同一分母",
            },
        },
        "code_snapshot": list(CODE_SNAPSHOT_FILES),
        "seed": SEED,
        "reproduction": [
            "python scripts/round2_stage13_train_freeze.py --run-id " + args.run_id,
            "python scripts/verify_round2_stage13.py --run " + str(run_dir),
        ],
    }
    _write_json(run_dir / "freeze_manifest.json", freeze)
    snapshot = run_dir / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for relative in CODE_SNAPSHOT_FILES:
        source = PROJECT_ROOT / relative
        if source.is_file():
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    manifest = {
        "stage": 13,
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "upstream": {
            "stage11": str(stage11),
            "stage12": str(stage12),
            "protocol_md5": _md5(stage11 / "protocol.json"),
        },
        "outputs": sorted(
            str(path.relative_to(run_dir)) for path in run_dir.rglob("*") if path.is_file()
        ),
        "seed": SEED,
    }
    _write_json(run_dir / "manifest.json", manifest)
    print(f"[stage13] 完成：{run_dir}", flush=True)
    print(json.dumps(freeze["status_counts"], ensure_ascii=False))
    return 0


def _model_fingerprint(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("notes", {}).get("bundle_id", "") + "|" + md5(
        path.read_bytes()
    ).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
