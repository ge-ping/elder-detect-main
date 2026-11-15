"""阶段 14 补充：轨迹分层与更大的半合成训练预算敏感性。

两部分：

1. **轨迹分层**（预注册情景维度）：把 S 分支的 M3−M2 按 ``slow_linear`` /
   ``rapid_step`` / ``stable`` / ``transient`` / ``reverse`` 分层，直接回答
   “终点型统计量在突然恶化时是否比块均值更有信息”；
2. **训练预算敏感性**（事后、明确标注）：把 S-only 的合成训练序列从
   600/单元提高到 3,000/单元，检验主表的“无增量”是信息不足还是检验力不足。
   这一项不替换主表，也不改变任何冻结规格。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from round2_stage14_evaluate import (  # noqa: E402
    CONDITIONS, LADDERS, MECHANISMS, NOISE_FAMILIES, SEED, _bootstrap_delta,
    _fit_and_predict, _metrics, conclusion_state,
)
from mobilise_d.round2 import R2_FEATURES  # noqa: E402
from mobilise_d.round2_scoring import (  # noqa: E402
    PersonSeries, r2_ds_pcd_config, score_person, level_and_change,
)
from mobilise_d.round2_synth import (  # noqa: E402
    GeneratorConfig, GeneratorParams, person_records, simulate_batch,
)

OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise" / "round2"
STAGE11 = OUTPUT_ROOT / "stage11" / "20261115-protocol-v1"
STAGE12 = OUTPUT_ROOT / "stage12" / "20261115-supplement-v1"
STAGE13 = OUTPUT_ROOT / "stage13" / "20261115-freeze-v2"
BOOTSTRAP = 2000
LARGE_TRAIN = 3000


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")


def trajectory_breakdown(paired_path: Path) -> pd.DataFrame:
    paired = pd.read_csv(paired_path, low_memory=False)
    subset = paired.loc[
        (paired["branch"] == "S") & (paired["ablation"] == "full")
        & paired["trajectory"].isin(("slow_linear", "rapid_step", "stable", "transient", "reverse"))
    ].copy()
    wide = subset.pivot_table(
        index=["key", "label", "condition", "horizon_days", "mechanism", "noise_family", "trajectory"],
        columns="model", values="prediction", aggfunc="first",
    ).reset_index()
    rows = []
    for keys, group in wide.groupby(
        ["condition", "horizon_days", "mechanism", "trajectory"]
    ):
        condition, horizon, mechanism, trajectory = keys
        group = group.dropna(subset=["M2", "M3", "label"])
        if len(group) < 50:
            continue
        delta = _bootstrap_delta(
            group["label"].to_numpy(), group["M2"].to_numpy(), group["M3"].to_numpy(),
            group["key"].astype(str).to_numpy(), samples=BOOTSTRAP // 2,
            seed=SEED + int(horizon),
        )
        row = {
            "condition": condition, "horizon_days": int(horizon),
            "mechanism": mechanism, "trajectory": trajectory,
            "n": int(len(group)), "positives": int(group["label"].sum()), **delta,
        }
        met2 = _metrics(group["label"].to_numpy(), group["M2"].to_numpy())
        met3 = _metrics(group["label"].to_numpy(), group["M3"].to_numpy())
        row.update({
            "auroc_M2": met2["auroc"], "auroc_M3": met3["auroc"],
            "brier_M2": met2["brier"], "brier_M3": met3["brier"],
        })
        row.update(conclusion_state(row))
        rows.append(row)
    return pd.DataFrame(rows)


def large_budget(paired_path: Path, stage12: Path, stage13: Path, run_dir: Path) -> pd.DataFrame:
    diagnostics = json.loads((stage13 / "training_diagnostics.json").read_text(encoding="utf-8"))
    config = GeneratorConfig()
    ds_config = r2_ds_pcd_config(precision_mode="session_se")
    test = pd.read_csv(stage12 / "synthetic_test_scores.csv")
    test = test.loc[test["status"] == "scored"].copy()
    gap = pd.to_numeric(test["event_gap_from_snapshot"], errors="coerce")
    test["duration"] = np.where((gap > 0) & (gap <= 7), gap, 7.0)
    test["event"] = ((gap > 0) & (gap <= 7)).astype(float)
    test["key"] = test["series_id"]
    rows = []
    for fold in (0, 1, 2):
        mean, sd = (
            float(diagnostics["folds"][str(fold)]["real"]["age_mean"]),
            float(diagnostics["folds"][str(fold)]["real"]["age_sd"]),
        )
        params = GeneratorParams.from_dict(
            json.loads((stage12 / f"generator_params_fold{fold}.json").read_text(encoding="utf-8"))["params"]
        )
        train_rows = []
        for mechanism in MECHANISMS:
            for noise in NOISE_FAMILIES:
                offset = MECHANISMS.index(mechanism) * 10 + NOISE_FAMILIES.index(noise)
                print(f"[sensitivity] fold{fold} {mechanism}|{noise} train={LARGE_TRAIN}", flush=True)
                _, series = simulate_batch(
                    params, count=LARGE_TRAIN, fold=fold, mechanism=mechanism,
                    noise_family=noise, seed=SEED + 1000 * fold + 100 + offset,
                    config=config, prefix="train_large", keep_series=True,
                    lambda0=params.lambda0.get(f"{mechanism}|{noise}"),
                )
                for person in series:
                    records = person_records(person, R2_FEATURES)
                    observed = np.asarray(person["observed"], dtype=float)
                    se = np.asarray(person["se"], dtype=float)
                    series_obj = PersonSeries(
                        person_id=person["series_id"],
                        days=tuple(record.date for record in records),
                        features={name: tuple(observed[:, i]) for i, name in enumerate(R2_FEATURES)},
                        feature_se={name: tuple(se[:, i]) for i, name in enumerate(R2_FEATURES)},
                        source="synthetic", meta={},
                    )
                    snapshot_day = records[config.baseline_days + config.monitoring_days - 1].date
                    result = score_person(
                        series_obj, ds_config, snapshot_day=snapshot_day,
                        baseline_dates=[r.date for r in records[: config.baseline_days]],
                    )
                    if result.get("status") != "scored":
                        continue
                    covariates = level_and_change(
                        series_obj, list(range(config.baseline_days)),
                        list(range(config.baseline_days, config.baseline_days + config.monitoring_days)),
                    )
                    gap_value = person["event_gap_from_snapshot"]
                    train_rows.append({
                        "series_id": person["series_id"],
                        "fold": fold,
                        "mechanism": mechanism,
                        "noise_family": noise,
                        "trajectory": person["trajectory"],
                        "age": person["age"],
                        "prior_fall_any": person["prior_fall_any"],
                        "T": result["T"],
                        "C": result["C"],
                        **covariates,
                        "duration": (
                            float(gap_value) if gap_value is not None and 0 < gap_value <= 7 else 7.0
                        ),
                        "event": float(gap_value is not None and 0 < gap_value <= 7),
                        "y_7d": float(gap_value is not None and 0 < gap_value <= 7),
                    })
        train_frame = pd.DataFrame(train_rows)
        train_frame.to_csv(run_dir / f"sensitivity_train_scores_fold{fold}.csv.gz",
                           index=False, compression="gzip")
        fold_test = test.loc[test["fold"] == fold]
        for model in ("M0", "M1", "M2", "M3"):
            fit, predictions, info = _fit_and_predict(
                train_real=train_frame.iloc[0:0], train_synthetic=train_frame,
                test=fold_test, features=LADDERS[model], condition="S-only",
                fold=fold, bundle=f"sensitivity-fold{fold}-{model}",
                age_mean=mean, age_sd=sd,
            )
            frame = fold_test[["key", "y_7d", "mechanism", "noise_family"]].copy()
            frame["prediction"] = predictions[7]
            frame["model"] = model
            frame["fold"] = fold
            frame["status"] = fit.status
            frame["train_rows"] = info["train_rows"]
            frame["train_events"] = info["train_events"]
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    stage12, stage13 = Path(args.stage12), Path(args.stage13)
    print("[stage14-sensitivity] 轨迹分层", flush=True)
    trajectory = trajectory_breakdown(run_dir / "paired_predictions.csv")
    trajectory.to_csv(run_dir / "comparison_by_trajectory.csv", index=False, encoding="utf-8")
    print("[stage14-sensitivity] 更大训练预算", flush=True)
    large = large_budget(run_dir / "paired_predictions.csv", stage12, stage13, run_dir)
    large.to_csv(run_dir / "power_sensitivity_predictions.csv", index=False, encoding="utf-8")
    rows = []
    for (mechanism, noise), group in large.groupby(["mechanism", "noise_family"]):
        wide = group.pivot_table(index=["key", "y_7d"], columns="model",
                                 values="prediction", aggfunc="first").reset_index()
        wide = wide.dropna(subset=["M2", "M3", "y_7d"])
        delta = _bootstrap_delta(
            wide["y_7d"].to_numpy(), wide["M2"].to_numpy(), wide["M3"].to_numpy(),
            wide["key"].astype(str).to_numpy(), samples=args.bootstrap, seed=SEED,
        )
        row = {
            "condition": "S-only", "scenario": f"{mechanism}|{noise}",
            "mechanism": mechanism, "noise_family": noise,
            "n": int(len(wide)), "positives": int(wide["y_7d"].sum()),
            "train_rows": int(group["train_rows"].iloc[0]),
            "train_events": int(group["train_events"].iloc[0]),
            **delta,
        }
        row.update(conclusion_state(row))
        rows.append(row)
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(run_dir / "power_sensitivity.csv", index=False, encoding="utf-8")
    _write_json(run_dir / "sensitivity_manifest.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trajectory_breakdown": {
            "status": "pre-specified scenario dimension",
            "file": "comparison_by_trajectory.csv",
            "bootstrap": BOOTSTRAP // 2,
        },
        "power_sensitivity": {
            "status": "post-hoc sensitivity; does not replace the frozen primary table",
            "reason": "主表 S-only 训练预算 600/单元（每折约 76 个事件），需要区分信息不足与检验力不足",
            "train_per_cell": LARGE_TRAIN,
            "file": "power_sensitivity.csv",
            "bootstrap": args.bootstrap,
        },
        "seed": SEED,
    })
    print(f"[stage14-sensitivity] 完成：{run_dir}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 14 轨迹分层与预算敏感性")
    parser.add_argument("--run-dir", default=str(
        OUTPUT_ROOT / "stage14" / "20261115-eval-v1"
    ))
    parser.add_argument("--stage12", default=str(STAGE12))
    parser.add_argument("--stage13", default=str(STAGE13))
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
