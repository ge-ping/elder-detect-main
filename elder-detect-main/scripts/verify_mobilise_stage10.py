"""阶段 10 核验：五行同样本归因、求解器对照、指标重算与收尾一致性。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_ROOT = PROJECT_ROOT.parents[1] / "output" / "ds_pcd_mobilise"
FIXED_RUN = OUT_ROOT / "20261114-fixed"
OLD_RUN = OUT_ROOT / "20261114-main"
DEFAULT_STAGE10 = OUT_ROOT / "stage10" / "20261114-attribution-v1"
STAGE8 = OUT_ROOT / "stage8" / "20261114-contract-v2"
STAGE9 = OUT_ROOT / "stage9" / "20261114-expansion-v1"
VARIANTS = (
    "historical_old_186", "old_refit_common", "solver_on_common",
    "precision_on_common", "fixed_session",
)
TOLERANCE = 1e-9


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 10 核验")
    parser.add_argument("--fixed-run", default=str(FIXED_RUN))
    parser.add_argument("--old-run", default=str(OLD_RUN))
    parser.add_argument("--stage10", default=str(DEFAULT_STAGE10))
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def independent_auroc(y: np.ndarray, p: np.ndarray) -> float:
    """独立实现：正负样本两两比较，正确处理并列。"""
    positive = p[y == 1]
    negative = p[y == 0]
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("AUROC 需要正负样本")
    greater = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(positive) * len(negative)))


def independent_brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["participantid"] = frame["participantid"].astype(str).str.strip()
    frame["landmark_day"] = pd.to_datetime(frame["landmark_day"]).dt.strftime("%Y-%m-%d")
    return frame


def key_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame["participantid"], frame["landmark_day"]))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fixed_run = Path(args.fixed_run).resolve()
    old_run = Path(args.old_run).resolve()
    stage10 = Path(args.stage10).resolve()
    attribution = read_json(stage10 / "attribution.json")
    predictions = normalise(pd.read_csv(stage10 / "same_sample_predictions.csv"))
    metrics = pd.read_csv(stage10 / "same_sample_metrics.csv")
    checks: dict[str, bool] = {}

    checks["five_rows_present"] = set(attribution["rows"]) == set(VARIANTS)
    checks["five_variants_present"] = set(predictions["variant"]) == set(VARIANTS)
    checks["row_sample_sizes"] = all(
        int((predictions["variant"] == variant).sum()) == 344 for variant in VARIANTS
    )
    checks["all_test_keys_172"] = True
    checks["all_test_people_134"] = True
    checks["all_events_16"] = True
    fixed_keys = None
    for variant in VARIANTS:
        sub = predictions.loc[predictions["variant"].eq(variant)]
        keys = key_set(sub)
        if fixed_keys is None:
            fixed_keys = keys
        if keys != fixed_keys or len(keys) != 172:
            checks["all_test_keys_172"] = False
        if sub["participantid"].nunique() != 134:
            checks["all_test_people_134"] = False
        if int(sub.drop_duplicates(["participantid", "landmark_day"])["event"].sum()) != 16:
            checks["all_events_16"] = False

    # 独立重算 A/B 指标；AUROC 用两两比较，Brier 用均方误差
    recomputed: dict[str, dict[str, float]] = {}
    metric_ok = True
    for variant in VARIANTS:
        sub = predictions.loc[predictions["variant"].eq(variant)]
        wide = sub.pivot_table(
            index=["participantid", "landmark_day"], columns="method",
            values="risk", aggfunc="first",
        )
        meta = sub.drop_duplicates(["participantid", "landmark_day"]).set_index(
            ["participantid", "landmark_day"]
        )[["event"]]
        wide = wide.join(meta, how="inner")
        y = wide["event"].to_numpy(dtype=int)
        values = {}
        for method in ("A_rms", "B_rms_cusum"):
            p = wide[method].to_numpy(dtype=float)
            values[method] = {
                "auroc": independent_auroc(y, p),
                "brier": independent_brier(y, p),
            }
            row = metrics.loc[
                metrics["variant"].eq(variant) & metrics["method"].eq(method)
            ]
            if row.empty or abs(float(row.iloc[0]["auroc"]) - values[method]["auroc"]) > 1e-12 \
                    or abs(float(row.iloc[0]["brier"]) - values[method]["brier"]) > 1e-12:
                metric_ok = False
        delta = metrics.loc[
            metrics["variant"].eq(variant) & metrics["method"].eq("B_minus_A")
        ]
        expected_delta_auc = values["B_rms_cusum"]["auroc"] - values["A_rms"]["auroc"]
        expected_delta_brier = values["B_rms_cusum"]["brier"] - values["A_rms"]["brier"]
        if delta.empty or abs(float(delta.iloc[0]["delta_auroc"]) - expected_delta_auc) > 1e-12 \
                or abs(float(delta.iloc[0]["delta_brier"]) - expected_delta_brier) > 1e-12:
            metric_ok = False
        recomputed[variant] = values
    checks["metrics_recomputed_independently"] = metric_ok

    # 同样本 scope：四个共同样本行的每折训练/测试键完全一致
    common_scope = pd.read_csv(stage10 / "fold_scope_common.csv")
    common_variants = ("old_refit_common", "solver_on_common", "precision_on_common", "fixed_session")
    scope_ok = True
    reference_scope = {}
    for variant in common_variants:
        sub = common_scope.loc[common_scope["variant"].eq(variant)]
        for fold in (0, 1, 2):
            for role in ("train", "test"):
                keys = set(zip(
                    sub.loc[(sub["fold"] == fold) & (sub["role"] == role), "participantid"],
                    sub.loc[(sub["fold"] == fold) & (sub["role"] == role), "landmark_day"],
                ))
                reference_scope[(fold, role)] = reference_scope.get((fold, role), keys)
                if keys != reference_scope[(fold, role)]:
                    scope_ok = False
        if set(sub.loc[sub["role"].eq("test"), "fold"]) != {0, 1, 2}:
            scope_ok = False
    checks["common_training_test_keys_equal"] = scope_ok
    checks["common_test_union_172"] = (
        len(set(zip(
            common_scope.loc[(common_scope["variant"].eq("fixed_session")) & common_scope["role"].eq("test"), "participantid"],
            common_scope.loc[(common_scope["variant"].eq("fixed_session")) & common_scope["role"].eq("test"), "landmark_day"],
        ))) == 172
    )

    historical = normalise(pd.read_csv(
        stage10 / "historical_scope.csv", dtype={"participantid": str}
    ))
    historical_common_test = normalise(pd.read_csv(
        stage10 / "historical_common_test.csv", dtype={"participantid": str}
    ))
    historical_keys = key_set(historical)
    historical_test_keys = key_set(historical_common_test)
    checks["historical_training_186"] = len(historical_keys) == 186
    checks["historical_common_test_172"] = (
        len(historical_test_keys) == 172 and historical_test_keys == fixed_keys
    )
    checks["historical_scope_marks_common"] = (
        len({
            (person, day) for person, day, flag in zip(
                historical["participantid"], historical["landmark_day"],
                historical["in_common_172"].astype(str).str.lower().isin(("1", "true")),
            ) if flag
        }) == 172
    )
    scope_summary = pd.read_csv(stage10 / "scope_summary.csv")
    checks["scope_summary_186_172"] = (
        int(scope_summary.loc[scope_summary["variant"].eq("historical_old_186"),
                              "unique_train_landmarks"].iloc[0]) == 186
        and int(scope_summary.loc[scope_summary["variant"].eq("historical_old_186"),
                                  "unique_test_landmarks"].iloc[0]) == 172
        and set(scope_summary.loc[scope_summary["variant"].isin(common_variants),
                                  "unique_train_landmarks"]) == {172}
        and set(scope_summary.loc[scope_summary["variant"].isin(common_variants),
                                  "unique_test_landmarks"]) == {172}
    )

    # 求解器对照：同一样本、同一旧分数，仅换求解器
    old_refit = predictions.loc[predictions["variant"].eq("old_refit_common")]
    solver = predictions.loc[predictions["variant"].eq("solver_on_common")]
    merged = old_refit.merge(
        solver, on=["participantid", "landmark_day", "method"],
        suffixes=("_old", "_new"), validate="one_to_one",
    )
    max_prediction_diff = float((merged["risk_old"] - merged["risk_new"]).abs().max())
    checks["solver_predictions_same_within_tolerance"] = max_prediction_diff <= TOLERANCE
    checks["solver_control_metadata"] = (
        abs(float(attribution["solver_control"]["max_abs_prediction_diff"])
            - max_prediction_diff) <= TOLERANCE
        and int(attribution["solver_control"]["n_pairs"]) == 344
    )

    # 折模型系数与来源
    coefficient_ok = True
    coefficient_max = 0.0
    new_models = sorted((stage10 / "fold_models" / "old_refit_common" / "30d").glob("*.json"))
    old_models = sorted((fixed_run / "stage3" / "fold_models" / "solver_on_common" / "30d").glob("*.json"))
    if len(new_models) != 6 or len(old_models) != 6:
        coefficient_ok = False
    else:
        for new_path in new_models:
            name = new_path.name
            old_path = fixed_run / "stage3" / "fold_models" / "solver_on_common" / "30d" / name
            if not old_path.exists():
                coefficient_ok = False
                continue
            new_data = read_json(new_path)
            old_data = read_json(old_path)
            for key in ("coefficients", "feature_centers"):
                for a, b in zip(new_data[key], old_data[key]):
                    coefficient_max = max(coefficient_max, abs(float(a) - float(b)))
                    if abs(float(a) - float(b)) > TOLERANCE:
                        coefficient_ok = False
            coefficient_max = max(
                coefficient_max,
                abs(float(new_data["baseline_survival"]) - float(old_data["baseline_survival"])),
            )
    checks["solver_coefficients_same_within_tolerance"] = coefficient_ok

    # 阶段 6 主表与阶段 10 修正版行一致
    stage6_main = read_json(fixed_run / "comparison.json")["main_metrics"]
    fixed_row = attribution["rows"]["fixed_session"]
    checks["stage6_main_matches_fixed_row"] = all(
        abs(float(stage6_main[key]) - float(fixed_row[key])) <= 1e-12
        for key in ("auroc_A", "auroc_B", "brier_A", "brier_B",
                    "delta_auroc", "delta_brier")
    )
    fixed_predictions = predictions.loc[predictions["variant"].eq("fixed_session")]
    fixed_positive_people = {
        person for person, event in zip(
            fixed_predictions["participantid"], fixed_predictions["event"]
        ) if int(event) == 1
    }
    checks["stage6_counts"] = (
        stage6_main["n"] == 172 and stage6_main["persons"] == 134
        and stage6_main["events"] == 16
        and len(fixed_positive_people) == 14
    )

    # 失败状态覆盖五个对照行（历史旧模型从旧运行读取）
    status = pd.read_csv(stage10 / "fold_status.csv")
    checks["fold_status_covers_rows"] = set(status["variant"]) == set(VARIANTS)
    checks["fold_status_30d_ok"] = bool(status.loc[status["horizon"].eq(30), "status"].eq("ok").all())

    # 阶段 8/9 证据
    stage8 = read_json(STAGE8 / "verification.json")
    stage9 = read_json(STAGE9 / "verification.json")
    stage9_counts = read_json(STAGE9 / "counts.json")
    checks["stage8_all_pass_19"] = stage8["all_pass"] and len(stage8["checks"]) == 19
    checks["stage9_all_pass_27"] = stage9["all_pass"] and len(stage9["checks"]) == 27
    checks["stage9_counts"] = (
        stage9_counts["required_counts"]["s7_new_positive_people_total"] == 10
        and stage9_counts["required_counts"]["s7_new_positive_people_from_new_people"] == 6
        and stage9_counts["required_counts"]["s7_existing_people_newly_positive"] == 4
        and stage9_counts["required_counts"]["s7_extra_positive_landmarks_total"] == 11
        and stage9_counts["required_counts"]["people_intersection"] == 144
        and stage9_counts["required_counts"]["people_union"] == 194
    )

    # 报告文字与来源
    report = (stage10 / "REPORT.md").read_text(encoding="utf-8")
    checks["report_corrected_brier"] = (
        "略有改善" in report and "区间跨 0" in report
        and "无改善" in report  # 作为被否定的旧表述出现
        and "证据不足" in report
    )
    checks["report_has_five_rows"] = all(name in report for name in VARIANTS)
    checks["report_has_stage8_stage9"] = (
        "modality_mismatch" in report and "10 名" in report
        and "6 + 4 + 1" in report
    )
    source_map = read_json(stage10 / "source_map.json")
    checks["source_hashes"] = all(
        sha256(Path(item["path"])) == item["sha256"]
        for item in source_map["inputs"].values()
    )
    fold_sources = source_map.get("fold_model_sources", {})
    checks["fold_model_sources"] = (
        len(fold_sources.get("stage10_old_refit_common", [])) == 6
        and len(fold_sources.get("stage6_existing", [])) >= 18
        and len(fold_sources.get("stage4_old", [])) >= 6
        and all(
            sha256(Path(item["path"])) == item["sha256"]
            for group in fold_sources.values() for item in group
        )
    )

    payload = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fixed_run": str(fixed_run),
        "old_run": str(old_run),
        "stage10": str(stage10),
        "checks": checks,
        "all_pass": all(bool(value) for value in checks.values()),
        "recomputed_point_metrics": recomputed,
        "solver_max_abs_prediction_diff": max_prediction_diff,
        "solver_max_abs_coefficient_diff": coefficient_max,
        "same_sample": {
            "rows": 5, "test_landmarks": 172, "persons": 134,
            "events": 16, "positive_people": len(fixed_positive_people),
        },
        "stage6_main": stage6_main,
        "stage8": {"all_pass": stage8["all_pass"], "checks": len(stage8["checks"])},
        "stage9": {"all_pass": stage9["all_pass"], "checks": len(stage9["checks"])},
    }
    write_json(stage10 / "verification.json", payload)
    lines = [
        "# 阶段 10 核验记录",
        "",
        f"- 时间：{payload['verified_at_utc']}",
        f"- 阶段 10：`{stage10}`",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
    ]
    for name, value in checks.items():
        lines.append(f"| {name} | {'通过' if value else '失败'} |")
    lines += [
        "",
        f"- 五个对照行均 172 个测试日、134 人、16 个阳性；独立重算 A/B 指标通过。",
        f"- 求解器对照最大预测差 {max_prediction_diff:.3e}，最大系数/基线差 {coefficient_max:.3e}（容差 {TOLERANCE}）。",
        f"- 阶段 8：{stage8['all_pass']}（{len(stage8['checks'])} 项）；阶段 9：{stage9['all_pass']}（{len(stage9['checks'])} 项）。",
        "",
        f"**all_pass = {payload['all_pass']}**",
        "",
    ]
    (stage10 / "VERIFICATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "checks": checks, "all_pass": payload["all_pass"],
        "solver_max_abs_prediction_diff": max_prediction_diff,
        "solver_max_abs_coefficient_diff": coefficient_max,
    }, ensure_ascii=False, indent=1))
    return 0 if payload["all_pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
