"""阶段 8 核验：模型输入契约、三类反例、正例复算与来源匹配。"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.baseline import build_baseline  # noqa: E402
from longitudinal.config import LongitudinalConfig  # noqa: E402
from longitudinal.ds_pcd import (  # noqa: E402
    DayRecord,
    DsPcdConfig,
    fit_baseline,
)
from longitudinal.research import research_prediction  # noqa: E402
from longitudinal.service import LongitudinalRiskService  # noqa: E402
from longitudinal.storage import ProfileStore  # noqa: E402
from longitudinal.survival import (  # noqa: E402
    CoxModelArtifact,
    predict_cumulative_risk,
)
from mobilise_d.observations import wearable_config  # noqa: E402

OUT_ROOT = PROJECT_ROOT.parents[1] / "output" / "ds_pcd_mobilise"
STAGE7 = OUT_ROOT / "stage7"
STAGE8 = OUT_ROOT / "stage8" / "20261114-contract-v2"
FIXED = OUT_ROOT / "20261114-fixed"
TOLERANCE = 1e-9
POPULATION = "65+ Parkinson (Mobilise-D CVS)"
FEATURE_VERSION = "mobilise-d-daily-v1"
SCORE_VERSION = "scoring-v2-causal-precision"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 8 契约核验")
    parser.add_argument("--stage7", default=str(STAGE7))
    parser.add_argument("--stage8", default=str(STAGE8))
    parser.add_argument("--fixed-run", default=str(FIXED))
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def artifact(path: Path) -> CoxModelArtifact:
    return CoxModelArtifact.from_dict(read_json(path))


def evaluate_wearable(store: ProfileStore, person: str, as_of: str, ds_config: DsPcdConfig,
                      *, population: str = POPULATION,
                      score_version: str = SCORE_VERSION):
    service = LongitudinalRiskService(
        store, wearable_config(), ds_pcd_config=ds_config,
        modality="wearable", population=population,
        feature_version=FEATURE_VERSION, score_version=score_version,
    )
    observations = {item.observation_date: item for item in store.load_observations()}
    return service.evaluate(person, observations[as_of], as_of=as_of, persist=False)


def evaluate_video(store: ProfileStore, person: str, as_of: str,
                   ds_config: DsPcdConfig | None = None,
                   ds_baseline=None):
    service = LongitudinalRiskService(
        store, LongitudinalConfig(), ds_pcd_config=ds_config,
        modality="video", population="community-dwelling older adults",
        feature_version="video-pose-v1", score_version="legacy-rms-v1",
    )
    observations = {item.observation_date: item for item in store.load_observations()}
    kwargs = {}
    if ds_baseline is not None:
        kwargs["ds_pcd_baseline_override"] = ds_baseline
    return service.evaluate(
        person, observations[as_of], as_of=as_of, persist=False, **kwargs,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stage7 = Path(args.stage7)
    stage8 = Path(args.stage8)
    fixed = Path(args.fixed_run)
    demo = read_json(stage7 / "demo_output.json")
    run_config = read_json(fixed / "config.json")
    session_config = DsPcdConfig.from_dict(run_config["ds_pcd"]["session_se"])
    none_config = DsPcdConfig.from_dict(run_config["ds_pcd"]["none"])

    # 正确穿戴前缀
    wearable_store = ProfileStore(stage7 / "demo_profile", str(demo["person"]))
    wearable = evaluate_wearable(
        wearable_store, str(demo["person"]), str(demo["landmark"]), session_config,
    )
    fold = int(demo["fold"])
    model_dir = stage8 / "models_contract" / "fixed_session" / "30d"
    model_b = artifact(model_dir / f"B_rms_cusum_fold{fold}.json")
    model_a = artifact(model_dir / f"A_rms_fold{fold}.json")
    positive = research_prediction(wearable, model_b)
    expected = pd.read_csv(fixed / "predictions.csv", dtype={"participantid": str})
    expected = expected.loc[
        expected["variant"].eq("fixed_session") & expected["horizon"].eq(30)
        & expected["method"].eq("B_rms_cusum")
        & expected["participantid"].eq(str(demo["person"]))
        & expected["landmark_day"].astype(str).eq(str(demo["landmark"]))
    ].iloc[0]
    positive_risk_diff = abs(float(positive.get("risk")) - float(expected["risk"])) if positive.get("risk") is not None else None

    # 反例 1/2：视频画像 → 穿戴 A/B；视频侧先计算有效的 C，证明不是只有缺失检查
    video_store = ProfileStore(stage7 / "old_profile", "elder_001")
    video_observations = video_store.load_observations()
    video_as_of = video_observations[-1].observation_date
    video_ds = DsPcdConfig(
        features=("gait_speed_proxy", "cadence_spm", "step_time_mean_s"),
        directions={"gait_speed_proxy": -1.0, "cadence_spm": -1.0, "step_time_mean_s": 1.0},
        precision_mode="none", null_samples=1000, quantile_points=200,
        outlier_z=100.0,
    )
    video_baseline = fit_baseline(
        [DayRecord(date=item.observation_date, features=dict(item.features),
                   feature_se=dict(item.feature_se))
         for item in video_observations],
        "elder_001", video_ds,
        baseline_dates=tuple(item.observation_date for item in video_observations[:14]),
    )
    video = evaluate_video(video_store, "elder_001", video_as_of, video_ds, video_baseline)
    video_plain = evaluate_video(video_store, "elder_001", video_as_of)
    video_a = research_prediction(video, model_a)
    video_b = research_prediction(video, model_b)

    # 反例 3：同一穿戴画像用 none 模式评分 → session_se B
    records = [
        DayRecord(date=item.observation_date, features=dict(item.features),
                  feature_se=dict(item.feature_se))
        for item in wearable_store.load_observations()
    ]
    none_baseline = fit_baseline(
        records, str(demo["person"]), none_config,
        baseline_dates=tuple(demo["baseline_dates"]),
    )
    none_service = LongitudinalRiskService(
        wearable_store, wearable_config(), ds_pcd_config=none_config,
        modality="wearable", population=POPULATION,
        feature_version=FEATURE_VERSION, score_version=SCORE_VERSION,
    )
    none_observations = {item.observation_date: item for item in wearable_store.load_observations()}
    none_result = none_service.evaluate(
        str(demo["person"]), none_observations[str(demo["landmark"])],
        as_of=str(demo["landmark"]), persist=False,
        ds_pcd_baseline_override=none_baseline,
    )
    none_b = research_prediction(none_result, model_b)

    # 缺失契约、特征版本不符、缺少来源
    original_b = artifact(fixed / "fold_models" / "fixed_session" / "30d" / f"B_rms_cusum_fold{fold}.json")
    missing_contract = research_prediction(wearable, original_b)
    bad_version_data = read_json(model_dir / f"B_rms_cusum_fold{fold}.json")
    bad_version_data["input_contract"]["feature_version"] = "video-pose-v1"
    bad_version = research_prediction(
        wearable, CoxModelArtifact.from_dict(bad_version_data),
    )
    no_provenance = research_prediction(
        replace(wearable, provenance={}), model_b,
    )
    bad_schema_data = read_json(model_dir / f"B_rms_cusum_fold{fold}.json")
    bad_schema_data["input_contract"]["schema_version"] = 99
    bad_schema = research_prediction(
        wearable, CoxModelArtifact.from_dict(bad_schema_data),
    )

    # 序列化往返
    roundtrip = CoxModelArtifact.from_dict(
        json.loads(json.dumps(model_b.to_dict(), ensure_ascii=False))
    )
    roundtrip_contract_equal = roundtrip.input_contract == model_b.input_contract
    roundtrip_risk = research_prediction(wearable, roundtrip)

    # 服务入口：使用声明契约的校准模型（演示用校准副本，不是临床模型）
    service_models = {}
    for horizon, base_name in ((7, "7d"), (30, "30d")):
        path = stage8 / "models_contract" / "fixed_session" / base_name / f"B_rms_cusum_fold{fold}.json"
        if not path.exists():
            # 7d B fold1 在阶段 6 是失败折；用 fold2 的 7d 契约仅验证入口行为
            path = stage8 / "models_contract" / "fixed_session" / base_name / "B_rms_cusum_fold2.json"
        data = read_json(path)
        data["calibrated"] = True
        data["bundle_id"] = f"stage8-service-demo-{horizon}d"
        service_models[horizon] = CoxModelArtifact.from_dict(data)
    service = LongitudinalRiskService(
        wearable_store, wearable_config(), models=service_models,
        ds_pcd_config=session_config, modality="wearable",
        population=POPULATION, feature_version=FEATURE_VERSION,
        score_version=SCORE_VERSION,
    )
    service_result = service.evaluate(
        str(demo["person"]), none_observations[str(demo["landmark"])],
        as_of=str(demo["landmark"]), persist=False,
    )
    mismatch_models = dict(service_models)
    mismatch_data = mismatch_models[30].to_dict()
    mismatch_data["input_contract"]["feature_version"] = "video-pose-v1"
    mismatch_models[30] = CoxModelArtifact.from_dict(mismatch_data)
    mismatch_service = LongitudinalRiskService(
        wearable_store, wearable_config(), models=mismatch_models,
        ds_pcd_config=session_config, modality="wearable",
        population=POPULATION, feature_version=FEATURE_VERSION,
        score_version=SCORE_VERSION,
    )
    mismatch_result = mismatch_service.evaluate(
        str(demo["person"]), none_observations[str(demo["landmark"])],
        as_of=str(demo["landmark"]), persist=False,
    )
    gate_blocked = False
    try:
        predict_cumulative_risk(
            {name: float(wearable.research_covariates[name]) for name in model_b.feature_names},
            model_b,
        )
    except ValueError:
        gate_blocked = True

    checks: dict[str, bool] = {
        "positive_contract_matches": positive.get("status") == "ok",
        "positive_risk_matches_stage6": positive_risk_diff is not None and positive_risk_diff <= TOLERANCE,
        "positive_calibrated_false": positive.get("calibrated") is False,
        "positive_research_only": positive.get("research_only") is True,
        "video_to_wearable_a_rejected": video_a.get("status") == "modality_mismatch" and video_a.get("risk") is None,
        "video_to_wearable_b_rejected": video_b.get("status") == "modality_mismatch" and video_b.get("risk") is None,
        "video_covariate_was_available": video.research_covariates.get("cusum") is not None,
        "none_to_session_b_rejected": none_b.get("status") == "precision_mode_mismatch" and none_b.get("risk") is None,
        "missing_contract_status": missing_contract.get("status") == "contract_missing" and missing_contract.get("risk") is None,
        "feature_version_mismatch": bad_version.get("status") == "feature_version_mismatch" and bad_version.get("risk") is None,
        "missing_provenance_status": no_provenance.get("status") == "missing_provenance" and no_provenance.get("risk") is None,
        "contract_invalid_status": bad_schema.get("status") == "contract_invalid" and bad_schema.get("risk") is None,
        "serialization_roundtrip": roundtrip_contract_equal and roundtrip_risk.get("status") == "ok"
        and abs(float(roundtrip_risk["risk"]) - float(positive["risk"])) <= TOLERANCE,
        "service_contract_matches": service_result.status == "calibrated"
        and service_result.risk_7d is not None and service_result.risk_30d is not None
        and all(item.get("status") == "ok" for item in service_result.model_input_status.values()),
        "service_contract_rejects_mismatch": mismatch_result.status == "model_input_mismatch"
        and mismatch_result.risk_7d is None and mismatch_result.risk_30d is None,
        "uncalibrated_gate_blocks": gate_blocked,
        "covariate_status_complete": set(
            wearable.research_covariate_status.get("covariates", {})
        ) == {"rms_distance", "cusum", "distance", "deviation_index", "trend_7d", "persistent_deviation_days"},
    }
    checks["old_profile_distance_available"] = (
        video_plain.status == "baseline_ready" and video_plain.distance is not None
        and video_plain.risk_7d is None and video_plain.risk_30d is None
    )
    checks["old_profile_cusum_status"] = (
        video_plain.research_covariate_status.get("cusum") is None
        or str(video_plain.research_covariate_status.get("cusum")).startswith(("missing:", "unavailable:"))
    )

    payload = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage7": str(stage7),
        "stage8": str(stage8),
        "source_run": str(fixed),
        "positive": {
            "person": demo["person"], "landmark": demo["landmark"], "fold": fold,
            "service": {
                "status": wearable.status,
                "distance": wearable.distance,
                "cusum": wearable.research_covariates.get("cusum"),
                "provenance": dict(wearable.provenance),
                "covariate_status": dict(wearable.research_covariate_status),
            },
            "research": positive,
            "stage6_risk": float(expected["risk"]),
            "risk_diff": positive_risk_diff,
        },
        "counterexamples": {
            "video_plain": {
                "status": video_plain.status, "distance": video_plain.distance,
                "risk_7d": video_plain.risk_7d, "risk_30d": video_plain.risk_30d,
                "cusum_status": video_plain.research_covariate_status.get("cusum"),
            },
            "video_to_wearable_a": video_a,
            "video_to_wearable_b": video_b,
            "video_result": {
                "status": video.status, "distance": video.distance,
                "ds_status": (video.ds_pcd or {}).get("status"),
                "cusum": video.research_covariates.get("cusum"),
            },
            "none_to_session_b": none_b,
            "none_result": {
                "status": none_result.status,
                "precision_mode": (none_result.ds_pcd or {}).get("precision_mode"),
                "cusum": none_result.research_covariates.get("cusum"),
            },
        },
        "contract_errors": {
            "missing_contract": missing_contract,
            "feature_version_mismatch": bad_version,
            "missing_provenance": no_provenance,
            "contract_invalid": bad_schema,
        },
        "serialization": {
            "field_equal": roundtrip_contract_equal,
            "risk": roundtrip_risk.get("risk"),
            "to_dict_equal": roundtrip.to_dict() == model_b.to_dict(),
        },
        "service_integration": {
            "matched": service_result.to_dict(),
            "mismatched": mismatch_result.to_dict(),
            "demo_note": "校准副本仅用于验证服务入口契约规则，不是临床模型。",
        },
        "checks": checks,
        "all_pass": all(bool(value) for value in checks.values()),
        "tolerance": TOLERANCE,
    }
    write_json(stage8 / "verification.json", payload)
    lines = [
        "# 阶段 8 输入契约核验",
        "",
        f"- 时间：{payload['verified_at_utc']}",
        f"- 阶段 7 演示：{demo['person']} / {demo['landmark']} / 折 {fold}",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
    ]
    for name, value in checks.items():
        lines.append(f"| {name} | {'通过' if value else '失败'} |")
    lines += [
        "",
        f"- 视频 → 穿戴 A：`{video_a.get('status')}`，风险 `{video_a.get('risk')}`",
        f"- 视频 → 穿戴 B：`{video_b.get('status')}`，风险 `{video_b.get('risk')}`",
        f"- `none` → `session_se` B：`{none_b.get('status')}`，风险 `{none_b.get('risk')}`",
        f"- 缺失契约：`{missing_contract.get('status')}`；特征版本不符：`{bad_version.get('status')}`；"
        f"缺少来源：`{no_provenance.get('status')}`",
        f"- 正例：研究风险 {positive.get('risk'):.12g}，阶段 6 {float(expected['risk']):.12g}，"
        f"差 {positive_risk_diff:.2e}",
        "",
        f"**all_pass = {payload['all_pass']}**",
        "",
    ]
    (stage8 / "VERIFICATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "checks": checks, "all_pass": payload["all_pass"],
        "positive_risk": positive.get("risk"), "risk_diff": positive_risk_diff,
        "video_a": video_a.get("status"), "video_b": video_b.get("status"),
        "none_b": none_b.get("status"),
    }, ensure_ascii=False, indent=1))
    return 0 if payload["all_pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
