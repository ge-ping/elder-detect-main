"""阶段 7 核验：服务协变量、研究接口、旧画像、缺失状态与扩样统计。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.ds_pcd import DsPcdConfig  # noqa: E402
from longitudinal.research import research_prediction  # noqa: E402
from longitudinal.service import LongitudinalRiskService  # noqa: E402
from longitudinal.storage import ProfileStore  # noqa: E402
from longitudinal.survival import CoxModelArtifact  # noqa: E402
from mobilise_d.observations import wearable_config  # noqa: E402

OUT_ROOT = PROJECT_ROOT.parents[1] / "output" / "ds_pcd_mobilise"
FIXED = OUT_ROOT / "20261114-fixed"
_CONTRACT_CANDIDATES = (
    OUT_ROOT / "stage8" / "20261114-contract-v2" / "models_contract",
    OUT_ROOT / "stage8" / "20261114-contract-v1" / "models_contract",
)
DEFAULT_CONTRACT_DIR = next(
    (item for item in _CONTRACT_CANDIDATES if item.exists()), None
)
STAGE7 = OUT_ROOT / "stage7"
TOLERANCE = 1e-9


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 7 核验")
    parser.add_argument("--stage7", default=str(STAGE7))
    parser.add_argument("--fixed-run", default=str(FIXED))
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_CONTRACT_DIR) if DEFAULT_CONTRACT_DIR is not None else None,
        help="阶段 8 带输入契约的模型目录；省略时优先使用阶段 8 导出，否则源模型",
    )
    return parser


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def expansion_from_stage1(fixed: Path) -> dict[str, Any]:
    landmarks = pd.read_csv(fixed / "data" / "landmarks.csv", dtype={"participantid": str})
    windows = pd.read_csv(fixed / "data" / "landmark_windows.csv", dtype={"participantid": str})
    windows = windows.loc[windows["horizon_days"].eq(30)].copy()
    windows["event_bool"] = windows["event"].astype(str).str.lower().eq("true")
    s14 = landmarks.loc[landmarks["baseline_eligible"].astype(str).str.lower().eq("true")]
    s7 = landmarks.loc[landmarks["n_prior_valid_days"].ge(7)]
    def counts(frame):
        keys = set(zip(frame["participantid"], frame["landmark_day"]))
        events = windows[[ (p, d) in keys for p, d in zip(windows["participantid"], windows["landmark_day"]) ]]
        positive = events.loc[events["event_bool"]]
        return {
            "landmarks": int(len(frame)),
            "people": int(frame["participantid"].nunique()),
            "positive_landmarks": int(len(positive)),
            "positive_people": int(positive["participantid"].nunique()),
        }
    people14 = set(s14["participantid"])
    people7 = set(s7["participantid"])
    keys14 = set(zip(s14["participantid"], s14["landmark_day"]))
    keys7 = set(zip(s7["participantid"], s7["landmark_day"]))
    positive7 = windows[
        windows["event_bool"]
        & pd.Series([(p, d) in keys7 for p, d in zip(windows["participantid"], windows["landmark_day"])], index=windows.index)
    ]
    new_positive = positive7.loc[~positive7["participantid"].isin(people14)]
    extra = positive7.loc[
        ~pd.Series([(p, d) in keys14 for p, d in zip(positive7["participantid"], positive7["landmark_day"])], index=positive7.index)
    ]
    return {
        "S14": counts(s14), "S7": counts(s7),
        "overlapping_people": len(people14 & people7),
        "new_people": len(people7 - people14),
        "new_positive_people": int(new_positive["participantid"].nunique()),
        "new_positive_people_ids": sorted(new_positive["participantid"].unique().tolist()),
        "new_positive_landmarks": int(len(new_positive)),
        "S7_extra_positive_landmarks_not_in_S14": int(len(extra)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stage7 = Path(args.stage7)
    fixed = Path(args.fixed_run)
    demo = _read_json(stage7 / "demo_output.json")
    mapping = _read_json(stage7 / "input_field_mapping.json")
    comparison = demo["comparison"]
    research = demo["research"]
    missing = demo["missing_ds_baseline"]
    mismatch = demo["config_mismatch"]
    old = demo["old_profile"]
    expected = demo["expected_stage6"]

    checks: dict[str, bool] = {}
    checks["service_numbers_match_stage6"] = all(
        comparison[key] <= TOLERANCE for key in
        ("distance_diff", "cusum_diff", "T_diff", "q_diff")
    ) and all(
        comparison[key] for key in
        ("status_match", "baseline_end_match", "precision_mode_match",
         "n_valid_days_match", "n_valid_days_before_match")
    )
    checks["research_risk_matches_stage6"] = (
        research.get("status") == "ok"
        and research.get("calibrated") is False
        and research.get("research_only") is True
        and research.get("risk_diff") is not None
        and research["risk_diff"] <= TOLERANCE
        and int(research.get("fold")) == int(demo["fold"])
    )
    checks["research_interface_contract"] = (
        research.get("features") == ["rms_distance", "cusum"]
        and research.get("covariate_status", {}).get("cusum") == "available"
        and research.get("horizon_days") == 30
    )
    clinical = demo["clinical_probability"]
    checks["clinical_probability_empty"] = (
        clinical.get("service_risk_7d") is None
        and clinical.get("service_risk_30d") is None
        and "尚未校准" in str(clinical.get("gate_error", ""))
    )
    checks["missing_ds_baseline_reported"] = (
        missing["ds_status"] == "no_baseline"
        and missing["cusum"] is None
        and missing["distance"] is not None
        and missing["research"]["status"] == "missing_covariates"
        and missing["research"]["missing"] == ["cusum"]
    )
    checks["config_mismatch_reported"] = (
        mismatch["ds_status"] == "config_mismatch"
        and mismatch["cusum"] is None
    )
    checks["old_profile_compatible"] = (
        old["baseline_version"] == 1
        and old["status"] == "baseline_ready"
        and old["distance"] is not None
        and old["risk_7d"] is None and old["risk_30d"] is None
    )
    checks["provenance_fields"] = (
        demo["service"]["provenance"].get("modality") == "wearable"
        and demo["service"]["provenance"].get("precision_mode") == "session_se"
        and demo["service"]["provenance"].get("feature_version") == "mobilise-d-daily-v1"
        and demo["service"]["provenance"].get("baseline_version") == 1
        and demo["service"]["provenance"].get("ds_pcd_baseline_version") == 1
    )
    checks["field_mapping"] = bool(
        all(
            key in mapping.get("wearable", {})
            for key in ("walking_speed_m_s", "cadence_spm", "stride_duration_s",
                        "feature_se", "quality")
        )
        and mapping.get("research_covariates", {}).get("rms_distance")
        and mapping.get("research_covariates", {}).get("cusum")
    )

    # 重新读取保存的画像并复算服务输出
    store = ProfileStore(stage7 / "demo_profile", str(demo["person"]))
    run_summary = _read_json(fixed / "stage2" / "replay_summary.json")
    ds_config = DsPcdConfig.from_dict(run_summary["modes"]["session_se"]["config"])
    service = LongitudinalRiskService(
        store, wearable_config(), ds_pcd_config=ds_config,
        ds_pcd_alarm_threshold=5.2, modality="wearable",
        population="65+ Parkinson (Mobilise-D CVS)",
        feature_version="mobilise-d-daily-v1",
        score_version="scoring-v2-causal-precision",
    )
    observations = {item.observation_date: item for item in store.load_observations()}
    repeated = service.evaluate(
        str(demo["person"]), observations[demo["landmark"]],
        as_of=demo["landmark"], persist=False,
    )
    repeated_payload = {
        "distance": repeated.distance,
        "cusum": (repeated.ds_pcd or {}).get("cusum"),
        "T": (repeated.ds_pcd or {}).get("T"),
        "q": (repeated.ds_pcd or {}).get("q"),
        "status": repeated.status,
        "ds_status": (repeated.ds_pcd or {}).get("status"),
        "valid_day_counts": dict(repeated.valid_day_counts),
    }
    demo_payload = {
        "distance": demo["service"]["distance"],
        "cusum": demo["service"]["ds_pcd"].get("cusum"),
        "T": demo["service"]["ds_pcd"].get("T"),
        "q": demo["service"]["ds_pcd"].get("q"),
        "status": demo["service"]["status"],
        "ds_status": demo["service"]["ds_pcd"].get("status"),
        "valid_day_counts": dict(demo["service"]["valid_day_counts"]),
    }
    replay_match = (
        abs(repeated_payload["distance"] - demo_payload["distance"]) <= TOLERANCE
        and abs(repeated_payload["cusum"] - demo_payload["cusum"]) <= TOLERANCE
        and abs(repeated_payload["T"] - demo_payload["T"]) <= TOLERANCE
        and abs(repeated_payload["q"] - demo_payload["q"]) <= TOLERANCE
        and repeated_payload["status"] == demo_payload["status"]
        and repeated_payload["ds_status"] == demo_payload["ds_status"]
        and repeated_payload["valid_day_counts"] == demo_payload["valid_day_counts"]
    )
    checks["saved_profile_reevaluation"] = replay_match

    # 研究模型 calibrated=false；服务概率门槛拒绝
    if args.model_dir:
        model_path = (
            Path(args.model_dir) / "fixed_session" / "30d"
            / f"B_rms_cusum_fold{demo['fold']}.json"
        )
    else:
        model_path = (
            fixed / "fold_models" / "fixed_session" / "30d"
            / f"B_rms_cusum_fold{demo['fold']}.json"
        )
    artifact = CoxModelArtifact.from_dict(_read_json(model_path))
    checks["research_model_not_calibrated"] = artifact.calibrated is False
    try:
        from longitudinal.survival import predict_cumulative_risk
        predict_cumulative_risk(
            {"rms_distance": demo_payload["distance"], "cusum": demo_payload["cusum"]},
            artifact,
        )
        gate_blocked = False
    except ValueError:
        gate_blocked = True
    checks["service_gate_blocks_uncalibrated"] = gate_blocked

    expansion = expansion_from_stage1(fixed)
    reported = demo["expansion"]
    checks["expansion_counts"] = (
        expansion["S14"] == reported["S14"]
        and expansion["S7"] == reported["S7"]
        and expansion["overlapping_people"] == reported["overlapping_people"]
        and expansion["new_people"] == reported["new_people"]
        and expansion["new_positive_people"] == reported["new_positive_people"]
        and expansion["new_positive_landmarks"] == reported["new_positive_landmarks"]
        and expansion["S7_extra_positive_landmarks_not_in_S14"]
        == reported["S7_extra_positive_landmarks_not_in_S14"]
    )
    payload = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage7": str(stage7),
        "checks": checks,
        "all_pass": all(checks.values()),
        "demo": {
            "person": demo["person"], "landmark": demo["landmark"], "fold": demo["fold"],
            "comparison": comparison, "research": research,
            "clinical": clinical, "missing": missing, "mismatch": mismatch,
            "old": old, "provenance": demo["service"]["provenance"],
        },
        "saved_profile_recheck": repeated_payload,
        "expansion": expansion,
        "tolerance": TOLERANCE,
    }
    _write_json(stage7 / "verification.json", payload)
    lines = [
        "# 阶段 7 核验记录",
        "",
        f"- 时间：{payload['verified_at_utc']}",
        f"- 演示：受试者 {demo['person']}，预测日 {demo['landmark']}，折 {demo['fold']}",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
    ]
    for name, value in checks.items():
        lines.append(f"| {name} | {'通过' if value else '失败'} |")
    lines += [
        "",
        f"- Stage 6 对照：distance diff {comparison['distance_diff']:.2e}，"
        f"C diff {comparison['cusum_diff']:.2e}，T diff {comparison['T_diff']:.2e}，"
        f"q diff {comparison['q_diff']:.2e}；研究风险 diff {research['risk_diff']:.2e}。",
        f"- 服务风险：7d={clinical['service_risk_7d']}，30d={clinical['service_risk_30d']}；"
        f"门槛：{clinical['gate_error']}。",
        f"- 旧画像：baseline v{old['baseline_version']}，status {old['status']}，"
        f"distance {old['distance']:.4f}，风险 {old['risk_7d']}/{old['risk_30d']}。",
        f"- S14 {expansion['S14']}；S7 {expansion['S7']}；重叠 {expansion['overlapping_people']}；"
        f"新增阳性人数 {expansion['new_positive_people']}；新增阳性时点 {expansion['new_positive_landmarks']}。",
        "",
        f"**all_pass = {payload['all_pass']}**",
        "",
    ]
    (stage7 / "VERIFICATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"checks": checks, "all_pass": payload["all_pass"]}, ensure_ascii=False, indent=1))
    return 0 if payload["all_pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
