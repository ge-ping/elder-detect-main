"""阶段 8：为阶段 6 折模型导出带输入契约的副本与来源映射。

只增加 ``input_contract``，数值参数、校准标志和训练摘要原样继承。
旧分数归因模型（solver_*）保留为 legacy 分数版本，正常新分数服务不会
匹配它们；这一点在契约的 score_version 中明确记录。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.baseline import (  # noqa: E402
    config_fingerprint as rms_config_fingerprint,
    rms_distance_config_fingerprint,
)
from longitudinal.ds_pcd import (  # noqa: E402
    DsPcdConfig,
    config_fingerprint as ds_pcd_fingerprint,
    scoring_config_fingerprint as ds_pcd_scoring_fingerprint,
)
from longitudinal.model_contract import (  # noqa: E402
    build_input_contract,
    contract_summary,
)
from mobilise_d.observations import wearable_config  # noqa: E402

OUT_ROOT = PROJECT_ROOT.parents[1] / "output" / "ds_pcd_mobilise"
DEFAULT_RUN = OUT_ROOT / "20261114-fixed"
DEFAULT_OUT = OUT_ROOT / "stage8" / "20261114-contract-v1"
POPULATION = "65+ Parkinson (Mobilise-D CVS)"
FEATURE_VERSION = "mobilise-d-daily-v1"

FEATURE_SPECS = {
    "rms_distance": {
        "definition": "root_mean_square_distance_from_personal_baseline",
        "source": "service.distance",
        "requires_ds_pcd": False,
    },
    "cusum": {
        "definition": "ds_pcd_cusum_directional_accumulation",
        "source": "service.ds_pcd.cusum",
        "requires_ds_pcd": True,
    },
}

VARIANT_RULES = {
    "fixed_session": {"score_version": "scoring-v2-causal-precision",
                      "precision_policy": "causal_prior_only", "legacy": False},
    "fixed_none": {"score_version": "scoring-v2-causal-precision",
                   "precision_policy": "causal_prior_only", "legacy": False},
    "precision_on_common": {"score_version": "scoring-v2-causal-precision",
                            "precision_policy": "causal_prior_only", "legacy": False},
    "solver_on_common": {"score_version": "legacy-stage4-v1",
                         "precision_policy": "full_archive", "legacy": True},
    "solver_none": {"score_version": "legacy-stage4-v1",
                    "precision_policy": "full_archive", "legacy": True},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 8 模型契约导出")
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    return parser


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run = Path(args.run)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"目标目录已存在且非空：{out}")
    config = load_json(run / "config.json")
    run_manifest = load_json(run / "manifest.json")
    rms_full_fingerprint = rms_config_fingerprint(wearable_config())
    rms_fingerprint = rms_distance_config_fingerprint(wearable_config())
    ds_configs = {
        mode: DsPcdConfig.from_dict(config["ds_pcd"][mode])
        for mode in ("session_se", "none")
    }
    ds_fingerprints = {
        mode: ds_pcd_fingerprint(item) for mode, item in ds_configs.items()
    }
    ds_scoring_fingerprints = {
        mode: ds_pcd_scoring_fingerprint(item) for mode, item in ds_configs.items()
    }
    output_root = out / "models_contract"
    entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for source in sorted((run / "fold_models").glob("*/*/*.json")):
        variant = source.parent.parent.name
        horizon_dir = source.parent.name
        if variant not in VARIANT_RULES:
            failures.append({"source": str(source), "reason": "unknown_variant"})
            continue
        rule = VARIANT_RULES[variant]
        method = source.stem.rsplit("_fold", 1)[0]
        try:
            fold = int(source.stem.rsplit("_fold", 1)[1])
            horizon = int(horizon_dir.rstrip("d"))
            raw = load_json(source)
        except (ValueError, IndexError, json.JSONDecodeError) as exc:
            failures.append({"source": str(source), "reason": f"parse_error: {exc}"})
            continue
        feature_names = [str(name) for name in raw.get("feature_names", [])]
        if not feature_names or any(name not in FEATURE_SPECS for name in feature_names):
            failures.append({"source": str(source),
                             "reason": f"unsupported_features: {feature_names}"})
            continue
        mode = "none" if variant.endswith("_none") else "session_se"
        requires_cusum = "cusum" in feature_names
        ds_contract: dict[str, Any] = {"required": False}
        if requires_cusum:
            ds = ds_configs[mode]
            ds_contract = {
                "required": True,
                "precision_mode": mode,
                "scoring_config_fingerprint": ds_scoring_fingerprints[mode],
                "features": list(ds.features),
                "directions": dict(ds.directions),
                "k_ref": ds.k_ref,
                "gap_reset_days": ds.gap_reset_days,
            }
        contract = build_input_contract(
            contract_id=f"mobilise-stage6-{variant}-{horizon_dir}-{method}",
            modality="wearable",
            population=POPULATION,
            feature_version=FEATURE_VERSION,
            score_version=rule["score_version"],
            rms_distance_config_fingerprint=rms_fingerprint,
            precision_policy=rule["precision_policy"],
            features={
                name: dict(FEATURE_SPECS[name]) for name in feature_names
            },
            ds_pcd=ds_contract,
        )
        raw["input_contract"] = contract
        target = output_root / variant / horizon_dir / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target, raw)
        entry = {
            "variant": variant,
            "horizon_days": horizon,
            "method": method,
            "fold": fold,
            "source": str(source.relative_to(run)),
            "source_sha256": sha256(source),
            "output": str(target.relative_to(out)),
            "output_sha256": sha256(target),
            "bundle_id": raw.get("bundle_id"),
            "feature_names": feature_names,
            "calibrated": bool(raw.get("calibrated", False)),
            "score_version": rule["score_version"],
            "legacy_scores": rule["legacy"],
            "contract": contract_summary(contract),
        }
        entries.append(entry)
        counts[variant] = counts.get(variant, 0) + 1
    # 明确记录阶段 6 已知失败折没有模型文件。
    expected = []
    for variant in VARIANT_RULES:
        horizons = ("7d", "30d") if variant.startswith("fixed_") else ("30d",)
        for horizon_dir in horizons:
            for method in ("A_rms", "B_rms_cusum"):
                for fold in (0, 1, 2):
                    expected.append((variant, horizon_dir, method, fold))
    present = {
        (item["variant"], f"{item['horizon_days']}d", item["method"], item["fold"])
        for item in entries
    }
    for variant, horizon_dir, method, fold in expected:
        if (variant, horizon_dir, method, fold) not in present:
            failures.append({
                "variant": variant, "horizon_days": int(horizon_dir.rstrip("d")),
                "method": method, "fold": fold,
                "reason": "source_model_missing_fold_failed",
            })
    source_map = {
        "stage": 8,
        "run_id": out.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_run": str(run),
        "source_run_manifest": "manifest.json",
        "source_run_config": "config.json",
        "source_config_sha256": sha256(run / "config.json"),
        "contract_schema_version": 1,
        "population": POPULATION,
        "feature_version": FEATURE_VERSION,
        "rms_distance_config_fingerprint": rms_fingerprint,
        "rms_config_fingerprint_full": rms_full_fingerprint,
        "ds_pcd_scoring_config_fingerprints": ds_scoring_fingerprints,
        "ds_pcd_config_fingerprints_full": ds_fingerprints,
        "ds_pcd_score_versions": {
            "session_se": VARIANT_RULES["fixed_session"]["score_version"],
            "none": VARIANT_RULES["fixed_none"]["score_version"],
        },
        "notes": [
            "副本只增加 input_contract；coefficients、feature_centers、baseline_survival、calibrated 与 trained_at 原样继承。",
            "solver_* 是旧分数归因模型，contract.score_version=legacy-stage4-v1；新的 causal 服务分数不会匹配它们。",
            "precision_on_common 使用新分数与旧 Cox 求解器，输入契约按新分数导出。",
        ],
        "models": entries,
        "failures": failures,
        "counts": counts,
    }
    write_json(out / "source_map.json", source_map)
    manifest = {
        "stage": 8,
        "run_id": out.name,
        "status": "completed",
        "created_at_utc": source_map["created_at_utc"],
        "source_run": str(run),
        "source_run_status": run_manifest.get("status"),
        "contract_schema_version": 1,
        "models_exported": len(entries),
        "models_by_variant": counts,
        "legacy_score_models": sum(1 for item in entries if item["legacy_scores"]),
        "failures": failures,
        "expected_models": len(expected),
        "notes": source_map["notes"],
    }
    write_json(out / "models_manifest.json", manifest)
    lines = [
        "# 阶段 8 模型输入契约副本",
        "",
        f"- 源运行：`{run}`",
        f"- 导出时间：{source_map['created_at_utc']}",
        f"- 契约 schema：{source_map['contract_schema_version']}",
        f"- 导出模型：{len(entries)} 个（{counts}）",
        f"- 缺失模型：{len(failures)} 个（阶段 6 失败折或未知文件）",
        "",
        "| 变体 | 时窗 | 方法 | 折 | 来源 | 契约 |",
        "|---|---|---|---|---|---|",
    ]
    for item in entries:
        lines.append(
            f"| {item['variant']} | {item['horizon_days']}d | {item['method']} | "
            f"{item['fold']} | `{item['source']}` | `{item['contract'].get('contract_id')}` |"
        )
    lines += [
        "",
        "每个副本保留原 `coefficients`、`feature_centers`、`baseline_survival`、`calibrated` 与 "
        "`trained_at`，只新增 `input_contract`。完整 SHA-256 来源映射见 `source_map.json`。",
        "",
        "`solver_*` 使用旧分数（`legacy-stage4-v1`），不会匹配新的 causal 服务分数；"
        "`precision_on_common` 使用新分数与旧 Cox，契约按新分数。",
        "",
    ]
    (out / "MODELS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "out": str(out), "models": len(entries), "counts": counts,
        "failures": len(failures),
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
