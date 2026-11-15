"""用修正后的相对风险推理重算阶段 11 的临床参照诊断。

背景：``longitudinal/cox_horizon.py`` 的线性预测量最初把原始单位系数再除以
尺度一次，导致预测风险被放大。该缺陷在阶段 13 冻结前发现（尚未查看任何
外层测试性能），修正后与 ``longitudinal.survival`` 的既有研究推理入口逐行
一致（1e-12）。本脚本只在**同一协议**下重算诊断数值，不改变任何冻结参数；
旧结果保留在 ``clinical_reference.json``，新结果写入
``clinical_reference_corrected.json``，差异写入 ``clinical_reference_diff.json``。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from round2_stage11_protocol import _clinical_reference  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise" / "round2"


def _flatten(reference: dict) -> dict:
    flat = {}
    for horizon, entry in reference["horizons"].items():
        flat[f"{horizon}.positives"] = entry["positives"]
        flat[f"{horizon}.auroc_estimable"] = entry.get("auroc_estimable")
        for name, model in entry["models"].items():
            flat[f"{horizon}.{name}.auroc"] = model.get("auroc")
            flat[f"{horizon}.{name}.brier"] = model.get("brier")
    return flat


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="重算阶段 11 临床参照诊断")
    parser.add_argument("--run", default=str(
        OUTPUT_ROOT / "stage11" / "20261115-protocol-v1"
    ))
    args = parser.parse_args(argv)
    run_dir = Path(args.run)
    real = pd.read_csv(run_dir / "real_candidates.csv", dtype={"participantid": str})
    folds = pd.read_csv(run_dir / "splits.csv", dtype={"participantid": str})
    corrected = _clinical_reference(real, folds)
    previous = json.loads((run_dir / "clinical_reference.json").read_text(encoding="utf-8"))
    (run_dir / "clinical_reference_corrected.json").write_text(
        json.dumps(corrected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    old_flat, new_flat = _flatten(previous), _flatten(corrected)
    diff = {
        "reason": "修正 cox_horizon 线性预测量（原始单位系数不再重复除以尺度）",
        "protocol_parameters_changed": False,
        "outcome_metrics_used_to_change_protocol": False,
        "affected_stages": ["stage11 clinical reference diagnostic only"],
        "unaffected": [
            "real_candidates / splits / landmarks / baseline_days",
            "protocol 冻结参数（窗口、模型阶梯、权重、特征、阈值目标等）",
            "stage12 半合成生成与评分（不使用 cox_horizon）",
        ],
        "differences": {
            key: {"old": old_flat.get(key), "new": new_flat.get(key)}
            for key in sorted(set(old_flat) | set(new_flat))
            if old_flat.get(key) != new_flat.get(key)
        },
    }
    (run_dir / "clinical_reference_diff.json").write_text(
        json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "changed": len(diff["differences"]),
        "examples": list(diff["differences"].items())[:6],
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
