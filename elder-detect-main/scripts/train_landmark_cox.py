"""使用真实纵向跌倒随访数据训练7天和30天 Landmark Cox 模型。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Sequence
import uuid

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EVIDENCE_IDS = ["mehdizadeh_2021", "cox_1972", "landmarking_2007"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练未来7天和30天 Landmark Cox 生存模型")
    parser.add_argument("--data", required=True, help="纵向 CSV 数据")
    parser.add_argument("--features", nargs="+", required=True, help="模型协变量列名")
    parser.add_argument("--out-dir", required=True, help="模型 JSON 输出目录")
    parser.add_argument(
        "--validation-metrics",
        default=None,
        help="独立验证指标 JSON；不提供时模型保持未校准",
    )
    parser.add_argument(
        "--mark-calibrated",
        action="store_true",
        help="仅在提供独立验证指标后允许将产物标记为已校准",
    )
    return parser


def _validation_ok(path: str | None) -> tuple[bool, dict]:
    if path is None:
        return False, {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    horizons = data.get("horizons")
    ok = type(data.get("independent_validation")) is bool
    ok = ok and data.get("independent_validation") is True
    ok = ok and bool(str(data.get("validation_id", "")).strip())
    ok = ok and isinstance(horizons, dict)
    if ok:
        for horizon in (7, 30):
            metrics = horizons.get(str(horizon))
            if not isinstance(metrics, dict):
                ok = False
                break
            auroc = metrics.get("auroc")
            slope = metrics.get("calibration_slope")
            if (
                isinstance(auroc, bool)
                or not isinstance(auroc, (int, float))
                or not math.isfinite(float(auroc))
                or not 0.0 <= float(auroc) <= 1.0
                or isinstance(slope, bool)
                or not isinstance(slope, (int, float))
                or not math.isfinite(float(slope))
                or float(slope) <= 0.0
            ):
                ok = False
                break
    return ok, data


def _step_survival_at(
    times: np.ndarray,
    survival: np.ndarray,
    horizon: float,
) -> float:
    """按右连续阶梯函数取得不晚于 horizon 的基线生存值。"""
    times = np.asarray(times, dtype=np.float64)
    survival = np.asarray(survival, dtype=np.float64)
    eligible = np.flatnonzero(times <= horizon)
    if len(eligible) == 0:
        return 1.0
    return float(survival[eligible[-1]])


def _landmark_rows(frame, horizon: int, features: list[str]):
    import pandas as pd

    rows = []
    for _, row in frame.iterrows():
        landmark = pd.to_datetime(row["landmark_date"])
        censor = pd.to_datetime(row["censor_date"])
        if censor <= landmark:
            continue
        horizon_end = landmark + pd.Timedelta(days=horizon)
        observed_end = min(censor, horizon_end)
        fall = pd.to_datetime(row.get("next_fall_date"), errors="coerce")
        event = bool(pd.notna(fall) and landmark < fall <= observed_end)
        duration = (fall - landmark).total_seconds() / 86400.0 if event else (
            observed_end - landmark
        ).total_seconds() / 86400.0
        item = {
            "person_id": str(row["person_id"]),
            "duration_days": max(float(duration), 1e-6),
            "event": int(event),
        }
        item.update({name: float(row[name]) for name in features})
        rows.append(item)
    if not rows:
        raise ValueError(f"{horizon}天窗口没有可训练的 Landmark 样本")
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import pandas as pd
        from lifelines import CoxPHFitter
    except ImportError as exc:
        raise RuntimeError("训练需要先安装 pandas 和 lifelines") from exc

    frame = pd.read_csv(args.data)
    required = {"person_id", "landmark_date", "next_fall_date", "censor_date", *args.features}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"训练 CSV 缺少列：{', '.join(missing)}")
    if frame[list(args.features)].isna().any().any():
        raise ValueError("训练特征不能包含缺失值")

    validation_ok, validation = _validation_ok(args.validation_metrics)
    if args.mark_calibrated and not validation_ok:
        raise ValueError("标记已校准必须提供同时覆盖7天和30天的独立验证指标")
    calibrated = bool(args.mark_calibrated and validation_ok)
    validation_hash = (
        hashlib.sha256(Path(args.validation_metrics).read_bytes()).hexdigest()
        if args.validation_metrics else None
    )
    bundle_id = uuid.uuid4().hex

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[int, dict] = {}
    for horizon in (7, 30):
        dataset = _landmark_rows(frame, horizon, list(args.features))
        fitter = CoxPHFitter()
        fitter.fit(
            dataset,
            duration_col="duration_days",
            event_col="event",
            cluster_col="person_id",
            robust=True,
        )
        times = np.asarray(fitter.baseline_survival_.index, dtype=np.float64)
        survival_values = np.asarray(
            fitter.baseline_survival_.iloc[:, 0], dtype=np.float64
        )
        baseline_survival = _step_survival_at(times, survival_values, horizon)
        centers = [float(dataset[name].mean()) for name in args.features]
        artifact = {
            "schema_version": 2,
            "model_type": "landmark_cox",
            "bundle_id": bundle_id,
            "horizon_days": horizon,
            "feature_names": list(args.features),
            "coefficients": [float(fitter.params_[name]) for name in args.features],
            "feature_centers": centers,
            "baseline_survival": baseline_survival,
            "calibrated": calibrated,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "evidence_ids": EVIDENCE_IDS,
            "training_summary": {
                "persons": int(dataset["person_id"].nunique()),
                "landmarks": int(len(dataset)),
                "events": int(dataset["event"].sum()),
                "validation": validation.get("horizons", {}).get(str(horizon)),
                "validation_id": validation.get("validation_id"),
                "validation_hash": validation_hash,
            },
        }
        artifacts[horizon] = artifact

    temporary_paths: dict[int, Path] = {}
    for horizon, artifact in artifacts.items():
        temporary = out_dir / f"risk_{horizon}d.json.tmp.{bundle_id}"
        temporary.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_paths[horizon] = temporary
    for horizon in (7, 30):
        path = out_dir / f"risk_{horizon}d.json"
        temporary_paths[horizon].replace(path)
        print(f"已生成 {horizon} 天模型：{path}（calibrated={calibrated}）")
    return 0


def _entrypoint() -> int:
    try:
        return main()
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
