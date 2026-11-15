"""阶段 14 前置：为组件消融计算同一条时间线的多个评分变体。

变体（全部使用 ``longitudinal.ds_pcd`` 共享核心，不复制公式）：

* ``full``：session_se 精度 + 方向统计量 T + CUSUM C（主规格）；
* ``none``：无精度权重（session_se→none），其余同 full；
* ``omnibus``：双侧 omnibus D 与同样 k_ref 的 CUSUM（不分方向但同样累积）；
* ``direction_only``：只取 full 的 T，不含累积（在建模层实现，无需另算）。

虚拟序列由冻结的生成器参数与同一随机种子重建，保证与阶段 12/13 的序列
一一对应；输出只含评分结果，不改动阶段 12/13 的产物。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import md5
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.ds_pcd import DsPcdConfig, normal_score, score_record, cusum_replay  # noqa: E402
from mobilise_d.round2 import R2_FEATURES  # noqa: E402
from mobilise_d.round2_scoring import (  # noqa: E402
    PersonSeries, r2_ds_pcd_config, score_person, series_from_frame,
)
from mobilise_d.round2_synth import (  # noqa: E402
    GeneratorConfig, GeneratorParams, simulate_batch, person_records,
)

OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "output" / "ds_pcd_mobilise" / "round2"
STAGE11 = OUTPUT_ROOT / "stage11" / "20261115-protocol-v1"
STAGE12 = OUTPUT_ROOT / "stage12" / "20261115-supplement-v1"
STAGE13 = OUTPUT_ROOT / "stage13" / "20261115-freeze-v2"
SEED = 20261115
MECHANISMS = ("level_dominant", "sustained_change", "change_no_effect")
NOISE_FAMILIES = ("gaussian_ar1", "heavy_tail_spike")
BUDGET_TRAIN = 600
BUDGET_TEST = 2000

CODE_SNAPSHOT_FILES = (
    "mobilise_d/round2_scoring.py",
    "mobilise_d/round2_synth.py",
    "longitudinal/ds_pcd.py",
    "scripts/round2_stage14_ablation_scores.py",
)


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


def _omnibus_cusum(daily: Sequence[dict], k_ref: float) -> tuple[float | None, float | None, float | None]:
    """用同一 k_ref 对 omnibus 的 q 做 CUSUM 重放（不分方向但同样累积）。"""
    q_values = []
    last = None
    for row in daily:
        p = row.get("omnibus_p")
        if p is None or not math.isfinite(float(p)):
            continue
        q = normal_score(float(p), -4.0, 6.0)
        q_values.append(q)
        last = row
    if not q_values:
        return None, None, None
    current = 0.0
    for q in q_values:
        current = max(0.0, current + q - k_ref)
    return float(current), float(q_values[-1]), float(last.get("omnibus_D")) if last else None


def _score_variants(
    series: PersonSeries,
    *,
    full_config: DsPcdConfig,
    none_config: DsPcdConfig,
    snapshot_day: str,
    baseline_dates: Sequence[str],
    k_ref: float,
    meta: dict,
) -> dict:
    full = score_person(
        series, full_config, snapshot_day=snapshot_day,
        baseline_dates=baseline_dates,
    )
    none = score_person(
        series, none_config, snapshot_day=snapshot_day,
        baseline_dates=baseline_dates,
    )
    c_omnibus, q_omnibus, d_snapshot = _omnibus_cusum(full.get("daily", []), k_ref)
    row = {
        **meta,
        "status_full": full.get("status"),
        "status_none": none.get("status"),
        "T": full.get("T"),
        "C": full.get("C"),
        "q": full.get("q"),
        "T_none": none.get("T"),
        "C_none": none.get("C"),
        "q_none": none.get("q"),
        "D_snapshot": d_snapshot,
        "C_omnibus": c_omnibus,
        "q_omnibus": q_omnibus,
        "mask_full": full.get("mask"),
        "mask_none": none.get("mask"),
        "baseline_days_used": full.get("baseline_days_used"),
    }
    return row


def _real_series(run_dir: Path) -> pd.DataFrame:
    real = pd.read_csv(STAGE11 / "real_candidates.csv", dtype={"participantid": str})
    baseline_days = pd.read_csv(STAGE11 / "baseline_days.csv", dtype={"participantid": str})
    person_daily = pd.read_csv(STAGE12 / "person_daily_features.csv", dtype={"participantid": str},
                               low_memory=False)
    real_scores = pd.read_csv(STAGE12 / "real_scores.csv", dtype={"participantid": str})
    scored = real_scores.loc[real_scores["status"] == "scored"]
    rows = []
    for row in scored.itertuples(index=False):
        person = str(row.participantid)
        landmark = str(row.landmark_day)
        baseline_list = sorted(
            baseline_days.loc[
                (baseline_days["participantid"] == person)
                & (baseline_days["landmark_day"] == landmark),
                "baseline_day",
            ].astype(str)
        )
        person_rows = person_daily.loc[person_daily["participantid"] == person]
        window = person_rows.loc[
            person_rows["day"].astype(str).isin(baseline_list)
            | (
                (person_rows["day"] >= str(row.baseline_end))
                & (person_rows["day"] <= landmark)
            )
        ]
        series = series_from_frame(window, person, features=R2_FEATURES, source="observed")
        rows.append({
            "series": series,
            "snapshot_day": landmark,
            "baseline_dates": baseline_list,
            "meta": {
                "branch": "R",
                "test_set": "real_test",
                "series_id": f"real-{person}-{landmark}",
                "participantid": person,
                "landmark_day": landmark,
                "fold": int(row.fold) if "fold" in row._fields else None,
                "age": row.age,
                "prior_fall_any": row.prior_fall_any,
            },
        })
    return rows


def run(args: argparse.Namespace) -> int:
    run_dir = Path(args.out_root) / f"stage14/{args.run_id}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"运行目录已存在且非空，拒绝覆盖：{run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    full_config = r2_ds_pcd_config(precision_mode="session_se")
    none_config = r2_ds_pcd_config(precision_mode="none")
    config = GeneratorConfig()
    splits = pd.read_csv(STAGE11 / "splits.csv", dtype={"participantid": str})
    real_fold = splits.set_index("participantid")["fold"].to_dict()

    rows: list[dict] = []
    # 真实候选
    print("[stage14] 真实候选多变体评分", flush=True)
    for item in _real_series(run_dir):
        meta = dict(item["meta"])
        meta["fold"] = int(real_fold.get(meta["participantid"], -1))
        rows.append(_score_variants(
            item["series"], full_config=full_config, none_config=none_config,
            snapshot_day=item["snapshot_day"], baseline_dates=item["baseline_dates"],
            k_ref=full_config.k_ref, meta=meta,
        ))
    # 半合成训练与测试
    for fold in sorted(splits["fold"].unique()):
        fold = int(fold)
        params_payload = json.loads(
            (STAGE12 / f"generator_params_fold{fold}.json").read_text(encoding="utf-8")
        )
        params = GeneratorParams.from_dict(params_payload["params"])
        for mechanism in MECHANISMS:
            for noise in NOISE_FAMILIES:
                offset = MECHANISMS.index(mechanism) * 10 + NOISE_FAMILIES.index(noise)
                for batch, budget, seed_base in (
                    ("train", BUDGET_TRAIN, 100),
                    ("test", BUDGET_TEST, 200),
                ):
                    print(f"[stage14] fold{fold} {mechanism}|{noise} {batch}", flush=True)
                    _, series = simulate_batch(
                        params, count=budget, fold=fold, mechanism=mechanism,
                        noise_family=noise, seed=SEED + 1000 * fold + seed_base + offset,
                        config=config, prefix=batch, keep_series=True,
                        lambda0=params.lambda0.get(f"{mechanism}|{noise}"),
                    )
                    for person in series:
                        records = person_records(person, R2_FEATURES)
                        observed = np.asarray(person["observed"], dtype=float)
                        se = np.asarray(person["se"], dtype=float)
                        series_obj = PersonSeries(
                            person_id=person["series_id"],
                            days=tuple(record.date for record in records),
                            features={
                                name: tuple(observed[:, index])
                                for index, name in enumerate(R2_FEATURES)
                            },
                            feature_se={
                                name: tuple(se[:, index])
                                for index, name in enumerate(R2_FEATURES)
                            },
                            source="synthetic",
                            meta={},
                        )
                        snapshot_day = records[
                            config.baseline_days + config.monitoring_days - 1
                        ].date
                        rows.append(_score_variants(
                            series_obj, full_config=full_config, none_config=none_config,
                            snapshot_day=snapshot_day,
                            baseline_dates=[record.date for record in records[: config.baseline_days]],
                            k_ref=full_config.k_ref,
                            meta={
                                "branch": "S",
                                "test_set": f"synthetic_{batch}",
                                "series_id": person["series_id"],
                                "participantid": "",
                                "landmark_day": "",
                                "fold": fold,
                                "mechanism": mechanism,
                                "noise_family": noise,
                                "trajectory": person["trajectory"],
                                "age": person["age"],
                                "prior_fall_any": person["prior_fall_any"],
                            },
                        ))
    frame = pd.DataFrame(rows)
    frame.to_csv(run_dir / "variant_scores.csv", index=False, encoding="utf-8")
    _write_json(run_dir / "variant_manifest.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "variants": {
            "full": {"precision_mode": "session_se", "features": "T,C"},
            "none": {"precision_mode": "none", "features": "T,C"},
            "omnibus": {"precision_mode": "session_se", "features": "D,C_omnibus"},
            "direction_only": {"precision_mode": "session_se", "features": "T"},
        },
        "rows": int(len(frame)),
        "by_test_set": frame.groupby("test_set").size().to_dict(),
        "k_ref": full_config.k_ref,
        "config_fingerprint": {
            "full": full_config.to_dict(),
            "none": none_config.to_dict(),
        },
        "regenerated_from": str(STAGE12),
        "seed": SEED,
        "note": "虚拟序列由冻结参数与同一随机种子重建，只输出评分，不改阶段 12/13 产物",
    })
    manifest = {
        "stage": "14-prep",
        "run_id": args.run_id,
        "outputs": sorted(str(path.relative_to(run_dir)) for path in run_dir.rglob("*") if path.is_file()),
        "upstream": {
            "stage12": str(STAGE12),
            "stage13": str(STAGE13),
        },
        "seed": SEED,
    }
    _write_json(run_dir / "manifest.json", manifest)
    snapshot = run_dir / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for relative in CODE_SNAPSHOT_FILES:
        source = PROJECT_ROOT / relative
        if source.is_file():
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    print(f"[stage14] 变体评分完成：{run_dir}（{len(frame)} 行）", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 14：多评分变体")
    parser.add_argument("--run-id", default="20261115-eval-v1")
    parser.add_argument("--out-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
