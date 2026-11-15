"""从个人每日观测建立14日稳健基线与可选的 DS-PCD 基线。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.baseline import build_baseline  # noqa: E402
from longitudinal.config import LongitudinalConfig  # noqa: E402
from longitudinal.ds_pcd import (  # noqa: E402
    BaselineUnavailable,
    DayRecord,
    DsPcdConfig,
    fit_baseline,
)
from longitudinal.quality import assess_quality  # noqa: E402
from longitudinal.storage import ProfileStore  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="建立个人14日步态姿态基线")
    parser.add_argument("--person-id", required=True, help="个人唯一标识")
    parser.add_argument("--profile-dir", default="profiles", help="个人画像根目录")
    parser.add_argument(
        "--detector", choices=("rms", "ds_pcd", "both"), default="rms",
        help="rms 只建旧基线；ds_pcd/both 追加 DS-PCD 基线",
    )
    parser.add_argument("--ds-pcd-config", default=None, help="DS-PCD 参数 JSON")
    parser.add_argument("--precision-mode", choices=("session_se", "none"), default=None)
    parser.add_argument("--null-samples", type=int, default=None)
    parser.add_argument("--quantile-points", type=int, default=None)
    parser.add_argument("--k-ref", type=float, default=None)
    parser.add_argument("--gap-reset-days", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = ProfileStore(args.profile_dir, args.person_id)
    observations = store.load_observations()
    config = LongitudinalConfig()
    baseline = build_baseline(observations, config)
    store.save_baseline(baseline)
    payload: dict = {"rms": baseline.to_dict()}
    if args.detector in ("ds_pcd", "both"):
        ds_config = DsPcdConfig()
        if args.ds_pcd_config:
            ds_config = DsPcdConfig.from_dict(json.loads(
                Path(args.ds_pcd_config).read_text(encoding="utf-8")
            ))
        overrides = {
            key: value for key, value in {
                "precision_mode": args.precision_mode,
                "null_samples": args.null_samples,
                "quantile_points": args.quantile_points,
                "k_ref": args.k_ref,
                "gap_reset_days": args.gap_reset_days,
            }.items() if value is not None
        }
        if overrides:
            ds_config = replace(ds_config, **overrides)
        records = [
            DayRecord(
                date=item.observation_date,
                features=dict(item.features),
                feature_se=dict(item.feature_se),
                valid=assess_quality(item, config).is_valid,
            )
            for item in observations
        ]
        try:
            ds_baseline = fit_baseline(
                records, args.person_id, ds_config,
                baseline_dates=baseline.valid_dates,
            )
        except BaselineUnavailable as exc:
            payload["ds_pcd"] = {"status": "unavailable", **exc.to_dict()}
            if args.detector == "ds_pcd":
                raise
            print(
                f"警告：DS-PCD 基线未建立（{exc.reason}）：{exc}", file=sys.stderr,
            )
        else:
            store.save_ds_pcd_baseline(ds_baseline)
            payload["ds_pcd"] = ds_baseline.to_dict()
    if args.detector == "rms":
        print(json.dumps(baseline.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _entrypoint() -> int:
    try:
        return main()
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
