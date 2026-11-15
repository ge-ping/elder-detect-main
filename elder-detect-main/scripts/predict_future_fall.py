"""计算个人基线偏离并按模型可用性输出未来跌倒风险。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.config import LongitudinalConfig  # noqa: E402
from longitudinal.ds_pcd import DsPcdConfig  # noqa: E402
from longitudinal.service import LongitudinalRiskService  # noqa: E402
from longitudinal.storage import ProfileStore  # noqa: E402
from longitudinal.survival import load_model_bundle  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="输出个人偏离状态和未来7天/30天跌倒风险")
    parser.add_argument("--person-id", required=True, help="个人唯一标识")
    parser.add_argument("--profile-dir", default="profiles", help="个人画像根目录")
    parser.add_argument("--model-7d", default=None, help="已校准7天 Cox 模型 JSON")
    parser.add_argument("--model-30d", default=None, help="已校准30天 Cox 模型 JSON")
    parser.add_argument(
        "--detector",
        choices=("rms", "ds_pcd", "both"),
        default="rms",
        help="rms 保持旧口径；ds_pcd/both 同时输出 DS-PCD 统计量",
    )
    parser.add_argument("--as-of", default=None, help="按该预测日截断历史（重放用）")
    parser.add_argument("--ds-pcd-config", default=None, help="DS-PCD 参数 JSON")
    parser.add_argument("--alarm-threshold", type=float, default=None,
                        help="DS-PCD 累计阈值（只用于展示告警）")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.model_7d) != bool(args.model_30d):
        raise ValueError("必须同时提供 --model-7d 和 --model-30d")
    store = ProfileStore(args.profile_dir, args.person_id)
    observations = store.load_observations()
    if not observations:
        raise ValueError("个人画像中没有每日观测")
    models = (
        load_model_bundle(args.model_7d, args.model_30d)
        if args.model_7d else None
    )
    ds_pcd_config = None
    if args.detector in ("ds_pcd", "both"):
        if args.ds_pcd_config:
            payload = json.loads(Path(args.ds_pcd_config).read_text(encoding="utf-8"))
            ds_pcd_config = DsPcdConfig.from_dict(payload)
        else:
            stored = store.load_ds_pcd_baseline()
            if stored and stored.get("config"):
                ds_pcd_config = DsPcdConfig.from_dict(stored["config"])
            else:
                ds_pcd_config = DsPcdConfig()
    service = LongitudinalRiskService(
        store, LongitudinalConfig(), models=models,
        ds_pcd_config=ds_pcd_config,
        ds_pcd_alarm_threshold=args.alarm_threshold,
    )
    if args.as_of:
        observation = next(
            (item for item in observations if item.observation_date == args.as_of), None
        )
        if observation is None:
            raise ValueError(f"画像中没有预测日 {args.as_of} 的观测")
    else:
        observation = observations[-1]
    result = service.evaluate(
        args.person_id, observation,
        as_of=args.as_of,
        strict_baseline_order=bool(args.as_of),
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _entrypoint() -> int:
    try:
        return main()
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
