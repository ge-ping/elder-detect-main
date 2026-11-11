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
from longitudinal.service import LongitudinalRiskService  # noqa: E402
from longitudinal.storage import ProfileStore  # noqa: E402
from longitudinal.survival import load_model_bundle  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="输出个人偏离状态和未来7天/30天跌倒风险")
    parser.add_argument("--person-id", required=True, help="个人唯一标识")
    parser.add_argument("--profile-dir", default="profiles", help="个人画像根目录")
    parser.add_argument("--model-7d", default=None, help="已校准7天 Cox 模型 JSON")
    parser.add_argument("--model-30d", default=None, help="已校准30天 Cox 模型 JSON")
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
    service = LongitudinalRiskService(store, LongitudinalConfig(), models=models)
    result = service.evaluate(args.person_id, observations[-1])
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
