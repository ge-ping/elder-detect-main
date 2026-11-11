"""从个人每日观测建立14日稳健基线。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.baseline import build_baseline  # noqa: E402
from longitudinal.config import LongitudinalConfig  # noqa: E402
from longitudinal.storage import ProfileStore  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="建立个人14日步态姿态基线")
    parser.add_argument("--person-id", required=True, help="个人唯一标识")
    parser.add_argument("--profile-dir", default="profiles", help="个人画像根目录")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = ProfileStore(args.profile_dir, args.person_id)
    baseline = build_baseline(store.load_observations(), LongitudinalConfig())
    store.save_baseline(baseline)
    print(json.dumps(baseline.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _entrypoint() -> int:
    try:
        return main()
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
