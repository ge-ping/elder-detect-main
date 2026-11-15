"""下载并校验 Mobilise-D CVS V1.0.0 原始发布包。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mobilise_d import sources  # noqa: E402


PACKAGE_CHOICES = {
    "clinical": (sources.CLINICAL,),
    "daily": (sources.DAILY,),
    "dictionary": (sources.DICTIONARY,),
    "bouts": (sources.BOUTS,),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="获取并校验 Mobilise-D CVS 原始包")
    parser.add_argument("--cache-dir", default=None, help="缓存目录，默认 anti-fall/_datasets/Mobilise-D-CVS/21060991")
    parser.add_argument(
        "--package",
        action="append",
        choices=sorted(PACKAGE_CHOICES),
        default=None,
        help="只处理指定包，可重复；缺省为四个包全部处理",
    )
    parser.add_argument("--verify-only", action="store_true", help="只校验已有缓存，不下载")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache_dir = Path(args.cache_dir) if args.cache_dir else sources.default_cache_dir()
    names = args.package or sorted(PACKAGE_CHOICES)
    selected: list[sources.ReleaseFile] = []
    for name in names:
        selected.extend(PACKAGE_CHOICES[name])
    report = {
        "cache_dir": str(cache_dir),
        "mode": "verify-only" if args.verify_only else "download",
        "release": sources.release_manifest(selected),
        "files": {},
    }
    status = 0
    for item in selected:
        target = cache_dir / item.name
        try:
            if args.verify_only:
                if not sources.is_cached(target, item):
                    raise FileNotFoundError(f"缓存缺失或哈希不符：{target}")
            else:
                print(f"获取 {item.name} ...", flush=True)
                sources.download(item, cache_dir)
            report["files"][item.name] = {
                "path": str(target),
                "bytes": target.stat().st_size,
                "md5": item.md5,
                "status": "verified",
            }
        except Exception as exc:  # 记录失败后继续其它包
            status = 1
            report["files"][item.name] = {
                "path": str(target),
                "expected_bytes": item.bytes,
                "expected_md5": item.md5,
                "status": "failed",
                "error": sources.describe_error(exc),
            }
    print(json.dumps(report["files"], ensure_ascii=False, indent=2))
    manifest_path = cache_dir / "fetch_report.json"
    if cache_dir.is_dir():
        sources.write_json(manifest_path, report)
        print(f"报告：{manifest_path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
