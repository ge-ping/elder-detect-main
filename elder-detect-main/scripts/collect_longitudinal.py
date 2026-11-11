"""导入一天的步态姿态摘要到个人纵向画像。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.pose_features import summarize_pose_sequence  # noqa: E402
from longitudinal.schema import DailyObservation  # noqa: E402
from longitudinal.storage import ProfileStore  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导入个人每日步态和姿态观测")
    parser.add_argument("--input", required=True, help="每日特征 JSON 或姿态缓存 NPZ")
    parser.add_argument("--person-id", required=True, help="个人唯一标识")
    parser.add_argument("--observed-at", required=True, help="ISO 8601 采集日期时间")
    parser.add_argument("--profile-dir", default="profiles", help="个人画像根目录")
    parser.add_argument("--source", default=None, help="数据来源说明，默认使用输入文件名")
    parser.add_argument("--replace", action="store_true", help="显式替换同一天已有观测")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="姿态置信度下限")
    return parser


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("features"), dict):
        raise ValueError("输入 JSON 必须包含对象类型的 features")
    if not isinstance(payload.get("quality"), dict):
        raise ValueError("输入 JSON 必须包含对象类型的 quality")
    return payload


def _load_npz(path: Path, min_confidence: float) -> dict:
    with np.load(path, allow_pickle=False) as data:
        required_fields = ("landmarks", "timestamps", "visibility")
        missing_fields = [name for name in required_fields if name not in data]
        if missing_fields:
            raise ValueError(
                "姿态 NPZ 必须同时包含 landmarks、timestamps、visibility；"
                f"当前缺少：{'、'.join(missing_fields)}"
            )
        landmarks = np.asarray(data["landmarks"])
        timestamps = np.asarray(data["timestamps"], dtype=np.float64)
        if len(timestamps) > 1 and float(np.median(np.diff(timestamps))) > 2.0:
            timestamps = timestamps / 1000.0
        confidence = np.asarray(data["visibility"], dtype=np.float64)
    return summarize_pose_sequence(
        landmarks,
        timestamps,
        confidence,
        min_confidence=min_confidence,
    ).to_dict()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_path = Path(args.input)
    if not source_path.exists():
        raise ValueError(f"输入文件不存在：{source_path}")
    if source_path.suffix.lower() == ".json":
        payload = _load_json(source_path)
    elif source_path.suffix.lower() == ".npz":
        payload = _load_npz(source_path, args.min_confidence)
    else:
        raise ValueError("输入格式只支持 .json 或 .npz")

    observation = DailyObservation.from_dict({
        "person_id": args.person_id,
        "observed_at": args.observed_at,
        "features": payload["features"],
        "quality": payload["quality"],
        "psychological": payload.get("psychological", {}),
        "source": args.source or source_path.name,
    })
    store = ProfileStore(args.profile_dir, args.person_id)
    store.append_observation(observation, replace=args.replace)
    print(json.dumps(observation.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _entrypoint() -> int:
    try:
        return main()
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
