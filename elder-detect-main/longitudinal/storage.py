"""基于文件的个人纵向画像存储。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import DailyObservation


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError("保存对象必须是字典或提供 to_dict()")


class ProfileStore:
    """保存一个人的每日观测、基线和最新风险结果。"""

    def __init__(self, root: str | Path, person_id: str) -> None:
        person_id = str(person_id).strip()
        if (
            not person_id
            or person_id in {".", ".."}
            or "/" in person_id
            or "\\" in person_id
        ):
            raise ValueError("person_id 不能包含路径字符")
        self.root = Path(root).resolve()
        self.person_id = person_id
        self.profile_dir = self.root / person_id
        self.observations_path = self.profile_dir / "observations.jsonl"
        self.baseline_path = self.profile_dir / "baseline.json"
        self.latest_risk_path = self.profile_dir / "latest_risk.json"

    def append_observation(
        self,
        observation: DailyObservation,
        replace: bool = False,
    ) -> None:
        if observation.person_id != self.person_id:
            raise ValueError("观测 person_id 与画像目录不一致")
        items = self.load_observations()
        existing = {
            item.observation_date: index for index, item in enumerate(items)
        }
        date = observation.observation_date
        if date in existing and not replace:
            raise ValueError(f"重复观测日期：{date}")
        if date in existing:
            items[existing[date]] = observation
        else:
            items.append(observation)
        items.sort(key=lambda item: item.observed_at)
        content = "".join(
            json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for item in items
        )
        self._atomic_write_text(self.observations_path, content)

    def load_observations(self) -> list[DailyObservation]:
        if not self.observations_path.exists():
            return []
        observations: list[DailyObservation] = []
        for line_number, line in enumerate(
            self.observations_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                observation = DailyObservation.from_dict(json.loads(line))
                if observation.person_id != self.person_id:
                    raise ValueError(
                        "观测 person_id 与画像目录 person_id 不一致"
                    )
                observations.append(observation)
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"观测文件第 {line_number} 行无效：{exc}"
                ) from exc
        return sorted(observations, key=lambda item: item.observed_at)

    def save_baseline(self, baseline: Any) -> None:
        self._save_json(self.baseline_path, _as_dict(baseline))

    def load_baseline(self) -> dict[str, Any] | None:
        if not self.baseline_path.exists():
            return None
        return json.loads(self.baseline_path.read_text(encoding="utf-8"))

    def save_latest_result(self, result: Any) -> None:
        self._save_json(self.latest_risk_path, _as_dict(result))

    def _save_json(self, path: Path, data: dict[str, Any]) -> None:
        content = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        self._atomic_write_text(path, content)

    def _atomic_write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
