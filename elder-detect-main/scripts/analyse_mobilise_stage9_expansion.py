"""阶段 9：修正 S7 扩样人数、预测日分解与报告口径。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mobilise_d.expansion import analyze_expansion  # noqa: E402

OUT_ROOT = PROJECT_ROOT.parents[1] / "output" / "ds_pcd_mobilise"
DEFAULT_RUN = OUT_ROOT / "20261114-fixed"
DEFAULT_OUT = OUT_ROOT / "stage9" / "20261114-expansion-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 9 S7 入组统计修正")
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    return parser


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["participantid"] = frame["participantid"].astype(str).str.strip()
    frame["landmark_day"] = pd.to_datetime(frame["landmark_day"]).dt.strftime("%Y-%m-%d")
    return frame


def _scorable(fixed: Path) -> dict[str, Any]:
    predictions = _normalise(pd.read_csv(
        fixed / "predictions.csv", dtype={"participantid": str}
    ))
    windows = _normalise(pd.read_csv(
        fixed / "data" / "landmark_windows.csv", dtype={"participantid": str}
    ))
    windows = windows.loc[windows["horizon_days"].eq(30)].copy()
    windows["event_bool"] = windows["event"].astype(str).str.lower().eq("true")
    event_by_key = dict(zip(
        zip(windows["participantid"], windows["landmark_day"]),
        windows["event_bool"],
    ))
    landmarks = _normalise(pd.read_csv(
        fixed / "data" / "landmarks.csv", dtype={"participantid": str}
    ))
    s14 = landmarks.loc[
        landmarks["baseline_eligible"].astype(str).str.lower().eq("true")
    ].copy()
    candidate_keys = set(zip(s14["participantid"], s14["landmark_day"]))
    base = predictions.loc[
        predictions["variant"].eq("fixed_session")
        & predictions["horizon"].eq(30)
    ]
    a = base.loc[base["method"].eq("A_rms")]
    b = base.loc[base["method"].eq("B_rms_cusum")]
    keys_a = set(zip(a["participantid"], a["landmark_day"]))
    keys_b = set(zip(b["participantid"], b["landmark_day"]))
    if keys_a != keys_b:
        raise ValueError("阶段 6 A/B 共同行键不一致")
    keys = keys_a
    people = {key[0] for key in keys}
    positive_keys = {key for key in keys if event_by_key.get(key, False)}
    positive_people = {key[0] for key in positive_keys}
    candidate_people = set(zip(s14["participantid"], s14["landmark_day"]))
    candidate_people = {key[0] for key in candidate_people}
    candidate_positive = {
        key for key in candidate_keys if event_by_key.get(key, False)
    }
    candidate_positive_people = {key[0] for key in candidate_positive}
    not_scorable = candidate_keys - keys
    return {
        "status": "estimated",
        "label": "S14 session_se 主分析",
        "source": "predictions.csv",
        "selection": "variant=fixed_session, horizon=30, method=A_rms；A/B 键一致",
        "candidate_landmarks": len(candidate_keys),
        "candidate_people": len(candidate_people),
        "landmarks": len(keys),
        "people": len(people),
        "positive_landmarks": len(positive_keys),
        "positive_people": len(positive_people),
        "not_scorable_landmarks": len(not_scorable),
        "not_scorable_people": len(candidate_people - people),
        "candidate_positive_landmarks": len(candidate_positive),
        "candidate_positive_people": len(candidate_positive_people),
        "not_scorable_positive_landmarks": len(candidate_positive - positive_keys),
        "not_scorable_positive_people": len(
            candidate_positive_people - positive_people
        ),
        "input_sha256": sha256(fixed / "predictions.csv"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run = Path(args.run).resolve()
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"目标目录已存在且非空：{out}")
    out.mkdir(parents=True, exist_ok=True)
    landmarks = pd.read_csv(run / "data" / "landmarks.csv", dtype={"participantid": str})
    windows = pd.read_csv(run / "data" / "landmark_windows.csv", dtype={"participantid": str})
    analysis = analyze_expansion(landmarks, windows)
    summary = analysis["summary"]
    scorable = _scorable(run)
    if not all(summary["identities"].values()):
        raise ValueError(f"集合恒等关系失败：{summary['identities']}")
    if (scorable["landmarks"], scorable["people"], scorable["positive_landmarks"],
            scorable["positive_people"]) != (172, 134, 16, 14):
        raise ValueError(f"S14 session_se 可评分数与冻结值不符：{scorable}")
    s7_scorable = {
        "status": "not_estimated",
        "reason": "本轮仅修正候选入组统计；7 日基线重新拟合、逐日精度质控和重新评分未执行。",
        "candidate_landmarks": summary["S7"]["landmarks"],
        "candidate_people": summary["S7"]["people"],
        "candidate_positive_landmarks": summary["S7"]["positive_landmarks"],
        "candidate_positive_people": summary["S7"]["positive_people"],
        "scorable_landmarks": None,
        "scorable_people": None,
        "scorable_positive_landmarks": None,
        "scorable_positive_people": None,
    }

    counts_rows = [
        ("s14_candidate_landmarks", summary["s14_candidate_landmarks"], "S14 候选预测日", "participantid+landmark_day", "S14=baseline_eligible=true；候选集，未做额外精度质控"),
        ("s7_candidate_landmarks", summary["s7_candidate_landmarks"], "S7 候选预测日", "participantid+landmark_day", "S7=n_prior_valid_days>=7；候选集"),
        ("s14_people", summary["s14_people"], "S14 人员", "participantid", "S14 候选集中至少一个预测日的受试者"),
        ("s7_people", summary["s7_people"], "S7 人员", "participantid", "S7 候选集中至少一个预测日的受试者"),
        ("people_intersection", summary["people_intersection"], "S14∩S7", "participantid", "两候选集人员交集"),
        ("people_union", summary["people_union"], "S14∪S7", "participantid", "两候选集人员并集"),
        ("s7_new_people", summary["new_people"], "S7−S14", "participantid", "S7 新增受试者"),
        ("s14_positive_people", summary["s14_positive_people"], "S14 阳性人员", "participantid", "S14 候选集内至少一个 30 天阳性预测日"),
        ("s7_positive_people", summary["s7_positive_people"], "S7 阳性人员", "participantid", "S7 候选集内至少一个 30 天阳性预测日"),
        ("s7_new_positive_people_total", summary["s7_new_positive_people_total"], "S7 阳性人员−S14 阳性人员", "participantid", "S7 相对 S14 的全部新增阳性受试者"),
        ("s7_new_positive_people_from_new_people", summary["s7_new_positive_people_from_new_people"], "S7 新增阳性人员∩S7 新增人员", "participantid", "来自 S14 从未纳入的新增受试者的阳性"),
        ("s7_existing_people_newly_positive", summary["s7_existing_people_newly_positive"], "S7 新增阳性人员∩S14 人员", "participantid", "S14 已纳入、但 S14 窗口内无阳性的受试者在 S7 新进入阳性集合"),
        ("s14_positive_landmarks", summary["S14"]["positive_landmarks"], "S14 阳性预测日", "participantid+landmark_day", "S14 候选集内 horizon=30、event=true"),
        ("s7_positive_landmarks", summary["S7"]["positive_landmarks"], "S7 阳性预测日", "participantid+landmark_day", "S7 候选集内 horizon=30、event=true"),
        ("s7_extra_positive_landmarks_total", summary["s7_extra_positive_landmarks_total"], "S7 阳性预测日−S14 阳性预测日", "participantid+landmark_day", "全部额外阳性预测日"),
        ("s7_extra_positive_landmarks_from_new_people", summary["s7_extra_positive_landmarks_from_new_people"], "额外阳性预测日∩S7 新增人员", "participantid+landmark_day", "来自新增受试者的额外阳性预测日"),
        ("s7_extra_positive_landmarks_from_existing_people_newly_positive", summary["s7_extra_positive_landmarks_from_existing_people_newly_positive"], "额外阳性预测日∩S14 人员新进入阳性集合", "participantid+landmark_day", "来自原有受试者新进入阳性集合的额外阳性预测日"),
        ("s7_extra_positive_landmarks_from_already_positive_people", summary["s7_extra_positive_landmarks_from_already_positive_people"], "额外阳性预测日∩S14 阳性人员", "participantid+landmark_day", "来自 S14 已阳性受试者的额外阳性预测日"),
        ("s14_session_scorable_landmarks", scorable["landmarks"], "S14 session_se 可评分预测日", "participantid+landmark_day", "阶段 6 fixed_session/30d/A_rms 键；A/B 一致"),
        ("s14_session_scorable_people", scorable["people"], "S14 session_se 可评分人员", "participantid", "上述可评分键的人员"),
        ("s14_session_scorable_positive_landmarks", scorable["positive_landmarks"], "S14 session_se 可评分阳性预测日", "participantid+landmark_day", "上述可评分键中 event=true"),
        ("s14_session_scorable_positive_people", scorable["positive_people"], "S14 session_se 可评分阳性人员", "participantid", "上述阳性预测日的人员"),
        ("s14_session_not_scorable_landmarks", scorable["not_scorable_landmarks"], "S14 候选−可评分", "participantid+landmark_day", "基线日缺少步速标准误等原因，见阶段 6 报告"),
        ("s7_scorable_landmarks", None, "S7 可评分预测日", "participantid+landmark_day", "未估计：7 日基线拟合、精度质控与重新评分未执行"),
        ("s7_scorable_people", None, "S7 可评分人员", "participantid", "未估计：同上"),
    ]
    counts_table = pd.DataFrame(counts_rows, columns=[
        "field", "value", "denominator", "dedup_key", "definition",
    ])
    counts_table.to_csv(out / "counts.csv", index=False, encoding="utf-8")

    counts = {
        "stage": 9,
        "run_id": out.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_run": str(run),
        "code_version": "stage9-expansion-v1",
        "inputs": {
            name: {
                "path": f"data/{name}",
                "sha256": sha256(run / "data" / name),
            }
            for name in ("landmarks.csv", "landmark_windows.csv")
        },
        "scorable_source": {
            "path": "predictions.csv",
            "sha256": scorable["input_sha256"],
        },
        "candidate_expansion": summary,
        "S14": summary["S14"],
        "S7": summary["S7"],
        "scorable": {
            "S14": scorable,
            "S7": s7_scorable,
            "definition": "S14 的 session_se 主分析可评分样本为阶段 6 fixed_session/30d 的 172 个共同键；S7 仍是候选层，未做 7 日基线拟合/精度质控。",
        },
        "required_counts": {
            row[0]: row[1] for row in counts_rows
        },
        "identities": summary["identities"],
        "definitions": summary["definitions"],
        "old_field_semantics": summary["old_field_semantics"],
        "caveat": summary["caveat"],
    }
    write_json(out / "counts.json", counts)
    analysis["landmark_sets"].to_csv(out / "landmark_sets.csv", index=False, encoding="utf-8")
    analysis["positive_people_sets"].to_csv(
        out / "positive_people_sets.csv", index=False, encoding="utf-8"
    )
    analysis["extra_positive_landmarks"].to_csv(
        out / "extra_positive_landmarks.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame([{
        "cohort": "S14",
        "status": scorable["status"],
        "candidate_landmarks": scorable["candidate_landmarks"],
        "scorable_landmarks": scorable["landmarks"],
        "scorable_people": scorable["people"],
        "scorable_positive_landmarks": scorable["positive_landmarks"],
        "scorable_positive_people": scorable["positive_people"],
        "not_scorable_landmarks": scorable["not_scorable_landmarks"],
        "note": "session_se 主分析；预测日键来自 predictions.csv fixed_session/30d",
    }, {
        "cohort": "S7",
        "status": s7_scorable["status"],
        "candidate_landmarks": s7_scorable["candidate_landmarks"],
        "scorable_landmarks": None,
        "scorable_people": None,
        "scorable_positive_landmarks": None,
        "scorable_positive_people": None,
        "not_scorable_landmarks": None,
        "note": s7_scorable["reason"],
    }]).to_csv(out / "scorable.csv", index=False, encoding="utf-8")

    extra = analysis["extra_positive_landmarks"]
    report = f"""# 阶段 9：修正 S7 扩样人数与报告口径

- 日期：2026-11-14
- 计划：`planning-files/ds-pcd-mobilise-implementation-plan.md`（Stage 9 与阶段 8–10 修复契约）
- 源运行：`{run}`
- 本阶段运行：`{out}`
- 代码版本：`stage9-expansion-v1`
- 输入：`data/landmarks.csv`（SHA-256 `{counts['inputs']['landmarks.csv']['sha256']}`）、`data/landmark_windows.csv`（SHA-256 `{counts['inputs']['landmark_windows.csv']['sha256']}`）

## 1. 修正后的结论

1. S7 相对 S14 新增 **10 名阳性受试者**，不是原文的 6 名。其中 **6 名**来自 S14 从未纳入的新增受试者，**4 名**来自 S14 已纳入但在 S14 窗口内没有阳性的受试者。
2. S7 比 S14 多 **11 个阳性预测日**，分解为 **11 = 6 + 4 + 1**：6 个来自新增受试者，4 个使原有受试者首次进入阳性集合，最后 1 个来自 S14 已阳性受试者的重复事件。
3. 两组的受试者交集为 **144**、并集为 **194**；S7 新增 50 人。原文“合并人数 144”应改为“交集 144、并集 194”。
4. 上述 S14/S7 都是**基线长度筛选后的候选集**。S14 的 `session_se` 主分析另有 **172 个可评分预测日、134 人、16 个阳性预测日、14 名阳性受试者**；S7 尚未做 7 日基线重新拟合和逐日精度质控，不能给出最终可评分数，也不能直接与 172 个 S14 可评分日配对。
5. “只新增 6 名独立阳性”的旧表述只对“新增受试者中的阳性”成立；它不能代表 S7 相对 S14 的全部新增阳性，也不足以单独作为停止扩样的推理。S7 是否值得继续需要另立探索方案，先完成 7 日基线拟合与精度质控。

## 2. 口径与分母

| 口径 | S14 | S7 | 去重键 | 定义 |
|---|---:|---:|---|---|
| 候选预测日 | 186 | 325 | `participantid+landmark_day` | S14=`baseline_eligible=true`；S7=`n_prior_valid_days>=7` |
| 候选人员 | 144 | 194 | `participantid` | 候选集中至少一个预测日 |
| 阳性预测日 | 17 | 28 | `participantid+landmark_day` | `horizon_days=30`、`event=true`，限定在对应候选集 |
| 阳性人员 | 15 | 25 | `participantid` | 对应候选集内至少一个阳性预测日 |

人员交集 **144**、并集 **194**、S7 新增 **50** 人。S14 阳性人员 15、S7 阳性人员 25，新增阳性人员 **10**。

## 3. 新增阳性人员分解

| 集合 | 人数 | 说明 |
|---|---:|---|
| S7 新增阳性受试者总数 | {summary['s7_new_positive_people_total']} | `S7阳性人员 − S14阳性人员` |
| 来自 S7 新增受试者 | {summary['s7_new_positive_people_from_new_people']} | `S7新增阳性人员 ∩ (S7人员−S14人员)` |
| 来自 S14 原有受试者新进入阳性集合 | {summary['s7_existing_people_newly_positive']} | `S7新增阳性人员 ∩ S14人员` |

6 + 4 = 10；明细见 `positive_people_sets.csv` 的 `new_positive_from_new_person` 与 `new_positive_from_existing_person` 列。

## 4. 额外阳性预测日分解

S7 阳性预测日 {summary['S7']['positive_landmarks']} − S14 阳性预测日 {summary['S14']['positive_landmarks']} = **{summary['s7_extra_positive_landmarks_total']}**。

| 来源 | 预测日数 | 人员 |
|---|---:|---|
| S7 新增受试者 | {summary['s7_extra_positive_landmarks_from_new_people']} | 6 名新受试者各 1 个 |
| S14 原有受试者新进入阳性集合 | {summary['s7_extra_positive_landmarks_from_existing_people_newly_positive']} | 4 名原有受试者各 1 个 |
| S14 已阳性受试者的重复事件 | {summary['s7_extra_positive_landmarks_from_already_positive_people']} | 1 名已阳性受试者 |
| **合计** | **{summary['s7_extra_positive_landmarks_total']}** | **11 = 6 + 4 + 1** |

逐行键、人员类别与 30 天结局见 `extra_positive_landmarks.csv`；全部 463 个候选预测日的 S14/S7/阳性标记见 `landmark_sets.csv`。

## 5. 候选入组 vs 最终可评分

| 队列 | 状态 | 候选预测日 | 可评分预测日 | 可评分人员 | 可评分阳性预测日 | 可评分阳性人员 |
|---|---|---:|---:|---:|---:|---:|
| S14 `session_se` 主分析 | 已评分（阶段 6） | 186 | 172 | 134 | 16 | 14 |
| S7 | 未估计 | 325 | — | — | — | — |

S14 从 186 个候选日降到 172 个可评分日，损失 14 个；其中 1 个阳性预测日、1 名阳性受试者（阶段 6 报告：基线日缺少步行速度标准误）。S7 的 28/25 只是候选层阳性计数，不能当作最终可评分阳性人数。

## 6. 旧字段含义（保留兼容）

- `new_positive_people`：历史字段，仅表示“S7 新增受试者中的阳性人数”= 6；新增字段 `s7_new_positive_people_total`=10 才是 S7 相对 S14 的全部新增阳性受试者。
- `new_positive_landmarks`：历史字段，仅表示“S7 新增受试者中的阳性预测日”= 6；`s7_extra_positive_landmarks_total`=11 是全部额外阳性预测日。
- `S7_extra_positive_landmarks_not_in_S14`：11，与新增字段一致。
- 人员交集/并集字段：`people_intersection`=144、`people_union`=194；不再使用“合并人数 144”。

## 7. 产物与核验

- `counts.json`：修正后的完整计数、分母、去重键、定义和集合恒等关系。
- `counts.csv`：便于引用的计数表。
- `landmark_sets.csv`：463 个候选预测日的 S14/S7/30 天阳性/人员/类别/额外阳性标记。
- `positive_people_sets.csv`：194 名人员集合与阳性/新增阳性类别。
- `extra_positive_landmarks.csv`：11 个额外阳性预测日及 6/4/1 分类。
- `scorable.csv`：S14 可评分与 S7 未估计状态。
- `verification.json`、`VERIFICATION.md`：独立集合核验与最小案例。

```bash
python scripts/analyse_mobilise_stage9_expansion.py
python scripts/verify_mobilise_stage9.py
python -m pytest tests -q
```

## 8. 边界

- 本阶段只做候选入组计数修正；没有训练、评分或外部验证。
- S7 的 7 日基线拟合、精度质控和重新评分属于后续探索研究，不能用本轮候选计数替代。
- 本阶段不改变阶段 6 的 30 天主结果（证据不足）和阶段 8 的输入契约结论。
"""
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    source_map = {
        "stage": 9,
        "run_id": out.name,
        "created_at_utc": counts["created_at_utc"],
        "source_run": str(run),
        "code_version": "stage9-expansion-v1",
        "inputs": counts["inputs"],
        "scorable_source": counts["scorable_source"],
        "script": "scripts/analyse_mobilise_stage9_expansion.py",
        "verification": "scripts/verify_mobilise_stage9.py",
        "test": "tests/test_stage9_expansion.py",
        "notes": [
            "所有计数从 data/landmarks.csv、data/landmark_windows.csv 和 predictions.csv 重新计算。",
            "S14/S7 是候选入组；S14 session_se 可评分 172/134/16/14 来自阶段 6 fixed_session 预测。",
            "S7 未做 7 日基线拟合/精度质控，不能给出最终可评分人数。",
        ],
    }
    write_json(out / "source_map.json", source_map)
    print(json.dumps({
        "out": str(out),
        "S14": summary["S14"], "S7": summary["S7"],
        "new_positive_total": summary["s7_new_positive_people_total"],
        "split": [summary["s7_new_positive_people_from_new_people"],
                  summary["s7_existing_people_newly_positive"]],
        "extra": summary["s7_extra_positive_landmarks_total"],
        "extra_split": [
            summary["s7_extra_positive_landmarks_from_new_people"],
            summary["s7_extra_positive_landmarks_from_existing_people_newly_positive"],
            summary["s7_extra_positive_landmarks_from_already_positive_people"],
        ],
        "scorable": {k: scorable[k] for k in ("landmarks", "people", "positive_landmarks", "positive_people")},
        "all_identities": all(summary["identities"].values()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
