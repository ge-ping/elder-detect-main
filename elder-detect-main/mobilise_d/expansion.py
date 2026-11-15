"""Mobilise-D S14/S7 候选入组与阳性人员集合统计。

S14 是“至少 14 个既往有效日”的基线长度筛选候选集；S7 是预设的
“至少 7 个既往有效日”探索入组。这里的计数全部是候选预测日层级，
不是完成 7 日基线拟合/精度质控后的可评分样本。
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


def _ids(series: pd.Series) -> pd.Series:
    values = []
    for value in series:
        text = str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        values.append(text)
    return pd.Series(values, index=series.index, dtype="object")


def _days(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="raise").dt.strftime("%Y-%m-%d")


def _bools(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes", "y"))


def _key_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame["participantid"], frame["landmark_day"]))


def _people(frame: pd.DataFrame) -> set[str]:
    return set(frame["participantid"])


def _prepare(
    landmarks: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    horizon_days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    landmarks = landmarks.copy()
    windows = windows.copy()
    for frame, name in ((landmarks, "landmarks"), (windows, "landmark_windows")):
        missing = {"participantid", "landmark_day"} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} 缺少字段：{sorted(missing)}")
        frame["participantid"] = _ids(frame["participantid"])
        frame["landmark_day"] = _days(frame["landmark_day"])
    if landmarks.duplicated(["participantid", "landmark_day"]).any():
        raise ValueError("landmarks 存在重复的 (participantid, landmark_day) 键")
    if "baseline_eligible" not in landmarks.columns:
        raise ValueError("landmarks 缺少 baseline_eligible")
    if "n_prior_valid_days" not in landmarks.columns:
        raise ValueError("landmarks 缺少 n_prior_valid_days")
    if "horizon_days" not in windows.columns or "event" not in windows.columns:
        raise ValueError("landmark_windows 缺少 horizon_days 或 event")
    windows = windows.loc[
        pd.to_numeric(windows["horizon_days"], errors="coerce").eq(horizon_days)
    ].copy()
    if windows.duplicated(["participantid", "landmark_day"]).any():
        raise ValueError(f"landmark_windows 在 {horizon_days} 天窗口存在重复键")
    return landmarks, windows


def _summary(
    s14: pd.DataFrame,
    s7: pd.DataFrame,
    positive14: set[tuple[str, str]],
    positive7: set[tuple[str, str]],
    people14: set[str],
    people7: set[str],
    positive_people14: set[str],
    positive_people7: set[str],
    new_positive_landmarks: set[tuple[str, str]],
    extra_positive_landmarks: set[tuple[str, str]],
    categories: Mapping[str, set[tuple[str, str]]],
) -> dict[str, Any]:
    new_people = people7 - people14
    new_positive_people = positive_people7 - positive_people14
    from_new_people = new_positive_people & new_people
    from_existing_people = new_positive_people & people14
    from_new_landmarks = {
        key for key in extra_positive_landmarks if key[0] in new_people
    }
    from_existing_landmarks = {
        key for key in extra_positive_landmarks if key[0] in from_existing_people
    }
    from_already_landmarks = {
        key for key in extra_positive_landmarks if key[0] in positive_people14
    }
    summary = {
        "S14": {
            "landmarks": int(len(s14)),
            "people": int(len(people14)),
            "positive_landmarks": int(len(positive14)),
            "positive_people": int(len(positive_people14)),
        },
        "S7": {
            "landmarks": int(len(s7)),
            "people": int(len(people7)),
            "positive_landmarks": int(len(positive7)),
            "positive_people": int(len(positive_people7)),
        },
        "s14_candidate_landmarks": int(len(s14)),
        "s7_candidate_landmarks": int(len(s7)),
        "s14_people": int(len(people14)),
        "s7_people": int(len(people7)),
        "people_intersection": int(len(people14 & people7)),
        "people_union": int(len(people14 | people7)),
        "overlapping_people": int(len(people14 & people7)),
        "new_people": int(len(new_people)),
        "s14_positive_people": int(len(positive_people14)),
        "s7_positive_people": int(len(positive_people7)),
        "s7_new_positive_people_total": int(len(new_positive_people)),
        "s7_new_positive_people_from_new_people": int(len(from_new_people)),
        "s7_existing_people_newly_positive": int(len(from_existing_people)),
        # 历史字段：只表示“新增受试者中的阳性”，不是全部新增阳性。
        "new_positive_people": int(len(from_new_people)),
        # 历史字段：只表示新增受试者中的阳性预测日。
        "new_positive_landmarks": int(len(from_new_landmarks)),
        "s7_positive_landmarks_from_new_people": int(len(from_new_landmarks)),
        "S7_extra_positive_landmarks_not_in_S14": int(len(extra_positive_landmarks)),
        "s7_extra_positive_landmarks_total": int(len(extra_positive_landmarks)),
        "s7_extra_positive_landmarks_from_new_people": int(len(from_new_landmarks)),
        "s7_extra_positive_landmarks_from_existing_people_newly_positive": int(
            len(from_existing_landmarks)
        ),
        "s7_extra_positive_landmarks_from_already_positive_people": int(
            len(from_already_landmarks)
        ),
        "s7_extra_positive_landmarks_unclassified": int(
            len(
                extra_positive_landmarks
                - from_new_landmarks
                - from_existing_landmarks
                - from_already_landmarks
            )
        ),
        "positive_keys": {
            "S14": sorted(f"{p}|{d}" for p, d in positive14),
            "S7": sorted(f"{p}|{d}" for p, d in positive7),
            "extra": sorted(f"{p}|{d}" for p, d in extra_positive_landmarks),
        },
        "positive_people_sets": {
            "S14": sorted(positive_people14),
            "S7": sorted(positive_people7),
            "new_total": sorted(new_positive_people),
            "from_new_people": sorted(from_new_people),
            "from_existing_people": sorted(from_existing_people),
        },
        "new_positive_landmark_categories": {
            "from_new_people": sorted(f"{p}|{d}" for p, d in from_new_landmarks),
            "from_existing_people_newly_positive": sorted(
                f"{p}|{d}" for p, d in from_existing_landmarks
            ),
            "from_already_positive_people": sorted(
                f"{p}|{d}" for p, d in from_already_landmarks
            ),
        },
        "identities": {
            "new_positive_people_6_plus_4": (
                len(from_new_people) + len(from_existing_people)
                == len(new_positive_people)
            ),
            "extra_positive_landmarks_6_plus_4_plus_1": (
                len(from_new_landmarks)
                + len(from_existing_landmarks)
                + len(from_already_landmarks)
                == len(extra_positive_landmarks)
            ),
            "s14_positive_landmarks_subset_of_s7": positive14 <= positive7,
            "s14_people_subset_of_s7": people14 <= people7,
            "s14_positive_people_subset_of_s7": positive_people14 <= positive_people7,
            "new_people_disjoint_s14_people": not (new_people & people14),
            "extra_positive_landmarks_equal_s7_minus_s14": (
                extra_positive_landmarks == positive7 - positive14
            ),
            "new_positive_landmarks_are_new_people": (
                new_positive_landmarks
                == {key for key in positive7 if key[0] in from_new_people}
            ),
            "positive_people_derived_from_keys": (
                {key[0] for key in positive7} == positive_people7
                and {key[0] for key in positive14} == positive_people14
            ),
            "categories_disjoint": (
                not (from_new_landmarks & from_existing_landmarks)
                and not (from_new_landmarks & from_already_landmarks)
                and not (from_existing_landmarks & from_already_landmarks)
            ),
        },
        "old_field_semantics": {
            "new_positive_people": "仅 S7 新增受试者中至少有一个 30 天阳性的受试者数；不是 S7 相对 S14 的全部新增阳性受试者。",
            "new_positive_landmarks": "仅 S7 新增受试者中的 30 天阳性预测日数；不是全部额外阳性预测日。",
            "S7_extra_positive_landmarks_not_in_S14": "S7 阳性键减去 S14 阳性键的全部额外阳性预测日。",
        },
        "definitions": {
            "candidate_landmark": "landmarks.csv 的 (participantid, landmark_day) 唯一候选预测日；S14=baseline_eligible=true，S7=n_prior_valid_days>=7。",
            "positive_landmark": "landmark_windows.csv 中 horizon_days=30、event=true，且键属于对应候选集的预测日。",
            "positive_person": "在对应候选集内至少有一个 30 天阳性预测日的受试者，按 participantid 去重。",
            "new_person": "S7 人员集合 − S14 人员集合。",
            "new_positive_person_total": "S7 阳性人员集合 − S14 阳性人员集合。",
            "extra_positive_landmark": "S7 阳性预测键 − S14 阳性预测键。",
            "candidate_vs_scorable": "候选入组由基线长度筛选；S14 的 session_se 主分析可评分样本另计（172/134/16/14）。S7 尚未做 7 日基线拟合和精度质控，本文件不给出 S7 可评分数。",
        },
        "caveat": "S7 只是候选入组统计，缩短基线会改变个人参考精度；不能把它当作已完成 7 日基线拟合或与 S14 的 172 个 session_se 可评分日直接配对。",
    }
    return summary


def analyze_expansion(
    landmarks: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    horizon_days: int = 30,
) -> dict[str, Any]:
    """返回摘要、逐预测日集合表和人员集合表。"""
    landmarks, windows = _prepare(landmarks, windows, horizon_days=horizon_days)
    s14 = landmarks.loc[_bools(landmarks["baseline_eligible"])].copy()
    s7 = landmarks.loc[
        pd.to_numeric(landmarks["n_prior_valid_days"], errors="coerce").ge(7)
    ].copy()
    people14, people7 = _people(s14), _people(s7)
    event_by_key = dict(zip(
        zip(windows["participantid"], windows["landmark_day"]),
        _bools(windows["event"]),
    ))
    s14_keys, s7_keys = _key_set(s14), _key_set(s7)
    missing = (s14_keys | s7_keys) - set(event_by_key)
    if missing:
        raise ValueError(f"候选预测日缺少 30 天结局：{sorted(missing)[:5]}")
    positive14 = {key for key in s14_keys if event_by_key[key]}
    positive7 = {key for key in s7_keys if event_by_key[key]}
    positive_people14 = {key[0] for key in positive14}
    positive_people7 = {key[0] for key in positive7}
    new_people = people7 - people14
    new_positive_people = positive_people7 - positive_people14
    from_new_people = new_positive_people & new_people
    from_existing_people = new_positive_people & people14
    extra_positive = positive7 - positive14
    categories = {
        "from_new_people": {key for key in extra_positive if key[0] in new_people},
        "from_existing_people_newly_positive": {
            key for key in extra_positive if key[0] in from_existing_people
        },
        "from_already_positive_people": {
            key for key in extra_positive if key[0] in positive_people14
        },
    }
    category_by_key = {
        key: category for category, keys in categories.items() for key in keys
    }
    summary = _summary(
        s14, s7, positive14, positive7, people14, people7,
        positive_people14, positive_people7,
        {key for key in positive7 if key[0] in from_new_people},
        extra_positive, categories,
    )

    landmark_table = landmarks.copy()
    landmark_table["in_s14"] = [
        key in s14_keys for key in zip(
            landmark_table["participantid"], landmark_table["landmark_day"]
        )
    ]
    landmark_table["in_s7"] = [
        key in s7_keys for key in zip(
            landmark_table["participantid"], landmark_table["landmark_day"]
        )
    ]
    landmark_table["event_30d"] = [
        bool(event_by_key.get(key, False)) for key in zip(
            landmark_table["participantid"], landmark_table["landmark_day"]
        )
    ]
    landmark_table["s14_positive"] = [
        key in positive14 for key in zip(
            landmark_table["participantid"], landmark_table["landmark_day"]
        )
    ]
    landmark_table["s7_positive"] = [
        key in positive7 for key in zip(
            landmark_table["participantid"], landmark_table["landmark_day"]
        )
    ]
    landmark_table["person_group"] = [
        "new_person" if person in new_people
        else ("s14_person" if person in people14 else "outside_s14")
        for person in landmark_table["participantid"]
    ]
    landmark_table["positive_group"] = [
        "not_in_s7" if not in_s7
        else (
            "negative" if not positive
            else ("new_person_positive" if person in from_new_people
                  else ("existing_person_newly_positive" if person in from_existing_people
                        else ("already_positive_person" if person in positive_people14
                              else "unclassified_positive")))
        )
        for person, in_s7, positive in zip(
            landmark_table["participantid"], landmark_table["in_s7"],
            landmark_table["s7_positive"],
        )
    ]
    landmark_table["extra_positive"] = [
        key in extra_positive for key in zip(
            landmark_table["participantid"], landmark_table["landmark_day"]
        )
    ]
    landmark_table["extra_positive_category"] = [
        category_by_key.get(key, "") for key in zip(
            landmark_table["participantid"], landmark_table["landmark_day"]
        )
    ]

    people_table = pd.DataFrame({"participantid": sorted(people14 | people7)})
    people_table["in_s14"] = people_table["participantid"].isin(people14)
    people_table["in_s7"] = people_table["participantid"].isin(people7)
    people_table["s14_positive"] = people_table["participantid"].isin(positive_people14)
    people_table["s7_positive"] = people_table["participantid"].isin(positive_people7)
    people_table["is_new_person"] = people_table["participantid"].isin(new_people)
    people_table["new_positive_total"] = people_table["participantid"].isin(new_positive_people)
    people_table["new_positive_from_new_person"] = people_table["participantid"].isin(from_new_people)
    people_table["new_positive_from_existing_person"] = people_table["participantid"].isin(from_existing_people)
    people_table["already_s14_positive"] = people_table["participantid"].isin(positive_people14)

    extra_table = landmark_table.loc[landmark_table["extra_positive"]].copy()
    extra_table = extra_table.sort_values(["participantid", "landmark_day"])

    return {
        "summary": summary,
        "landmark_sets": landmark_table.sort_values(["participantid", "landmark_day"]),
        "positive_people_sets": people_table,
        "extra_positive_landmarks": extra_table,
        "identities": summary["identities"],
    }


def compute_expansion_from_frames(
    landmarks: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    horizon_days: int = 30,
) -> dict[str, Any]:
    """只返回可直接 JSON 序列化的摘要（阶段 7 兼容入口）。"""
    return analyze_expansion(
        landmarks, windows, horizon_days=horizon_days
    )["summary"]
