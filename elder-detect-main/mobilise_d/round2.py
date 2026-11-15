"""Round 2（阶段 11–15）：多队列装配、1/3/7 天标签、数据缺口矩阵与人员划分。

与阶段 1 的差别（均为本轮计划的显式要求）：

* 预测窗口固定为 1、3、7 天，7 天为主分析；不再使用 7/30 天或半年终点。
* 真实分支 R 只用有**逐次跌倒日期**的队列（本发布包中只有 PD）；MS 只有
  半年总数、PFF/COPD 只有半年回忆，都不能拆成短窗真值。
* 主特征组按 C6 在读取性能之前固定：步速、步长、段间跨步周期变异。
* 基线资格按主特征组的日表覆盖判定；时段间变异由阶段 12 的逐段资料补足。

本模块不训练风险模型、不生成增强数据。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from io import BytesIO
import math
from typing import Any, Iterable, Mapping, Sequence
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .clinical import ClinicalData
from .cohort import CohortConfig, CohortResult, build_cohort
from .daily import FIELD_SPECS, WEAR_COLUMNS
from .sources import member_by_suffix


HORIZONS_DAYS: tuple[int, ...] = (1, 3, 7)
PRIMARY_HORIZON_DAYS: int = 7
COHORTS: tuple[str, ...] = ("pd", "ms", "pff", "copd")
COHORT_COLUMN: dict[str, str] = {name: f"cohort {name}" for name in COHORTS}
# 有逐次跌倒日期的队列（其余队列的半年总数不能拆成短窗真值）。
DATED_COHORTS: tuple[str, ...] = ("pd",)

# C6：本轮主特征组。名称 → 日表列（均为已核验来源），方向向量预注册。
R2_FEATURES: tuple[str, ...] = (
    "gait_speed_mean",
    "stride_length_mean_cm",
    "stride_duration_bout_cv",
)
R2_DIRECTIONS: dict[str, float] = {
    "gait_speed_mean": -1.0,
    "stride_length_mean_cm": -1.0,
    "stride_duration_bout_cv": 1.0,
}
R2_SOURCE_COLUMN: dict[str, str] = {
    "gait_speed_mean": "ws_30_avg_d",
    "stride_length_mean_cm": "strlen_30_avg_d",
    "stride_duration_bout_cv": "strdur_all_var_d",
}
# 阶段 11 的日表级基线资格：前三列均有定义且当日至少两个步行段，
# 段间变异与精度在阶段 12 由逐段资料补足（缺失时以 2 特征掩码评分）。
R2_BASELINE_COLUMNS: tuple[str, ...] = (
    "ws_30_avg_d",
    "strlen_30_avg_d",
    "strdur_all_avg_d",
    "strdur_all_var_d",
)
R2_MIN_BOUTS_PER_DAY: int = 2

# 历史三特征（仅作历史消融/敏感性，不作为本轮主特征组）。
HISTORY_FEATURES: tuple[str, ...] = ("walking_speed_m_s", "cadence_spm", "stride_duration_s")
HISTORY_DIRECTIONS: dict[str, float] = {
    "walking_speed_m_s": -1.0,
    "cadence_spm": -1.0,
    "stride_duration_s": 1.0,
}
HISTORY_SOURCE: dict[str, str] = {
    "walking_speed_m_s": "ws_30_avg_d",
    "cadence_spm": "cadence_all_avg_d",
    "stride_duration_s": "strdur_all_avg_d",
}

EXTENDED_FEATURES: tuple[str, ...] = R2_FEATURES + (
    "walking_speed_p90",
    "walk_minutes_per_wear_hour",
)
EXTENDED_DIRECTIONS: dict[str, float] = {
    **R2_DIRECTIONS,
    "walking_speed_p90": -1.0,
    # 活动量受佩戴时长控制但仍缺方向依据：只作双侧通道。
    "walk_minutes_per_wear_hour": 0.0,
}


@dataclass(frozen=True)
class Round2Config:
    """阶段 11 冻结的队列与标签口径。"""

    min_age: float = 65.0
    cohorts: tuple[str, ...] = COHORTS
    baseline_days: int = 14
    alignment_days: int = 14
    diary_window_months: int = 6
    horizons: tuple[int, ...] = HORIZONS_DAYS
    min_bouts_per_day: int = R2_MIN_BOUTS_PER_DAY
    low_wear_waking_h: float = 10.0

    def cohort_config(self, cohort: str) -> CohortConfig:
        return CohortConfig(
            min_age=self.min_age,
            cohort_column=COHORT_COLUMN[cohort],
            baseline_days=self.baseline_days,
            alignment_days=self.alignment_days,
            diary_window_months=self.diary_window_months,
            horizons=self.horizons,
            feature_columns=R2_BASELINE_COLUMNS,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_age": self.min_age,
            "cohorts": list(self.cohorts),
            "baseline_days": self.baseline_days,
            "alignment_days": self.alignment_days,
            "diary_window_months": self.diary_window_months,
            "horizons_days": list(self.horizons),
            "primary_horizon_days": PRIMARY_HORIZON_DAYS,
            "min_bouts_per_day": self.min_bouts_per_day,
            "low_wear_waking_h": self.low_wear_waking_h,
            "baseline_columns": list(R2_BASELINE_COLUMNS),
            "main_features": list(R2_FEATURES),
            "main_directions": dict(R2_DIRECTIONS),
            "history_features": list(HISTORY_FEATURES),
            "history_directions": dict(HISTORY_DIRECTIONS),
            "extended_features": list(EXTENDED_FEATURES),
            "extended_directions": dict(EXTENDED_DIRECTIONS),
            "dated_fall_cohorts": list(DATED_COHORTS),
        }


@dataclass
class Round2Cohort:
    """多队列装配结果；每张表都带 ``cohort`` 列。"""

    participants: pd.DataFrame
    daily: pd.DataFrame
    visits: pd.DataFrame
    diaries: pd.DataFrame
    reports: pd.DataFrame
    fall_history: pd.DataFrame
    gates: pd.DataFrame
    landmarks: pd.DataFrame
    baseline_days: pd.DataFrame
    windows: pd.DataFrame
    flow: pd.DataFrame
    counts: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)

    @property
    def short_windows(self) -> pd.DataFrame:
        return self.windows.loc[
            self.windows["horizon_days"].isin(HORIZONS_DAYS)
        ].copy()


def load_fall_history(clinical_path) -> pd.DataFrame:
    """读取 T1–T5 的半年跌倒回忆（``falls-injuries``），只作临床参照。"""
    with ZipFile(clinical_path) as archive:
        payload = archive.read(member_by_suffix(archive, "falls-injuries.csv"))
    frame = pd.read_csv(BytesIO(payload), dtype=str)
    frame = frame.rename(columns={"visit-number": "visit_number"})
    frame["participantid"] = frame["participantid"].astype(str)
    frame["visit_number"] = frame["visit_number"].astype(str).str.lower()
    frame["fallshist"] = frame["fallshist"].map({"T": True, "F": False})
    frame["fallsnum"] = pd.to_numeric(frame["fallsnum"], errors="coerce")
    frame["fallsinjury"] = frame["fallsinjury"].map({"T": True, "F": False})
    frame["day"] = pd.to_datetime(frame["visit-date"], errors="coerce").dt.normalize()
    return frame


def load_baseline_fall_history(clinical_path) -> pd.DataFrame:
    """T1 基线的过去 12 个月跌倒史（``descriptives1``）。"""
    with ZipFile(clinical_path) as archive:
        payload = archive.read(member_by_suffix(archive, "descriptives1.csv"))
    frame = pd.read_csv(BytesIO(payload), dtype=str)
    frame["participantid"] = frame["participantid"].astype(str)
    out = pd.DataFrame({
        "participantid": frame["participantid"],
        "fallhistyr": frame["fallhistyr"].map({"T": True, "F": False}),
        "fallnumyr": pd.to_numeric(frame["fallnumyr"], errors="coerce"),
    })
    return out.groupby("participantid", as_index=False).max(numeric_only=False)


def _annotate_cohort(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    out = frame.copy()
    out["cohort"] = cohort
    return out


def _concat(frames: Sequence[pd.DataFrame], columns: Sequence[str] | None = None) -> pd.DataFrame:
    kept = [frame for frame in frames if frame is not None and len(frame)]
    if not kept:
        return pd.DataFrame(columns=list(columns or ()))
    out = pd.concat(kept, ignore_index=True, sort=False)
    if columns is not None:
        for column in columns:
            if column not in out.columns:
                out[column] = None
        out = out[list(dict.fromkeys(list(columns) + [c for c in out.columns if c not in columns]))]
    return out


def build_round2_cohort(
    clinical: ClinicalData,
    daily: pd.DataFrame,
    config: Round2Config,
    *,
    fall_history: pd.DataFrame | None = None,
    baseline_history: pd.DataFrame | None = None,
) -> Round2Cohort:
    """按队列分别装配，再合并为一张带 ``cohort`` 列的表。"""
    frame = daily.copy()
    frame["participantid"] = frame["participantid"].astype(str)
    frame["visit_number"] = frame["visit_number"].astype(str).str.lower()
    for column in R2_BASELINE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["wb_all_sum_d"] = pd.to_numeric(frame["wb_all_sum_d"], errors="coerce")
    frame["has_history_gait"] = (
        frame[["ws_30_avg_d", "cadence_all_avg_d", "strdur_all_avg_d"]]
        .notna().all(axis=1)
        & frame[["ws_30_avg_d", "cadence_all_avg_d", "strdur_all_avg_d"]].gt(0).all(axis=1)
    )

    results: list[tuple[str, CohortResult]] = []
    for cohort in config.cohorts:
        result = build_cohort(clinical, frame, config.cohort_config(cohort))
        results.append((cohort, result))

    participants = _concat(
        [_annotate_cohort(result.participants, cohort) for cohort, result in results]
    )
    daily_out = _concat(
        [_annotate_cohort(result.daily, cohort) for cohort, result in results]
    )
    visits = _concat([_annotate_cohort(result.visits, cohort) for cohort, result in results])
    diaries = _concat([_annotate_cohort(result.diaries, cohort) for cohort, result in results])
    reports = _concat([_annotate_cohort(result.reports, cohort) for cohort, result in results])
    gates = _concat([_annotate_cohort(result.gates, cohort) for cohort, result in results])
    landmarks = _concat(
        [_annotate_cohort(result.landmarks, cohort) for cohort, result in results]
    )
    baseline_days = _concat(
        [_annotate_cohort(result.baseline_days, cohort) for cohort, result in results]
    )
    windows = _concat(
        [_annotate_cohort(result.windows, cohort) for cohort, result in results]
    )
    flow = _concat(
        [pd.concat([pd.DataFrame({"cohort": [cohort]}).reindex(result.flow.index).ffill(),
                    result.flow], axis=1) for cohort, result in results]
    )

    for table in (participants, daily_out, visits, diaries, reports, gates,
                  landmarks, baseline_days, windows):
        if len(table) and "participantid" in table.columns:
            table["participantid"] = table["participantid"].astype(str)

    if len(landmarks):
        landmarks = _add_prior_fall_covariate(
            landmarks, gates, diaries, baseline_history, config
        )
        landmarks["age"] = landmarks["participantid"].map(
            participants.set_index("participantid")["age"]
        ).astype(float)

    counts = _counts(results, config)
    notes = {
        "per_cohort": {
            cohort: {
                **result.counts,
                "notes": result.notes,
                "flow_steps": int(len(result.flow)),
            }
            for cohort, result in results
        },
        "dated_fall_reports_total": int(reports["day"].notna().sum()) if len(reports) else 0,
        "dated_fall_reports_by_cohort": (
            reports.loc[reports["day"].notna()].groupby("cohort")["participantid"].nunique().to_dict()
            if len(reports) else {}
        ) if "cohort" in reports.columns else {},
    }
    return Round2Cohort(
        participants=participants, daily=daily_out, visits=visits, diaries=diaries,
        reports=reports, fall_history=fall_history if fall_history is not None else pd.DataFrame(),
        gates=gates, landmarks=landmarks, baseline_days=baseline_days,
        windows=windows, flow=flow, counts=counts, notes=notes,
    )


def _add_prior_fall_covariate(
    landmarks: pd.DataFrame,
    gates: pd.DataFrame,
    diaries: pd.DataFrame,
    baseline_history: pd.DataFrame | None,
    config: Round2Config,
) -> pd.DataFrame:
    """预测时点已知的既往跌倒 covariate。

    优先级：预测日所在访视的半年日记（与该访视的逐次报告一致时）→ T1 的
    过去 12 个月跌倒史。两者都不可用则记 ``unknown``，不填空值。
    """
    out = landmarks.copy()
    gate_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if len(gates):
        gate_index = gates.set_index(["participantid", "visit_number"])
        for key, row in gate_index.iterrows():
            gate_lookup[(str(key[0]), str(key[1]))] = row.to_dict()
    for row in out.itertuples(index=False):
        key = (str(row.participantid), str(row.visit_number))
        gate = gate_lookup.get(key, {})
        diary_total = gate.get("fallstotal")
        diary_ok = bool(gate.get("fall_reports_consistent", False)) and pd.notna(diary_total)
        prior_6m = int(diary_total) if diary_ok else None
        out.loc[
            (out["participantid"] == row.participantid)
            & (out["landmark_day"] == row.landmark_day)
            & (out["visit_number"] == row.visit_number),
            "prior_fall_6m",
        ] = prior_6m
        out.loc[
            (out["participantid"] == row.participantid)
            & (out["landmark_day"] == row.landmark_day)
            & (out["visit_number"] == row.visit_number),
            "prior_fall_6m_known",
        ] = bool(prior_6m is not None)
    out["prior_fall_6m"] = pd.to_numeric(out.get("prior_fall_6m"), errors="coerce")
    out["prior_fall_6m_known"] = out.get("prior_fall_6m_known", pd.Series(False, index=out.index)).astype(bool)

    if baseline_history is not None and len(baseline_history):
        history = baseline_history.set_index("participantid")
        out["fallhistyr"] = out["participantid"].map(history["fallhistyr"])
        out["fallnumyr"] = pd.to_numeric(
            out["participantid"].map(history["fallnumyr"]), errors="coerce"
        )
    else:
        out["fallhistyr"] = None
        out["fallnumyr"] = np.nan
    out["fallhistyr"] = out["fallhistyr"].astype("object")

    def resolve(row) -> tuple[float, str]:
        if row["prior_fall_6m_known"] and pd.notna(row["prior_fall_6m"]):
            return float(row["prior_fall_6m"] > 0), "diary_6m"
        value = row["fallhistyr"]
        if isinstance(value, (bool, np.bool_)):
            return float(bool(value)), "baseline_12m"
        return float("nan"), "unknown"

    resolved = [resolve(row) for _, row in out.iterrows()]
    out["prior_fall_any"] = [item[0] for item in resolved]
    out["prior_fall_source"] = [item[1] for item in resolved]
    return out


def _counts(results: Sequence[tuple[str, CohortResult]], config: Round2Config) -> dict:
    counts: dict[str, Any] = {"per_cohort": {}, "pooled": {"horizons": {}}}
    for cohort, result in results:
        counts["per_cohort"][cohort] = result.counts
    for horizon in config.horizons:
        rows = []
        for cohort, result in results:
            part = result.windows.loc[
                (result.windows["horizon_days"] == horizon)
                & (result.windows["status"] == "eligible")
            ]
            rows.append(part)
        part = _concat(rows)
        if not len(part):
            counts["pooled"]["horizons"][f"{horizon}d"] = {
                "eligible_landmarks": 0, "eligible_people": 0,
                "positive_landmarks": 0, "positive_people": 0,
                "baseline_ready_eligible_landmarks": 0,
                "baseline_ready_positive_landmarks": 0,
            }
            continue
        positive = part.loc[part["event"] == True]  # noqa: E712
        ready = part.loc[part["baseline_eligible"]]
        ready_positive = ready.loc[ready["event"] == True]  # noqa: E712
        counts["pooled"]["horizons"][f"{horizon}d"] = {
            "eligible_landmarks": int(len(part)),
            "eligible_people": int(part["participantid"].nunique()),
            "positive_landmarks": int(len(positive)),
            "positive_people": int(positive["participantid"].nunique()),
            "baseline_ready_eligible_landmarks": int(len(ready)),
            "baseline_ready_eligible_people": int(ready["participantid"].nunique()),
            "baseline_ready_positive_landmarks": int(len(ready_positive)),
            "baseline_ready_positive_people": int(ready_positive["participantid"].nunique()),
        }
    counts["dated_report_people_by_cohort"] = {}
    for cohort, result in results:
        dated = result.reports.loc[result.reports["day"].notna()]
        counts["dated_report_people_by_cohort"][cohort] = int(dated["participantid"].nunique())
    return counts


def short_window_table(windows: pd.DataFrame) -> pd.DataFrame:
    """把逐窗行透视为每个预测日一行：``y_1d / y_3d / y_7d``。

    同一预测日的三个标签来自同一事件时间，只要求已知标签满足
    ``y_1d <= y_3d <= y_7d``；未知标签保持空值，不填阴性。
    """
    frame = windows.loc[windows["horizon_days"].isin(HORIZONS_DAYS)].copy()
    if not len(frame):
        return pd.DataFrame(columns=["participantid", "landmark_day"])
    keys = ["participantid", "cohort", "visit_number", "diary_visit_number", "landmark_day"]
    frame["day_key"] = frame["landmark_day"].astype(str)
    event = frame.pivot_table(
        index=keys, columns="horizon_days", values="event", aggfunc="first"
    )
    status = frame.pivot_table(
        index=keys, columns="horizon_days", values="status", aggfunc="first"
    )
    first = frame.pivot_table(
        index=keys, columns="horizon_days", values="first_event_day", aggfunc="first"
    )
    counts = frame.pivot_table(
        index=keys, columns="horizon_days", values="n_reports_in_window", aggfunc="first"
    )
    out = pd.DataFrame(index=event.index)
    for horizon in HORIZONS_DAYS:
        label = f"{horizon}d"
        out[f"y_{label}"] = event.get(horizon)
        out[f"status_{label}"] = status.get(horizon)
        out[f"first_event_{label}"] = first.get(horizon)
        out[f"n_reports_{label}"] = counts.get(horizon)
    out = out.reset_index()
    out["y_1d_num"] = out["y_1d"].map({True: 1.0, False: 0.0})
    out["y_3d_num"] = out["y_3d"].map({True: 1.0, False: 0.0})
    out["y_7d_num"] = out["y_7d"].map({True: 1.0, False: 0.0})
    return out


def nested_label_audit(short: pd.DataFrame) -> dict:
    """短窗标签的连接核验：嵌套、同日事件、窗口边界与覆盖状态。"""
    known = short.dropna(subset=["y_1d_num", "y_3d_num", "y_7d_num"])
    violations = known.loc[
        (known["y_1d_num"] > known["y_3d_num"]) | (known["y_3d_num"] > known["y_7d_num"])
    ]
    boundary: list[dict[str, Any]] = []
    for row in short.itertuples(index=False):
        landmark = pd.Timestamp(row.landmark_day)
        for horizon in HORIZONS_DAYS:
            value = getattr(row, f"first_event_{horizon}d", None)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            gap = int((pd.Timestamp(str(value)) - landmark).days)
            boundary.append({
                "participantid": row.participantid,
                "landmark_day": row.landmark_day,
                "horizon_days": horizon,
                "first_event_day": str(value),
                "gap_days": gap,
                "in_window": bool(0 < gap <= horizon),
                "at_boundary": bool(gap in HORIZONS_DAYS),
            })
    boundary_frame = pd.DataFrame(boundary)
    status_counts = {}
    for horizon in HORIZONS_DAYS:
        column = f"status_{horizon}d"
        if column in short.columns:
            status_counts[f"{horizon}d"] = short[column].value_counts(dropna=False).to_dict()
    return {
        "rows": int(len(short)),
        "rows_with_all_three_known": int(len(known)),
        "nested_violations": int(len(violations)),
        "nested_violation_keys": violations[[
            c for c in ("participantid", "landmark_day", "y_1d_num", "y_3d_num", "y_7d_num")
            if c in violations.columns
        ]].to_dict("records")[:20],
        "boundary_rows": int(len(boundary_frame)),
        "boundary_violations": int((~boundary_frame["in_window"]).sum()) if len(boundary_frame) else 0,
        "events_at_exact_horizon_boundary": (
            boundary_frame.loc[boundary_frame["at_boundary"], "horizon_days"].value_counts().to_dict()
            if len(boundary_frame) else {}
        ),
        "status_counts": status_counts,
        "positives_by_horizon": {
            f"{h}d": int((short[f"y_{h}d_num"] == 1).sum()) for h in HORIZONS_DAYS
        },
        "unknown_by_horizon": {
            f"{h}d": int(short[f"y_{h}d_num"].isna().sum()) for h in HORIZONS_DAYS
        },
    }


def assign_person_folds(
    landmarks: pd.DataFrame,
    shares: int = 3,
    *,
    seed: int = 20261115,
    stratify_by: Sequence[str] = ("any_event", "cohort"),
) -> pd.DataFrame:
    """人员级分层折；同一人的全部预测日落在同一折。"""
    frame = landmarks.copy()
    frame["participantid"] = frame["participantid"].astype(str)
    person = frame.groupby("participantid", as_index=False).agg(
        cohort=("cohort", "first"),
        any_event=("y_7d_num", lambda values: float(np.nanmax(values)) if values.notna().any() else 0.0),
        n_landmarks=("landmark_day", "size"),
    )
    rng = np.random.default_rng(seed)
    folds = np.full(len(person), -1, dtype=int)
    strata: list[str] = []
    for _, row in person.iterrows():
        key = "|".join(str(row[column]) for column in stratify_by if column in person.columns)
        strata.append(key)
    person["stratum"] = strata
    for stratum, group in person.groupby("stratum", sort=True):
        order = group.sort_values("participantid").index.to_numpy()
        rng.shuffle(order)
        for position, index in enumerate(order):
            folds[person.index.get_loc(index)] = position % shares
    person["fold"] = folds
    return person[["participantid", "cohort", "any_event", "n_landmarks", "stratum", "fold"]]


def data_gap_matrix(
    cohort: Round2Cohort,
    config: Round2Config,
    *,
    history_gap: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """逐字段的数据缺口表：来源、单位、覆盖、缺口类型与可用分支。"""
    daily = cohort.daily
    rows: list[dict[str, Any]] = []
    spec_lookup = {spec["column"]: spec for spec in FIELD_SPECS}

    def coverage(column: str, cohort_name: str | None = None) -> tuple[int, int, int]:
        frame = daily if cohort_name is None else daily.loc[daily["cohort"] == cohort_name]
        values = pd.to_numeric(frame.get(column), errors="coerce")
        valid = values.notna() & values.gt(0)
        return int(valid.sum()), int(len(frame)), int(frame.loc[valid, "participantid"].nunique())

    for column in (
        "ws_30_avg_d", "ws_30_p90_d", "ws_10_p90_d", "strlen_30_avg_d",
        "strdur_all_avg_d", "strdur_all_var_d", "strdur_30_avg_d", "cadence_all_avg_d",
        "wb_all_sum_d", "wb_30_sum_d", "walkdur_all_sum_d", "wbsteps_all_sum_d",
        "total_worn_during_waking_h_d",
    ):
        spec = spec_lookup.get(column, {})
        valid, total, people = coverage(column)
        per_cohort = {
            name: coverage(column, name)[0] for name in config.cohorts
        }
        per_cohort_total = {
            name: coverage(column, name)[1] for name in config.cohorts
        }
        raw_null = int(pd.to_numeric(daily[column], errors="coerce").isna().sum()) if column in daily else total
        rows.append({
            "group": "daily_gait",
            "field": column,
            "source": "Aggregated DMO Data.zip / daily_agg_filtered.csv",
            "unit": spec.get("unit", "unknown"),
            "unit_verified": column not in ("strdur_all_var_d",),
            "rows_defined": valid,
            "rows_total": total,
            "coverage": round(valid / total, 4) if total else None,
            "people_defined": people,
            "raw_null": raw_null,
            "no_long_walking_bout": (
                int(pd.to_numeric(daily["wb_all_sum_d"], errors="coerce").gt(0).sum()
                    - pd.to_numeric(daily[column], errors="coerce").notna().sum())
                if column.startswith(("ws_30", "strlen_30", "strdur_30")) else 0
            ),
            "low_wear_lt_10h": int(
                pd.to_numeric(daily[WEAR_COLUMNS[1]], errors="coerce").lt(config.low_wear_waking_h).sum()
            ),
            "per_cohort_defined": per_cohort,
            "per_cohort_total": per_cohort_total,
            "supplement_type": "observed",
            "usable_branches": "S,R",
            "evaluation_limit": "",
        })

    # 逐段推导字段（阶段 12 生成）
    for column, unit, note in (
        ("measurement_se_gait_speed", "m/s", "同日步行段 bootstrap；段数不足用既往合并标准差"),
        ("measurement_se_stride_length", "cm", "同速，段级步长×100"),
        ("stride_duration_bout_cv", "1", "日表 var_d 单位不可核验，改由逐段资料的段间方差/均值重算"),
        ("measurement_se_stride_duration_bout_cv", "1", "逐段 bootstrap 重算 CV"),
        ("n_bouts", "count", "同日步行段数"),
    ):
        rows.append({
            "group": "bout_derived",
            "field": column,
            "source": "Walking Bout DMO Data.zip / CVS-T*-wb-dmo.zip",
            "unit": unit,
            "unit_verified": True,
            "rows_defined": None,
            "rows_total": int(len(daily)),
            "coverage": None,
            "people_defined": None,
            "raw_null": None,
            "no_long_walking_bout": None,
            "low_wear_lt_10h": None,
            "per_cohort_defined": {},
            "per_cohort_total": {},
            "supplement_type": "derived",
            "usable_branches": "S,R",
            "evaluation_limit": note,
        })

    # 临床与结局字段
    parts = {
        "age": ("Clinical Data.zip / descriptives1.csv", "years", int(cohort.participants["age"].notna().sum())),
        "fallhistyr": ("Clinical Data.zip / descriptives1.csv", "boolean", int(cohort.participants["participantid"].map(
            cohort.landmarks.drop_duplicates("participantid").set_index("participantid")["fallhistyr"]
        ).notna().sum()) if len(cohort.landmarks) else 0),
        "prior_fall_any": ("derived: diary(t-1) 或 fallhistyr", "boolean",
                           int(cohort.landmarks["prior_fall_any"].notna().sum()) if len(cohort.landmarks) else 0),
        "dated_fall_reports": ("Clinical Data.zip / falls-reports.csv", "date",
                               int(cohort.reports["day"].notna().sum()) if len(cohort.reports) else 0),
        "falls_diary_total": ("Clinical Data.zip / falls-diaries.csv", "count",
                              int(cohort.diaries["fallstotal"].notna().sum()) if len(cohort.diaries) else 0),
        "falls_history_recall": ("Clinical Data.zip / falls-injuries.csv", "boolean",
                                 int(len(cohort.fall_history)) if len(cohort.fall_history) else 0),
    }
    for field_name, (source, unit, defined) in parts.items():
        rows.append({
            "group": "clinical_outcome",
            "field": field_name,
            "source": source,
            "unit": unit,
            "unit_verified": True,
            "rows_defined": defined,
            "rows_total": int(len(cohort.landmarks)) if field_name in ("prior_fall_any",) else None,
            "coverage": (
                round(defined / len(cohort.landmarks), 4) if field_name == "prior_fall_any" and len(cohort.landmarks) else None
            ),
            "people_defined": None,
            "raw_null": None,
            "no_long_walking_bout": None,
            "low_wear_lt_10h": None,
            "per_cohort_defined": {
                name: int((cohort.landmarks["cohort"] == name).sum()) if len(cohort.landmarks) else 0
                for name in config.cohorts
            },
            "per_cohort_total": {
                name: int((cohort.landmarks["cohort"] == name).sum()) if len(cohort.landmarks) else 0
                for name in config.cohorts
            },
            "supplement_type": "observed",
            "usable_branches": "S,R" if field_name in ("age", "prior_fall_any", "dated_fall_reports") else "S",
            "evaluation_limit": (
                "只有 PD 有逐次跌倒日期；MS 仅半年总数、PFF/COPD 仅半年回忆，均不能构造 1/3/7 天真值"
                if field_name == "dated_fall_reports" else ""
            ),
        })

    # 预测时点层面的缺口
    if len(cohort.landmarks):
        total = int(len(cohort.landmarks))
        for name, mask, detail in (
            ("baseline_eligible", ~cohort.landmarks["baseline_eligible"].astype(bool),
             f"预测日前不足 {config.baseline_days} 个主特征有效日"),
            ("prior_fall_known", cohort.landmarks["prior_fall_source"].eq("unknown"),
             "既往跌倒史不可用（日记不一致且无 T1 12 个月史）"),
            ("low_wear_block", cohort.landmarks["block_days_wear_lt_10h"].gt(0),
             "采集块内有清醒佩戴 <10h 的日期"),
        ):
            rows.append({
                "group": "landmark",
                "field": name,
                "source": "derived: landmark table",
                "unit": "boolean",
                "unit_verified": True,
                "rows_defined": int((~mask).sum()),
                "rows_total": total,
                "coverage": round(float((~mask).mean()), 4),
                "people_defined": int(cohort.landmarks.loc[~mask, "participantid"].nunique()),
                "raw_null": None,
                "no_long_walking_bout": None,
                "low_wear_lt_10h": None,
                "per_cohort_defined": {
                    name: int(((~mask) & (cohort.landmarks["cohort"] == name)).sum())
                    for name in config.cohorts
                },
                "per_cohort_total": {
                    name: int((cohort.landmarks["cohort"] == name).sum())
                    for name in config.cohorts
                },
                "supplement_type": "observed" if name != "baseline_eligible" else "derived",
                "usable_branches": "S,R",
                "evaluation_limit": detail,
            })
    frame = pd.DataFrame(rows)
    if history_gap:
        frame.attrs["history_gap"] = dict(history_gap)
    return frame


def real_candidate_table(cohort: Round2Cohort) -> pd.DataFrame:
    """R 分支候选表：真实预测日、既往协变量与 1/3/7 天标签。"""
    windows = short_window_table(cohort.short_windows)
    if not len(windows):
        return windows
    landmarks = cohort.landmarks.copy()
    keep = [
        "participantid", "cohort", "visit_number", "landmark_day", "diary_visit_number",
        "diary_day", "covered_from", "n_valid_days_in_block", "block_first_day",
        "block_last_day", "block_max_gap_days", "n_prior_valid_days", "baseline_eligible",
        "baseline_days_used", "baseline_first_day", "baseline_end_day",
        "days_from_baseline_end_to_landmark", "monitoring_start_day",
        "n_monitored_days_in_block", "landmark_walking_speed_m_s", "landmark_wear_waking_h",
        "block_min_wear_waking_h", "block_days_wear_lt_10h", "age",
        "prior_fall_6m", "prior_fall_6m_known", "fallhistyr", "fallnumyr",
        "prior_fall_any", "prior_fall_source",
    ]
    merged = windows.merge(
        landmarks[[c for c in keep if c in landmarks.columns]],
        on=["participantid", "cohort", "visit_number", "landmark_day"],
        how="left", validate="one_to_one",
    )
    merged["r_branch_eligible"] = (
        merged["baseline_eligible"].astype(bool)
        & merged["cohort"].isin(DATED_COHORTS)
        & merged["prior_fall_any"].notna()
    )
    return merged


def daily_field_audit(daily: pd.DataFrame) -> pd.DataFrame:
    """逐列覆盖与缺口分类：原始空值、无长步行段、低佩戴。"""
    rows = []
    total = int(len(daily))
    wear = pd.to_numeric(daily.get(WEAR_COLUMNS[1]), errors="coerce")
    bouts = pd.to_numeric(daily.get("wb_all_sum_d"), errors="coerce")
    long_bouts = pd.to_numeric(daily.get("wb_30_sum_d"), errors="coerce")
    for column in daily.columns:
        series = pd.to_numeric(daily[column], errors="coerce")
        if series.notna().sum() == 0:
            continue
        defined = series.notna() & series.gt(0) if series.min() >= 0 else series.notna()
        rows.append({
            "column": column,
            "rows": total,
            "defined": int(defined.sum()),
            "coverage": round(float(defined.mean()), 4),
            "raw_null": int(series.isna().sum()),
            "zero": int(series.eq(0).sum()),
            "no_long_walking_bout": int((long_bouts.fillna(0) == 0).sum()),
            "low_wear_lt_10h": int(wear.lt(10.0).sum()),
            "median": float(series.median()) if series.notna().any() else None,
        })
    return pd.DataFrame(rows)
