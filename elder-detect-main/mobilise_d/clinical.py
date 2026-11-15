"""读取 Clinical Data.zip 中的受试者、访视、跌倒日记与逐次跌倒报告。

发布方用 ``participantid`` + ``visit-number``（t1..t5）标识一次临床访视；
跌倒日记的 ``fallstotal`` 是过去六个月的跌倒总数，``fallstotal`` 必须等于
同一次访视的逐次报告条数，逐次报告的 ``falldate`` 才是可用的日期证据。
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from .sources import member_by_suffix


VISIT_TABLE_SUFFIXES = ("descriptives1.csv", "status2-5.csv")


def _read(archive: ZipFile, suffix: str) -> pd.DataFrame:
    return pd.read_csv(BytesIO(archive.read(member_by_suffix(archive, suffix))), dtype=str)


def _normalize_day(value: pd.Series) -> pd.Series:
    return pd.to_datetime(value, errors="coerce").dt.normalize()


@dataclass(frozen=True)
class ClinicalData:
    """临床表格的规范化视图，日期字段统一为无时分的日历日。"""

    participants: pd.DataFrame
    visits: pd.DataFrame
    diaries: pd.DataFrame
    reports: pd.DataFrame

    def notes(self) -> dict:
        undated = int(self.reports["day"].isna().sum())
        return {
            "participants": int(len(self.participants)),
            "visit_rows": int(len(self.visits)),
            "visits_with_date": int(self.visits["day"].notna().sum()),
            "diary_rows": int(len(self.diaries)),
            "diary_rows_with_date": int(self.diaries["day"].notna().sum()),
            "fall_report_rows": int(len(self.reports)),
            "fall_report_rows_undated": undated,
            "fall_report_rows_dated": int(len(self.reports) - undated),
            "participants_with_dated_report": int(
                self.reports.loc[self.reports["day"].notna(), "participantid"].nunique()
            ),
        }


def load_clinical(path: Path) -> ClinicalData:
    """从 Clinical Data.zip 读取本阶段需要的四张表。"""
    with ZipFile(path) as archive:
        cohort_table = _read(archive, "Cohort Site-2026-06-05-10h09m34s.csv")
        baseline = _read(archive, "descriptives1.csv")
        status = _read(archive, "status2-5.csv")
        diaries = _read(archive, "falls-diaries.csv")
        reports = _read(archive, "falls-reports.csv")

    cohort_columns = [
        column for column in cohort_table.columns
        if column.startswith("cohort ") or column == "participant site"
    ]
    participants = cohort_table[["participantid"] + cohort_columns].rename(
        columns={"participant site": "site"}
    )
    ages = baseline[["participantid", "age"]].copy()
    ages["age"] = pd.to_numeric(ages["age"], errors="coerce")
    ages = ages.groupby("participantid", as_index=False)["age"].max()
    participants = participants.merge(ages, on="participantid", how="left")
    participants["is_pd"] = participants["cohort pd"].eq("T")
    participants["older_pd"] = participants["is_pd"] & participants["age"].ge(65)

    visit_frames = []
    for frame, source in ((baseline, "descriptives1"), (status, "status2-5")):
        part = frame[["participantid", "visit-number", "visit-date"]].copy()
        part["visit_source"] = source
        visit_frames.append(part)

    visits = pd.concat(visit_frames, ignore_index=True).rename(
        columns={"visit-number": "visit_number", "visit-date": "visit_date_raw"}
    )
    visits["day"] = _normalize_day(visits["visit_date_raw"])
    if "pstatus" in status.columns:
        status_flag = status[["participantid", "visit-number", "pstatus"]].rename(
            columns={"visit-number": "visit_number"}
        )
        visits = visits.merge(status_flag, on=["participantid", "visit_number"], how="left")
    undated = visits["day"].isna()
    visits = visits.loc[~undated].copy()

    diaries = diaries.rename(
        columns={"visit-number": "visit_number", "visit-date": "visit_date_raw"}
    )
    diaries["day"] = _normalize_day(diaries["visit_date_raw"])
    diaries["fallstotal"] = pd.to_numeric(diaries["fallstotal"], errors="coerce")
    diaries["fallsreports"] = pd.to_numeric(diaries["fallsreports"], errors="coerce")

    reports = reports.rename(columns={"visit-number": "visit_number"})
    reports["day"] = _normalize_day(reports["falldate"])
    reports["fallrptnum"] = pd.to_numeric(reports["fallrptnum"], errors="coerce")

    return ClinicalData(
        participants=participants,
        visits=visits,
        diaries=diaries,
        reports=reports,
    )
