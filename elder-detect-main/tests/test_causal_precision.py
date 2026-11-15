"""逐日冻结标准误（因果精度）的单元测试。"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mobilise_d.precision import (  # noqa: E402
    PrecisionConfig,
    apply_causal_pooled_fallback,
    daily_direct_stats,
    daily_precision,
    daily_precision_causal,
)

CONFIG = PrecisionConfig(samples=100, min_bouts=5)


def bout_frame(days: dict[str, int], *, person: str = "p1", visit: str = "t1") -> pd.DataFrame:
    rows = []
    base = {"averagecadence": 90.0, "averagestridespeed": 0.8, "averagestrideduration": 1.2}
    for day, count in days.items():
        for index in range(count):
            rows.append({
                "participantid": person,
                "datatype": "wb",
                "visit_number": visit,
                "wbday": 1,
                "day": pd.Timestamp(day),
                "duration": 40.0,
                "averagecadence": base["averagecadence"] + index * 0.5,
                "averagestridespeed": base["averagestridespeed"] + index * 0.01,
                "averagestrideduration": base["averagestrideduration"] + index * 0.002,
                "initialcontact_event_number": 50.0,
            })
    return pd.DataFrame(rows)


class CausalPrecisionTest(unittest.TestCase):
    def setUp(self):
        self.days = {"2024-01-01": 2, "2024-01-02": 6, "2024-01-03": 3}
        self.frame = bout_frame(self.days)

    def test_direct_stats_do_not_use_other_days(self):
        stats = daily_direct_stats(self.frame, config=CONFIG)
        first = stats.loc[stats["day"].eq("2024-01-01")].iloc[0]
        second = stats.loc[stats["day"].eq("2024-01-02")].iloc[0]
        self.assertEqual(int(first["n_cadence_spm"]), 2)
        self.assertTrue(pd.isna(first["se_cadence_spm"]))
        self.assertAlmostEqual(float(first["sd_cadence_spm"]), 0.5 / np.sqrt(2.0), places=10)
        self.assertIn(second["method_cadence_spm"], ("analytic_iid", "block2_bootstrap"))
        self.assertFalse(pd.isna(second["se_cadence_spm"]))

    def test_causal_fallback_uses_only_prior_days(self):
        causal = daily_precision_causal(self.frame, config=CONFIG)
        first = causal.loc[causal["day"].eq("2024-01-01")].iloc[0]
        second = causal.loc[causal["day"].eq("2024-01-02")].iloc[0]
        third = causal.loc[causal["day"].eq("2024-01-03")].iloc[0]
        self.assertTrue(pd.isna(first["se_cadence_spm"]))
        self.assertEqual(first["se_source_cadence_spm"], "unavailable")
        self.assertEqual(first["reason_cadence_spm"], "no_prior_pool")
        self.assertEqual(second["se_source_cadence_spm"], "direct")
        self.assertEqual(third["se_source_cadence_spm"], "pooled_prior")
        self.assertEqual(int(third["pooled_source_days_cadence_spm"]), 1)
        self.assertEqual(int(third["pooled_bouts_cadence_spm"]), 6)
        expected = float(second["sd_cadence_spm"]) / np.sqrt(3.0)
        self.assertAlmostEqual(third["se_cadence_spm"], expected, places=10)
        self.assertEqual(third["information_cutoff_day_cadence_spm"], "2024-01-03")

    def test_legacy_fallback_uses_future_days(self):
        legacy = daily_precision(self.frame, config=CONFIG)
        first = legacy.loc[legacy["day"].eq("2024-01-01")].iloc[0]
        third = legacy.loc[legacy["day"].eq("2024-01-03")].iloc[0]
        self.assertEqual(first["method_cadence_spm"], "pooled_sd")
        self.assertIsNotNone(first["se_cadence_spm"])
        self.assertAlmostEqual(
            float(first["se_cadence_spm"]),
            float(legacy.loc[legacy["day"].eq("2024-01-02"), "sd_cadence_spm"].iloc[0])
            / np.sqrt(2.0),
        )
        self.assertEqual(third["method_cadence_spm"], "pooled_sd")

    def test_prefix_matches_full_causal_table(self):
        full = daily_precision_causal(self.frame, config=CONFIG)
        prefix = daily_precision_causal(
            self.frame.loc[self.frame["day"].le("2024-01-02")], config=CONFIG,
        )
        columns = ["day", "se_cadence_spm", "se_walking_speed_m_s", "se_stride_duration_s",
                   "se_source_cadence_spm", "pooled_source_days_cadence_spm"]
        full_part = full.loc[full["day"].le("2024-01-02"), columns].reset_index(drop=True)
        prefix = prefix[columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(full_part, prefix)

    def test_future_anomaly_does_not_change_prior_rows(self):
        full = daily_precision_causal(self.frame, config=CONFIG)
        extended = pd.concat([
            self.frame,
            bout_frame({"2024-01-04": 6}, visit="t2"),
        ], ignore_index=True)
        extended.loc[extended["day"].eq("2024-01-04"), "averagecadence"] += 40.0
        changed = daily_precision_causal(extended, config=CONFIG)
        columns = [column for column in full.columns if column != "fallback_walking_speed_m_s"]
        pd.testing.assert_frame_equal(
            full[columns].reset_index(drop=True),
            changed.loc[changed["day"].le("2024-01-03"), columns].reset_index(drop=True),
        )

    def test_no_prior_pool_is_reported_with_reason(self):
        frame = bout_frame({"2024-01-01": 2}, person="p2")
        causal = daily_precision_causal(frame, config=CONFIG)
        row = causal.iloc[0]
        self.assertTrue(pd.isna(row["se_cadence_spm"]))
        self.assertEqual(row["reason_cadence_spm"], "no_prior_pool")
        self.assertEqual(row["se_source_cadence_spm"], "unavailable")

    def test_causal_fallback_accepts_precomputed_direct_stats(self):
        stats = daily_direct_stats(self.frame, config=CONFIG)
        direct = apply_causal_pooled_fallback(stats, CONFIG)
        rebuilt = daily_precision_causal(self.frame, config=CONFIG)
        pd.testing.assert_frame_equal(
            direct.reset_index(drop=True), rebuilt.reset_index(drop=True)
        )


if __name__ == "__main__":
    unittest.main()
