"""DS-PCD 统计层的仿真复核：零假设误报、方向特异性与累积行为。

口径：每人 14 个基线日 + 21 个监测日（含监测起始日 day 0）。监测期内
不模拟真实跌倒结局，只检查统计量与 CUSUM 的性质。阈值 ``h`` 只在
“已标注为仿真零假设”的序列上按 30 天内至少一次报警概率 ≤ 10% 标定，
并与阈值一起保存目标时窗、样本量和实测误报率。

运行：
    python scripts/sim_ds_pcd.py --out output/ds_pcd_mobilise/stage2/simulation
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.ds_pcd import (  # noqa: E402
    DayRecord,
    DsPcdConfig,
    fit_baseline,
    score_timeline,
)

FEATURES = ("gait_speed_proxy", "cadence_spm", "step_time_mean_s")
DIRECTIONS = {"gait_speed_proxy": -1.0, "cadence_spm": -1.0, "step_time_mean_s": 1.0}
# 仿真中三个特征的日间波动与测量噪声比例（与项目旧仿真口径同源）
TAU = np.array([1.0, 1.0, 1.0])
C30 = np.array([0.5, 0.4, 0.4])
CORRELATION = np.array([
    [1.0, 0.6, -0.55],
    [0.6, 1.0, -0.9],
    [-0.55, -0.9, 1.0],
])
DECLINE = np.array([-1.0, -0.7, 0.7])
BASELINE_DAYS = 14
MONITOR_DAYS = 21
RAMP_DAYS = 7
TARGET_FAR = 0.10


def _config(precision_mode: str, samples: int) -> DsPcdConfig:
    return DsPcdConfig(
        features=FEATURES,
        directions=DIRECTIONS,
        precision_mode=precision_mode,
        null_samples=samples,
        quantile_points=1000,
        tau_floor_relative=1e-3,
        outlier_z=100.0,
    )


def simulate_person_days(
    rng: np.random.Generator,
    count: int,
    *,
    steps: np.ndarray,
    shift: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """生成一个人的连续观测；返回 (x, s)。"""
    chol = np.linalg.cholesky(CORRELATION)
    natural = rng.standard_normal((count, 3)) @ chol.T
    noise_scale = C30 * np.sqrt(30.0 / steps[:, None])
    measurement = (rng.standard_normal((count, 3)) @ chol.T) * noise_scale
    x = natural * TAU[None, :] + measurement
    if shift is not None:
        profile = np.clip(
            (np.arange(count) - BASELINE_DAYS + 1) / RAMP_DAYS, 0.0, 1.0
        )
        profile[:BASELINE_DAYS] = 0.0
        x = x + shift[None, :] * profile[:, None]
    return x, noise_scale


def _records(
    x: np.ndarray,
    s: np.ndarray,
    start_day: int = 0,
    session_ids: np.ndarray | None = None,
) -> list[DayRecord]:
    records = []
    for index in range(len(x)):
        if session_ids is not None:
            day = int(session_ids[index])
        else:
            day = start_day + index
        records.append(DayRecord(
            date=(np.datetime64("2024-01-01") + np.timedelta64(day, "D")).astype(str),
            features={name: float(x[index, k]) for k, name in enumerate(FEATURES)},
            feature_se={name: float(s[index, k]) for k, name in enumerate(FEATURES)},
        ))
    return records


def run_scenario(
    persons: int,
    *,
    shift: np.ndarray | None,
    precision_mode: str,
    samples: int,
    seed: int,
    alarm_threshold: float | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    config = _config(precision_mode, samples)
    statistics = []
    cusums = []
    p_values = []
    for _ in range(persons):
        steps = np.exp(rng.uniform(np.log(20), np.log(150), BASELINE_DAYS + MONITOR_DAYS))
        x, s = simulate_person_days(rng, BASELINE_DAYS + MONITOR_DAYS, steps=steps, shift=shift)
        records = _records(x, s)
        try:
            timeline = score_timeline(records, "sim", config, alarm_threshold=alarm_threshold)
        except Exception:
            statistics.append(None)
            cusums.append(None)
            continue
        scored = timeline.scored_rows()
        statistics.append([row.statistic_t for row in scored])
        cusums.append([row.cusum for row in scored])
        p_values.extend(row.p_value for row in scored if row.p_value is not None)
    return {
        "statistics": statistics,
        "cusums": cusums,
        "p_values": p_values,
        "persons": persons,
        "seed": seed,
        "precision_mode": precision_mode,
    }


def _max_cusum(result: dict[str, Any]) -> np.ndarray:
    return np.asarray([
        max(series) if series else np.nan for series in result["cusums"]
    ], dtype=np.float64)


def calibrate_threshold(null: dict[str, Any], target_far: float = TARGET_FAR) -> dict[str, Any]:
    maximum = _max_cusum(null)
    usable = maximum[np.isfinite(maximum)]
    grid = np.round(np.arange(0.1, 25.01, 0.1), 1)
    chosen = None
    for threshold in grid:
        far = float((usable >= threshold).mean())
        if far <= target_far:
            chosen = {
                "threshold": float(threshold),
                "false_alarm_rate": far,
                "window_days": MONITOR_DAYS,
                "sample_size": int(len(usable)),
                "target": target_far,
            }
            break
    if chosen is None:
        chosen = {
            "threshold": float(grid[-1]),
            "false_alarm_rate": float((usable >= grid[-1]).mean()),
            "window_days": MONITOR_DAYS,
            "sample_size": int(len(usable)),
            "target": target_far,
            "note": "网格上限仍未达到目标误报率",
        }
    # Wilson 95% 区间，便于说明仿真不确定性
    n = chosen["sample_size"]
    p = chosen["false_alarm_rate"]
    if n:
        z = 1.959963984540054
        denominator = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denominator
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        chosen["wilson_95_interval"] = [max(0.0, centre - half), min(1.0, centre + half)]
    return chosen


def summarise(scenario: dict[str, Any], threshold: float) -> dict[str, Any]:
    statistics = [np.asarray(s, dtype=float) for s in scenario["statistics"] if s]
    cusums = [np.asarray(s, dtype=float) for s in scenario["cusums"] if s]
    maximum = np.asarray([max(s) for s in cusums], dtype=float)
    detected = maximum >= threshold
    delays = []
    for series in cusums:
        hits = np.flatnonzero(series >= threshold)
        if len(hits):
            delays.append(int(hits[0]))
    p_values = np.asarray(scenario["p_values"], dtype=float)
    return {
        "persons": scenario["persons"],
        "precision_mode": scenario["precision_mode"],
        "median_T_day1": float(np.median([s[0] for s in statistics if len(s)])),
        "median_T_last": float(np.median([s[-1] for s in statistics if len(s)])),
        "mean_cusum_by_day": [
            float(np.mean([s[index] for s in cusums if len(s) > index]))
            for index in range(MONITOR_DAYS)
        ],
        "detection_rate": float(detected.mean()),
        "median_delay_days": float(np.median(delays)) if delays else None,
        "max_cusum_median": float(np.median(maximum)),
        "p_value_mean": float(p_values.mean()) if len(p_values) else None,
        "p_value_below_0_05": float((p_values < 0.05).mean()) if len(p_values) else None,
        "p_value_below_0_10": float((p_values < 0.10).mean()) if len(p_values) else None,
    }


def run(
    out_dir: Path,
    *,
    persons: int = 400,
    samples: int = 4000,
    seed: int = 20261113,
) -> dict[str, Any]:
    started = time.time()
    payload: dict[str, Any] = {
        "protocol": {
            "baseline_days": BASELINE_DAYS,
            "monitor_days": MONITOR_DAYS,
            "includes_start_day": True,
            "ramp_days": RAMP_DAYS,
            "features": list(FEATURES),
            "directions": DIRECTIONS,
            "decline_shift": DECLINE.tolist(),
            "target_false_alarm_rate": TARGET_FAR,
            "seed": seed,
            "persons": persons,
            "null_samples": samples,
        },
        "results": {},
    }
    for mode in ("session_se", "none"):
        null = run_scenario(
            persons, shift=None, precision_mode=mode, samples=samples, seed=seed,
        )
        calibration = calibrate_threshold(null)
        threshold = calibration["threshold"]
        null_summary = summarise(null, threshold)
        # 三个场景使用同一随机流：同一受试者的自然波动与噪声相同，
        # 只差方向上的偏移，便于配对比较。
        decline = run_scenario(
            persons, shift=DECLINE, precision_mode=mode, samples=samples,
            seed=seed, alarm_threshold=threshold,
        )
        reverse = run_scenario(
            persons, shift=-DECLINE, precision_mode=mode, samples=samples,
            seed=seed, alarm_threshold=threshold,
        )
        payload["results"][mode] = {
            "calibration": calibration,
            "null": null_summary,
            "decline": summarise(decline, threshold),
            "reverse_direction": summarise(reverse, threshold),
        }
    payload["elapsed_seconds"] = round(time.time() - started, 1)
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "simulation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    primary = payload["results"]["session_se"]
    threshold_payload = {
        "threshold": primary["calibration"]["threshold"],
        "target_window_days": MONITOR_DAYS,
        "target_false_alarm_rate": TARGET_FAR,
        "measured_false_alarm_rate": primary["calibration"]["false_alarm_rate"],
        "wilson_95_interval": primary["calibration"].get("wilson_95_interval"),
        "sample_size": primary["calibration"]["sample_size"],
        "source": "simulation null（明确标注的仿真零假设，不是真实 30 天误报率）",
        "precision_mode": "session_se",
        "seed": seed,
        "persons": persons,
        "generated_at_utc": payload["generated_at_utc"],
        "protocol": payload["protocol"],
    }
    (out_dir / "threshold.json").write_text(
        json.dumps(threshold_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# DS-PCD 仿真复核",
        "",
        f"- 生成时间：{payload['generated_at_utc']}",
        f"- 每人 {BASELINE_DAYS} 个基线日 + {MONITOR_DAYS} 个监测日（含监测起始日）",
        f"- 受试者数：{persons}；零分布抽样：{samples}",
        f"- 仿真用时：{payload['elapsed_seconds']} 秒",
        "",
        "## 阈值（只在仿真零假设下标定）",
        "",
        f"- h = {threshold_payload['threshold']:.1f}，目标 {MONITOR_DAYS} 天内误报 ≤ "
        f"{TARGET_FAR:.0%}，实测 {threshold_payload['measured_false_alarm_rate']:.3f}"
        f"（95% 区间 [{float(threshold_payload['wilson_95_interval'][0]):.3f}, "
        f"{float(threshold_payload['wilson_95_interval'][1]):.3f}]）",
        "",
        "## 各模式结果",
        "",
        "| 模式 | 零假设误报 | 方向一致变化检出 | 中位延迟 | 反向变化检出 | 反向中位 T |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, result in payload["results"].items():
        lines.append(
            f"| {mode} | {result['null']['detection_rate']:.3f} | "
            f"{result['decline']['detection_rate']:.3f} | "
            f"{result['decline']['median_delay_days']} | "
            f"{result['reverse_direction']['detection_rate']:.3f} | "
            f"{result['reverse_direction']['median_T_last']:.2f} |"
        )
    lines += [
        "",
        "说明：仿真只检查统计层性质，不提供真实跌倒预测性能；"
        "真实数据只有半年一次访视，无法据此估计连续监测 30 天的真实误报率。",
        "",
    ]
    (out_dir / "simulation_report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DS-PCD 仿真复核")
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT.parents[1] / "output" / "ds_pcd_mobilise" / "stage2" / "simulation"),
    )
    parser.add_argument("--persons", type=int, default=400)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261113)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(
        Path(args.out), persons=args.persons, samples=args.samples, seed=args.seed,
    )
    primary = payload["results"]["session_se"]
    print(json.dumps({
        "threshold": primary["calibration"]["threshold"],
        "null_far": primary["null"]["detection_rate"],
        "decline_detection": primary["decline"]["detection_rate"],
        "reverse_detection": primary["reverse_direction"]["detection_rate"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
