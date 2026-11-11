#!/usr/bin/env python
"""
env_labels_to_risk.py — EnvRisk R1 第三段（独立工具）：VLM 弱标签 → risk_环境 sidecar

读 env_labels/<ds>/<stem>.json（视频级聚合）→ 按 §8 映射口径算 risk_环境
→ 写 sidecar  pose_cache/<ds>/<stem>.env_risk.npy

⚠️ 日常推荐直接用 env_perceive.py（R1 标签优先、R0 规则回退，单一入口）。
本脚本保留为「只转标签、不动 R0」的轻量工具，逻辑与 env_perceive 的 R1
分支共用 model.env.env_risk_model 的 risk_from_label / save_env_sidecar。

    用法:
        python scripts/env_labels_to_risk.py
    （纯口径 B：VLM 标签 → 同 R0 规则表，无 --blend）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.env.env_risk_model import risk_from_label, save_env_sidecar

CACHE_DIR = PROJECT_ROOT / "pose_cache"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--labels_root", default="env_labels")
    p.add_argument("--cache", default=str(CACHE_DIR))
    args = p.parse_args()
    labels_root = PROJECT_ROOT / args.labels_root
    cache = Path(args.cache)
    if not labels_root.exists():
        print(f"[ERR] labels_root 不存在: {labels_root}")
        return

    jsons = [j for j in sorted(labels_root.rglob("*.json"))
             if "_raw" not in j.parts]
    if not jsons:
        print(f"[ERR] 在 {labels_root} 下未找到任何聚合标签 json")
        return

    print(f"待映射 {len(jsons)} 个视频标签 → sidecar（纯口径 B：VLM 标签 → 规则表）")
    ok, skipped, warned = 0, 0, 0
    for jp in jsons:
        ds = jp.parent.name
        stem = jp.stem
        try:
            agg = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"  [SKIP] {ds}/{stem} 读取失败: {e}")
            skipped += 1
            continue

        if "corridor_clear" not in agg:
            print(f"  [SKIP] {ds}/{stem} 旧格式 room/light/floor 不兼容 HOME FAST")
            skipped += 1
            continue
        sc = risk_from_label(agg)          # 现返回 dict（total/normalized/level/coverage/need_review...）
        risk = sc["normalized"]
        npz = cache / ds / f"{stem}.npz"
        meta = {
            "source": "R1_vlm",
            "ds": ds, "stem": stem,
            "room_type": agg.get("room_type"),
            "lighting": agg.get("lighting"),
            "homefast_total": sc["total"],
            "homefast_n_observed": sc["n_observed"],
            "homefast_coverage": sc["coverage"],
            "homefast_low_coverage": sc["low_coverage"],
            "homefast_level": sc["level"],
            "vlm_confidence": agg.get("confidence"),
            "need_review": sc["need_review"],
        }
        saved, T = save_env_sidecar(npz, risk, meta)
        if not saved:
            warned += 1
            print(f"  [WARN] {ds}/{stem}: 写 sidecar 失败（{meta.get('error')}）")
            continue
        ok += 1
        flag = "⚠标量" if T < 0 else f"(T={T})"
        print(f"  [OK] {ds}/{stem} risk_环境={risk:.3f} {flag}")

    print(f"\nDone. 写 sidecar {ok} 个；WARN {warned}；SKIP {skipped}")


if __name__ == "__main__":
    main()
