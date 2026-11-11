#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vlm_baseline_calibrate.py — VLM 离线静态基线标定（一次性，部署时调用）

VLM(方舟 Doubao) 看场景首帧 → 输出地面区域顶部 y + 固定障碍物 bbox → 存静态基线 JSON。
运行时 EnvMaskBuilder bg_diff 加载此基线 + MOG2 新增物 + HSV 反光 + 人脚挖孔。

⚠️ 仅部署时调用一次，运行时不再调 VLM（边缘端纯算法，符合「一开始用后面不用」约定）。

用法:
  python scripts/vlm_baseline_calibrate.py --frame "<path>/cas100001.png" --scene Subject.1_Fall_backwards
  python scripts/vlm_baseline_calibrate.py --frame ... --mock   # 无网测试
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_config import ENV_VLM_ARK_BASE_URL, ENV_VLM_TIMEOUT  # noqa: E402


def _load_dotenv(path=PROJECT_ROOT / ".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


# ---- Prompts：标定静态基线（地面区域 + 固定障碍物）----
SYS_PROMPT = (
    "你是居家跌倒环境标定专家。给你一张室内监控画面，"
    "请标定静态环境基线：地面区域位置 + 画面中的固定障碍物（家具/地毯等，不要标人）。"
    "严格输出 JSON，不要多余文字。"
)
USER_PROMPT = (
    "分析这张监控画面，输出静态环境基线 JSON：\n"
    "{\n"
    '  "ground_top_y": 地面区域顶部y坐标(归一化0-1,0=画面顶1=画面底,地面在下方。如0.6表示画面下40%是地面),\n'
    '  "obstacles": [固定障碍物列表,每个{"bbox":[x,y,w,h]归一化(左上角xy+宽高,0-1),"type":"椅子/桌子/地毯/床/电线/纸箱等"},...],\n'
    '  "confidence": 0.0-1.0\n'
    "}\n"
    "要求：1)只标固定障碍物(家具/地毯/杂物),不要标人 2)bbox 用归一化坐标[左上x,左上y,宽,高] "
    "3)看不出障碍物填空数组 4)ground_top_y 是地面开始的 y(画面下方是地面) 5)只输出 JSON。"
)


def _b64_img(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _call_ark(frame_path: Path, model: str, timeout: float = ENV_VLM_TIMEOUT) -> str:
    from openai import OpenAI
    key = os.environ.get("ARK_API_KEY")
    if not key:
        raise RuntimeError("未设置 ARK_API_KEY（检查 .env）")
    client = OpenAI(api_key=key, base_url=ENV_VLM_ARK_BASE_URL,
                    timeout=timeout, max_retries=0)
    b64 = _b64_img(frame_path)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ],
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def _parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None


def main():
    p = argparse.ArgumentParser(description="VLM 离线静态基线标定")
    p.add_argument("--frame", required=True, help="场景首帧 PNG/JPG 路径")
    p.add_argument("--scene", required=True, help="场景名（如 Subject.1_Fall_backwards）")
    p.add_argument("--model", default=os.environ.get("ARK_VLM_MODEL", "Doubao-Seed-1.6-vision"))
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "env_baselines"))
    p.add_argument("--mock", action="store_true", help="无网测试（占位基线）")
    args = p.parse_args()

    frame = Path(args.frame)
    if not frame.exists():
        print(f"[ERR] 帧不存在: {frame}")
        return

    print(f"=== VLM 静态基线标定: {args.scene} ===")
    print(f"  帧源: {frame}")
    print(f"  模型: {args.model}")

    if args.mock:
        result = {
            "ground_top_y": 0.65,
            "obstacles": [{"bbox": [0.1, 0.7, 0.15, 0.1], "type": "[mock]椅子"}],
            "confidence": 1.0, "mock": True,
        }
        print("  [mock] 占位基线")
    else:
        print("  调用方舟 ...")
        try:
            text = _call_ark(frame, args.model)
            result = _parse(text)
            if result is None:
                print(f"[ERR] JSON 解析失败，原始返回:\n{text}")
                return
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] VLM 调用失败: {e}")
            return
        print(f"  VLM 返回: ground_top_y={result.get('ground_top_y')}, "
              f"obstacles={len(result.get('obstacles', []))} 个, "
              f"confidence={result.get('confidence')}")

    result["scene"] = args.scene
    result["frame_source"] = str(frame)
    result["model"] = args.model

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.scene}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  静态基线 → {out}")
    print(f"  内容: {json.dumps(result, ensure_ascii=False)}")
    print("  运行时 EnvMaskBuilder bg_diff 将加载此基线 + MOG2 新增物 + HSV 反光 + 人脚挖孔")


if __name__ == "__main__":
    main()
