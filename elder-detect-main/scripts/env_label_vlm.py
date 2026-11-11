#!/usr/bin/env python
"""
env_label_vlm.py — EnvRisk R1 第二段：帧 + prompt → VLM API → 结构化标签

逐帧调 VLM（零样本，不训练）→ 解析 JSON → 视频级聚合（众数/中位数）
→ 写 env_labels/<ds>/<stem>.json（聚合）+ env_labels/_raw/<ds>/<stem>.jsonl（逐帧，审计）

⚠️ API 留空约定：provider / model / key 均不硬编码
    - provider: --provider 或环境变量 ENV_VLM_PROVIDER；默认 ark
    - key:       仅从环境变量读（ARK_API_KEY / OPENAI_API_KEY / DASHSCOPE_API_KEY / GOOGLE_API_KEY）
                 优先读取项目根 .env（_load_dotenv，零依赖）
    - 没有 key 又想跑通链路 → 用 --mock（占位标签，零网络）

用法:
    python scripts/env_label_vlm.py --index frames/index.csv
    python scripts/env_label_vlm.py --index frames/index.csv --provider ark --model Doubao-Seed-1.6-vision
    python scripts/env_label_vlm.py --index frames/index.csv --provider openai --model gpt-4o-mini
    python scripts/env_label_vlm.py --index frames/index.csv --provider dashscope --model qwen-vl-max
    python scripts/env_label_vlm.py --index frames/index.csv --mock   # 无网验证
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_config import (
    ENV_VLM_PROVIDER, ENV_VLM_MODEL, ENV_VLM_ARK_BASE_URL, ENV_VLM_CONCURRENCY, ENV_VLM_RETRY,
    ENV_VLM_TIMEOUT, ENV_VLM_MIN_CONF, ENV_VLM_PROMPT_VER,
    ENV_VLM_ROOM_ENUM, ENV_VLM_LIGHT_ENUM,
    HOME_FAST_VLM_TAGS, HOME_FAST_ITEMS,
)

# ---- 加载 .env（API key 不硬编码，仅从环境变量 / .env 读取） ----
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


# ---- Prompts（文档 §6） -------------------------------------------------
SYS_PROMPT = (
    "你是老年人居家跌倒风险的环境评估专家。给你一张室内监控画面，"
    "请只评估「环境本身」对老人跌倒的危险度，不要评估画面里人的姿态或动作。"
    "严格按 HOME FAST 居家跌倒环境量表逐项输出 JSON，不要输出任何多余文字。"
    "对于画面中没有、看不清、或你无法判断的项目，必须输出 JSON null，绝对不要猜测。"
)
USER_PROMPT = (
    "分析这张居家监控画面，严格按 HOME FAST 跌倒环境量表逐项输出 JSON：\n"
    "{\n"
    '  "room_type":          "从 [bathroom, kitchen, stairs, bedroom, living_room, hallway, clinical, outdoor] 选一个；看不出填 null",\n'
    '  "lighting":           "从 [bright, dim, dark] 选一个；看不出填 null",\n'
    '  "corridor_clear":     "bool：走道无电线/纸箱等杂物遮挡？ true=无隐患, false=有杂物(隐患)",\n'
    '  "floor_intact":       "bool：全屋地面/地毯铺装完好？ true=完好, false=翘起/破损(隐患)",\n'
    '  "floor_antislip":     "bool：厨房/卫浴地面有防滑铺装/防滑垫？ true=有, false=光面无防滑(隐患)",\n'
    '  "mat_fixed":          "bool：零散地垫完全固定（防滑底/粘贴）？ true=固定, false=翘起未固定(隐患)",\n'
    '  "bedside_light":      "bool：床上可直接开灯（有床头灯/手电）？ true=可, false=必须下床开灯(隐患)",\n'
    '  "outdoor_light":      "bool：户外台阶/入户楼道夜间有灯？ true=有, false=无灯(隐患)",\n'
    '  "toilet_grab_bar":   "bool：马桶旁有辅助扶手？ true=有, false=无扶手(隐患)",\n'
    '  "bath_accessible":    "bool：浴缸/淋浴可独立安全进出（有借力/浴板）？ true=可, false=过高无借力(隐患)",\n'
    '  "bath_grab_bar":     "bool：浴缸/淋浴配有牢固承重扶手（非毛巾架）？ true=有, false=无(隐患)",\n'
    '  "bath_antislip_mat": "bool：浴区内铺设防滑垫？ true=有, false=无(隐患)",\n'
    '  "near_toilet":        "bool：卧室距离卫生间很近（<=2道门）？ true=近, false=需出门如厕(隐患)",\n'
    '  "storage_reachable":  "bool：常用物品在腰-肩高度、无需登高弯腰？ true=是, false=需梯子/大幅弯腰(隐患)",\n'
    '  "indoor_stair_rail":  "bool：室内台阶有全长扶手？ true=有, false=无/单侧(隐患)",\n'
    '  "outdoor_stair_rail": "bool：室外连续台阶有扶手？ true=有, false=无(隐患)",\n'
    '  "stair_regular":      "bool：台阶高度/宽度一致、行走不费力？ true=规整, false=过高/宽窄不一(隐患)",\n'
    '  "stair_edge_clear":   "bool：台阶边缘清晰可分辨（有对比色/警示条）？ true=清晰, false=同色看不清(隐患)",\n'
    '  "entry_safe":         "bool：入户门有落脚平台、门槛不高、进出平衡安全？ true=安全, false=无平台/高门槛(隐患)",\n'
    '  "yard_path_intact":   "bool：庭院步道完好、无开裂/杂物占道？ true=完好, false=破损/占道(隐患)",\n'
    '  "shoe_antislip":     "bool：日常鞋/拖鞋防滑合脚（低跟防滑）？ true=是, false=平底无防滑/磨损(隐患)",\n'
    '  "pet_present":        "bool：画面中是否有宠物（猫狗等）？ true=有, false=无",\n'
    '  "pet_safe":           "bool：宠物是否不窜到老人脚下？ true=安全, false=频繁窜脚下(隐患)",\n'
    '  "confidence":         "0.0~1.0，你对本次判断的把握",\n'
    '  "reason":             "一句中文，说明主要危险因素"\n'
    "}\n"
    "评分语义：每个 bool 字段 true=该项安全（无隐患），false=存在隐患（判 1 分）。\n"
    "【禁止瞎猜，宁缺毋滥】若某字段画面中看不到、看不清、或你无法判断，"
    "必须填 JSON null（不要猜 true/false，也不要硬填枚举值）。\n"
    "例如画面里没有浴室，toilet_grab_bar / bath_accessible / bath_grab_bar / "
    "bath_antislip_mat 一律填 null；看不到楼梯，indoor_stair_rail / stair_* 填 null。"
    "这些 null 会被当作 N/A，不计入评分，比瞎猜更安全。\n"
    "只输出 JSON，不要任何多余文字。"
)


# ---- 环境变量 key 映射 --------------------------------------------------
_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "google": "GOOGLE_API_KEY",
    "ark": "ARK_API_KEY",
}


def _api_key(provider: str) -> str:
    name = _ENV_KEYS.get(provider)
    if name is None:
        raise ValueError(f"未知 provider: {provider}（可选 ark/openai/dashscope/google）")
    key = os.environ.get(name)
    if not key:
        raise RuntimeError(
            f"未设置环境变量 {name}（可在项目根 .env 写入，已自动加载）。"
            f"或使用 --mock 跑通链路而不调真实 API。"
        )
    return key


# ---- 各 provider 调用（lazy import，避免无 SDK 时 import 即崩） --------
def _call_openai_style(client, model: str, b64: str, sys_p: str, usr_p: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_p},
            {"role": "user", "content": [
                {"type": "text", "text": usr_p},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ],
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def _call_google(model: str, b64: str, sys_p: str, usr_p: str) -> str:
    import google.generativeai as genai
    gm = genai.GenerativeModel(model_name=model, system_instruction=sys_p)
    # google 用 PIL 传图
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    resp = gm.generate_content(
        [usr_p, img],
        generation_config={"response_mime_type": "application/json"},
    )
    return resp.text


def _client(provider: str, model: str, timeout: float = ENV_VLM_TIMEOUT):
    key = _api_key(provider)
    # max_retries=0：SDK 层不重试，重试全交给 _call_once（可打印 retry 进度）
    if provider in ("openai",):
        from openai import OpenAI
        return ("openai", OpenAI(api_key=key, timeout=timeout, max_retries=0), model)
    if provider == "dashscope":
        from openai import OpenAI
        # 阿里官方 openai 兼容端点
        client = OpenAI(
            api_key=key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=timeout, max_retries=0,
        )
        return ("openai", client, model)
    if provider == "ark":
        from openai import OpenAI
        # 火山方舟 OpenAI 兼容端点（Doubao-Seed-1.6-vision 等视觉模型）
        client = OpenAI(
            api_key=key,
            base_url=ENV_VLM_ARK_BASE_URL,
            timeout=timeout, max_retries=0,
        )
        return ("openai", client, model)
    if provider == "google":
        import google.generativeai as genai
        genai.configure(api_key=key)
        return ("google", None, model)
    raise ValueError(f"未知 provider: {provider}")


_MOCK_LABEL = {
    "room_type": "unknown", "lighting": "bright",
    "corridor_clear": True, "floor_intact": True, "floor_antislip": True,
    "mat_fixed": True, "bedside_light": True, "outdoor_light": True,
    "toilet_grab_bar": True, "bath_accessible": True, "bath_grab_bar": True,
    "bath_antislip_mat": True, "near_toilet": True, "storage_reachable": True,
    "indoor_stair_rail": True, "outdoor_stair_rail": True, "stair_regular": True,
    "stair_edge_clear": True, "entry_safe": True, "yard_path_intact": True,
    "shoe_antislip": True, "pet_present": False, "pet_safe": True,
    "confidence": 1.0, "reason": "[mock] 未调真实 API",
}


def _call_once(provider: str, client, model: str, b64: str,
               retry: int, use_mock: bool) -> dict:
    if use_mock:
        return dict(_MOCK_LABEL)
    last = None
    for attempt in range(1, retry + 1):
        try:
            if provider == "google":
                text = _call_google(model, b64, SYS_PROMPT, USER_PROMPT)
            else:  # openai / dashscope / ark 走 openai 风格
                text = _call_openai_style(client, model, b64, SYS_PROMPT, USER_PROMPT)
            return _parse_label(text)
        except Exception as e:  # noqa: BLE001
            last = e
            wait = 2 ** attempt
            print(f"    [retry {attempt}/{retry}] {e}; sleep {wait}s", flush=True)
            time.sleep(wait)
    return {"parse_error": True, "reason": f"调用失败: {last}"}


# ---- 解析 + 质控（文档 §7） --------------------------------------------
def _clamp(x, lo=0.0, hi=1.0):
    try:
        return float(min(max(float(x), lo), hi))
    except (TypeError, ValueError):
        return 0.0


def _parse_label(text: str) -> dict:
    raw = text.strip()
    try:
        d = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group(0))
            except Exception:
                return {"parse_error": True, "reason": "JSON 解析失败", "raw": raw[:200]}
        else:
            return {"parse_error": True, "reason": "无 JSON", "raw": raw[:200]}
    # 枚举校验
    if d.get("room_type") not in ENV_VLM_ROOM_ENUM:
        d["room_type"] = "unknown"
    if d.get("lighting") not in ENV_VLM_LIGHT_ENUM:
        d["lighting"] = "bright"
    # bool 字段校验：非 bool → None（from_labels 跳过，不计入分母）
    for k, (qid, kind) in HOME_FAST_VLM_TAGS.items():
        if kind in ("pet_present", "pet_safe", "light_not_bright_is_risk"):
            continue  # 这三类在下面单独处理
        if not isinstance(d.get(k), bool):
            d[k] = None
    # pet_* 校验
    if not isinstance(d.get("pet_present"), bool):
        d["pet_present"] = None
    if not isinstance(d.get("pet_safe"), bool):
        d["pet_safe"] = None
    d["confidence"] = _clamp(d.get("confidence"))
    return d


# ---- 视频级聚合（众数/中位数） ---------------------------------------
def _mode_prefer_danger(values, danger_rank: dict):
    """众数；平票取 danger_rank 更高者。"""
    if not values:
        return None
    uniq, counts = np.unique(np.array(values), return_counts=True)
    top = counts.max()
    cands = uniq[counts == top]
    if len(cands) == 1:
        return cands[0]
    # 平票：取危险度更高
    return max(cands, key=lambda v: danger_rank.get(v, 0.0))


def aggregate(frame_labels: list[dict]) -> dict:
    if not frame_labels:
        return {}
    rooms = [d.get("room_type") for d in frame_labels if d.get("room_type")]
    lights = [d.get("lighting") for d in frame_labels if d.get("lighting")]
    confs = [d.get("confidence", 0.0) for d in frame_labels if "confidence" in d]

    # bool 标签：取众数（平票/少数 → False 更保守，有隐患嫌疑）
    bool_agg = {}
    for k, (qid, kind) in HOME_FAST_VLM_TAGS.items():
        if kind == "light_not_bright_is_risk":
            continue
        vals = [d.get(k) for d in frame_labels if isinstance(d.get(k), bool)]
        if not vals:
            bool_agg[k] = None
        else:
            t = sum(1 for v in vals if v)
            bool_agg[k] = True if t > len(vals) - t else False

    room = _mode_prefer_danger(rooms, {}) if rooms else "unknown"
    light = _mode_prefer_danger(lights, {}) if lights else "bright"
    conf = float(np.mean(confs)) if confs else 0.0
    need_review = conf < ENV_VLM_MIN_CONF

    out = {"room_type": room, "lighting": light}
    out.update({k: bool_agg[k] for k in bool_agg})
    out.update({
        "confidence": round(conf, 4), "need_review": need_review,
    })
    return out


# ---- 单视频处理 -------------------------------------------------------
def process_video(row: dict, provider: str, client, model: str,
                  retry: int, use_mock: bool, raw_dir: Path,
                  out_dir: Path, prompt_ver: str) -> dict | None:
    ds, stem = row["ds"], row["stem"]
    frame_paths = [p for p in row["frame_paths"].split(";") if p]
    if not frame_paths:
        return None
    # 视频级聚合 json 已存在 → 跳过（断点续跑）
    agg_path = out_dir / ds / f"{stem}.json"
    if agg_path.exists():
        print(f"  [SKIP] {ds}/{stem} 已标注")
        return None

    (raw_dir / ds).mkdir(parents=True, exist_ok=True)
    (out_dir / ds).mkdir(parents=True, exist_ok=True)
    raw_lines = []
    frame_labels = []
    n = len(frame_paths)
    for i, fp in enumerate(frame_paths, 1):
        b64 = base64.b64encode(Path(PROJECT_ROOT / fp).read_bytes()).decode()
        t0 = time.time()
        print(f"    {ds}/{stem} 帧 {i}/{n} 调用中…", flush=True)
        lab = _call_once(provider, client, model, b64, retry, use_mock)
        dt = time.time() - t0
        flag = " ⚠parse_error" if lab.get("parse_error") else ""
        print(f"    {ds}/{stem} 帧 {i}/{n} 完成 {dt:.1f}s{flag}", flush=True)
        lab["frame"] = fp
        raw_lines.append(json.dumps(lab, ensure_ascii=False))
        frame_labels.append(lab)

    agg = aggregate(frame_labels)
    agg.update({
        "ds": ds, "stem": stem,
        "n_frames_labeled": len(frame_labels),
        "prompt_ver": prompt_ver, "provider": provider, "model": model,
    })
    (raw_dir / ds / f"{stem}.jsonl").write_text(
        "\n".join(raw_lines), encoding="utf-8")
    agg_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="frames/index.csv")
    p.add_argument("--out", default="env_labels")
    p.add_argument("--provider", default=None, help="ark|openai|dashscope|google（默认 ENV_VLM_PROVIDER）")
    p.add_argument("--model", default=None, help="默认 ENV_VLM_MODEL")
    p.add_argument("--concurrency", type=int, default=ENV_VLM_CONCURRENCY)
    p.add_argument("--retry", type=int, default=ENV_VLM_RETRY)
    p.add_argument("--timeout", type=float, default=ENV_VLM_TIMEOUT,
                   help="单次 VLM 调用超时(秒)，超时抛错触发 retry")
    p.add_argument("--prompt_ver", default=ENV_VLM_PROMPT_VER)
    p.add_argument("--mock", action="store_true",
                   help="不调真实 API，用占位标签跑通链路（零网络）")
    args = p.parse_args()

    index = PROJECT_ROOT / args.index
    if not index.exists():
        print(f"[ERR] index 不存在: {index}")
        return
    rows = list(csv.DictReader(index.read_text(encoding="utf-8").splitlines()))

    # provider / model 优先级：CLI --xxx > .env ARK_VLM_* > build_config 默认
    provider = args.provider or os.environ.get("ARK_VLM_PROVIDER") or ENV_VLM_PROVIDER
    model = args.model or os.environ.get("ARK_VLM_MODEL") or ENV_VLM_MODEL

    if args.mock:
        print(f"[mock] 模式：不调真实 API，占位标签。provider={provider} model={model}")
        client = None
    else:
        try:
            kind, client, model = _client(provider, model, args.timeout)
            print(f"provider={provider} model={model} timeout={args.timeout}s")
        except Exception as e:
            print(f"[ERR] 初始化 API 客户端失败: {e}")
            return

    out_dir = PROJECT_ROOT / args.out
    raw_dir = out_dir / "_raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"待标注视频: {len(rows)}，并发={args.concurrency}，retry={args.retry}", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(process_video, row, provider, client, model,
                      args.retry, args.mock, raw_dir, out_dir, args.prompt_ver): row
            for row in rows
        }
        for fut in as_completed(futs):
            row = futs[fut]
            try:
                agg = fut.result()
                if agg is not None:
                    done += 1
                    print(f"  [OK] {agg['ds']}/{agg['stem']} "
                          f"room={agg['room_type']} light={agg['lighting']} "
                          f"conf={agg['confidence']}"
                          f"{' ⚠need_review' if agg['need_review'] else ''}")
            except Exception as e:  # noqa: BLE001
                print(f"  [ERR] {row['ds']}/{row['stem']}: {e}")
    print(f"\nDone. 新标注 {done}/{len(rows)} 个视频 → {out_dir}")


if __name__ == "__main__":
    main()
