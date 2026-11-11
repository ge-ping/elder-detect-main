#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_pipeline.py — 全流程编排：VLM 打标 → 掩码标定 → 实时处理
=================================================================
给定单个视频文件，依次跑完三段（模型/基线自动探测，无需手动指定）：

  ① VLM 打标
       env_extract_frames.py --video_root <视频>          抽 K 帧 → frames/index.csv
       env_label_vlm.py    --index frames/index.csv       逐帧 HOME FAST 打标
                                                        → env_labels/<ds>/<stem>.json
  ② 掩码标定（VLM 看首帧 → 静态环境基线/掩码）
       vlm_baseline_calibrate.py --frame <首帧> --scene <stem>
                                                        → env_baselines/<stem>.json
       （运行时 EnvMaskBuilder.from_baseline 加载：ground_top_y + 固定障碍物 bbox
        + MOG2 新增物 + HSV 反光 + 人脚挖孔，纯算法不再调 VLM）
  ③ 实时处理
       demo_gate.py <视频> --baseline env_baselines/<stem>.json
                                                        （cv2 窗口，q 退出）

用法：
  # 最简：mock 跑通整条链路（不调真实 API，占位标签 + 占位基线）
  python demo/run_pipeline.py "你的视频.mp4" --mock

  # 真实 VLM 打标（需项目根 .env 里 ARK_API_KEY；provider 默认 ark）
  python demo/run_pipeline.py "你的视频.mp4" --provider ark

  # 只跑离线打标+掩码，不弹实时窗口（适合先批量预处理）
  python demo/run_pipeline.py "你的视频.mp4" --stages 1,2

  # 断点续跑：已存在的 env_labels / env_baselines 自动跳过
  python demo/run_pipeline.py "你的视频.mp4" --resume

注：③ 实时窗口是交互式的（按 q 退出）。① ② 是离线、可无头运行。
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(script: str, *args, mock_mode: bool = False):
    """以当前 python 在项目根目录跑一个脚本，透传输出。"""
    cmd = [sys.executable, str(PROJECT_ROOT / script), *args]
    print(f"\n$ {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode
    if rc != 0:
        raise RuntimeError(f"{script} 退出码 {rc}，流程中断")
    return rc


def _read_index_row(index_csv: Path, stem: str):
    if not index_csv.exists():
        return None
    with index_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("stem") == stem:
                return row
    return None


def main():
    p = argparse.ArgumentParser(
        description="全流程：VLM 打标 → 掩码标定 → 实时处理（只给视频即可）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python demo/run_pipeline.py \"视频.mp4\" --mock\n"
               "  python demo/run_pipeline.py \"视频.mp4\" --provider ark\n"
               "  python demo/run_pipeline.py \"视频.mp4\" --stages 1,2   # 只离线打标+掩码\n"
               "  python demo/run_pipeline.py \"视频.mp4\" --resume       # 断点续跑")
    p.add_argument("video", help="输入视频文件（必填）")
    # VLM 相关
    p.add_argument("--mock", action="store_true",
                   help="① ② 都用占位输出，不调真实 VLM API（零网络跑通链路）")
    p.add_argument("--provider", default=None,
                   help="① VLM provider：ark|openai|dashscope|google（默认 ENV_VLM_PROVIDER）")
    p.add_argument("--model", default=None, help="① VLM 模型名（默认 ENV_VLM_MODEL）")
    p.add_argument("--frames-per-video", type=int, default=None,
                   help="① 每视频抽帧数（默认 ENV_VLM_FRAMES_PER_VIDEO）")
    # 流程控制
    p.add_argument("--stages", default="1,2,3",
                   help="只跑指定阶段，逗号分隔。默认 1,2,3（全部）")
    p.add_argument("--resume", action="store_true",
                   help="已存在的 env_labels / env_baselines 自动跳过（断点续跑）")
    # ③ 实时 demo 透传
    p.add_argument("--force-e", type=float, default=0.5)
    p.add_argument("--process-every", type=int, default=4)
    p.add_argument("--resize-width", type=int, default=720)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save", action="store_true", help="③ 另存标注视频")
    p.add_argument("--profile", action="store_true", help="③ 打印每阶段耗时")
    args = p.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"[ERR] 视频不存在: {video}")
        return
    video_abs = str(video.resolve())

    stages = {s.strip() for s in args.stages.split(",") if s.strip() in ("1", "2", "3")}
    if not stages:
        print("[ERR] --stages 必须含 1/2/3")
        return

    stem = video.stem
    frames_dir = PROJECT_ROOT / "frames"
    index_csv = frames_dir / "index.csv"
    labels_dir = PROJECT_ROOT / "env_labels"
    baselines_dir = PROJECT_ROOT / "env_baselines"
    baseline_json = baselines_dir / f"{stem}.json"

    print(f"=== 全流程编排 ===\n  视频 : {video_abs}\n  场景 : {stem}"
          f"\n  mock={args.mock}  stages={sorted(stages)}  resume={args.resume}")

    # ----------------------------------------------------------------
    # ① VLM 打标
    # ----------------------------------------------------------------
    if "1" in stages:
        need_label = True
        if args.resume and list(labels_dir.rglob(f"{stem}.json")):
            need_label = False
            print(f"\n[resume] 跳过 ①：env_labels/**/{stem}.json 已存在")
        if need_label:
            print("\n========== ① VLM 打标 ==========")
            # 1a 抽帧
            ex_args = [f"--video_root", video_abs]
            if args.frames_per_video:
                ex_args += ["--frames_per_video", str(args.frames_per_video)]
            _run("scripts/env_extract_frames.py", *ex_args)
            # 1b 逐帧 VLM 打标
            lb_args = ["--index", str(index_csv)]
            if args.mock:
                lb_args.append("--mock")
            if args.provider:
                lb_args += ["--provider", args.provider]
            if args.model:
                lb_args += ["--model", args.model]
            _run("scripts/env_label_vlm.py", *lb_args)
    else:
        print("\n[skip] ① VLM 打标（--stages 未含 1）")

    # 取首帧用于 ② 掩码标定（与 env_extract_frames 单文件约定一致）
    row = _read_index_row(index_csv, stem)
    if row is None:
        # 单文件模式 ds 可能是视频所在目录名，这里兜底扫描 index
        if index_csv.exists():
            with index_csv.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            row = rows[0] if rows else None
    first_frame = None
    if row and row.get("frame_paths"):
        first_frame = PROJECT_ROOT / row["frame_paths"].split(";")[0]
    if first_frame is None or not first_frame.exists():
        # 退而求其次：frames 目录下该 stem 的首张 jpg
        cand = sorted((frames_dir / stem).rglob("*.jpg")) if (frames_dir / stem).exists() else []
        if not cand:
            cand = sorted(frames_dir.rglob(f"{stem}/*.jpg"))
        first_frame = cand[0] if cand else None

    # ----------------------------------------------------------------
    # ② 掩码标定
    # ----------------------------------------------------------------
    if "2" in stages:
        need_mask = True
        if args.resume and baseline_json.exists():
            need_mask = False
            print(f"\n[resume] 跳过 ②：{baseline_json.name} 已存在")
        if need_mask:
            if first_frame is None or not first_frame.exists():
                print(f"[WARN] 找不到首帧用于掩码标定（frames 未抽帧？），跳过 ②")
            else:
                print("\n========== ② 掩码标定（VLM 看首帧 → 静态基线）==========")
                cal_args = ["--frame", str(first_frame), "--scene", stem]
                if args.mock:
                    cal_args.append("--mock")
                if args.model:
                    cal_args += ["--model", args.model]
                _run("scripts/vlm_baseline_calibrate.py", *cal_args)
    else:
        print("\n[skip] ② 掩码标定（--stages 未含 2）")

    # ----------------------------------------------------------------
    # ③ 实时处理
    # ----------------------------------------------------------------
    if "3" in stages:
        if not baseline_json.exists():
            print(f"\n[WARN] 掩码基线不存在: {baseline_json}（请先跑 ② 或 --resume）。"
                  f"\n  仍可用通用基线，或 Ctrl+C 退出。")
        print("\n========== ③ 实时处理（cv2 窗口，按 q 退出）==========")
        gate_args = [video_abs]
        if baseline_json.exists():
            gate_args += ["--baseline", str(baseline_json)]
        gate_args += ["--force-e", str(args.force_e),
                      "--process-every", str(args.process_every),
                      "--resize-width", str(args.resize_width)]
        if args.gpu:
            gate_args.append("--gpu")
        if args.cpu:
            gate_args.append("--cpu")
        if args.save:
            gate_args.append("--save")
        if args.profile:
            gate_args.append("--profile")
        _run("demo/demo_gate.py", *gate_args)
    else:
        print("\n[skip] ③ 实时处理（--stages 未含 3）。"
              f"\n  之后可单独运行：python demo/demo_gate.py \"{video_abs}\""
              f" --baseline {baseline_json}")


if __name__ == "__main__":
    main()
