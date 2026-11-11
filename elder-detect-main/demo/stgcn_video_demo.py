#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demo/stgcn_video_demo.py — ST-GCN 姿态跌倒风险 · 视频演示（纯姿态，不含环境/I 门控）

输入一段视频，逐帧 MediaPipe 提 17 关节 3D 世界坐标 → 髋中心归一化到 [-1,1]
→ 35 帧滑窗 → ST-GCN 前向 → 连续风险 risk∈[0,1]。
画面叠加：17 关节骨架 + 风险条 + 状态 + 报警边框。
输出：实时 cv2 窗口（q 退出 / 空格暂停）+ 标注视频 mp4 + 风险曲线 png。

复用既有模块（零新依赖、与训练零漂移）：
  - pose.extractor.PoseExtractor / PoseExtractorConfig
  - model.stgcn.stgcn.load_stgcn
  - model.stgcn.graph.FALL_JOINTS_17 / FALL_EDGES / skeleton_from_poseresult

防误报机制（默认已开启）:
  - EMA 平滑 (--smooth-alpha 0.2)：压瞬时毛刺。
  - 抬高 WARN 阈值 (--warn-thresh 0.45)：坐/蹲/弯腰的中间风险不再触发橙色警告。
  - 迟滞去抖 AlarmDebouncer (--hysteresis 0.08 / --min-hold 3 / --exit-hold 6)：
    进入用高阈值、退出用低阈值 + 升降级都需连续帧，彻底消除阈值附近的橙色横跳闪烁。

用法 (项目根, (elder) 环境, Windows cmd):
  python demo/stgcn_video_demo.py "视频.mp4" --model model/stgcn/stgcn_v1/run_02/best_model.pt
  更严(几乎不误报): python demo/stgcn_video_demo.py "视频.mp4" --warn-thresh 0.5 --hysteresis 0.1 --exit-hold 8
  退回旧行为(易误报): python demo/stgcn_video_demo.py "视频.mp4" --warn-thresh 0.3 --hysteresis 0 --min-hold 0 --smooth-alpha 0
  python demo/stgcn_video_demo.py "视频.mp4" --cpu       # 无显卡
  python demo/stgcn_video_demo.py "视频.mp4" --no-save   # 只看不存
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pose.extractor import PoseExtractor, PoseExtractorConfig  # noqa: E402
from model.stgcn.stgcn import load_stgcn  # noqa: E402
from model.stgcn.graph import (  # noqa: E402
    FALL_JOINTS_17, FALL_EDGES, skeleton_from_poseresult,
)
from features.landmark_filter import LandmarkSmoother  # noqa: E402

TASK_PATH = PROJECT_ROOT / "model/mediapipe_pose/pose_landmarker_lite.task"
WARN, DANGER = 0.3, 0.6
ALARM_COLORS = {"SAFE": (0, 200, 0), "WARNING": (0, 165, 255), "DANGER": (0, 0, 255)}


def load_stgcn_model(model_path: str):
    """读 ST-GCN 模型 + model_info.json（含 window_size）。"""
    mp = Path(model_path)
    if not mp.is_absolute():
        mp = PROJECT_ROOT / mp
    if not mp.exists():
        raise FileNotFoundError(f"找不到 ST-GCN 模型: {mp}")
    model = load_stgcn(str(mp))
    info = {}
    ip = mp.parent / "model_info.json"
    if ip.exists():
        info = json.loads(ip.read_text(encoding="utf-8"))
    window = int(info.get("window_size", 35))
    return model, window, info


def status_of(risk: float):
    if risk >= DANGER:
        return "DANGER", (0, 0, 255)
    if risk >= WARN:
        return "WARNING", (0, 165, 255)
    return "SAFE", (0, 200, 0)


def debounce(ema: float, danger_cnt: int, warn_cnt: int, min_hold: int):
    """[旧] 无迟滞去抖：仅"升级"需连续帧，"降级"立即——在阈值附近会反复横跳闪烁。
    保留仅为兼容，主循环已改用 AlarmDebouncer。min_hold<=0 表示立即报警。
    """
    if ema >= DANGER:
        danger_cnt += 1
        warn_cnt += 1
    elif ema >= WARN:
        danger_cnt = 0
        warn_cnt += 1
    else:
        danger_cnt = 0
        warn_cnt = 0
    if min_hold <= 0:
        alarm = "DANGER" if ema >= DANGER else ("WARNING" if ema >= WARN else "SAFE")
    elif danger_cnt >= min_hold:
        alarm = "DANGER"
    elif warn_cnt >= min_hold:
        alarm = "WARNING"
    else:
        alarm = "SAFE"
    return alarm, danger_cnt, warn_cnt


class AlarmDebouncer:
    """带迟滞(hysteresis)的报警状态机 —— 消除阈值附近的橙/红反复横跳闪烁。

    核心思想：进入与退出用**不同的阈值**（错开一个迟滞带），抖动落在带内时保持
    当前状态不翻转，这样坐下/弯腰导致 risk 在 warn 阈值附近抖时，不会反复亮灭。

    - 升级 (SAFE→WARNING→DANGER)：ema 需连续 ≥enter_hold 帧达到目标进入阈值，
      一次可直达目标等级（真跌倒 risk 冲高时快速点亮 DANGER，仅损失 enter_hold 帧）。
    - 降级 (DANGER→WARNING→SAFE)：ema 需连续 ≥exit_hold 帧 < (阈值 − hysteresis)，
      且逐级下降，避免突变与闪烁。

    参数:
      warn / danger : 进入阈值
      hysteresis    : 退出比进入低多少（迟滞带宽），落在带内保持原状态
      enter_hold    : 升级所需连续帧（=旧 --min-hold）
      exit_hold     : 降级所需连续帧（通常比 enter_hold 大，报警"黏"一点更稳）
    """
    LEVELS = ("SAFE", "WARNING", "DANGER")

    def __init__(self, warn=0.3, danger=0.6, hysteresis=0.05,
                 enter_hold=3, exit_hold=5):
        self.warn = warn
        self.danger = danger
        self.hyst = max(0.0, hysteresis)
        self.enter_hold = max(1, enter_hold)
        self.exit_hold = max(1, exit_hold)
        self.level = 0
        self._up = 0
        self._down = 0

    def reset(self):
        self.level = 0
        self._up = 0
        self._down = 0

    def update(self, ema: float) -> str:
        if ema >= self.danger:
            target = 2
        elif ema >= self.warn:
            target = 1
        else:
            target = 0
        if target > self.level:
            # 升级：连续 enter_hold 帧后直达目标等级
            self._up += 1
            self._down = 0
            if self._up >= self.enter_hold:
                self.level = target
                self._up = 0
        elif target < self.level:
            # 降级：需跌破"退出阈值(进入阈值−迟滞)"并连续 exit_hold 帧，逐级降
            exit_thr = (self.danger if self.level == 2 else self.warn) - self.hyst
            if ema < exit_thr:
                self._down += 1
                self._up = 0
                if self._down >= self.exit_hold:
                    self.level -= 1
                    self._down = 0
            else:
                # 落在迟滞带内：保持当前等级，不翻转（这一步消除橙色横跳）
                self._up = 0
                self._down = 0
        else:
            self._up = 0
            self._down = 0
        return self.LEVELS[self.level]


def draw(frame: np.ndarray, pr, risk: float, alarm: str, raw: float = None,
         low_conf: bool = False, min_conf: float = 0.5):
    """画骨架 + 风险条 + 状态 + 报警边框。

    risk  : 显示用风险值（已 EMA 平滑）
    alarm : 去抖后的报警等级（SAFE/WARNING/DANGER），决定边框与文字颜色
    raw   : 原始未平滑风险（平滑开启且偏差大时附带显示，便于判读毛刺）
    low_conf: 该帧为低置信度（仅画面提示，risk 由调用方保持上一有效值）
    """
    vis = frame.copy()
    h, w = vis.shape[:2]
    # 17 关节骨架（用 2D 归一化坐标贴合画面）
    if pr is not None:
        pts = {}
        for i, part in enumerate(FALL_JOINTS_17):
            try:
                kp = pr.get(part)
            except KeyError:
                continue
            if getattr(kp, "visibility", 0) >= 0.5:
                pts[i] = (int(kp.x * w), int(kp.y * h))
        for a, b in FALL_EDGES:
            if a in pts and b in pts:
                cv2.line(vis, pts[a], pts[b], (0, 255, 255), 2)
        for i in pts:
            cv2.circle(vis, pts[i], 4, (0, 255, 0), -1)
    # 顶部信息条
    col = ALARM_COLORS[alarm]
    cv2.rectangle(vis, (0, 0), (w, 80), (0, 0, 0), -1)
    label = f"ST-GCN  Risk: {risk:.3f}   {alarm}"
    if raw is not None and abs(raw - risk) > 0.02:
        label += f"  (raw {raw:.3f})"
    cv2.putText(vis, label, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    # 风险条
    bx, by, bw, bh = 8, 44, 320, 18
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (80, 80, 80), -1)
    cv2.rectangle(vis, (bx, by), (bx + int(bw * min(risk, 1.0)), by + bh), col, -1)
    cv2.putText(vis, f"{risk:.3f}", (bx + bw + 8, by + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    # 报警边框（仅去抖后等级驱动，单帧毛刺不再触发）
    if alarm == "DANGER":
        cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
    elif alarm == "WARNING":
        cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 165, 255), 3)
    if low_conf and pr is not None:
        cv2.putText(vis, f"LOW CONF {pr.mean_confidence:.2f}<{min_conf:.2f}  risk held",
                    (8, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return vis


def make_curve(risks: list[float], path: Path, warn: float = WARN, danger: float = DANGER):
    """纯 cv2 画风险曲线（零 matplotlib 依赖）。阈值线按实际 warn/danger 绘制。"""
    H, W = 420, 1100
    img = np.full((H, W, 3), 255, np.uint8)
    y_warn = int(H - warn * H)
    y_dang = int(H - danger * H)
    cv2.line(img, (0, y_warn), (W, y_warn), (255, 165, 0), 1)
    cv2.line(img, (0, y_dang), (W, y_dang), (0, 0, 255), 1)
    cv2.putText(img, f"WARNING {warn:.2f}", (6, y_warn - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
    cv2.putText(img, f"DANGER {danger:.2f}", (6, y_dang - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    n = len(risks)
    if n > 1:
        xs = np.linspace(0, W - 1, n)
        for i in range(1, n):
            r0, r1 = risks[i - 1], risks[i]
            y0 = int(H - min(max(r0, 0), 1) * H)
            y1 = int(H - min(max(r1, 0), 1) * H)
            col = (0, 200, 0) if r1 < warn else ((0, 165, 255) if r1 < danger else (0, 0, 255))
            cv2.line(img, (int(xs[i - 1]), y0), (int(xs[i]), y1), col, 1)
    cv2.putText(img, f"ST-GCN risk curve  (frames={n})", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser(
        description="ST-GCN 姿态跌倒风险 · 视频演示（纯姿态）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="输入视频文件（必填）")
    ap.add_argument("--model", default="model/stgcn/stgcn_v1/run_01/best_model.pt",
                    help="ST-GCN best_model.pt（默认自动定位 run_01）")
    ap.add_argument("--resize-width", type=int, default=720,
                    help="解码/推理缩放宽度（默认720，越大越准越慢）")
    ap.add_argument("--process-every", type=int, default=1,
                    help="每 N 帧推理 1 次（默认1=逐帧；ST-GCN 很快，可逐帧）")
    ap.add_argument("--smooth-alpha", type=float, default=0.2,
                    help="风险 EMA 平滑：新样本权重(0~1)，越小越平滑；0=关闭平滑")
    ap.add_argument("--min-hold", type=int, default=3,
                    help="升级去抖：风险需连续 ≥N 帧超阈值才点亮报警(=enter_hold)，抑制瞬时误报；0→按1处理")
    ap.add_argument("--exit-hold", type=int, default=6,
                    help="降级去抖：风险需连续 ≥N 帧跌破退出阈值才熄灭报警，报警更'黏'防闪烁")
    ap.add_argument("--warn-thresh", type=float, default=0.45,
                    help="WARNING 进入阈值（默认0.45，已抬高防坐/蹲/弯腰误报；原始为0.3）")
    ap.add_argument("--danger-thresh", type=float, default=0.6,
                    help="DANGER 进入阈值（默认0.6）")
    ap.add_argument("--hysteresis", type=float, default=0.08,
                    help="迟滞带宽：退出阈值=进入阈值−此值，抖动落在带内保持原状态不翻转，消除橙色横跳")
    ap.add_argument("--gpu", action="store_true", help="强制 GPU")
    ap.add_argument("--cpu", action="store_true", help="强制 CPU")
    ap.add_argument("--no-save", action="store_true", help="不保存标注视频/曲线")
    ap.add_argument("--out-dir", default=None, help="自定义保存目录")
    # 坐标层 OneEuro 时序平滑（默认按模型自动：模型平滑训练则开，否则关 → 零漂移）
    ap.add_argument("--smooth-landmarks", dest="smooth_landmarks", action="store_true", default=None,
                    help="坐标层 OneEuro 滤波（默认：按模型训练时是否启用自动决定）")
    ap.add_argument("--no-smooth-landmarks", dest="smooth_landmarks", action="store_false",
                    help="关闭坐标滤波")
    ap.add_argument("--min-cutoff", type=float, default=1.0, help="OneEuro min_cutoff（越大越平滑）")
    ap.add_argument("--beta", type=float, default=0.02, help="OneEuro beta（快速运动自适应平滑）")
    ap.add_argument("--dcutoff", type=float, default=1.0, help="OneEuro dcutoff")
    # 帧级姿态置信度门控（方案A）
    ap.add_argument("--min-confidence", type=float, default=0.5,
                    help="帧级置信度门控：PoseResult.mean_confidence 低于此值视为低置信度，"
                         "该帧不更新模型/滑窗/报警（risk 保持上一有效值），恢复后自然接续")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERR] 打不开视频: {args.video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model, window, info = load_stgcn_model(args.model)

    # 坐标层 OneEuro 平滑：CLI 未指定时，按模型训练是否启用自动决定（保证零漂移）
    ls_info = info.get("landmark_smoothing", {})
    cli_specified = args.smooth_landmarks is not None
    if args.smooth_landmarks is None:
        args.smooth_landmarks = bool(ls_info.get("enabled", False))
    if ls_info.get("enabled"):
        args.min_cutoff = ls_info.get("min_cutoff", args.min_cutoff)
        args.beta = ls_info.get("beta", args.beta)
        args.dcutoff = ls_info.get("dcutoff", args.dcutoff)
    if args.smooth_landmarks and not ls_info.get("enabled"):
        print("  [WARN] 坐标滤波已手动开启，但所用模型非平滑数据训练 → 存在 train/infer 漂移风险！")
    smoother = LandmarkSmoother(
        enabled=bool(args.smooth_landmarks),
        min_cutoff=args.min_cutoff, beta=args.beta, dcutoff=args.dcutoff,
        n_landmarks=len(FALL_JOINTS_17), channels=3,
        fps=ls_info.get("fps", 25.0))
    if args.cpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = True
    else:
        use_gpu = torch.cuda.is_available()

    # 统一目标分辨率（cap.set 不可靠，read 后 resize）
    if args.resize_width and W and args.resize_width < W:
        target_w = args.resize_width
        target_h = int(round(H * target_w / W))
    else:
        target_w, target_h = (W or 640), (H or 480)

    device = torch.device("cuda" if use_gpu else "cpu")
    model.to(device).eval()

    ext = PoseExtractor(PoseExtractorConfig(
        model_path=str(TASK_PATH), use_gpu=use_gpu,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        min_pose_presence_confidence=0.3))
    ext.__enter__()

    print(f"=== ST-GCN 视频 demo ===\n  视频 : {args.video}\n  模型 : {args.model}"
          f"\n  窗口 : {window}  关节 : {len(FALL_JOINTS_17)}  3D\n"
          f"  原始 {W}x{H} → {target_w}x{target_h} @ {fps:.1f}fps  gpu={use_gpu}\n"
          f"  防误报: WARN≥{args.warn_thresh} DANGER≥{args.danger_thresh} "
          f"迟滞={args.hysteresis} 升级保持={args.min_hold}帧 降级保持={args.exit_hold}帧 "
          f"EMA α={args.smooth_alpha}\n"
          f"  坐标滤波(OneEuro): {'ON' if smoother.enabled else 'OFF'}  "
          f"min_cutoff={args.min_cutoff} beta={args.beta} dcutoff={args.dcutoff}"
          f"{'  [依模型自动]' if not cli_specified else ''}")

    # 保存
    writer = None
    save_dir = None
    if not args.no_save:
        if args.out_dir:
            save_dir = Path(args.out_dir)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = PROJECT_ROOT / "demo" / "runs" / f"stgcn_{Path(args.video).stem}_{ts}"
        save_dir.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(save_dir / "annotated.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_w, target_h))
        print(f"  保存 → {save_dir}")

    cv2.namedWindow("ST-GCN demo (q=quit, space=pause)", cv2.WINDOW_NORMAL)
    print("  按 q 退出，空格暂停")

    fsm = AlarmDebouncer(
        warn=args.warn_thresh, danger=args.danger_thresh,
        hysteresis=args.hysteresis,
        enter_hold=max(1, args.min_hold), exit_hold=max(1, args.exit_hold))

    sbuf = deque(maxlen=window)
    risks: list[float] = []
    raw_risks: list[float] = []
    last_risk = 0.0
    last_pr = None
    ema = 0.0
    alarm = "SAFE"
    n_low_conf = 0
    t = 0
    paused = False
    play_delay_ms = max(1, int(round(1000.0 / fps))) if fps > 0 else 40
    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    print("视频读取完毕，开始收尾（释放模型 / 写出结果）...")
                    break
                if frame.shape[1] != target_w or frame.shape[0] != target_h:
                    small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                else:
                    small = frame
                raw_risk = 0.0
                low_conf = False
                if t % max(args.process_every, 1) == 0:
                    pr = ext.detect_frame(small, int(t * 1000 / fps))
                    last_pr = pr
                    has_person = pr is not None
                    # 帧级置信度门控（方案A）：平均关节置信度过低 → 视为低置信度，
                    # 保持上一有效 risk / 报警等级，不更新滑窗/EMA/状态机，避免误报。
                    low_conf = bool(has_person and pr.mean_confidence < args.min_confidence)
                    valid = has_person and not low_conf
                    if valid:
                        skel = skeleton_from_poseresult(pr, FALL_JOINTS_17)  # (V,3)
                        skel = smoother.update(skel)                         # 坐标层 OneEuro 平滑
                        sbuf.append(skel)
                        if len(sbuf) >= window:
                            x = (torch.from_numpy(np.stack(sbuf)).float()
                                 .permute(2, 0, 1).unsqueeze(0).to(device))
                            with torch.no_grad():
                                raw_risk = float(model(x).item())
                        # 平滑：人消失时原始风险为0，EMA 自然衰减
                        if args.smooth_alpha > 0:
                            ema = args.smooth_alpha * raw_risk + (1 - args.smooth_alpha) * ema
                        else:
                            ema = raw_risk
                        alarm = fsm.update(ema)
                    elif has_person:  # 低置信度：保持上一有效风险/报警，不更新滑窗与状态机
                        pass
                    else:  # 真无人：清空状态机，不维持任何报警
                        if args.smooth_alpha > 0:
                            ema = (1 - args.smooth_alpha) * ema
                        else:
                            ema = 0.0
                        fsm.reset()
                        alarm = "SAFE"
                    last_risk = ema
                else:
                    raw_risk = 0.0
                    ema = last_risk
                risks.append(float(ema))
                raw_risks.append(float(raw_risk))
                if low_conf:
                    n_low_conf += 1
                vis = draw(small, last_pr, ema, alarm,
                           raw_risk if args.smooth_alpha > 0 else None,
                           low_conf=low_conf, min_conf=args.min_confidence)
                cv2.imshow("ST-GCN demo (q=quit, space=pause)", vis)
                if writer:
                    writer.write(vis)
            key = cv2.waitKey(play_delay_ms if not paused else 0) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
            t += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()          # 先关窗口，避免末尾画面"假死"卡顿
        if writer:
            writer.release()             # 写出标注视频（长视频收尾略慢，正常）
        ext.__exit__(None, None, None)   # 释放 MediaPipe（CPU 回退时稍慢，正常）
        print("收尾完成，已退出。")

    if save_dir and risks:
        wt, dt = args.warn_thresh, args.danger_thresh
        make_curve(risks, save_dir / "risk_curve.png", warn=wt, danger=dt)
        # 原始风险（平滑前）越过"标准阈值 0.3/0.6"的帧数 —— 反映模型本身有多毛躁
        raw_danger_std = int(sum(1 for r in raw_risks if r >= DANGER))
        raw_warn_std = int(sum(1 for r in raw_risks if WARN <= r < DANGER))
        # 平滑后越过"实际生效阈值"的帧数
        eff_danger = int(sum(1 for r in risks if r >= dt))
        eff_warn = int(sum(1 for r in risks if wt <= r < dt))
        (save_dir / "run_meta.json").write_text(json.dumps({
            "model": str(Path(args.model)),
            "window": window,
            "video": str(args.video),
            "fps": fps,
            "n_frames": len(risks),
            "smooth_alpha": args.smooth_alpha,
            "min_confidence": args.min_confidence,
            "n_low_conf_frames": n_low_conf,
            "min_hold_enter": args.min_hold,
            "exit_hold": args.exit_hold,
            "warn_thresh": wt,
            "danger_thresh": dt,
            "hysteresis": args.hysteresis,
            "mean_risk": float(np.mean(risks)),
            "max_risk": float(np.max(risks)),
            "n_danger_eff": eff_danger,
            "n_warning_eff": eff_warn,
            "raw_over_0.6": raw_danger_std,
            "raw_in_0.3_0.6": raw_warn_std,
            "suppressed_danger_vs_raw": raw_danger_std - eff_danger,
        }, indent=2), encoding="utf-8")
        print(f"  完成 → {save_dir}  (max_risk={np.max(risks):.3f}, "
              f"DANGER帧={eff_danger}, WARNING帧={eff_warn}, "
              f"原始越0.6毛刺={raw_danger_std} → 去抖后={eff_danger})")


if __name__ == "__main__":
    main()
