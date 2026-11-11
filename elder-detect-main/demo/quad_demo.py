#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demo/quad_demo.py — 2×2 四组合跌倒风险对比

以 compare_demo.py（已验证的黄金参考）为基础改写：
  - 下排两格 = MediaPipe 后端，逐字照搬 compare_demo 的已实现逻辑
    （ST-GCN run_03 / LSTM run_04、双通道 OneEuro 平滑、帧级置信度门控、
     EMA、AlarmDebouncer 迟滞去抖状态机）
  - 上排两格 = YOLO 后端，按完全相同结构对齐 run_05 训练分布
    （YOLO+ST-GCN = coco13 2D / run_05；YOLO+LSTM = 29 维 2D 特征 / run_05）

布局（每格是一条完整「姿态后端 → 风险模型」流水线，输入同源、可横向对比）：

    ┌──────────────────┬──────────────────┐
    │ YOLO + LSTM      │ YOLO + ST-GCN    │   ← 上排：YOLO 2D 后端（13 关节）
    ├──────────────────┼──────────────────┤
    │ MediaPipe+LSTM   │ MediaPipe+STGCN  │   ← 下排：MediaPipe 3D 后端（33 关节）
    └──────────────────┴──────────────────┘

预处理严格对齐各自训练配置（平滑对象/顺序/窗口/归一化），保证零漂移。

用法（elder 环境，Windows cmd）:
  python demo\\quad_demo.py "视频.mp4"
  python demo\\quad_demo.py "视频.mp4" --yolo-model yolov8m-pose.pt
  python demo\\quad_demo.py "视频.mp4" --resize-width 480 --cpu
  python demo\\quad_demo.py "视频.mp4" --no-save
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# —— MediaPipe 后端（与 compare_demo 一致）——
from pose.extractor import PoseExtractor, PoseExtractorConfig          # noqa: E402
from pose.keypoint import BodyPart, CORE_JOINTS_13, PoseResult        # noqa: E402
# —— YOLO 后端 ——
from pose.backend import create_pose_backend                           # noqa: E402
# —— 模型 ——
from model.stgcn.stgcn import load_stgcn                               # noqa: E402
from model.stgcn.graph import (                                        # noqa: E402
    FALL_JOINTS_17, FALL_EDGES,
    COCO13_JOINTS, COCO13_EDGES,
    skeleton_from_poseresult, skeleton_from_poseresult_2d,
)
from model.lstm.risk_model import RiskRegressor                       # noqa: E402
# —— 特征 / 平滑 ——
from features.feature_builder import FrameFeatureBuilder               # noqa: E402
from features.gait import GaitSpeedExtractor                          # noqa: E402
from features.feature_builder_2d import build_frame_features, LANK, RANK  # noqa: E402
from features.landmark_filter import LandmarkSmoother                 # noqa: E402

TASK_PATH = PROJECT_ROOT / "model/mediapipe_pose/pose_landmarker_lite.task"

# 运行时报警阈值（沿用 compare_demo 的防误报调校）
DEF_WARN, DEF_DANGER = 0.45, 0.6


# ==================================================================
# 迟滞去抖状态机（逐字来自 compare_demo）
# ==================================================================
class AlarmDebouncer:
    """迟滞去抖状态机：进入用高阈值、退出用低阈值，抖动落带内保持不翻转。"""

    LEVELS = ("SAFE", "WARNING", "DANGER")

    def __init__(self, warn_thresh=DEF_WARN, danger_thresh=DEF_DANGER, hysteresis=0.08,
                 enter_hold=3, exit_hold=6):
        self.warn = warn_thresh
        self.danger = danger_thresh
        self.hyst = hysteresis
        self.enter_hold = enter_hold
        self.exit_hold = exit_hold
        self.level = 0
        self._up = 0
        self._down = 0

    def update(self, ema: float):
        t = 2 if ema >= self.danger else (1 if ema >= self.warn else 0)
        if t > self.level:
            self._up += 1
            self._down = 0
            if self._up >= self.enter_hold:
                self.level = t
                self._up = 0
        elif t < self.level:
            exit_thr = (self.danger if self.level == 2 else self.warn) - self.hyst
            if ema < exit_thr:
                self._down += 1
                self._up = 0
                if self._down >= self.exit_hold:
                    self.level -= 1
                    self._down = 0
            else:
                self._up = 0
                self._down = 0
        else:
            self._up = 0
            self._down = 0
        return self.LEVELS[self.level]


# ==================================================================
# 绘制（逐字来自 compare_demo 的 draw_pane）
# ==================================================================
def draw_pane(frame, pr, risk, level, algo, joints, edges, warn, danger, win, warm,
              low_conf=False, min_conf=0.5):
    vis = frame.copy()
    h, w = vis.shape[:2]
    if pr is not None:
        pts = {}
        for i, part in enumerate(joints):
            try:
                kp = pr.get(part)
            except KeyError:
                continue
            if getattr(kp, "visibility", 0) >= 0.5:
                pts[i] = (int(kp.x * w), int(kp.y * h))
        if edges:
            for a, b in edges:
                if a in pts and b in pts:
                    cv2.line(vis, pts[a], pts[b], (0, 255, 255), 2)
        for i in pts:
            cv2.circle(vis, pts[i], 4, (0, 255, 0), -1)
    col = (0, 0, 255) if level == "DANGER" else ((0, 165, 255) if level == "WARNING" else (0, 200, 0))
    cv2.rectangle(vis, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.putText(vis, algo, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(vis, f"Risk {risk:.3f}  {level}", (8, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    bx, by, bw, bh = 8, 58, 300, 14
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (80, 80, 80), -1)
    cv2.rectangle(vis, (bx, by), (bx + int(bw * min(risk, 1.0)), by + bh), col, -1)
    wx = bx + int(bw * warn)
    dx = bx + int(bw * danger)
    cv2.line(vis, (wx, by), (wx, by + bh), (0, 165, 255), 1)
    cv2.line(vis, (dx, by), (dx, by + bh), (0, 0, 255), 1)
    if level == "DANGER":
        cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
    elif level == "WARNING":
        cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 165, 255), 3)
    if warm:
        cv2.putText(vis, f"warming up (win={win})", (w - 230, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    if low_conf and pr is not None:
        cv2.putText(vis, f"LOW CONF {pr.mean_confidence:.2f}<{min_conf:.2f}  risk held",
                    (w - 380, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return vis


def make_curve(risk_series: dict, path, warn, danger):
    """纯 cv2 画四模型风险曲线。"""
    H, W = 460, 1100
    img = np.full((H, W, 3), 255, np.uint8)
    y_warn = int(H - warn * H)
    y_dang = int(H - danger * H)
    cv2.line(img, (0, y_warn), (W, y_warn), (255, 165, 0), 1)
    cv2.line(img, (0, y_dang), (W, y_dang), (0, 0, 255), 1)
    cv2.putText(img, f"WARNING {warn:.2f}", (6, y_warn - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
    cv2.putText(img, f"DANGER {danger:.2f}", (6, y_dang - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    colors = {"yolo_lstm": (30, 80, 220), "yolo_stgcn": (220, 80, 30),
              "mp_lstm": (0, 150, 0), "mp_stgcn": (180, 0, 180)}
    labels = {"yolo_lstm": "YOLO+LSTM", "yolo_stgcn": "YOLO+ST-GCN",
              "mp_lstm": "MP+LSTM", "mp_stgcn": "MP+ST-GCN"}

    def plot(series, color):
        n = len(series)
        if n < 2:
            return
        xs = np.linspace(0, W - 1, n)
        for i in range(1, n):
            y0 = int(H - min(max(series[i - 1], 0), 1) * H)
            y1 = int(H - min(max(series[i], 0), 1) * H)
            cv2.line(img, (int(xs[i - 1]), y0), (int(xs[i]), y1), color, 1)

    for k in ["yolo_lstm", "yolo_stgcn", "mp_lstm", "mp_stgcn"]:
        if k in risk_series:
            plot(risk_series[k], colors[k])
            cv2.putText(img, labels[k], (W - 200, 20 + list(risk_series).index(k) * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[k], 1)
    cv2.imwrite(str(path), img)


# ==================================================================
# 模型加载
# ==================================================================
def _read_info(pt_path: str) -> dict:
    return json.loads((Path(pt_path).parent / "model_info.json").read_text(encoding="utf-8"))


def load_stgcn_model(pt_path: str, device):
    info = _read_info(pt_path)
    m = load_stgcn(pt_path).to(device).eval()
    return m, info.get("window_size", 60), info


def load_lstm_model(pt_path: str, device):
    info = _read_info(pt_path)
    m = RiskRegressor(input_dim=info.get("input_dim", 29))
    m.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    m.to(device).eval()
    return m, info.get("window_size", 35), info


# ==================================================================
# 单格风险推进（置信度门控 + EMA + 去抖，与 compare_demo 同逻辑）
# ==================================================================
def dispatch_panel(ema, deb, raw_risk, valid, has_person, low_conf, alpha):
    """返回 (ema, level, raw)。"""
    if valid:
        raw = raw_risk
        ema = (1 - alpha) * ema + alpha * raw_risk if alpha > 0 else raw_risk
        level = deb.update(ema)
    elif has_person:  # 低置信度：保持上一有效 risk / 报警等级，不更新窗口/EMA/状态机
        raw = 0.0
        level = deb.LEVELS[deb.level]
    else:  # 真无人：ema 衰减、fsm 降级、risk→0
        raw = 0.0
        ema = (1 - alpha) * ema if alpha > 0 else 0.0
        level = deb.update(ema)
    return ema, level, raw


def main():
    ap = argparse.ArgumentParser(description="2×2 四组合跌倒风险对比（YOLO/MediaPipe × LSTM/ST-GCN）")
    # —— MediaPipe 两格（默认与 compare_demo 一致）——
    ap.add_argument("video")
    ap.add_argument("--stgcn-model", default="model/stgcn/stgcn_v1/run_03/best_model.pt",
                    help="MediaPipe ST-GCN 权重 (默认 run_03, 17关节3D, 窗口60)")
    ap.add_argument("--lstm-model", default="model/lstm/risk_v1/run_04/best_model.pt",
                    help="MediaPipe LSTM 权重 (默认 run_04, 29维, 窗口35, 平滑训练)")
    # —— YOLO 两格（run_05 / 2D）——
    ap.add_argument("--yolo-stgcn-model", default="model/stgcn/stgcn_v1/run_05/best_model.pt",
                    help="YOLO ST-GCN 权重 (默认 run_05, coco13 2D, 窗口60)")
    ap.add_argument("--yolo-lstm-model", default="model/lstm/risk_v1/run_05/best_model.pt",
                    help="YOLO LSTM 权重 (默认 run_05, 29维2D特征, 窗口60)")
    ap.add_argument("--yolo-model", type=str, default="yolov8m-pose.pt",
                   help="YOLO 姿态权重（默认 yolov8m-pose.pt；也可 yolov8n/s/l-pose.pt）")
    ap.add_argument("--yolo-conf", type=float, default=0.5, help="YOLO 检测置信度阈值")
    # —— 通用 ——
    ap.add_argument("--cpu", action="store_true", help="强制 CPU")
    ap.add_argument("--resize-width", type=int, default=720, help="每个面板宽度")
    ap.add_argument("--no-save", action="store_true", help="只看不存")
    ap.add_argument("--out-dir", default=None, help="保存目录 (默认 demo/runs/quad_...)")
    ap.add_argument("--smooth-alpha", type=float, default=0.25, help="风险 EMA 平滑系数(0~1)")
    ap.add_argument("--warn-thresh", type=float, default=DEF_WARN)
    ap.add_argument("--danger-thresh", type=float, default=DEF_DANGER)
    ap.add_argument("--hysteresis", type=float, default=0.08)
    ap.add_argument("--enter-hold", type=int, default=3)
    ap.add_argument("--exit-hold", type=int, default=6)
    ap.add_argument("--smooth-landmarks", dest="smooth_landmarks", action="store_true", default=None,
                    help="坐标层 OneEuro 滤波（默认：按模型训练是否启用自动决定）")
    ap.add_argument("--no-smooth-landmarks", dest="smooth_landmarks", action="store_false",
                    help="关闭坐标滤波")
    ap.add_argument("--min-cutoff", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--dcutoff", type=float, default=1.0)
    ap.add_argument("--min-confidence", type=float, default=0.5,
                    help="帧级姿态置信度门控阈值（PoseResult.mean_confidence 低于此值→保持风险）")
    ap.add_argument("--min-detection", type=float, default=0.3, help="MediaPipe 检测阈值")
    # 跳帧（默认每帧都跑，保证滑动窗口时序一致；仅在需要提速时调大）
    ap.add_argument("--process-every", type=int, default=1, help="每 N 帧重算 YOLO 姿态")
    ap.add_argument("--mp-process-every", type=int, default=None, help="每 N 帧重算 MediaPipe 姿态")
    args = ap.parse_args()

    use_gpu = (not args.cpu) and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    # ---- 加载 4 个模型 ----
    mp_stgcn_model, mp_stgcn_win, mp_stgcn_info = load_stgcn_model(args.stgcn_model, device)
    mp_lstm_model, mp_lstm_win, mp_lstm_info = load_lstm_model(args.lstm_model, device)
    yolo_stgcn_model, yolo_stgcn_win, yolo_stgcn_info = load_stgcn_model(args.yolo_stgcn_model, device)
    yolo_lstm_model, yolo_lstm_win, yolo_lstm_info = load_lstm_model(args.yolo_lstm_model, device)

    fb = FrameFeatureBuilder()
    gait = GaitSpeedExtractor()

    # ---- 坐标层 OneEuro 平滑（与各自训练分布一致；CLI 可整体覆盖）----
    mp_ls_info = mp_stgcn_info.get("landmark_smoothing", {})
    mp_ll_info = mp_lstm_info.get("landmark_smoothing", {})
    yolo_ls_info = yolo_stgcn_info.get("landmark_smoothing", {})
    yolo_ll_info = yolo_lstm_info.get("landmark_smoothing", {})

    if args.smooth_landmarks is None:
        # MediaPipe：ST-GCN 默认 OFF（run_03 无平滑记录），LSTM 默认 ON（run_04 fallback）
        mp_s_on = bool(mp_ls_info.get("enabled", False))
        mp_l_on = bool(mp_ll_info.get("enabled", True))
        # YOLO：run_05 训练默认平滑 ON
        yolo_s_on = bool(yolo_ls_info.get("enabled", True))
        yolo_l_on = bool(yolo_ll_info.get("enabled", True))
    else:
        mp_s_on = mp_l_on = yolo_s_on = yolo_l_on = args.smooth_landmarks

    if mp_ls_info.get("enabled"):
        args.min_cutoff = mp_ls_info.get("min_cutoff", args.min_cutoff)
        args.beta = mp_ls_info.get("beta", args.beta)
        args.dcutoff = mp_ls_info.get("dcutoff", args.dcutoff)
    if mp_ll_info.get("enabled"):
        args.min_cutoff = mp_ll_info.get("min_cutoff", args.min_cutoff)
        args.beta = mp_ll_info.get("beta", args.beta)
        args.dcutoff = mp_ll_info.get("dcutoff", args.dcutoff)
    mp_fps = mp_ls_info.get("fps", 25.0)
    yolo_fps = yolo_ls_info.get("fps", 25.0)

    # MediaPipe: ST-GCN 17×3；LSTM 双通道 33×2 / 33×3（与 build_from_cache 一致）
    smoother_s = LandmarkSmoother(enabled=bool(mp_s_on), min_cutoff=args.min_cutoff,
                                  beta=args.beta, dcutoff=args.dcutoff,
                                  n_landmarks=len(FALL_JOINTS_17), channels=3, fps=mp_fps)
    smoother_l2d = LandmarkSmoother(enabled=bool(mp_l_on), min_cutoff=args.min_cutoff,
                                    beta=args.beta, dcutoff=args.dcutoff,
                                    n_landmarks=33, channels=2, fps=mp_fps)
    smoother_l3d = LandmarkSmoother(enabled=bool(mp_l_on), min_cutoff=args.min_cutoff,
                                    beta=args.beta, dcutoff=args.dcutoff,
                                    n_landmarks=33, channels=3, fps=mp_fps)
    # YOLO: 两路都是 13×2（与 build_from_cache_yolo / build_skeleton_yolo 一致）
    yolo_smoother_s = LandmarkSmoother(enabled=bool(yolo_s_on), min_cutoff=args.min_cutoff,
                                       beta=args.beta, dcutoff=args.dcutoff,
                                       n_landmarks=len(COCO13_JOINTS), channels=2, fps=yolo_fps)
    yolo_smoother_l = LandmarkSmoother(enabled=bool(yolo_l_on), min_cutoff=args.min_cutoff,
                                       beta=args.beta, dcutoff=args.dcutoff,
                                       n_landmarks=len(COCO13_JOINTS), channels=2, fps=yolo_fps)

    # ---- 视频 ----
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERR] 无法打开视频: {args.video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_w = args.resize_width
    target_h = int(H0 * target_w / W0) if W0 else target_w

    # ---- 两个后端 ----
    mp_ext = PoseExtractor(PoseExtractorConfig(
        model_path=str(TASK_PATH),
        use_gpu=use_gpu,
        min_detection_confidence=args.min_detection,
        min_tracking_confidence=args.min_detection,
        min_pose_presence_confidence=args.min_detection))
    mp_ext.__enter__()
    yolo_bk = create_pose_backend("yolo", device="cuda" if use_gpu else "cpu",
                                  model=args.yolo_model, conf=args.yolo_conf)
    yolo_bk.__enter__()

    # ---- 滑动窗口缓冲 ----
    mp_sbuf = deque(maxlen=mp_stgcn_win)
    mp_lbuf = deque(maxlen=mp_lstm_win)
    yolo_sbuf = deque(maxlen=yolo_stgcn_win)
    yolo_lbuf = deque(maxlen=yolo_lstm_win)

    # ---- 去抖状态机 ----
    mp_fsm_s = AlarmDebouncer(args.warn_thresh, args.danger_thresh, args.hysteresis,
                              args.enter_hold, args.exit_hold)
    mp_fsm_l = AlarmDebouncer(args.warn_thresh, args.danger_thresh, args.hysteresis,
                              args.enter_hold, args.exit_hold)
    yolo_fsm_s = AlarmDebouncer(args.warn_thresh, args.danger_thresh, args.hysteresis,
                                args.enter_hold, args.exit_hold)
    yolo_fsm_l = AlarmDebouncer(args.warn_thresh, args.danger_thresh, args.hysteresis,
                                args.enter_hold, args.exit_hold)

    # ---- 各格状态 ----
    state = {
        "yolo_lstm":  {"ema": 0.0, "level": "SAFE", "raw": 0.0, "buf": yolo_lbuf, "fsm": yolo_fsm_l, "warm": True},
        "yolo_stgcn": {"ema": 0.0, "level": "SAFE", "raw": 0.0, "buf": yolo_sbuf, "fsm": yolo_fsm_s, "warm": True},
        "mp_lstm":    {"ema": 0.0, "level": "SAFE", "raw": 0.0, "buf": mp_lbuf, "fsm": mp_fsm_l, "warm": True},
        "mp_stgcn":   {"ema": 0.0, "level": "SAFE", "raw": 0.0, "buf": mp_sbuf, "fsm": mp_fsm_s, "warm": True},
    }

    curves = {k: [] for k in state}
    rows = []
    n_low_conf = n_no_person = 0

    yolo_prev_ankle = None
    last_yolo_pr = last_mp_pr = None

    # ---- 保存 ----
    save_dir = writer = csv_f = csv_w = None
    if not args.no_save:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        vname = Path(args.video).stem
        save_dir = Path(args.out_dir) if args.out_dir else (
            PROJECT_ROOT / "demo" / "runs" / f"quad_{vname}_{stamp}")
        save_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_dir / "annotated.mp4"),
                                 fourcc, fps, (target_w * 2, target_h * 2))
        csv_f = open(save_dir / "frames.csv", "w", newline="", encoding="utf-8")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["frame", "ts_ms", "has_person_yolo", "has_person_mp",
                        "mean_conf_yolo", "mean_conf_mp",
                        "yolo_lstm_raw", "yolo_lstm_risk", "yolo_lstm_alarm",
                        "yolo_stgcn_raw", "yolo_stgcn_risk", "yolo_stgcn_alarm",
                        "mp_lstm_raw", "mp_lstm_risk", "mp_lstm_alarm",
                        "mp_stgcn_raw", "mp_stgcn_risk", "mp_stgcn_alarm"])
        print(f"  保存 → {save_dir}")

    print(f"=== 2×2 四组合跌倒风险对比 ===\n"
          f"  视频 : {args.video}\n"
          f"  MediaPipe ST-GCN : {args.stgcn_model}  窗口={mp_stgcn_win}\n"
          f"  MediaPipe LSTM   : {args.lstm_model}  窗口={mp_lstm_win}\n"
          f"  YOLO ST-GCN      : {args.yolo_stgcn_model}  窗口={yolo_stgcn_win}\n"
          f"  YOLO LSTM        : {args.yolo_lstm_model}  窗口={yolo_lstm_win}\n"
          f"  面板 {target_w}x{target_h} @ {fps:.1f}fps  gpu={use_gpu}\n"
          f"  防误报: WARN={args.warn_thresh} DANGER={args.danger_thresh} "
          f"hyst={args.hysteresis} enter={args.enter_hold} exit={args.exit_hold} smooth={args.smooth_alpha}\n"
          f"  坐标滤波(OneEuro): MP-STGCN={'ON' if smoother_s.enabled else 'OFF'}  "
          f"MP-LSTM={'ON' if smoother_l3d.enabled else 'OFF'}  "
          f"YOLO-STGCN={'ON' if yolo_smoother_s.enabled else 'OFF'}  "
          f"YOLO-LSTM={'ON' if yolo_smoother_l.enabled else 'OFF'}\n"
          f"  帧级置信度门控(方案A): 阈值={args.min_confidence}\n"
          f"  操作: q=退出  空格=暂停")

    try:
        t = 0
        paused = False
        last_t = time.perf_counter()
        fps_ema = 0.0
        play_delay_ms = max(1, int(round(1000.0 / fps))) if fps > 0 else 40

        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    print("视频读取完毕，开始收尾...")
                    break
                small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR) if (
                    frame.shape[1] != target_w or frame.shape[0] != target_h) else frame
                ts_ms = int(t * 1000 / fps)

                pe_yolo = max(args.process_every, 1)
                pe_mp = max(args.mp_process_every if args.mp_process_every is not None else args.process_every, 1)

                # ---------- 两个后端推理（同帧同源）----------
                if t % pe_yolo == 0:
                    last_yolo_pr = yolo_bk.detect_frame(small, ts_ms)
                if t % pe_mp == 0:
                    last_mp_pr = mp_ext.detect_frame(small, ts_ms)

                # ========== YOLO 车道（每 step 帧推进）==========
                if t % pe_yolo == 0:
                    yolo_pr = last_yolo_pr
                    yolo_has = yolo_pr is not None
                    yolo_low = bool(yolo_has and yolo_pr.mean_confidence < args.min_confidence)
                    yolo_valid = yolo_has and not yolo_low

                    yolo_s_risk = yolo_l_risk = 0.0
                    if yolo_valid:
                        # --- YOLO + ST-GCN：归一化 2D → OneEuro → 窗口 ---
                        skel = skeleton_from_poseresult_2d(yolo_pr)        # (13,2) 髋心归一化
                        skel = yolo_smoother_s.update(skel)
                        yolo_sbuf.append(skel)
                        if len(yolo_sbuf) >= yolo_stgcn_win:
                            x = (torch.from_numpy(np.stack(yolo_sbuf)).float()
                                 .permute(2, 0, 1).unsqueeze(0).to(device))
                            with torch.no_grad():
                                yolo_s_risk = float(yolo_stgcn_model(x).item())
                        # --- YOLO + LSTM：原始 2D → OneEuro → gait → 29 维特征 → 窗口 ---
                        lm2d = np.array([[kp.x, kp.y] for kp in yolo_pr.landmarks], dtype=np.float32)
                        lm2d = yolo_smoother_l.update(lm2d)
                        ankle_mid = (lm2d[LANK] + lm2d[RANK]) / 2.0
                        gait_v = 0.0
                        if yolo_prev_ankle is not None:
                            gait_v = float(np.linalg.norm(ankle_mid - yolo_prev_ankle) * 25.0)
                        yolo_prev_ankle = ankle_mid
                        vec = build_frame_features(lm2d, gait_v)            # (29,)
                        yolo_lbuf.append(np.asarray(vec, dtype=np.float32))
                        if len(yolo_lbuf) >= yolo_lstm_win:
                            x = torch.from_numpy(np.stack(yolo_lbuf)).float().unsqueeze(0).to(device)
                            with torch.no_grad():
                                yolo_l_risk = float(yolo_lstm_model(x).item())

                    e, lv, raw = dispatch_panel(state["yolo_stgcn"]["ema"], yolo_fsm_s,
                                                yolo_s_risk, yolo_valid, yolo_has, yolo_low,
                                                args.smooth_alpha)
                    state["yolo_stgcn"]["ema"], state["yolo_stgcn"]["level"], state["yolo_stgcn"]["raw"] = e, lv, raw
                    e, lv, raw = dispatch_panel(state["yolo_lstm"]["ema"], yolo_fsm_l,
                                                yolo_l_risk, yolo_valid, yolo_has, yolo_low,
                                                args.smooth_alpha)
                    state["yolo_lstm"]["ema"], state["yolo_lstm"]["level"], state["yolo_lstm"]["raw"] = e, lv, raw
                    state["yolo_stgcn"]["warm"] = yolo_valid and len(yolo_sbuf) < yolo_stgcn_win
                    state["yolo_lstm"]["warm"] = yolo_valid and len(yolo_lbuf) < yolo_lstm_win

                # ========== MediaPipe 车道（逐字来自 compare_demo）==========
                if t % pe_mp == 0:
                    mp_pr = last_mp_pr
                    mp_has = mp_pr is not None
                    mp_low = bool(mp_has and mp_pr.mean_confidence < args.min_confidence)
                    mp_valid = mp_has and not mp_low

                    s_risk = l_risk = 0.0
                    if mp_valid:
                        # --- ST-GCN 分支 ---
                        skel = skeleton_from_poseresult(mp_pr, FALL_JOINTS_17)   # (17,3)
                        skel = smoother_s.update(skel)
                        mp_sbuf.append(skel)
                        if len(mp_sbuf) >= mp_stgcn_win:
                            x = (torch.from_numpy(np.stack(mp_sbuf)).float()
                                 .permute(2, 0, 1).unsqueeze(0).to(device))
                            with torch.no_grad():
                                s_risk = float(mp_stgcn_model(x).item())
                        # --- LSTM 分支（先对 2D+3D 双通道做 OneEuro 平滑）---
                        if mp_l_on and mp_pr.world_landmarks is not None:
                            lm2d = np.array([[kp.x, kp.y] for kp in mp_pr.landmarks], dtype=np.float32)
                            wl3d = np.array([[kp.x, kp.y, kp.z] for kp in mp_pr.world_landmarks], dtype=np.float32)
                            lm2d = smoother_l2d.update(lm2d)
                            wl3d = smoother_l3d.update(wl3d)
                            for i, kp in enumerate(mp_pr.landmarks):
                                kp.x, kp.y = float(lm2d[i, 0]), float(lm2d[i, 1])
                            for i, kp in enumerate(mp_pr.world_landmarks):
                                kp.x, kp.y, kp.z = float(wl3d[i, 0]), float(wl3d[i, 1]), float(wl3d[i, 2])
                        gs = gait.compute_per_frame(mp_pr)
                        vec = fb.build(mp_pr, gs)   # (29,)
                        mp_lbuf.append(np.asarray(vec, dtype=np.float32))
                        if len(mp_lbuf) >= mp_lstm_win:
                            x = torch.from_numpy(np.stack(mp_lbuf)).float().unsqueeze(0).to(device)
                            with torch.no_grad():
                                l_risk = float(mp_lstm_model(x).item())

                    e, lv, raw = dispatch_panel(state["mp_stgcn"]["ema"], mp_fsm_s,
                                                s_risk, mp_valid, mp_has, mp_low, args.smooth_alpha)
                    state["mp_stgcn"]["ema"], state["mp_stgcn"]["level"], state["mp_stgcn"]["raw"] = e, lv, raw
                    e, lv, raw = dispatch_panel(state["mp_lstm"]["ema"], mp_fsm_l,
                                                l_risk, mp_valid, mp_has, mp_low, args.smooth_alpha)
                    state["mp_lstm"]["ema"], state["mp_lstm"]["level"], state["mp_lstm"]["raw"] = e, lv, raw
                    state["mp_stgcn"]["warm"] = mp_valid and len(mp_sbuf) < mp_stgcn_win
                    state["mp_lstm"]["warm"] = mp_valid and len(mp_lbuf) < mp_lstm_win

                # ========== 绘制 2×2 ==========
                tl = draw_pane(small, last_yolo_pr, state["yolo_lstm"]["ema"], state["yolo_lstm"]["level"],
                               "YOLO + LSTM", COCO13_JOINTS, COCO13_EDGES,
                               args.warn_thresh, args.danger_thresh, yolo_lstm_win, state["yolo_lstm"]["warm"],
                               low_conf=bool(last_yolo_pr is not None and last_yolo_pr.mean_confidence < args.min_confidence),
                               min_conf=args.min_confidence)
                tr = draw_pane(small, last_yolo_pr, state["yolo_stgcn"]["ema"], state["yolo_stgcn"]["level"],
                               "YOLO + ST-GCN", COCO13_JOINTS, COCO13_EDGES,
                               args.warn_thresh, args.danger_thresh, yolo_stgcn_win, state["yolo_stgcn"]["warm"],
                               low_conf=bool(last_yolo_pr is not None and last_yolo_pr.mean_confidence < args.min_confidence),
                               min_conf=args.min_confidence)
                bl = draw_pane(small, last_mp_pr, state["mp_lstm"]["ema"], state["mp_lstm"]["level"],
                               "MediaPipe + LSTM", CORE_JOINTS_13, None,
                               args.warn_thresh, args.danger_thresh, mp_lstm_win, state["mp_lstm"]["warm"],
                               low_conf=bool(last_mp_pr is not None and last_mp_pr.mean_confidence < args.min_confidence),
                               min_conf=args.min_confidence)
                br = draw_pane(small, last_mp_pr, state["mp_stgcn"]["ema"], state["mp_stgcn"]["level"],
                               "MediaPipe + ST-GCN", FALL_JOINTS_17, FALL_EDGES,
                               args.warn_thresh, args.danger_thresh, mp_stgcn_win, state["mp_stgcn"]["warm"],
                               low_conf=bool(last_mp_pr is not None and last_mp_pr.mean_confidence < args.min_confidence),
                               min_conf=args.min_confidence)

                canvas = np.zeros((target_h * 2, target_w * 2, 3), dtype=np.uint8)
                canvas[0:target_h, 0:target_w] = tl
                canvas[0:target_h, target_w:target_w * 2] = tr
                canvas[target_h:target_h * 2, 0:target_w] = bl
                canvas[target_h:target_h * 2, target_w:target_w * 2] = br

                for k in state:
                    curves[k].append(float(state[k]["ema"]))
                if last_yolo_pr is None and last_mp_pr is None:
                    n_no_person += 1

                if writer:
                    writer.write(canvas)
                if csv_w:
                    rows.append({
                        "frame": t, "ts_ms": ts_ms,
                        "has_person_yolo": int(last_yolo_pr is not None),
                        "has_person_mp": int(last_mp_pr is not None),
                        "mean_conf_yolo": round(float(last_yolo_pr.mean_confidence), 4) if last_yolo_pr else 0.0,
                        "mean_conf_mp": round(float(last_mp_pr.mean_confidence), 4) if last_mp_pr else 0.0,
                        "yolo_lstm_raw": round(float(state["yolo_lstm"]["raw"]), 4) if (t % pe_yolo == 0 and last_yolo_pr is not None) else "",
                        "yolo_lstm_risk": round(float(state["yolo_lstm"]["ema"]), 4),
                        "yolo_lstm_alarm": state["yolo_lstm"]["level"],
                        "yolo_stgcn_raw": round(float(state["yolo_stgcn"]["raw"]), 4) if (t % pe_yolo == 0 and last_yolo_pr is not None) else "",
                        "yolo_stgcn_risk": round(float(state["yolo_stgcn"]["ema"]), 4),
                        "yolo_stgcn_alarm": state["yolo_stgcn"]["level"],
                        "mp_lstm_raw": round(float(state["mp_lstm"]["raw"]), 4) if (t % pe_mp == 0 and last_mp_pr is not None) else "",
                        "mp_lstm_risk": round(float(state["mp_lstm"]["ema"]), 4),
                        "mp_lstm_alarm": state["mp_lstm"]["level"],
                        "mp_stgcn_raw": round(float(state["mp_stgcn"]["raw"]), 4) if (t % pe_mp == 0 and last_mp_pr is not None) else "",
                        "mp_stgcn_risk": round(float(state["mp_stgcn"]["ema"]), 4),
                        "mp_stgcn_alarm": state["mp_stgcn"]["level"],
                    })

                now = time.perf_counter()
                dt = now - last_t
                last_t = now
                if dt > 0:
                    inst = 1.0 / dt
                    fps_ema = inst if fps_ema == 0 else fps_ema * 0.9 + inst * 0.1
                t += 1
            else:
                canvas = np.zeros((target_h * 2, target_w * 2, 3), dtype=np.uint8)

            cv2.putText(canvas, f"FPS={fps_ema:.1f}", (8, target_h * 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("quad 2x2  [q=quit, space=pause]", canvas)
            key = cv2.waitKey(0 if paused else play_delay_ms) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if writer:
            writer.release()
        if csv_f:
            csv_f.close()
        yolo_bk.__exit__(None, None, None)
        mp_ext.__exit__(None, None, None)
        print("收尾完成，已退出。")

    if save_dir:
        make_curve(curves, save_dir / "risk_curve.png", args.warn_thresh, args.danger_thresh)
        (save_dir / "run_meta.json").write_text(json.dumps({
            "video": str(args.video),
            "fps": fps,
            "work_res": [target_w, target_h],
            "total_frames": t,
            "avg_display_fps": round(fps_ema, 2),
            "warn_thresh": args.warn_thresh,
            "danger_thresh": args.danger_thresh,
            "smooth_alpha": args.smooth_alpha,
            "min_confidence": args.min_confidence,
            "panels": {
                "yolo_lstm": {"backend": "yolo", "risk": "lstm", "model": str(Path(args.yolo_lstm_model)),
                              "window": yolo_lstm_win, "smooth": yolo_smoother_l.enabled},
                "yolo_stgcn": {"backend": "yolo", "risk": "stgcn", "model": str(Path(args.yolo_stgcn_model)),
                               "window": yolo_stgcn_win, "smooth": yolo_smoother_s.enabled},
                "mp_lstm": {"backend": "mediapipe", "risk": "lstm", "model": str(Path(args.lstm_model)),
                            "window": mp_lstm_win, "smooth": smoother_l3d.enabled},
                "mp_stgcn": {"backend": "mediapipe", "risk": "stgcn", "model": str(Path(args.stgcn_model)),
                             "window": mp_stgcn_win, "smooth": smoother_s.enabled},
            },
            "yolo_weights": args.yolo_model,
            "outputs": ["annotated.mp4", "risk_curve.png", "frames.csv", "run_meta.json"],
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        with (save_dir / "frames.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "ts_ms", "has_person_yolo", "has_person_mp",
                        "mean_conf_yolo", "mean_conf_mp",
                        "yolo_lstm_raw", "yolo_lstm_risk", "yolo_lstm_alarm",
                        "yolo_stgcn_raw", "yolo_stgcn_risk", "yolo_stgcn_alarm",
                        "mp_lstm_raw", "mp_lstm_risk", "mp_lstm_alarm",
                        "mp_stgcn_raw", "mp_stgcn_risk", "mp_stgcn_alarm"])
            for r in rows:
                w.writerow([r["frame"], r["ts_ms"], r["has_person_yolo"], r["has_person_mp"],
                            r["mean_conf_yolo"], r["mean_conf_mp"],
                            r["yolo_lstm_raw"], r["yolo_lstm_risk"], r["yolo_lstm_alarm"],
                            r["yolo_stgcn_raw"], r["yolo_stgcn_risk"], r["yolo_stgcn_alarm"],
                            r["mp_lstm_raw"], r["mp_lstm_risk"], r["mp_lstm_alarm"],
                            r["mp_stgcn_raw"], r["mp_stgcn_risk"], r["mp_stgcn_alarm"]])

        print(f"  完成 -> {save_dir}\n"
              f"    产物: annotated.mp4 / risk_curve.png / frames.csv / run_meta.json")
        for k in ["yolo_lstm", "yolo_stgcn", "mp_lstm", "mp_stgcn"]:
            s = curves[k]
            if s:
                print(f"    {k:12s}: max_risk={max(s):.3f}  "
                      f"DANGER帧={sum(1 for r in s if r >= args.danger_thresh)}  "
                      f"WARNING帧={sum(1 for r in s if args.warn_thresh <= r < args.danger_thresh)}")


if __name__ == "__main__":
    main()
