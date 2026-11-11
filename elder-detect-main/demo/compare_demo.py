"""
ST-GCN vs LSTM 并排对比 demo（同一视频、同一帧、同一套后处理）

左: ST-GCN (run_03, 17 关节 3D, 窗口 60)
右: LSTM   (run_04, 29 维特征, 窗口 35, 坐标已平滑)

两路共用一个 MediaPipe PoseExtractor（output_world_landmarks=True），
每帧各跑一次推理，各自维护:
  - 滑动窗口缓冲
  - EMA 平滑 (--smooth-alpha)
  - 迟滞去抖状态机 (--warn-thresh/--danger-thresh/--hysteresis/--enter-hold/--exit-hold)
从而公平对比两个模型在同一段视频上的风险曲线、谁先报警、谁误报。

用法 (项目根, (elder) 环境, Windows cmd):
  python demo\\compare_demo.py "视频.mp4"
  python demo\\compare_demo.py "视频.mp4" --cpu
  python demo\\compare_demo.py "视频.mp4" --stgcn-model model\\stgcn\\stgcn_v1\\run_01\\best_model.pt   # 换成 S1
  python demo\\compare_demo.py "视频.mp4" --no-save
  python demo\\compare_demo.py "视频.mp4" --warn-thresh 0.5 --hysteresis 0.1 --exit-hold 8   # 更严

  保存 (默认开启, --no-save 关闭) -> demo/runs/compare_<视频名>_<时间戳>/:
    annotated.mp4  并排标注视频
    risk_curve.png 双模型风险曲线
    frames.csv     逐帧明细 (frame/ts_ms/has_person/mean_conf/low_conf/
                             stgcn_raw/stgcn_risk/stgcn_alarm/
                             lstm_raw/lstm_risk/lstm_alarm)
    run_meta.json  汇总统计
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

from pose.extractor import PoseExtractor, PoseExtractorConfig          # noqa: E402
from model.stgcn.stgcn import load_stgcn                               # noqa: E402
from model.stgcn.graph import FALL_JOINTS_17, FALL_EDGES, skeleton_from_poseresult  # noqa: E402
from model.lstm.risk_model import RiskRegressor                       # noqa: E402
from features.feature_builder import FrameFeatureBuilder               # noqa: E402
from features.gait import GaitSpeedExtractor                           # noqa: E402
from pose.keypoint import CORE_JOINTS_13                              # noqa: E402
from features.landmark_filter import LandmarkSmoother                  # noqa: E402

TASK_PATH = PROJECT_ROOT / "model/mediapipe_pose/pose_landmarker_lite.task"

# ---- 运行时报警阈值（默认沿用 stgcn_video_demo 的防误报调校）----
DEF_WARN, DEF_DANGER = 0.45, 0.6


class AlarmDebouncer:
    """迟滞去抖状态机：进入用高阈值、退出用低阈值，抖动落带内保持不翻转。

    解决“坐下/蹲下/弯腰时风险在阈值附近横跳闪烁”的误报。
    """

    LEVELS = ("SAFE", "WARNING", "DANGER")

    def __init__(self, warn_thresh=DEF_WARN, danger_thresh=DEF_DANGER, hysteresis=0.08,
                 enter_hold=3, exit_hold=6):
        self.warn = warn_thresh
        self.danger = danger_thresh
        self.hyst = hysteresis
        self.enter_hold = enter_hold
        self.exit_hold = exit_hold
        self.level = 0            # 0=SAFE 1=WARNING 2=DANGER
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


def draw_pane(frame, pr, risk, level, algo, joints, edges, warn, danger, win, warm,
              low_conf=False, min_conf=0.5):
    """单模型面板：骨架 + 风险条 + 状态 + 报警边框。
    low_conf: 该帧为低置信度（仅画面提示，risk 由调用方保持上一有效值）。"""
    vis = frame.copy()
    h, w = vis.shape[:2]
    # 骨架
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
    # 顶部信息条
    col = (0, 0, 255) if level == "DANGER" else ((0, 165, 255) if level == "WARNING" else (0, 200, 0))
    cv2.rectangle(vis, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.putText(vis, algo, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(vis, f"Risk {risk:.3f}  {level}", (8, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    # 风险条
    bx, by, bw, bh = 8, 58, 300, 14
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (80, 80, 80), -1)
    cv2.rectangle(vis, (bx, by), (bx + int(bw * min(risk, 1.0)), by + bh), col, -1)
    wx = bx + int(bw * warn)
    dx = bx + int(bw * danger)
    cv2.line(vis, (wx, by), (wx, by + bh), (0, 165, 255), 1)
    cv2.line(vis, (dx, by), (dx, by + bh), (0, 0, 255), 1)
    # 报警边框
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


def make_curve(stgcn_risks, lstm_risks, path, warn, danger):
    """纯 cv2 画双模型风险曲线（零 matplotlib 依赖）。"""
    H, W = 440, 1100
    img = np.full((H, W, 3), 255, np.uint8)
    y_warn = int(H - warn * H)
    y_dang = int(H - danger * H)
    cv2.line(img, (0, y_warn), (W, y_warn), (255, 165, 0), 1)
    cv2.line(img, (0, y_dang), (W, y_dang), (0, 0, 255), 1)
    cv2.putText(img, f"WARNING {warn:.2f}", (6, y_warn - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
    cv2.putText(img, f"DANGER {danger:.2f}", (6, y_dang - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, "ST-GCN (blue)", (W - 200, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 80, 220), 1)
    cv2.putText(img, "LSTM (green)", (W - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)

    def plot(series, color):
        n = len(series)
        if n < 2:
            return
        xs = np.linspace(0, W - 1, n)
        for i in range(1, n):
            y0 = int(H - min(max(series[i - 1], 0), 1) * H)
            y1 = int(H - min(max(series[i], 0), 1) * H)
            cv2.line(img, (int(xs[i - 1]), y0), (int(xs[i]), y1), color, 1)

    plot(stgcn_risks, (30, 80, 220))
    plot(lstm_risks, (0, 150, 0))
    cv2.imwrite(str(path), img)


def load_stgcn_model(path, device):
    info = json.loads((Path(path).parent / "model_info.json").read_text(encoding="utf-8"))
    m = load_stgcn(path).to(device).eval()
    return m, info.get("window_size", 35), info


def load_lstm_model(path, device):
    info = json.loads((Path(path).parent / "model_info.json").read_text(encoding="utf-8"))
    m = RiskRegressor(input_dim=info.get("input_dim", 29))
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m.to(device).eval()
    return m, info.get("window_size", 35), info


def main():
    ap = argparse.ArgumentParser(description="ST-GCN vs LSTM 并排对比 demo")
    ap.add_argument("video")
    ap.add_argument("--stgcn-model", default="model/stgcn/stgcn_v1/run_03/best_model.pt",
                    help="ST-GCN 权重 (默认 run_03, 窗口60)")
    ap.add_argument("--lstm-model", default="model/lstm/risk_v1/run_04/best_model.pt",
                    help="LSTM 权重 (默认 run_04, 窗口35, 平滑训练)")
    ap.add_argument("--cpu", action="store_true", help="强制 CPU")
    ap.add_argument("--resize-width", type=int, default=720, help="每个面板的宽度")
    ap.add_argument("--no-save", action="store_true", help="只看不存")
    ap.add_argument("--out-dir", default=None, help="保存目录 (默认 demo/runs/compare_...)")
    ap.add_argument("--smooth-alpha", type=float, default=0.25,
                    help="风险 EMA 平滑：新样本权重(0~1)，越小越平滑；0=关闭")
    ap.add_argument("--warn-thresh", type=float, default=DEF_WARN,
                    help="WARNING 进入阈值 (默认0.45，防橙色横跳)")
    ap.add_argument("--danger-thresh", type=float, default=DEF_DANGER, help="DANGER 进入阈值")
    ap.add_argument("--hysteresis", type=float, default=0.08, help="迟滞带宽")
    ap.add_argument("--enter-hold", type=int, default=3, help="升级需连续 N 帧")
    ap.add_argument("--exit-hold", type=int, default=6, help="降级需连续 N 帧")
    # 坐标层 OneEuro 时序平滑（同时作用于 ST-GCN 与 LSTM 两路，与各自训练分布一致；默认按模型自动）
    ap.add_argument("--smooth-landmarks", dest="smooth_landmarks", action="store_true", default=None,
                    help="坐标层 OneEuro 滤波（默认：按模型训练是否启用自动决定，覆盖两路）")
    ap.add_argument("--no-smooth-landmarks", dest="smooth_landmarks", action="store_false",
                    help="关闭坐标滤波（两路同时关闭）")
    ap.add_argument("--min-cutoff", type=float, default=1.0, help="OneEuro min_cutoff")
    ap.add_argument("--beta", type=float, default=0.02, help="OneEuro beta")
    ap.add_argument("--dcutoff", type=float, default=1.0, help="OneEuro dcutoff")
    # 帧级姿态置信度门控（方案A）：平均关节置信度低于阈值 → 该帧保持上一有效风险，
    # 不更新滑窗/EMA/状态机，避免遮挡/运动模糊导致的低置信度噪声帧污染滑窗误报。
    ap.add_argument("--min-confidence", type=float, default=0.5,
                    help="帧级姿态置信度门控阈值：PoseResult.mean_confidence 低于此值视为低置信度，"
                         "该帧不更新模型/滑窗/报警（risk 保持上一有效值），恢复后自然接续")
    args = ap.parse_args()

    use_gpu = (not args.cpu) and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    stgcn_model, stgcn_win, stgcn_info = load_stgcn_model(args.stgcn_model, device)
    lstm_model, lstm_win, lstm_info = load_lstm_model(args.lstm_model, device)
    fb = FrameFeatureBuilder()
    gait = GaitSpeedExtractor()

    # 坐标层 OneEuro 时序平滑 —— 两路都做，与各自训练分布一致（保证零漂移）
    # ST-GCN 分支：从 stgcn_info 读；LSTM 分支：run_04 用平滑数据训练，
    #   但 train_risk.py 未把平滑配置写入 model_info，故 LSTM 未在 model_info 记录时
    #   默认开启（匹配最新 run_04 训练分布）。
    ls_info = stgcn_info.get("landmark_smoothing", {})
    ll_info = lstm_info.get("landmark_smoothing", {})
    cli_specified = args.smooth_landmarks is not None
    if args.smooth_landmarks is None:
        s_on = bool(ls_info.get("enabled", False))
        l_on = bool(ll_info.get("enabled", True))   # LSTM fallback: 默认开启
    else:
        s_on = l_on = args.smooth_landmarks

    # ST-GCN 用 17 关节 3D；LSTM 用 33 关节 2D+3D 双通道（与 build_from_cache 训练侧完全一致）
    if ls_info.get("enabled"):
        args.min_cutoff = ls_info.get("min_cutoff", args.min_cutoff)
        args.beta = ls_info.get("beta", args.beta)
        args.dcutoff = ls_info.get("dcutoff", args.dcutoff)
    if ll_info.get("enabled"):
        args.min_cutoff = ll_info.get("min_cutoff", args.min_cutoff)
        args.beta = ll_info.get("beta", args.beta)
        args.dcutoff = ll_info.get("dcutoff", args.dcutoff)
    s_fps = ls_info.get("fps", 25.0)
    l_fps = ll_info.get("fps", 25.0)

    smoother_s = LandmarkSmoother(
        enabled=bool(s_on),
        min_cutoff=args.min_cutoff, beta=args.beta, dcutoff=args.dcutoff,
        n_landmarks=len(FALL_JOINTS_17), channels=3, fps=s_fps)
    # LSTM 双通道：2D(33x2) + 3D(33x3)，与 build_from_cache 训练侧完全一致
    smoother_l2d = LandmarkSmoother(
        enabled=bool(l_on),
        min_cutoff=args.min_cutoff, beta=args.beta, dcutoff=args.dcutoff,
        n_landmarks=33, channels=2, fps=l_fps)
    smoother_l3d = LandmarkSmoother(
        enabled=bool(l_on),
        min_cutoff=args.min_cutoff, beta=args.beta, dcutoff=args.dcutoff,
        n_landmarks=33, channels=3, fps=l_fps)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERR] 无法打开视频: {args.video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_w = args.resize_width
    target_h = int(H0 * target_w / W0) if W0 else target_w

    ext = PoseExtractor(PoseExtractorConfig(
        model_path=str(TASK_PATH),
        use_gpu=use_gpu,
        min_detection_confidence=0.3, min_tracking_confidence=0.3,
        min_pose_presence_confidence=0.3))
    ext.__enter__()

    sbuf = deque(maxlen=stgcn_win)
    lbuf = deque(maxlen=lstm_win)
    fsm_s = AlarmDebouncer(args.warn_thresh, args.danger_thresh, args.hysteresis,
                           args.enter_hold, args.exit_hold)
    fsm_l = AlarmDebouncer(args.warn_thresh, args.danger_thresh, args.hysteresis,
                           args.enter_hold, args.exit_hold)

    s_risks, l_risks = [], []
    rows = []   # 逐帧明细（用于 frames.csv）
    n_low_conf = n_no_person = 0
    ema_s = ema_l = 0.0
    raw_s = raw_l = 0.0
    last_pr = None
    t = 0
    paused = False
    play_delay_ms = max(1, int(round(1000.0 / fps))) if fps > 0 else 40

    save_dir = None
    writer = None
    if not args.no_save:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        vname = Path(args.video).stem
        save_dir = Path(args.out_dir) if args.out_dir else (
            PROJECT_ROOT / "demo" / "runs" / f"compare_{vname}_{stamp}")
        save_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_dir / "annotated.mp4"),
                                 fourcc, fps, (target_w * 2, target_h))

    print(f"=== ST-GCN vs LSTM 对比 demo ===\n  视频 : {args.video}\n"
          f"  ST-GCN : {args.stgcn_model}  窗口={stgcn_win}\n"
          f"  LSTM   : {args.lstm_model}  窗口={lstm_win}\n"
          f"  面板 {target_w}x{target_h} @ {fps:.1f}fps  gpu={use_gpu}\n"
          f"  防误报: WARN={args.warn_thresh} DANGER={args.danger_thresh} "
          f"hyst={args.hysteresis} enter={args.enter_hold} exit={args.exit_hold} "
          f"smooth={args.smooth_alpha}\n"
          f"  坐标滤波(OneEuro): ST-GCN={'ON' if smoother_s.enabled else 'OFF'}  "
          f"LSTM={'ON' if smoother_l3d.enabled else 'OFF'}"
          f"  min_cutoff={args.min_cutoff} beta={args.beta} dcutoff={args.dcutoff}"
          f"{'  [依模型自动]' if not cli_specified else ''}")
    print(f"  帧级置信度门控(方案A): 阈值={args.min_confidence}  "
          f"(平均关节置信度低于此值→保持风险/不更新滑窗)")
    print("  操作: q=退出  空格=暂停")

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

                ts_ms = int(t * 1000 / fps)
                pr = ext.detect_frame(small, ts_ms)
                has_person = pr is not None
                # 帧级置信度门控（方案A）：平均关节置信度低于阈值 → 视为低置信度，
                # 该帧不计为有效观测 —— 不更新滑窗/EMA/状态机，risk 保持上一有效值，
                # 仅画面提示；恢复后自然接续，避免遮挡/运动模糊噪声污染滑窗导致误报。
                low_conf = bool(has_person and pr.mean_confidence < args.min_confidence)
                valid = has_person and not low_conf

                s_risk = l_risk = 0.0
                if valid:
                    last_pr = pr
                    # ---- ST-GCN 分支 ----
                    skel = skeleton_from_poseresult(pr, FALL_JOINTS_17)  # (17,3)
                    skel = smoother_s.update(skel)                       # 坐标层 OneEuro 平滑
                    sbuf.append(skel)
                    if len(sbuf) >= stgcn_win:
                        x = (torch.from_numpy(np.stack(sbuf)).float()
                             .permute(2, 0, 1).unsqueeze(0).to(device))
                        with torch.no_grad():
                            s_risk = float(stgcn_model(x).item())
                    # ---- LSTM 分支（先对 2D+3D 双通道做 OneEuro 平滑，匹配 run_04 训练分布）----
                    if l_on and pr.world_landmarks is not None:
                        lm2d = np.array([[kp.x, kp.y] for kp in pr.landmarks], dtype=np.float32)
                        wl3d = np.array([[kp.x, kp.y, kp.z] for kp in pr.world_landmarks], dtype=np.float32)
                        lm2d = smoother_l2d.update(lm2d)
                        wl3d = smoother_l3d.update(wl3d)
                        for i, kp in enumerate(pr.landmarks):
                            kp.x, kp.y = float(lm2d[i, 0]), float(lm2d[i, 1])
                        for i, kp in enumerate(pr.world_landmarks):
                            kp.x, kp.y, kp.z = float(wl3d[i, 0]), float(wl3d[i, 1]), float(wl3d[i, 2])
                    gs = gait.compute_per_frame(pr)
                    vec = fb.build(pr, gs)  # (29,)
                    lbuf.append(np.asarray(vec, dtype=np.float32))
                    if len(lbuf) >= lstm_win:
                        x = torch.from_numpy(np.stack(lbuf)).float().unsqueeze(0).to(device)
                        with torch.no_grad():
                            l_risk = float(lstm_model(x).item())

                # ---- 帧级置信度门控状态分发 ----
                if valid:
                    raw_s, raw_l = s_risk, l_risk
                    ema_s = (1 - args.smooth_alpha) * ema_s + args.smooth_alpha * s_risk if args.smooth_alpha > 0 else s_risk
                    ema_l = (1 - args.smooth_alpha) * ema_l + args.smooth_alpha * l_risk if args.smooth_alpha > 0 else l_risk
                    lvl_s = fsm_s.update(ema_s)
                    lvl_l = fsm_l.update(ema_l)
                elif has_person:  # 低置信度：保持上一有效 risk / 报警等级，不更新窗口/EMA/状态机
                    raw_s, raw_l = 0.0, 0.0
                    lvl_s = fsm_s.LEVELS[fsm_s.level]
                    lvl_l = fsm_l.LEVELS[fsm_l.level]
                else:  # 真无人：原行为（ema 衰减、fsm 降级、risk→0）
                    raw_s, raw_l = 0.0, 0.0
                    ema_s = (1 - args.smooth_alpha) * ema_s if args.smooth_alpha > 0 else 0.0
                    ema_l = (1 - args.smooth_alpha) * ema_l if args.smooth_alpha > 0 else 0.0
                    lvl_s = fsm_s.update(ema_s)
                    lvl_l = fsm_l.update(ema_l)
                s_risks.append(float(ema_s))
                l_risks.append(float(ema_l))
                if low_conf:
                    n_low_conf += 1
                elif not has_person:
                    n_no_person += 1

                mean_conf = float(pr.mean_confidence) if has_person else 0.0
                rows.append({
                    "frame": t,
                    "ts_ms": ts_ms,
                    "has_person": int(has_person),
                    "mean_conf": round(mean_conf, 4),
                    "low_conf": int(low_conf),
                    "stgcn_raw": (round(float(s_risk), 4) if valid else ""),
                    "stgcn_risk": round(float(ema_s), 4),
                    "stgcn_alarm": lvl_s,
                    "lstm_raw": (round(float(l_risk), 4) if valid else ""),
                    "lstm_risk": round(float(ema_l), 4),
                    "lstm_alarm": lvl_l,
                })

                warm_s = valid and len(sbuf) < stgcn_win
                warm_l = valid and len(lbuf) < lstm_win

                pane_s = draw_pane(small, pr if has_person else None, ema_s, lvl_s,
                                   "ST-GCN", FALL_JOINTS_17, FALL_EDGES,
                                   args.warn_thresh, args.danger_thresh, stgcn_win, warm_s,
                                   low_conf=low_conf, min_conf=args.min_confidence)
                pane_l = draw_pane(small, pr if has_person else None, ema_l, lvl_l,
                                   "LSTM", CORE_JOINTS_13, None,
                                   args.warn_thresh, args.danger_thresh, lstm_win, warm_l,
                                   low_conf=low_conf, min_conf=args.min_confidence)
                combined = np.hstack([pane_s, pane_l])

                cv2.imshow("ST-GCN (left) vs LSTM (right)  [q=quit, space=pause]", combined)
                if writer:
                    writer.write(combined)
                t += 1

            key = cv2.waitKey(0 if paused else play_delay_ms) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
    finally:
        cap.release()
        cv2.destroyAllWindows()          # 先关窗口，避免末尾画面"假死"卡顿
        if writer:
            writer.release()             # 写出标注视频（长视频收尾略慢，正常）
        ext.__exit__(None, None, None)   # 释放 MediaPipe（CPU 回退时稍慢，正常）
        print("收尾完成，已退出。")

    if save_dir and s_risks:
        make_curve(s_risks, l_risks, save_dir / "risk_curve.png",
                   args.warn_thresh, args.danger_thresh)
        (save_dir / "run_meta.json").write_text(json.dumps({
            "stgcn_model": str(Path(args.stgcn_model)),
            "lstm_model": str(Path(args.lstm_model)),
            "stgcn_window": stgcn_win,
            "lstm_window": lstm_win,
            "video": str(args.video),
            "fps": fps,
            "n_frames": len(s_risks),
            "warn_thresh": args.warn_thresh,
            "danger_thresh": args.danger_thresh,
            "smooth_alpha": args.smooth_alpha,
            "min_confidence": args.min_confidence,
            "n_valid_frames": len(s_risks) - n_low_conf - n_no_person,
            "n_low_conf_frames": n_low_conf,
            "n_no_person_frames": n_no_person,
            "stgcn": {
                "mean_risk": float(np.mean(s_risks)),
                "max_risk": float(np.max(s_risks)),
                "n_danger": int(sum(1 for r in s_risks if r >= args.danger_thresh)),
                "n_warning": int(sum(1 for r in s_risks if args.warn_thresh <= r < args.danger_thresh)),
            },
            "lstm": {
                "mean_risk": float(np.mean(l_risks)),
                "max_risk": float(np.max(l_risks)),
                "n_danger": int(sum(1 for r in l_risks if r >= args.danger_thresh)),
                "n_warning": int(sum(1 for r in l_risks if args.warn_thresh <= r < args.danger_thresh)),
            },
        }, indent=2), encoding="utf-8")

        # 逐帧明细 CSV
        csv_path = save_dir / "frames.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "ts_ms", "has_person", "mean_conf", "low_conf",
                        "stgcn_raw", "stgcn_risk", "stgcn_alarm",
                        "lstm_raw", "lstm_risk", "lstm_alarm"])
            for r in rows:
                w.writerow([r["frame"], r["ts_ms"], r["has_person"], r["mean_conf"],
                            r["low_conf"], r["stgcn_raw"], r["stgcn_risk"], r["stgcn_alarm"],
                            r["lstm_raw"], r["lstm_risk"], r["lstm_alarm"]])

        print(f"  完成 -> {save_dir}\n"
              f"    产物: annotated.mp4 / risk_curve.png / frames.csv / run_meta.json\n"
              f"    ST-GCN: max_risk={np.max(s_risks):.3f}  DANGER帧={sum(1 for r in s_risks if r >= args.danger_thresh)}  WARNING帧={sum(1 for r in s_risks if args.warn_thresh <= r < args.danger_thresh)}\n"
              f"    LSTM  : max_risk={np.max(l_risks):.3f}  DANGER帧={sum(1 for r in l_risks if r >= args.danger_thresh)}  WARNING帧={sum(1 for r in l_risks if args.warn_thresh <= r < args.danger_thresh)}")


if __name__ == "__main__":
    main()
