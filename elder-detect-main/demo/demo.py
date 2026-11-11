#!/usr/bin/env python
"""
demo.py — 实时跌倒风险预测

用法:
    python demo/demo.py --model model/lstm/v5_w25/run_01/best_model.pt
    python demo/demo.py --model ... --video test.mp4
    python demo/demo.py --model ... --input_dim 29 --window_size 35  # 旧模型
"""
from __future__ import annotations

import argparse
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

from pose import PoseExtractor, load_pose_config
from features.feature_builder import FrameFeatureBuilder
from features.gait import GaitSpeedExtractor
from model.lstm.model import FallLSTM
from model.lstm.risk_model import RiskRegressor


# ==================================================================
# DemoRunner
# ==================================================================

class DemoRunner:
    def __init__(
        self,
        model_path: str,
        window_size: int = 25,
        input_dim: int = 29,
        pose_config: str = "config/pose_config.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        task: str = "classification",
    ):
        self.device = device
        self.window_size = window_size
        self.task = task

        # 加载模型：按 task 分流，原分类分支完全不动
        if task == "regression":
            self.model = RiskRegressor(input_dim=input_dim).to(device)
        else:
            self.model = FallLSTM(input_dim=input_dim).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.model.eval()

        # 特征管线
        pose_cfg = load_pose_config(pose_config)
        self._pose_cfg = pose_cfg
        self._extractor: PoseExtractor | None = None
        self._feat_builder = FrameFeatureBuilder()
        self._gait = GaitSpeedExtractor()

        # 窗口缓冲
        self._buffer: deque[np.ndarray] = deque(maxlen=window_size)
        self._poses: deque = deque(maxlen=window_size)  # 存储 PoseResult 用于绘制

    # ==================================================================
    # 单帧处理
    # ==================================================================

    def process_frame(self, frame: np.ndarray) -> dict:
        """处理一帧 → 返回推理结果"""
        h, w = frame.shape[:2]

        # MediaPipe
        if self._extractor is None:
            return self._empty_result(h, w, "Initializing...")
        ts_ms = cv2.getTickCount() // 10  # 模拟时间戳
        result = self._extractor.detect_frame(frame, timestamp_ms=ts_ms)

        if result is None:
            return self._empty_result(h, w, "No person")

        self._poses.append(result)

        # 特征
        gs = self._gait.compute_per_frame(result)
        vec = self._feat_builder.build(result, gs)
        self._buffer.append(vec)

        # 窗口未满
        if len(self._buffer) < self.window_size:
            return {
                "status": "Buffering",
                "risk": 0.0,
                "probs": np.zeros(3, dtype=np.float32),
                "landmarks": [(int(kp.x * w), int(kp.y * h))
                              for kp in result.landmarks[:13]],
                "buffer_pct": len(self._buffer) / self.window_size,
                "has_person": True,
            }

        # LSTM 推理
        window = np.stack(list(self._buffer), axis=0)  # (W, 29)
        x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)  # (1, W, 29)
        with torch.no_grad():
            out = self.model(x)
            if self.task == "regression":
                # 回归模型直出 risk ∈ [0,1]，删除概率合成公式
                risk = float(out.item())
                probs = np.zeros(3, dtype=np.float32)  # 回归无三分类概率
            else:
                logits = out
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # (3,)
                risk = float(probs[0] * 0.1 + probs[1] * 0.6 + probs[2] * 0.9)

        status = "DANGER" if risk >= 0.6 else ("WARNING" if risk >= 0.3 else "SAFE")

        return {
            "status": status,
            "risk": risk,
            "probs": probs,
            "landmarks": [(int(kp.x * w), int(kp.y * h))
                          for kp in result.landmarks[:13]],
            "buffer_pct": 1.0,
            "has_person": True,
        }

    # ==================================================================
    # 摄像头
    # ==================================================================

    def run_camera(self, camera_id: int = 0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Cannot open camera"); return

        self._extractor = PoseExtractor(self._pose_cfg)
        cv2.namedWindow("Fall Risk Demo", cv2.WINDOW_NORMAL)  # 可变窗口：可拖动缩放
        print(f"Demo running (camera {camera_id}, window={self.window_size}). Press Q to quit.")

        tick = cv2.getTickCount()
        frame_count = 0
        last_frame = None

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            last_frame = frame

            info = self.process_frame(frame)
            self._draw_overlay(frame, info)

            # FPS
            frame_count += 1
            if frame_count % 30 == 0:
                new_tick = cv2.getTickCount()
                fps = 30.0 / ((new_tick - tick) / cv2.getTickFrequency())
                self._fps = fps
                tick = new_tick

            cv2.imshow("Fall Risk Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    # ==================================================================
    # 视频文件
    # ==================================================================

    def run_video(self, video_path: str, save: bool = False, process_every_n: int = 1):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open: {video_path}"); return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        delay = max(1, int(1000 / fps))

        writer = None
        out_path = None
        if save:
            save_dir = PROJECT_ROOT / "test" / "demo"
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = str(save_dir / f"{Path(video_path).stem}.demo.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        self._extractor = PoseExtractor(self._pose_cfg)
        cv2.namedWindow("Fall Risk Demo", cv2.WINDOW_NORMAL)  # 可变窗口：可拖动缩放
        print(f"Demo running (video={video_path}, fps={fps:.1f}, window={self.window_size}).")
        print(f"  process_every_n={process_every_n}，每 {process_every_n} 帧做一次推理；其余帧复用上一次结果")
        print("  Space = pause, Q = quit")

        paused = False
        last_frame = None
        last_info = self._empty_result(h, w, "Buffering")
        frame_count = 0
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret: break
                last_frame = frame
                frame_count += 1

                proc_start = time.perf_counter()
                if frame_count % process_every_n == 0:
                    info = self.process_frame(frame)
                    last_info = info
                else:
                    info = last_info
                self._draw_overlay(frame, info)
                if writer:
                    writer.write(frame)
                elapsed_ms = int((time.perf_counter() - proc_start) * 1000)
                display = frame
            else:
                display = last_frame
                elapsed_ms = 0

            if display is None:
                display = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.imshow("Fall Risk Demo", display)

            key = cv2.waitKey(max(1, delay - elapsed_ms)) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
                print(f"  {'Paused' if paused else 'Resumed'}")

        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
        if out_path:
            print(f"Saved: {out_path}")

        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
        if out_path:
            print(f"Saved: {out_path}")

    # ==================================================================
    # 绘制
    # ==================================================================

    _fps: float = 0.0

    def _draw_overlay(self, frame: np.ndarray, info: dict):
        h, w = frame.shape[:2]
        # 底部面板
        panel_h = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_h), (w, h), (30, 30, 30), -1)
        frame[:] = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # 骨架
        for x, y in info.get("landmarks", []):
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        y0 = h - panel_h + 20
        status = info["status"]
        colors = {"SAFE": (0, 255, 0), "WARNING": (0, 215, 255), "DANGER": (0, 0, 255),
                  "Buffering": (180, 180, 180), "No person": (100, 100, 100)}
        color = colors.get(status, (255,255,255))

        cv2.putText(frame, f"Status: {status}", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if self.task == "classification":
            cv2.putText(frame, f"P(N)={info['probs'][0]:.3f}  P(U)={info['probs'][1]:.3f}  "
                        f"P(F)={info['probs'][2]:.3f}", (10, y0 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        else:
            cv2.putText(frame, f"[Risk Regression] continuous risk", (10, y0 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        # 风险条
        risk = info["risk"]
        bar_x, bar_y, bar_w, bar_h = 10, y0 + 52, 200, 14
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        fill = int(bar_w * min(risk, 1.0))
        bar_color = (0, 255, 0) if risk < 0.3 else ((0, 215, 255) if risk < 0.6 else (0, 0, 255))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
        cv2.putText(frame, f"Risk: {risk:.3f}", (bar_x, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, bar_color, 1)

        # 缓冲 / FPS
        buf_pct = info.get("buffer_pct", 0)
        cv2.putText(frame, f"Buffer: {int(buf_pct * self.window_size)}/{self.window_size}  "
                    f"FPS: {self._fps:.1f}", (bar_x, bar_y + bar_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    @staticmethod
    def _empty_result(h, w, status):
        return {"status": status, "risk": 0.0, "probs": np.zeros(3),
                "landmarks": [], "buffer_pct": 0.0, "has_person": False}


# ==================================================================
# Main
# ==================================================================

def main():
    parser = argparse.ArgumentParser(description="Fall Risk Demo")
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", default=None)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--input_dim", type=int, default=None)
    parser.add_argument("--window_size", "-w", type=int, default=None)
    parser.add_argument("--task", type=str, default=None,
                        choices=["classification", "regression"],
                        help="覆盖 model_info.json 的 task（默认自动读）")
    parser.add_argument("--process_every_n", type=int, default=1,
                        help="每 N 帧做一次推理（其余帧复用上次结果），提升播放流畅度；默认每帧都推理")
    parser.add_argument("--save", action="store_true", help="保存标注视频")
    parser.add_argument("--pose_config", default="config/pose_config.yaml")
    args = parser.parse_args()

    # 自动读 model_info.json
    info_path = Path(args.model).parent / "model_info.json"
    input_dim = args.input_dim
    window_size = args.window_size
    task = args.task if hasattr(args, "task") else "classification"

    if info_path.exists():
        info = json.loads(info_path.read_text())
        if input_dim is None:
            input_dim = info.get("input_dim", 29)
        if window_size is None:
            window_size = info.get("window_size", 25)
        if "task" in info:
            task = info["task"]
        print(f"Loaded model_info: dim={input_dim}, window={window_size}, "
              f"version={info.get('version','?')}, task={task}")

    input_dim = input_dim or 29
    window_size = window_size or 25

    runner = DemoRunner(
        model_path=args.model,
        window_size=window_size,
        input_dim=input_dim,
        pose_config=args.pose_config,
        task=task,
    )

    if args.video:
        runner.run_video(args.video, save=args.save, process_every_n=args.process_every_n)
    else:
        runner.run_camera(args.camera)


if __name__ == "__main__":
    main()
