"""上传视频的可选 MediaPipe 姿态摘要适配层。"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class ModelUnavailableError(RuntimeError):
    """当前环境没有可用姿态模型。"""


class MediaPipeVideoAnalyzer:
    """仅在模型和依赖可用时延迟加载现有姿态后端。"""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        capture_factory: Any | None = None,
        extractor_factory: Any | None = None,
        max_source_frames: int = 3000,
        frame_stride: int = 2,
    ) -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.model_path = Path(model_path) if model_path else (
            project_root / "model" / "mediapipe_pose" / "pose_landmarker_lite.task"
        )
        if max_source_frames < 1 or frame_stride < 1:
            raise ValueError("原始帧上限和抽帧间隔必须是正整数")
        self.capture_factory = capture_factory
        self.extractor_factory = extractor_factory
        self.max_source_frames = int(max_source_frames)
        self.frame_stride = int(frame_stride)

    def analyze(self, video_path: Path) -> dict[str, Any]:
        if not self.model_path.is_file():
            raise ModelUnavailableError("当前未安装姿态模型")

        try:
            import numpy as np
            from longitudinal.pose_features import summarize_pose_sequence
        except Exception as exc:
            raise ModelUnavailableError("当前环境缺少姿态分析依赖") from exc

        if self.capture_factory is None:
            try:
                import cv2
            except Exception as exc:
                raise ModelUnavailableError("当前环境缺少视频读取依赖") from exc
            capture = cv2.VideoCapture(str(video_path))
        else:
            capture = self.capture_factory(str(video_path))

        if self.extractor_factory is None:
            try:
                from pose.extractor import PoseExtractor, PoseExtractorConfig
            except Exception as exc:
                capture.release()
                raise ModelUnavailableError("当前环境缺少姿态分析依赖") from exc
            config: Any = PoseExtractorConfig(
                model_path=str(self.model_path),
                output_world_landmarks=False,
                frame_skip=1,
                use_gpu=False,
            )
            extractor = PoseExtractor(config)
        else:
            config = {
                "model_path": str(self.model_path),
                "output_world_landmarks": False,
                "frame_skip": 1,
                "use_gpu": False,
            }
            extractor = self.extractor_factory(config)

        landmarks: list[Any] = []
        visibility: list[Any] = []
        timestamps: list[float] = []
        try:
            if not capture.isOpened():
                raise RuntimeError("无法读取上传的视频文件")
            fps = float(capture.get(5))
            if not np.isfinite(fps) or fps <= 0:
                fps = 30.0
            source_index = 0
            with extractor:
                while source_index < self.max_source_frames:
                    readable, frame = capture.read()
                    if not readable:
                        break
                    current_index = source_index
                    source_index += 1
                    if current_index % self.frame_stride != 0:
                        continue
                    timestamp_ms = int(current_index / fps * 1000)
                    result = extractor.detect_frame(frame, timestamp_ms)
                    if result is None:
                        continue
                    frame_points = np.full((33, 3), np.nan, dtype=np.float64)
                    frame_visibility = np.zeros(33, dtype=np.float64)
                    for keypoint in result.landmarks:
                        index = int(keypoint.name.value)
                        if not 0 <= index < 33:
                            continue
                        frame_points[index] = (keypoint.x, keypoint.y, keypoint.z)
                        frame_visibility[index] = float(keypoint.visibility)
                    landmarks.append(frame_points)
                    visibility.append(frame_visibility)
                    timestamps.append(timestamp_ms / 1000.0)
        except Exception as exc:
            raise RuntimeError("视频姿态分析未完成") from exc
        finally:
            capture.release()

        if len(landmarks) < 3:
            raise RuntimeError("视频中未获得足够的有效人体姿态帧")
        summary = summarize_pose_sequence(
            np.asarray(landmarks),
            np.asarray(timestamps),
            np.asarray(visibility),
        )
        return {"mode": "real_observation", **summary.to_dict()}
