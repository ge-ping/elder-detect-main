"""上传视频的可选 MediaPipe 姿态摘要适配层。"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class ModelUnavailableError(RuntimeError):
    """当前环境没有可用姿态模型。"""


class MediaPipeVideoAnalyzer:
    """仅在模型和依赖可用时延迟加载现有姿态后端。"""

    def __init__(self, model_path: str | Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.model_path = Path(model_path) if model_path else (
            project_root / "model" / "mediapipe_pose" / "pose_landmarker_lite.task"
        )

    def analyze(self, video_path: Path) -> dict[str, Any]:
        if not self.model_path.is_file():
            raise ModelUnavailableError("当前未安装姿态模型")

        try:
            import numpy as np
            from pose.extractor import PoseExtractor, PoseExtractorConfig
            from longitudinal.pose_features import summarize_pose_sequence
        except Exception as exc:
            raise ModelUnavailableError("当前环境缺少姿态分析依赖") from exc

        landmarks: list[Any] = []
        visibility: list[Any] = []
        timestamps: list[float] = []
        config = PoseExtractorConfig(
            model_path=str(self.model_path),
            output_world_landmarks=False,
            frame_skip=2,
            use_gpu=False,
        )
        try:
            with PoseExtractor(config) as extractor:
                for result in extractor.process_video(str(video_path), max_frames=1500):
                    frame_points = np.full((33, 3), np.nan, dtype=np.float64)
                    frame_visibility = np.zeros(33, dtype=np.float64)
                    for keypoint in result.landmarks:
                        index = int(keypoint.name.value)
                        frame_points[index] = (keypoint.x, keypoint.y, keypoint.z)
                        frame_visibility[index] = float(keypoint.visibility)
                    landmarks.append(frame_points)
                    visibility.append(frame_visibility)
                    timestamps.append(float(result.timestamp_ms) / 1000.0)
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise RuntimeError("视频姿态分析未完成") from exc

        if len(landmarks) < 3:
            raise RuntimeError("视频中未获得足够的有效人体姿态帧")
        summary = summarize_pose_sequence(
            np.asarray(landmarks),
            np.asarray(timestamps),
            np.asarray(visibility),
        )
        return {"mode": "real_observation", **summary.to_dict()}
