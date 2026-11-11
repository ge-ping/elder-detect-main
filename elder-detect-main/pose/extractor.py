"""
MediaPipe Pose 姿态提取器核心类

封装 MediaPipe Tasks API（0.10.x），提供：
- 图片/视频帧 → 33个人体关键点（x,y ∈ [0,1] MediaPipe 原生归一化, z=相对深度, visibility）
- 支持关键点筛选（论文4的3点、论文5的13点）
- 世界坐标系3D关键点输出（论文1的3D重心计算）
- 生成器模式逐帧处理，节省内存

对齐论文5（注意力GRU三分类）的 MediaPipe 使用方式。
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Optional
from dataclasses import dataclass, field

# MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .keypoint import (
    PoseResult, Keypoint, BodyPart,
    ALL_JOINTS_33, JOINT_SET_MAP,
)


@dataclass
class PoseExtractorConfig:
    """PoseExtractor 运行时配置

    Attributes:
        model_path: .task 模型文件的绝对或相对路径
        min_detection_confidence: 人体检测置信度阈值
        min_tracking_confidence: 关键点跟踪置信度阈值
        min_pose_presence_confidence: 姿态存在置信度阈值（论文1：低于60%丢弃帧）
        output_world_landmarks: 是否输出3D世界坐标
        selected_joints: 输出的关键点列表，None=全部33点
        frame_skip: 跳帧间隔，1=不跳帧，>1=每N帧处理1帧（加速）
    """
    model_path: str
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.6
    output_world_landmarks: bool = True
    selected_joints: Optional[list[BodyPart]] = None
    frame_skip: int = 1
    use_gpu: bool = True  # True=GPU delegate, False=CPU+XNNPACK

    def __post_init__(self):
        if self.frame_skip < 1:
            raise ValueError(f"frame_skip must be >= 1, got {self.frame_skip}")


class PoseExtractor:
    """MediaPipe Pose 姿态提取器

    封装 MediaPipe Tasks API，将视频帧转为结构化的 PoseResult。

    使用方式:
        >>> config = PoseExtractorConfig(model_path="model/mediapipe_pose/pose_landmarker_lite.task")
        >>> with PoseExtractor(config) as extractor:
        ...     for result in extractor.process_video("video.mp4"):
        ...         print(result.array.shape)  # (33, 4) 或 (13, 4)

    论文参考:
        - 论文4: selected_joints=MINIMAL_JOINTS_3, 仅3点轻量化
        - 论文5: selected_joints=CORE_JOINTS_13, 13核心关节三分类
        - 论文1: output_world_landmarks=True, 3D重心计算
    """

    def __init__(self, config: PoseExtractorConfig):
        self.config = config
        self._validate_model()
        self._detector: Optional[vision.PoseLandmarker] = None
        self._frame_counter: int = 0

    # ========== 模型验证 ==========

    def _validate_model(self) -> None:
        """验证模型文件存在且格式正确"""
        path = Path(self.config.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                f"Please download the model first. See model/mediapipe_pose/download_models.py"
            )
        if path.suffix != ".task":
            raise ValueError(f"Expected .task model file, got: {path.suffix}")
        if not path.is_file():
            raise ValueError(f"Model path is not a file: {path}")

    # ========== 延迟初始化 ==========

    def _ensure_initialized(self) -> None:
        """延迟初始化 MediaPipe detector，GPU 失败自动回落 CPU"""
        if self._detector is not None:
            return

        mp_options = vision.PoseLandmarkerOptions(
            base_options=self._build_base_options(),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            min_pose_presence_confidence=self.config.min_pose_presence_confidence,
            output_segmentation_masks=False,
        )
        try:
            self._detector = vision.PoseLandmarker.create_from_options(mp_options)
        except Exception:
            if self.config.use_gpu:
                print("  [WARN] GPU create_from_options failed, retrying with CPU", flush=True)
                mp_options.base_options = mp_python.BaseOptions(
                    model_asset_path=str(self.config.model_path),
                    delegate=mp_python.BaseOptions.Delegate.CPU,
                )
                self._detector = vision.PoseLandmarker.create_from_options(mp_options)
            else:
                raise

    def _build_base_options(self):
        delegate = (
            mp_python.BaseOptions.Delegate.GPU
            if self.config.use_gpu else
            mp_python.BaseOptions.Delegate.CPU
        )
        return mp_python.BaseOptions(
            model_asset_path=str(self.config.model_path),
            delegate=delegate,
        )

    # ========== 核心 API ==========

    def detect_frame(self, frame: np.ndarray, timestamp_ms: int) -> Optional[PoseResult]:
        """处理单帧图像（用于实时摄像头流）

        Args:
            frame: BGR numpy 数组 (H, W, 3)，来自 cv2.VideoCapture
            timestamp_ms: 帧时间戳（毫秒），必须单调递增
                          摄像头流用 cap.get(cv2.CAP_PROP_POS_MSEC)
                          视频文件用 frame_index / fps * 1000

        Returns:
            PoseResult（landmarks.x/y ∈ [0,1] 归一化坐标）
            或 None（未检测到人体 / 置信度不足时）

        Raises:
            ValueError: 输入帧格式不正确
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None")
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected BGR 3-channel image, got shape {frame.shape}")

        self._ensure_initialized()
        assert self._detector is not None

        h, w = frame.shape[:2]

        # BGR → RGB（MediaPipe 要求）
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # 执行推理
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return None

        return self._build_result(result, w, h, timestamp_ms)

    def process_video(
        self,
        video_path: str,
        resize: Optional[tuple[int, int]] = None,
        max_frames: Optional[int] = None,
    ) -> Iterator[PoseResult]:
        """处理视频文件，生成器模式逐帧返回结果

        Args:
            video_path: 视频文件路径
            resize: 可选，统一 resize 到 (width, height)，加速推理
            max_frames: 可选，最多处理多少帧（调试用）

        Yields:
            每一帧若检测到人体则 yield PoseResult（无检测则跳过该帧）

        Example:
            >>> extractor = PoseExtractor(config)
            >>> for result in extractor.process_video("walk.mp4"):
            ...     arr = result.array  # (N, 4) numpy
            ...     # 送入下游 LSTM
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # 默认假设 30fps

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        processed = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames and processed >= max_frames:
                    break

                # 跳帧
                if frame_idx % self.config.frame_skip != 0:
                    frame_idx += 1
                    continue

                if resize:
                    frame = cv2.resize(frame, resize)

                timestamp_ms = int(frame_idx / fps * 1000)
                result = self.detect_frame(frame, timestamp_ms)
                if result is not None:
                    yield result
                    processed += 1

                frame_idx += 1
        finally:
            cap.release()

    def process_image(self, image_path: str) -> Optional[PoseResult]:
        """处理单张图片

        Args:
            image_path: 图片文件路径

        Returns:
            PoseResult 或 None
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
        return self.detect_frame(frame, timestamp_ms=0)

    def process_batch(
        self,
        frames: list[np.ndarray],
        timestamps_ms: Optional[list[int]] = None,
    ) -> list[Optional[PoseResult]]:
        """批量处理多帧（不做 batch inference，逐帧调用。MediaPipe VIDEO 模式不支持 batch）

        Args:
            frames: BGR 帧列表
            timestamps_ms: 对应时间戳列表，None 则自动生成

        Returns:
            与 frames 等长的结果列表，无检测的位置为 None
        """
        if timestamps_ms is None:
            timestamps_ms = [i * 33 for i in range(len(frames))]  # 默认 30fps

        if len(frames) != len(timestamps_ms):
            raise ValueError(
                f"frames and timestamps_ms must have same length, "
                f"got {len(frames)} vs {len(timestamps_ms)}"
            )

        results = []
        for frame, ts in zip(frames, timestamps_ms):
            results.append(self.detect_frame(frame, ts))
        return results

    def process_webcam(
        self,
        camera_id: int = 0,
        display: bool = False,
        exit_key: str = "q",
    ) -> Iterator[PoseResult]:
        """实时摄像头处理（生成器）

        Args:
            camera_id: 摄像头设备 ID
            display: 是否弹出窗口显示（调试用）
            exit_key: display=True 时的退出按键

        Yields:
            每帧的 PoseResult

        Example:
            >>> extractor = PoseExtractor(config)
            >>> for result in extractor.process_webcam(display=True):
            ...     # result.array 送入模型
            ...     pass  # Ctrl+C 或按 q 退出
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {camera_id}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                result = self.detect_frame(frame, timestamp_ms)

                if display:
                    if result:
                        self._draw_landmarks_on_frame(frame, result)
                    cv2.imshow("PoseExtractor", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(exit_key):
                        break

                yield result  # None 也 yield，让调用方自行处理
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

    # ========== 内部方法 ==========

    def _build_result(
        self,
        mp_result: vision.PoseLandmarkerResult,
        width: int,
        height: int,
        timestamp_ms: int,
    ) -> PoseResult:
        """将 MediaPipe 原始输出转为我们的 PoseResult"""
        selected = self.config.selected_joints or ALL_JOINTS_33
        selected_set = set(selected)

        # 2D 关键点 — 保留 MediaPipe 原生 [0, 1] 归一化坐标
        landmarks = []
        if mp_result.pose_landmarks:
            for i, lm in enumerate(mp_result.pose_landmarks[0]):
                part = BodyPart(i)
                if part not in selected_set:
                    continue
                landmarks.append(Keypoint(
                    x=lm.x,   # [0, 1] 归一化，不需要 * width
                    y=lm.y,   # [0, 1] 归一化，不需要 * height
                    z=lm.z,   # 相对深度（相对于髋部中心）
                    visibility=float(lm.visibility or 0.0),
                    name=part,
                ))

        # 世界坐标关键点（3D，单位：米）
        world_landmarks = None
        if self.config.output_world_landmarks and mp_result.pose_world_landmarks:
            world_landmarks = []
            for i, lm in enumerate(mp_result.pose_world_landmarks[0]):
                part = BodyPart(i)
                if part not in selected_set:
                    continue
                world_landmarks.append(Keypoint(
                    x=lm.x, y=lm.y, z=lm.z,
                    visibility=float(lm.visibility or 0.0),
                    name=part,
                ))

        return PoseResult(
            landmarks=landmarks,
            world_landmarks=world_landmarks,
            timestamp_ms=timestamp_ms,
            frame_width=width,
            frame_height=height,
        )

    @staticmethod
    def _draw_landmarks_on_frame(
        frame: np.ndarray,
        result: PoseResult,
        color: tuple[int, int, int] = (0, 255, 0),
        radius: int = 4,
    ) -> None:
        """在帧上绘制关键点（调试可视化）"""
        h, w = frame.shape[:2]
        for kp in result.landmarks:
            if kp.visibility >= 0.5:
                cx, cy = int(kp.x * w), int(kp.y * h)  # [0,1] → 像素
                cv2.circle(frame, (cx, cy), radius, color, -1)
                cv2.putText(frame, str(kp.name.value), (cx + 5, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # 绘制骨架连线（简化版：躯干+四肢）
        _BONE_CONNECTIONS = [
            (BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER),
            (BodyPart.LEFT_SHOULDER, BodyPart.LEFT_ELBOW),
            (BodyPart.LEFT_ELBOW, BodyPart.LEFT_WRIST),
            (BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW),
            (BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST),
            (BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP),
            (BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_HIP),
            (BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP),
            (BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE),
            (BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE),
            (BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE),
            (BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE),
        ]
        try:
            for a, b in _BONE_CONNECTIONS:
                ka = result.get(a)
                kb = result.get(b)
                if ka.visibility >= 0.5 and kb.visibility >= 0.5:
                    cv2.line(frame, (int(ka.x * w), int(ka.y * h)),
                             (int(kb.x * w), int(kb.y * h)),
                             (255, 0, 0), 2)
        except KeyError:
            pass  # 筛选后的关键点集可能不包含所有骨骼端点

    # ========== 资源管理 ==========

    def __enter__(self) -> "PoseExtractor":
        self._ensure_initialized()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """释放 MediaPipe 资源"""
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def __del__(self) -> None:
        if hasattr(self, "_detector") and self._detector is not None:
            self._detector.close()
            self._detector = None
