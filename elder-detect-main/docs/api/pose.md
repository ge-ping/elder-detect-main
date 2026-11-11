# Pose API 文档

> **模块路径**: `pose/`  
> **依赖**: `mediapipe>=0.10.0`, `opencv-python>=4.8`, `numpy>=1.24`, `pyyaml>=6.0`  
> **运行环境**: 见仓库根 `requirements.txt`（项目 Python 环境，无需 conda）

---

## 概述

`pose` 模块封装了 **MediaPipe Pose Landmarker**（Tasks API），将摄像头/视频/图片帧转换为结构化的 **33 个人体关键点**。坐标保留 **MediaPipe 原生归一化**格式（x, y ∈ [0, 1]，z 为相对于髋部中心的深度），输出 `(N, 4)` numpy 数组，可直接送入下游 LSTM 时序模型。

**论文对齐**:
| 论文 | 需求 | 对应配置 |
|------|------|----------|
| 论文4 (MediaPipe 失衡预判) | 3 个关键点，低算力 | `selected_joints: "minimal_3"` |
| 论文5 (注意力GRU 三分类) | 13 个核心关节 | `selected_joints: "core_13"` |
| 论文1 (双目3D 倒地检测) | 世界坐标 3D 重心 | `output_world_landmarks: true` |

---

## 快速上手

```python
from pose import PoseExtractor, PoseExtractorConfig, load_pose_config

# 方式1: 从 YAML 配置加载
config = load_pose_config("config/pose_config.yaml")

# 方式2: 代码直接配置
config = PoseExtractorConfig(
    model_path="model/mediapipe_pose/pose_landmarker_lite.task",
    selected_joints=None,            # None = 全部33点
    output_world_landmarks=True,
)

with PoseExtractor(config) as extractor:
    result = extractor.detect_frame(frame, timestamp_ms=0)
    if result:
        arr = result.array              # (33, 4) numpy → 送 LSTM
        nose = result.get(BodyPart.NOSE) # 按名称取点
```

---

## 模块结构

```
pose/
├── __init__.py        # 公开导出
├── keypoint.py        # 数据模型: Keypoint, PoseResult, BodyPart 枚举
├── extractor.py       # 核心类: PoseExtractor + PoseExtractorConfig
└── config_loader.py   # YAML 配置加载: load_pose_config(), create_config_from_dict()
```

---

## API 总览

### PoseExtractor 公有方法

| 方法 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `detect_frame()` | 单帧 BGR + 时间戳 | `PoseResult \| None` | 实时摄像头逐帧推理 |
| `process_video()` | 视频文件路径 | `Iterator[PoseResult]` | 离线视频逐帧处理 |
| `process_image()` | 图片文件路径 | `PoseResult \| None` | 单张图片检测 |
| `process_batch()` | 帧列表 + 时间戳列表 | `list[PoseResult \| None]` | 多帧批量处理 |
| `process_webcam()` | 摄像头 ID | `Iterator[PoseResult \| None]` | 实时摄像头生成器 |
| `close()` | — | `None` | 释放 MediaPipe 资源 |

### 配置加载函数

| 函数 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `load_pose_config()` | YAML 文件路径 `str` | `PoseExtractorConfig` | 从配置文件加载 |
| `create_config_from_dict()` | 配置字典 `dict` | `PoseExtractorConfig` | 代码中直接创建 |

---

## 核心类

### PoseExtractor

```python
class PoseExtractor:
    def __init__(self, config: PoseExtractorConfig)
    def detect_frame(frame: np.ndarray, timestamp_ms: int) -> PoseResult | None
    def process_video(video_path: str, resize=None, max_frames=None) -> Iterator[PoseResult]
    def process_image(image_path: str) -> PoseResult | None
    def process_batch(frames: list[np.ndarray], timestamps_ms=None) -> list[PoseResult | None]
    def process_webcam(camera_id=0, display=False, exit_key="q") -> Iterator[PoseResult | None]
    def close() -> None
```

**资源管理**: 支持 `with` 上下文管理器，自动释放 MediaPipe 资源。支持延迟初始化（首次调用推理方法时才创建 detector）。

---

#### `detect_frame(frame, timestamp_ms) → PoseResult | None`

处理单帧图像，是其他所有方法的底层调用。

**输入**:

| 参数 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `frame` | `np.ndarray` | ✅ | BGR 格式 3 通道图像，shape `(H, W, 3)`，dtype `uint8` |
| `timestamp_ms` | `int` | ✅ | 毫秒时间戳，必须单调递增（MediaPipe VIDEO 模式要求） |

**输入约束**:
- `frame` 不能为 `None` 或空数组，否则抛出 `ValueError`
- `frame.shape` 必须为 `(H, W, 3)`，灰度图需先转为 BGR
- `timestamp_ms` 摄像头流用 `cap.get(cv2.CAP_PROP_POS_MSEC)`，视频文件用 `frame_index / fps * 1000`

**输出**:

| 条件 | 返回值 | 说明 |
|------|--------|------|
| 检测到人体且置信度 ≥ 阈值 | `PoseResult` | 包含关键点列表、世界坐标、时间戳、帧尺寸 |
| 未检测到人体 | `None` | 帧中无人或置信度不足 |
| 输入格式错误 | 抛出 `ValueError` | 空帧、非3通道 |

**示例**:
```python
cap = cv2.VideoCapture(0)
extractor = PoseExtractor(config)
while True:
    ret, frame = cap.read()
    result = extractor.detect_frame(frame, timestamp_ms=cap.get(cv2.CAP_PROP_POS_MSEC))
    if result:
        print(result.array.shape)  # (33, 4) 或 (13, 4) 或 (3, 4)
```

---

#### `process_video(video_path, resize=None, max_frames=None) → Iterator[PoseResult]`

处理视频文件，**生成器模式**逐帧 yield，不会因视频过长撑爆内存。跳帧由 `config.frame_skip` 控制。

**输入**:

| 参数 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `video_path` | `str` | ✅ | 视频文件路径，支持 `cv2.VideoCapture` 可打开的所有格式 |
| `resize` | `tuple[int, int] \| None` | ❌ | 可选，统一缩放为 `(width, height)`，加速推理 |
| `max_frames` | `int \| None` | ❌ | 可选，最多处理 N 帧（调试用），None = 全部 |

**输出**:

| 类型 | 说明 |
|------|------|
| `Iterator[PoseResult]` | 生成器，每检测到一帧 yield 一个 `PoseResult`。无人帧**跳过不 yield**，因此输出数量 ≤ 输入帧数 |

**异常**:
- `RuntimeError`: 视频文件无法打开（路径不存在、编码不支持）

**示例**:
```python
from pose import load_pose_config, PoseExtractor

config = load_pose_config("config/pose_config.yaml")
all_sequences = []

with PoseExtractor(config) as extractor:
    for result in extractor.process_video("walk.mp4", resize=(320, 240)):
        all_sequences.append(result.array)

# all_sequences → (T, N, 4) → 送入时序模型
import numpy as np
features = np.stack(all_sequences, axis=0)
print(f"Sequence: {features.shape}")  # (T, 33, 4)
```

---

#### `process_image(image_path) → PoseResult | None`

处理单张图片，等同于 `cv2.imread()` + `detect_frame(frame, timestamp_ms=0)`。

**输入**:

| 参数 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `image_path` | `str` | ✅ | 图片文件路径，支持 `cv2.imread` 可读的所有格式（jpg, png, bmp 等） |

**输出**:

| 条件 | 返回值 | 说明 |
|------|--------|------|
| 检测到人体 | `PoseResult` | `timestamp_ms=0` |
| 未检测到人体 | `None` | — |
| 文件不存在/不可读 | 抛出 `FileNotFoundError` | — |

**示例**:
```python
result = extractor.process_image("person.jpg")
if result:
    print(f"{result.num_keypoints} keypoints detected")
    print(f"Mean confidence: {result.mean_confidence:.3f}")
```

---

#### `process_batch(frames, timestamps_ms=None) → list[PoseResult | None]`

批量处理多帧。**注意**：MediaPipe VIDEO 模式不支持真正的 batch inference，内部逐帧调用 `detect_frame()`。

**输入**:

| 参数 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `frames` | `list[np.ndarray]` | ✅ | BGR 帧列表，每个元素 shape `(H, W, 3)` |
| `timestamps_ms` | `list[int] \| None` | ❌ | 对应时间戳列表，None 则按 30fps 自动生成 `[0, 33, 66, ...]` |

**输入约束**:
- `frames` 和 `timestamps_ms` 长度必须一致（若提供后者），否则抛出 `ValueError`

**输出**:

| 类型 | 说明 |
|------|------|
| `list[PoseResult \| None]` | 与 `frames` **等长**的列表，无检测的位置为 `None`，**不会跳过无人帧** |

**示例**:
```python
frames = [frame1, frame2, frame3]
results = extractor.process_batch(frames)
# results = [PoseResult(...), None, PoseResult(...)]
# len(results) == len(frames)  # 始终等长

for i, (frame, result) in enumerate(zip(frames, results)):
    if result is not None:
        print(f"Frame {i}: {result.num_keypoints} keypoints")
```

---

#### `process_webcam(camera_id=0, display=False, exit_key="q") → Iterator[PoseResult | None]`

实时摄像头处理生成器。与 `process_video` 不同：无人帧也 **yield `None`**（不跳过），让调用方自行处理帧率控制。

**输入**:

| 参数 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `camera_id` | `int` | ❌ | 摄像头设备 ID，默认 `0`（第一个摄像头） |
| `display` | `bool` | ❌ | 是否弹出 OpenCV 窗口绘制关键点（调试用），默认 `False` |
| `exit_key` | `str` | ❌ | `display=True` 时的退出按键，默认 `"q"` |

**输出**:

| 类型 | 说明 |
|------|------|
| `Iterator[PoseResult \| None]` | 每帧 yield 一个 `PoseResult` 或 `None`。**无人帧也 yield None**，不会跳过 |

**异常**:
- `RuntimeError`: 摄像头无法打开（设备不存在或被占用）

**示例**:
```python
extractor = PoseExtractor(config)
for result in extractor.process_webcam(display=True):
    if result is None:
        continue  # 无人，跳过
    arr = result.array  # (N, 4)
    # → 送入下游模型
```

---

#### `close() → None`

释放底层 MediaPipe `PoseLandmarker` 资源。支持多次调用（幂等）。若使用 `with` 语句则自动调用，无需手动。

**输入**: 无

**输出**: `None`

---

### PoseExtractorConfig

```python
@dataclass
class PoseExtractorConfig:
    model_path: str                            # .task 模型路径（必填）
    min_detection_confidence: float = 0.5      # 人体检测阈值
    min_tracking_confidence: float = 0.5       # 关键点跟踪阈值
    min_pose_presence_confidence: float = 0.6  # 姿态存在阈值（论文1：<60%丢弃）
    output_world_landmarks: bool = True        # 是否输出 3D 世界坐标
    selected_joints: list[BodyPart] | None = None  # 关键点筛选
    frame_skip: int = 1                        # 跳帧间隔
```

**字段详情**:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | `str` | **必填** | `.task` 模型文件路径，必须存在且以 `.task` 结尾 |
| `min_detection_confidence` | `float` | `0.5` | 人体检测置信度阈值 (0~1)，低于此值认为帧中无人 |
| `min_tracking_confidence` | `float` | `0.5` | 关键点跟踪置信度阈值 (0~1)，影响连续帧跟踪稳定性 |
| `min_pose_presence_confidence` | `float` | `0.6` | 姿态存在置信度阈值 (0~1)，论文1：低于60%丢弃帧 |
| `output_world_landmarks` | `bool` | `True` | 是否输出 3D 世界坐标（米），论文1 重心计算需要 |
| `selected_joints` | `list[BodyPart] \| None` | `None` | 关键点筛选列表，None = 全部 33 点 |
| `frame_skip` | `int` | `1` | 跳帧间隔，`1`=不跳帧，`>1`=每 N 帧处理 1 帧 |

**验证**: `frame_skip < 1` 抛出 `ValueError`

---

### PoseResult

单帧姿态提取结果，封装了所有下游所需的数据。

```python
@dataclass
class PoseResult:
    landmarks: list[Keypoint]           # 筛选后的关键点
    world_landmarks: list[Keypoint] | None  # 世界坐标 3D 点
    timestamp_ms: int                   # 帧时间戳
    frame_width: int                    # 原始帧宽 (px)
    frame_height: int                   # 原始帧高 (px)
```

**字段**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `landmarks` | `list[Keypoint]` | 筛选后的关键点列表，长度 = `selected_joints` 配置的 N（33/13/3） |
| `world_landmarks` | `list[Keypoint] \| None` | 世界坐标 3D 点（单位：米），`output_world_landmarks=False` 时为 `None` |
| `timestamp_ms` | `int` | 帧毫秒时间戳 |
| `frame_width` | `int` | 原始帧宽度（像素），用于需要时 `x * frame_width` 转像素 |
| `frame_height` | `int` | 原始帧高度（像素），用于需要时 `y * frame_height` 转像素 |

**属性（numpy 数组输出）**:

| 属性 | 类型 | shape | dtype | 说明 |
|------|------|-------|-------|------|
| `.num_keypoints` | `int` | — | — | 实际关键点数量 N |
| `.array` | `np.ndarray` | `(N, 4)` | `float32` | `[x, y, z, visibility]` → **直接送 LSTM** |
| `.array_2d` | `np.ndarray` | `(N, 2)` | `float32` | 仅 `[x, y]`（[0, 1] 归一化） |
| `.array_3d` | `np.ndarray` | `(N, 3)` | `float32` | 仅 `[x, y, z]` 坐标 |
| `.confidence_array` | `np.ndarray` | `(N,)` | `float32` | 各点置信度 |
| `.mean_confidence` | `float` | — | — | 平均置信度 |
| `.valid_mask` | `np.ndarray` | `(N,)` | `bool` | 置信度 ≥ 0.6 的布尔掩码 |

**方法**:

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `.get(part)` | `BodyPart` | `Keypoint` | 从 `landmarks` 按部位名取点，不存在则 `KeyError` |
| `.get_world(part)` | `BodyPart` | `Keypoint` | 从 `world_landmarks` 按部位名取 3D 世界坐标点 |
| `.to_dict()` | — | `dict[str, list[float]]` | `{"NOSE": [x,y,z,v], ...}` JSON 序列化 |
| `.clone_with_normalized_coords()` | — | `PoseResult` | x,y 除以帧宽高，归一化到 0~1 |
| `.clone_with_3d_normalized()` | — | `PoseResult` | world_landmarks 归一化到 [-1, 1]，原点移到髋部中心 |

**`.to_dict()` 输出格式**:
```python
{
    "NOSE": [x, y, z, visibility],       # 4 个 float
    "LEFT_SHOULDER": [x, y, z, visibility],
    # ... 共 N 个 key
}
```

**`.clone_with_normalized_coords()` 输出**:
- 坐标已是 MediaPipe 原生 [0, 1] 归一化，此方法返回**深拷贝**（值不变）
- 需要像素坐标时手动乘以 `frame_width` / `frame_height`

**`.get_world(part)` 输入/输出**:

| 项目 | 说明 |
|------|------|
| 输入 `part` | `BodyPart` 枚举值，如 `BodyPart.LEFT_HIP` |
| 输出 | `Keypoint`（坐标单位为**米**） |
| 异常 `ValueError` | `world_landmarks` 为 `None`（`output_world_landmarks=False`） |
| 异常 `KeyError` | 关键点被筛选掉或不存在 |

**`.clone_with_3d_normalized()` 输入/输出**:

| 项目 | 说明 |
|------|------|
| 输入 | —（无参数） |
| 输出 | **新** `PoseResult`，`landmarks` 不变，`world_landmarks` 归一化到 [-1, 1] |
| 异常 `ValueError` | `world_landmarks` 为 `None` |
| 异常 `KeyError` | `LEFT_HIP` 或 `RIGHT_HIP` 不在 `world_landmarks` 中 |

对齐论文1：
> "Each axis key points are normalized, which range from **-1 to 1**,
> and the origin of the 3D space starts from the **hip center (0, 0, 0)**."

处理步骤：
1. **平移**：将左髋+右髋中点移至世界坐标原点 (0, 0, 0)
2. **缩放**：所有世界坐标除以 `max(|x|, |y|, |z|)`，使全体落入 [-1, 1]

```python
# 世界坐标是米制，归一化前（以左髋为例）
# get_world(BodyPart.LEFT_HIP)  → Keypoint(x=0.09, y=0.85, z=-3.2, ...)

norm = result.clone_with_3d_normalized()

# 归一化后：髋部中心 → (0, 0, 0)，所有点 ∈ [-1, 1]
# norm.world_landmarks[x]  → [-1, 1]
# norm.world_landmarks[y]  → [-1, 1]
# norm.world_landmarks[z]  → [-1, 1]
# norm.landmarks            → 不变（[0, 1] 归一化坐标）
```

**两种归一化对比**：

| | `clone_with_normalized_coords()` | `clone_with_3d_normalized()` |
|------|------|------|
| 处理对象 | `landmarks` + `world_landmarks` | 仅 `world_landmarks` |
| 归一化方式 | 除以帧宽高 | 除以 `max(|x|,|y|,|z|)` |
| 原点 | 图像左上角 (0, 0) | 髋部中心 (0, 0, 0) |
| 值域 | x, y ∈ [0, 1] | x, y, z ∈ [-1, 1] |
| 用途 | 消除分辨率差异 | 论文1 3D 重心/姿态计算 |

---

### Keypoint

```python
@dataclass
class Keypoint:
    x: float          # 归一化 x [0, 1]
    y: float          # 归一化 y [0, 1]
    z: float          # 深度（相对髋部中心，世界坐标系为米）
    visibility: float # 置信度 0~1
    name: BodyPart    # 部位枚举
```

**字段**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `x` | `float` | 归一化 x [0, 1]（相对于图像宽度） |
| `y` | `float` | 归一化 y [0, 1]（相对于图像高度） |
| `z` | `float` | 深度值，相对髋部中心。`landmarks` 中为 MediaPipe 归一化值，`world_landmarks` 中为米 |
| `visibility` | `float` | 置信度 `0.0 ~ 1.0` |
| `name` | `BodyPart` | 对应的 BodyPart 枚举值 |

**属性**:

| 属性 | 类型 | 说明 |
|------|------|------|
| `.confidence` | `float` | `= visibility`（别名） |
| `.is_valid` | `bool` | `visibility >= 0.6` |

**方法**:

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `.to_tuple()` | — | `tuple[float, float, float, float]` | `(x, y, z, visibility)` |

---

### BodyPart 枚举

MediaPipe 33 个关键点，从 0 到 32：

| 索引 | 枚举名 | 说明 |
|------|--------|------|
| 0 | `NOSE` | 鼻尖 |
| 1 | `LEFT_EYE_INNER` | 左眼内角 |
| 2 | `LEFT_EYE` | 左眼 |
| 3 | `LEFT_EYE_OUTER` | 左眼外角 |
| 4 | `RIGHT_EYE_INNER` | 右眼内角 |
| 5 | `RIGHT_EYE` | 右眼 |
| 6 | `RIGHT_EYE_OUTER` | 右眼外角 |
| 7 | `LEFT_EAR` | 左耳 |
| 8 | `RIGHT_EAR` | 右耳 |
| 9 | `MOUTH_LEFT` | 左嘴角 |
| 10 | `MOUTH_RIGHT` | 右嘴角 |
| **11** | **`LEFT_SHOULDER`** | 左肩 |
| **12** | **`RIGHT_SHOULDER`** | 右肩 |
| **13** | **`LEFT_ELBOW`** | 左肘 |
| **14** | **`RIGHT_ELBOW`** | 右肘 |
| **15** | **`LEFT_WRIST`** | 左腕 |
| **16** | **`RIGHT_WRIST`** | 右腕 |
| 17 | `LEFT_PINKY` | 左小指 |
| 18 | `RIGHT_PINKY` | 右小指 |
| 19 | `LEFT_INDEX` | 左食指 |
| 20 | `RIGHT_INDEX` | 右食指 |
| 21 | `LEFT_THUMB` | 左拇指 |
| 22 | `RIGHT_THUMB` | 右拇指 |
| **23** | **`LEFT_HIP`** | 左髋 |
| **24** | **`RIGHT_HIP`** | 右髋 |
| **25** | **`LEFT_KNEE`** | 左膝 |
| **26** | **`RIGHT_KNEE`** | 右膝 |
| **27** | **`LEFT_ANKLE`** | 左踝 |
| **28** | **`RIGHT_ANKLE`** | 右踝 |
| 29 | `LEFT_HEEL` | 左脚跟 |
| 30 | `RIGHT_HEEL` | 右脚跟 |
| 31 | `LEFT_FOOT_INDEX` | 左脚尖 |
| 32 | `RIGHT_FOOT_INDEX` | 右脚尖 |

> **粗体** = `CORE_JOINTS_13`（13 个核心关节，去除了面部和手部细节）

**预定义分组**:

| 常量 | 类型 | 数量 | 包含 | 用途 |
|------|------|------|------|------|
| `ALL_JOINTS_33` | `list[BodyPart]` | 33 | 全部关键点 | 通用场景 |
| `CORE_JOINTS_13` | `list[BodyPart]` | 13 | NOSE + 双肩/肘/腕/髋/膝/踝 | 论文5：注意力GRU 三分类 |
| `MINIMAL_JOINTS_3` | `list[BodyPart]` | 3 | NOSE + LEFT_SHOULDER + LEFT_HIP | 论文4：轻量 LSTM |
| `JOINT_SET_MAP` | `dict` | 3 | `{"all_33"\|"core_13"\|"minimal_3" → list}` | YAML 字符串 → 列表映射 |

---

## 配置加载

### `load_pose_config(config_path) → PoseExtractorConfig`

从 YAML 文件加载配置。

**输入**:

| 参数 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `config_path` | `str` | ✅ | YAML 文件路径，支持相对路径（相对于项目根）和绝对路径 |

**输出**:

| 类型 | 说明 |
|------|------|
| `PoseExtractorConfig` | 解析完成的配置对象，model_path 已拼接为绝对路径 |

**异常**:
- `FileNotFoundError`: 配置文件不存在
- `ValueError`: 配置文件为空或缺少 `pose` section

**YAML 格式** (`config/pose_config.yaml`):

```yaml
pose:
  model_name: "mediapipe_pose"              # 对应 model/{name}/
  model_version: "lite"                     # lite | full | heavy
  min_detection_confidence: 0.5             # 人体检测阈值 (0~1)
  min_tracking_confidence: 0.5              # 关键点跟踪阈值 (0~1)
  min_pose_presence_confidence: 0.6        # 姿态存在阈值 (0~1)
  output_world_landmarks: true              # 是否输出 3D 世界坐标
  selected_joints: null                     # null | "all_33" | "core_13" | "minimal_3"
  frame_skip: 1                             # 跳帧间隔 (>= 1)
```

**示例**:
```python
from pose import load_pose_config

config = load_pose_config("config/pose_config.yaml")
# config.model_path = "d:/myproject/elder_decete/model/mediapipe_pose/pose_landmarker_lite.task"
```

---

### `create_config_from_dict(pose_cfg) → PoseExtractorConfig`

从字典创建配置，无需 YAML 文件，适合程序化场景。

**输入**:

| 参数 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `pose_cfg` | `dict[str, Any]` | ✅ | 对应 YAML 中 `pose` section 的字典 |

**输入字典字段**（全部可选，有默认值）:

| 键 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name` | `str` | `"mediapipe_pose"` | 模型目录名 |
| `model_version` | `str` | `"lite"` | `"lite"` / `"full"` / `"heavy"` |
| `min_detection_confidence` | `float` | `0.5` | 人体检测阈值 |
| `min_tracking_confidence` | `float` | `0.5` | 关键点跟踪阈值 |
| `min_pose_presence_confidence` | `float` | `0.6` | 姿态存在阈值 |
| `output_world_landmarks` | `bool` | `True` | 是否输出世界坐标 |
| `selected_joints` | `str \| list[str] \| None` | `None` | 关键点选择 |
| `frame_skip` | `int` | `1` | 跳帧间隔 |

**输出**:

| 类型 | 说明 |
|------|------|
| `PoseExtractorConfig` | model_path 根据 `model_name` + `model_version` 自动拼接为 `model/{name}/pose_landmarker_{version}.task` |

**示例**:
```python
from pose import create_config_from_dict, CORE_JOINTS_13

# 论文5 配置
config = create_config_from_dict({
    "model_name": "mediapipe_pose",
    "model_version": "lite",
    "selected_joints": "core_13",
})

# 也可以直接传 BodyPart 名列表
config = create_config_from_dict({
    "model_name": "mediapipe_pose",
    "model_version": "full",
    "selected_joints": ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"],
})
```

---

## 模型

### 版本对比

| 版本 | 文件名 | 大小 | CPU 延迟 | 精度 | 适用场景 |
|------|--------|------|----------|------|----------|
| `lite` | `pose_landmarker_lite.task` | 5.5MB | ~15ms | ★★★ | 实时摄像头、边缘设备（论文4/8） |
| `full` | `pose_landmarker_full.task` | 12MB | ~30ms | ★★★★ | 服务器端实时 |
| `heavy` | `pose_landmarker_heavy.task` | 20MB | ~50ms | ★★★★★ | 离线高精度批处理 |

### 下载

```bash
python model/mediapipe_pose/download_models.py          # 全部三个版本
python model/mediapipe_pose/download_models.py --lite   # 仅 lite
python model/mediapipe_pose/download_models.py --full   # 仅 full
python model/mediapipe_pose/download_models.py --heavy  # 仅 heavy
```

模型文件加入 `.gitignore`，不会被提交。

---

## 完整示例

### 实时摄像头 + 躯干倾角预警

```python
import cv2
import numpy as np
from pose import PoseExtractor, PoseExtractorConfig, BodyPart

config = PoseExtractorConfig(
    model_path="model/mediapipe_pose/pose_landmarker_lite.task",
)
cap = cv2.VideoCapture(0)

with PoseExtractor(config) as extractor:
    while True:
        ret, frame = cap.read()
        ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = extractor.detect_frame(frame, ts)

        # 输入: BGR (H,W,3) + int
        # 输出: PoseResult | None
        if result is None:
            continue  # 无人，跳过

        arr = result.array  # (N, 4) float32

        # --- 论文5 躯干倾角 ---
        ls = result.get(BodyPart.LEFT_SHOULDER)   # → Keypoint
        rs = result.get(BodyPart.RIGHT_SHOULDER)  # → Keypoint
        lh = result.get(BodyPart.LEFT_HIP)        # → Keypoint
        rh = result.get(BodyPart.RIGHT_HIP)       # → Keypoint

        shoulder_mid = np.array([(ls.x+rs.x)/2, (ls.y+rs.y)/2])
        hip_mid = np.array([(lh.x+rh.x)/2, (lh.y+rh.y)/2])
        angle = np.arctan2(shoulder_mid[0]-hip_mid[0], -(shoulder_mid[1]-hip_mid[1]))
        angle_deg = np.degrees(angle)

        if abs(angle_deg) > 30:
            print(f"WARNING: Torso tilt {angle_deg:.1f} deg!")

        # arr → 时序缓冲区 → LSTM
```

### 视频文件批处理

```python
from pose import load_pose_config, PoseExtractor
import numpy as np

config = load_pose_config("config/pose_config.yaml")
all_sequences = []

# 输入: 视频路径 str
# 输出: Iterator[PoseResult]
with PoseExtractor(config) as extractor:
    for result in extractor.process_video("walk.mp4", resize=(320, 240)):
        all_sequences.append(result.array)  # (N, 4) float32

# 堆叠为时序特征 → (T, N, 4)
features = np.stack(all_sequences, axis=0)
print(f"Sequence: {features.shape}")  # (T, 33, 4)
```

### 批量处理 + 时间戳对齐

```python
frames = [frame1, frame2, frame3]           # list[np.ndarray]
timestamps = [0, 33, 66]                    # list[int]

# 输入: list[np.ndarray] + list[int]
# 输出: list[PoseResult | None]  # 始终与输入等长
results = extractor.process_batch(frames, timestamps)
for i, r in enumerate(results):
    if r:
        print(f"Frame {i} @ {r.timestamp_ms}ms: {r.num_keypoints} pts")
```

### 图片检测 + 序列化

```python
# 输入: 图片路径 str
# 输出: PoseResult | None
result = extractor.process_image("test/photo/image.png")
if result:
    # 序列化为 JSON 友好格式
    data = result.to_dict()  # → dict[str, list[float]]
    # {"NOSE": [320.5, 240.3, -0.12, 0.95], ...}

    # clone_with_normalized_coords — 坐标已原生 [0,1]，返回拷贝
    norm = result.clone_with_normalized_coords()  # → PoseResult (值不变)

    # 3D 世界坐标归一化（论文1：[-1, 1]，髋部中心原点）
    norm3d = result.clone_with_3d_normalized()    # → PoseResult (world x,y,z ∈ [-1,1])
    lh = norm3d.get_world(BodyPart.LEFT_HIP)      # → Keypoint (≈ 0, 0.1, 0) 髋部中心附近
```

### 命令行演示

```bash
# 摄像头实时（按 q 退出，按 s 截图）
python test/code/demo_pose.py

# 图片检测 → 输出到 test/photo/pose_output.jpg
python test/code/demo_pose.py --image test/photo/image.png

# 视频检测
python test/code/demo_pose.py --video walk.mp4
```

---

## 数据流

```
┌──────────┐   detect_frame()         ┌─────────────┐   .array          ┌──────────┐
│ Camera / │ ──────────────────────→  │ PoseResult   │ ──────────────→  │  (N, 4)  │
│ Video    │   Input: BGR (H,W,3)     │              │   (N,4) float32  │  LSTM    │
│          │   + timestamp_ms (int)   │ 33 Keypoints │   x,y ∈ [0,1]    │          │
│          │                          │ x,y ∈ [0,1]  │   z = 相对深度     │          │
└──────────┘                          └─────────────┘                  └──────────┘
     │                                      │
     │  process_image(str)                  ├── .get(BodyPart.NOSE)
     │  Input: 文件路径                      │   Input: BodyPart
     │  Output: PoseResult | None           │   Output: Keypoint | KeyError
     │                                      │
     │  process_video(str)                  ├── .get_world(BodyPart.LEFT_HIP)
     │  Input: 视频路径 + resize + max      │   Input: BodyPart
     │  Output: Iterator[PoseResult]        │   Output: Keypoint (米) | ValueError/KeyError
     │                                      │
     │  process_batch(list[ndarray])        ├── .to_dict()
     │  Input: 帧列表 + 时间戳列表            │   Input: —
     │  Output: list[PoseResult|None]       │   Output: dict[str, list[float]]
     │                                      │
     │  process_webcam(int)                 ├── .clone_with_normalized_coords()
     │  Input: 摄像头 ID + display          │   Input: —
     │  Output: Iterator[PoseResult|None]   │   Output: PoseResult (x,y ∈ [0,1])
     │                                      │
     │                                      ├── .clone_with_3d_normalized()
     │                                      │   Input: —
     │                                      │   Output: PoseResult (world x,y,z ∈ [-1,1])
     │                                      │   原点: 髋部中心 (0,0,0)
     │                                      │
     │                                      ├── .world_landmarks
     │                                      │   Output: list[Keypoint] | None
     │                                      │   (世界坐标, 单位: 米)
     │                                      │
     │                                      └── .mean_confidence
     │                                          Output: float
     │
     └── close()
         Input: —
         Output: None
```

---

## 异常处理

| 异常 | 触发条件 | 抛出方法 |
|------|----------|----------|
| `FileNotFoundError` | 模型文件不存在 | `PoseExtractor.__init__()` |
| `FileNotFoundError` | 配置文件不存在 | `load_pose_config()` |
| `FileNotFoundError` | 图片文件不存在/不可读 | `process_image()` |
| `ValueError` | 非 `.task` 文件 | `PoseExtractor.__init__()` |
| `ValueError` | 输入帧为空或非3通道 | `detect_frame()` |
| `ValueError` | `frame_skip < 1` | `PoseExtractorConfig.__post_init__()` |
| `ValueError` | 配置文件为空或缺少 `pose` section | `load_pose_config()` |
| `ValueError` | `frames` 与 `timestamps_ms` 长度不一致 | `process_batch()` |
| `ValueError` | `world_landmarks` 为 `None` | `PoseResult.get_world()`, `PoseResult.clone_with_3d_normalized()` |
| `KeyError` | `.get_world()` 查询未被筛选的关键点 | `PoseResult.get_world()` |
| `KeyError` | `LEFT_HIP` 或 `RIGHT_HIP` 不在 `world_landmarks` 中 | `PoseResult.clone_with_3d_normalized()` |
| `KeyError` | `.get()` 查询未被筛选的关键点 | `PoseResult.get()` |
| `RuntimeError` | 视频文件打开失败 | `process_video()` |
| `RuntimeError` | 摄像头打开失败 | `process_webcam()` |

**典型异常处理**:
```python
try:
    config = PoseExtractorConfig(model_path="wrong_path.task")
    extractor = PoseExtractor(config)
except FileNotFoundError:
    print("请先下载模型: python model/mediapipe_pose/download_models.py --lite")
except ValueError as e:
    print(f"配置错误: {e}")
```
