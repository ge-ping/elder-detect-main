# elder_decete — 老年人跌倒风险实时预测系统

## 纵向预警 Web Demo

演示页包含视频上传、长者画像、14日个人基线变化图、多维偏离距离和数据质量说明。页面默认使用明确标记的示例画像；没有真实纵向跌倒标签时，7天和30天概率保持为空。

```powershell
python scripts/run_web_demo.py
```

浏览器打开 `http://127.0.0.1:5000`。支持 MP4、MOV、AVI、MKV 和 WebM，单个文件不超过 200MB。

- 若 `model/mediapipe_pose/pose_landmarker_lite.task` 与 MediaPipe 依赖可用，上传视频会生成本次真实姿态摘要。
- 若模型未安装，接口会明确进入“示例回退”模式；页面展示的数据不代表上传视频的分析结果。
- 即使本次视频已分析，历史基线仍需至少14个真实有效观测日；概率还需要真实纵向标签与独立验证。
- Web Demo 不替代临床评估或紧急处置。

运行接口测试：`python -m unittest tests.test_web_demo -v`。

基于**三因子融合**的跌倒风险预警框架：**B（姿态行为）× E（环境）× P（个体生理）**，
以「阈值调节」范式输出连续风险值 `risk ∈ [0,1]`：

- `risk < 0.3` → SAFE（安全）
- `0.3 ≤ risk < 0.6` → WARNING（注意）
- `risk ≥ 0.6` → DANGER（高危）

本项目已实现 **B 因子**的两个风险模型（ST-GCN / LSTM），及其在多种姿态后端
（MediaPipe 3D、YOLO-Pose 2D）上的实时推理 demo；E 因子（HOME-FAST 环境量表）
与 I 门控（运动意图）也已接入实时门控链路。

---

## 1. 目录结构

```
elder_decete/
├── demo/                      # 推理/可视化 demo（源码入库，runs/ 与 *.mp4 排除）
│   ├── quad_demo.py           # ★ 四宫格：YOLO/MP × ST-GCN/LSTM 全组合对比
│   ├── compare_demo.py        # MediaPipe 下 ST-GCN vs LSTM 并排（黄金参考实现）
│   ├── demo_gate.py           # 实时门控：B + I（意图）因子，逐帧报警
│   ├── demo_env_risk.py       # 环境风险 E 因子（HOME FAST 量表）
│   ├── run_pipeline.py        # 全流程编排：VLM打标→掩码标定→实时门控
│   ├── yolo_pose_demo.py      # YOLO-Pose 单独姿态可视化
│   ├── stgcn_video_demo.py    # ST-GCN 单独风险可视化
│   └── intent_gate_demo*.py   # 意图门控可视化系列
├── model/
│   ├── stgcn/                 # ST-GCN 风险模型（fall17 / coco13 双骨架图）
│   │   ├── stgcn.py           # 网络 + load_stgcn() 加载器
│   │   ├── graph.py           # 骨架图（FALL_JOINTS_17 / COCO13_JOINTS）
│   │   └── stgcn_v1/run_03|run_05/   # ★ 已入库权重（见 §4）
│   ├── lstm/                  # LSTM 风险回归（RiskRegressor）
│   │   ├── risk_model.py      # RiskRegressor 定义
│   │   └── risk_v1/run_04|run_05/     # ★ 已入库权重
│   └── mediapipe_pose/        # MediaPipe .task（clone 后需自下载，见 §3）
├── pose/                      # 姿态后端
│   ├── backend.py             # create_pose_backend() 统一入口
│   ├── yolo_backend.py        # YOLO-Pose 后端（13 核心关节 2D）
│   └── coco_joints.py         # COCO 关节索引映射
├── features/                  # 特征 / 平滑 / 门控
│   ├── landmark_filter.py     # OneEuro 关键点平滑
│   ├── feature_builder_2d.py  # YOLO 2D → 29 维特征
│   ├── intent.py / intent_gate.py  # 运动意图 I 门控
│   └── env_mask.py            # 环境掩码
├── scripts/                   # 数据构建 / 训练 / 评估 / 融合
│   ├── build_cache*.py        # MediaPipe/YOLO 关键点缓存
│   ├── build_skeleton*.py     # 骨架序列窗口（滑窗）
│   ├── build_from_cache*.py   # → 29 维 LSTM 特征
│   ├── train_stgcn.py / train_risk.py / train_lstm.py  # 训练
│   ├── eval_stgcn.py / eval_risk.py / eval_model.py    # 评估
│   ├── fuse_risk.py           # 三因子融合
│   ├── env_*.py / prevfall_*.py  # 环境 / 基线标定
│   └── run_train_stgcn.bat    # 一键训练 ST-GCN
├── config/pose_config.yaml
├── docs/                      # 设计文档（RTMPose_STGCN_*.md 等）
├── env_baselines/ env_labels/ # 环境标定/VLM 标签 JSON
├── requirements.txt
└── .env.example               # 方舟 VLM key 模板（复制为 .env）
```

---

## 2. 环境搭建

```bash
# 1) 依赖（已含 ultralytics / openai / pillow）
pip install -r requirements.txt

# 2) MediaPipe 姿态模型（.task 不入库，clone 后自下载）
python model/mediapipe_pose/download_models.py --lite

# 3) YOLO-Pose 检测器权重（~50MB，不入库，ultralytics 自动下载到根目录）
python -c "from ultralytics import YOLO; YOLO('yolov8m-pose.pt')"

# 4) 如需 R1 VLM 打标（环境 E 因子），复制环境变量模板
cp .env.example .env
#   编辑 .env 填入 ARK_API_KEY / ARK_VLM_MODEL(endpoint id) / ARK_VLM_BASE_URL
#   注意：方舟视觉模型须用「接入点 endpoint id」而非模型名调用。
```

> ⚠️ 本项目依赖 `numpy<2`（MediaPipe 限制）。请勿在装了 numpy≥2 的环境里直接 `pip install`，
> 详见 `docs/RTMPose_STGCN_*.md` 的阶段 0 环境隔离说明。

---

## 3. 已发布的最优模型（随仓库提供，clone 即可用）

| 角色 | 路径 | 关节集 / 特征 | 输入形状 | 窗口 | 关键指标 |
|---|---|---|---|---|---|
| 🏆 **最优 ST-GCN**（MediaPipe-3D） | `model/stgcn/stgcn_v1/run_03` | fall17（17关节，3D 世界坐标） | `(N,3,60,17)` | 60 | **R²=0.847 / macro-F1=0.752 / AUC-ROC=0.986** |
| ST-GCN（YOLO-2D） | `model/stgcn/stgcn_v1/run_05` | coco13（13关节，2D） | `(N,2,60,13)` | 60 | R²=0.688 / macro-F1=0.680 |
| LSTM（MediaPipe） | `model/lstm/risk_v1/run_04` | 29 维特征（3D） | `(N,35,29)` | 35 | — |
| LSTM（YOLO） | `model/lstm/risk_v1/run_05` | 29 维特征（2D） | `(N,60,29)` | 60 | — |

> 每个 run 目录含 `best_model.pt` + `model_info.json`（结构超参）+ `metrics.json` + `config.txt` + 评估图。

### 规范加载示例

```python
import torch
from model.stgcn.stgcn import load_stgcn
from model.lstm.risk_model import RiskRegressor

# ST-GCN：传 .pt 路径，自动读同目录 model_info.json 构造网络
stgcn = load_stgcn("model/stgcn/stgcn_v1/run_03/best_model.pt")
stgcn.eval()

# LSTM：按 input_dim 构造后 load_state_dict
lstm = RiskRegressor(input_dim=29)
lstm.load_state_dict(
    torch.load("model/lstm/risk_v1/run_04/best_model.pt",
               map_location="cpu", weights_only=True))
lstm.eval()

# 推理（batch 维度在前）
x_stgcn = torch.randn(1, 3, 60, 17)   # ST-GCN run_03
x_lstm  = torch.randn(1, 35, 29)      # LSTM  run_04
with torch.no_grad():
    risk_stgcn = float(stgcn(x_stgcn)[0])
    risk_lstm  = float(lstm(x_lstm)[0])
```

---

## 4. 快速运行（demo）

```bash
# ★ 四宫格：YOLO+LSTM / YOLO+ST-GCN / MediaPipe+LSTM / MediaPipe+ST-GCN
python demo\quad_demo.py "test/video/Lecture room/video (2).avi" ^
    --yolo-model yolov8m-pose.pt --mp-process-every 3

# MediaPipe 下 ST-GCN vs LSTM 并排（黄金参考）
python demo\compare_demo.py "test/video/Lecture room/video (2).avi"

# 实时门控（B + I 意图因子，webcam 或视频）
python demo\demo_gate.py "test/video/Lecture room/video (2).avi"

# 环境风险 E 因子（HOME FAST 量表）
python demo\demo_env_risk.py "test/video/Lecture room/video (2).avi"
```

运行产物（标注视频 / 风险曲线 / 逐帧 CSV / 元数据）写入 `demo/runs/<名称>_<时间戳>/`。

---

## 5. 训练 / 复现流水线

```bash
# —— MediaPipe 路径（3D）——
python scripts\build_cache.py            # 关键点缓存 → pose_cache/
python scripts\build_skeleton.py         # 骨架滑窗 → datasets/experiment/skeleton_w35|60/
python scripts\train_stgcn.py            # 或 scripts\run_train_stgcn.bat 一键
python scripts\train_risk.py             # LSTM（build_from_cache.py 产出 29 维）

# —— YOLO 路径（2D）——
python scripts\build_cache_yolo.py
python scripts\build_skeleton_yolo.py
python scripts\train_stgcn.py            # coco13 图
python scripts\train_risk.py             # build_from_cache_yolo.py 产出 29 维

# —— 评估 / 融合 ——
python scripts\eval_stgcn.py
python scripts\eval_risk.py
python scripts\fuse_risk.py               # 三因子融合
```

> 训练数据 `datasets/`、缓存 `pose_cache*/`、原始视频 `test/` **均不入库**，需自行准备并跑上述脚本重建。

---

## 6. 如何修改 / 扩展（关键文件地图）

| 想改什么 | 改哪个文件 |
|---|---|
| 超参数（窗口/阈值/损失权重） | `scripts/build_config.py`（单一事实源，CLI `--xxx` 覆盖） |
| 换 / 调姿态后端 | `pose/backend.py`、`pose/yolo_backend.py`、`pose/coco_joints.py` |
| ST-GCN 结构 / 骨架图 | `model/stgcn/stgcn.py`、`model/stgcn/graph.py`（`FALL_JOINTS_17` / `COCO13_JOINTS`） |
| LSTM 模型 | `model/lstm/risk_model.py`、`model/lstm/model.py`（`FallLSTM`） |
| 实时门控 / 报警状态机 | `demo/compare_demo.py` 的 `AlarmDebouncer`（四宫格沿用） |
| 特征 / 关键点平滑 | `features/feature_builder_2d.py`、`features/landmark_filter.py`（OneEuro） |
| 环境风险 E 因子 | `model/env/env_risk_model.py`、`scripts/env_label_vlm.py` |

---

## 7. 仓库约定

- **4 个模型 run 已用 `git add -f` 强制入库**（默认 `.gitignore` 忽略整个 `model/`），
  clone 后可直接拿权重跑 demo，无需重训。
- 以下**不入库**（体积大或可重建）：
  - `datasets/`(16G)、`test/`(4.1G)、`pose_cache/`(71M)、`pose_cache_yolo/`(15M)
  - `demo/runs/`、`*.mp4`（推理产物/视频）
  - `yolov8m-pose.pt`(50.8M)、`yolov8n-pose.pt`(~6M)（检测器权重，`YOLO()` 可重下）
  - `model/**/*.task`（MediaPipe 权重，clone 后 `download_models.py` 自下载）
  - `temp/`、`frames/`、`.env`（临时/密钥）
- 提交用 `git add -f model/stgcn/stgcn_v1/run_03 model/stgcn/stgcn_v1/run_05 model/lstm/risk_v1/run_04 model/lstm/risk_v1/run_05` 锁定权重。

---

## 8. 纵向个人基线跌倒预警

项目新增独立的天/周级纵向预警模块：连续记录同一老人的步态和姿态摘要，以14个有效观测日建立个人稳健基线，再判断后续观测的多维偏离、趋势和持续时间。

当前仓库不含真实纵向跌倒随访标签，因此默认只输出 `deviation_index`，不会虚构未来概率。只有加载已独立验证且 `calibrated=true` 的 Landmark Cox 模型后，才输出 `risk_7d` 和 `risk_30d`。

```powershell
# 每天导入观测
python scripts/collect_longitudinal.py --input daily.json --person-id elder_001 --observed-at 2026-11-01 --profile-dir profiles

# 至少14个有效日后建立基线
python scripts/build_personal_baseline.py --person-id elder_001 --profile-dir profiles

# 输出最新偏离；无校准模型时7天/30天概率为 null
python scripts/predict_future_fall.py --person-id elder_001 --profile-dir profiles
```

算法依据、字段定义、训练数据格式和完整命令见 [`docs/纵向个人基线跌倒预警.md`](docs/纵向个人基线跌倒预警.md)。
