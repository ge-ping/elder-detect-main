# RTMPose + ST-GCN 跌倒风险预测 —— 开发任务清单

> 配套设计文档：`docs/RTMPose_STGCN_跌倒风险设计.md`
> 分支：`feature/rtmpose-stgcn-fall-risk`（从 `main` 拉取）
> 状态：任务规划，**未实现任何代码**
> 关键约定：**RTMPose body-26 去面部 5 点 → V=21，保留脚；通道 C=3（x,y,conf），先不加手工特征**

---

## 任务总览（按阶段）

每个任务含：目标 / 关键子任务 / 产出物 / 依赖 / 验收标准。勾选框 `[ ]` 用于实现阶段逐条推进。

---

## 阶段 0：环境与依赖
**目标**：搭好 RTMPose + ST-GCN 训练/推理基础环境。
**详细安装指令（可直接复制执行）见 `docs/RTMPose_STGCN_环境配置.md`，本机运行、不在仓库内执行。**

- [ ] 0.1 安装 OpenMMLab 栈：`mmengine` → `mmcv`(2.x) → `mmdet`(3.x，含 RTMDet) → `mmpose`(1.x，含 RTMPose + PoseTracker)
- [ ] 0.2 安装 `mmdeploy` + 后端（TensorRT / ONNXRuntime，阶段 9 才用，可先跳过）
- [ ] 0.3 安装 `torch` / `torchvision`（按本机 CUDA 选，示例 cu118/cu121 + torch 2.1.0）
- [ ] 0.4 下载模型权重：（`model/` 整体被忽略，仅本地使用，不入库）
  - RTMPose：`rtmpose-m_8xb512-700e_body8-halpe26-256x192`（官方 alias `body26`，body-26），存入 `model/rtmpose/`
  - **RTMDet（检测器，ByteTrack 关联必需）**：`rtmdet_m_300e_coco`（或轻量 `rtmdet_tiny_300e_coco`），存入 `model/rtmdet/`；`PoseTracker` 链路须先出检测框再进 RTMPose，**缺检测器则 track 不成立**
- [ ] 0.5 固化 `requirements-stgcn.txt`（锁定版本，可入库），`model/` 权重不提交
- [ ] 0.6 冒烟测试：单张图跑通 RTMPose body-26 推理（`MMPoseInferencer(pose2d='body26')`），输出 26 关键点

**产出**：可运行 RTMPose 的环境 + 权重文件（见环境配置文档）
**验收**：对任意测试帧输出 26 个 `(x,y,conf)` 关键点；`pose2d='body26'` 自动调用 RTMDet 验证检测前置就绪

---

## 阶段 1：姿态检测模块
**目标**：封装 RTMPose 为可复用检测器，输出本构建选定的 21 点骨架。

- [ ] 1.1 新建 `pose/rtmpose_detector.py`：封装 RTMPose 推理（图像/视频帧 → 关键点）
- [ ] 1.2 关键点筛选：从 26 点中剔除面部 5 点（鼻/眼/耳）→ 输出 **V=21**
- [ ] 1.3 跨帧 ID 绑定（**本阶段关键，时序任务命门**）：
  - 复用 MMPose `PoseTracker`（链路：**RTMDet 检测框 → RTMPose 关键点 → ByteTrack 关联**），逐帧输出带 `track_id` 的骨架
  - ⚠️ RTMPose 本身只出关键点、**不自带 track**；ByteTrack 依赖检测框，故必须串入 RTMDet 检测器（不能只装 RTMPose）
  - 强约束：滑窗 T=60 内必须始终是**同一 `track_id`** 的连续骨架；ID 跳变 = 窗口作废（见 1.6）
  - 主目标锁定（lock-and-switch）：首帧稳定检测后选定 `primary_id`（首选取「面积最大 且 conf 达标」），之后**持续跟随该 ID**；仅当其**连续丢失 > K 帧**（K 对齐 ByteTrack `track_buffer`）才重选，避免逐帧 argmax 的 ID 抖动
  - 多人：默认只跑 `primary_id`（省算力）；可选「每人各跑一条风险流」（养老院等多人场景）
- [ ] 1.4 置信度处理：低 conf 关节标记/插值，避免窗口空洞
- [ ] 1.5 输出结构：`PoseResult(track_id:int, keypoints:(21,3), bbox, score)`（对齐设计 §5.1）；`score`=该 track 姿态置信度
- [ ] 1.6 跟踪失效缓冲策略（缓解「窗口作废」）：
  - `keypoint_buffer` 按 `track_id` 索引；当前 `primary_id` 改变或连续丢失 > K 帧时 **flush 缓冲**（丢弃不满窗序列，禁止跨人拼接进同一窗口）
  - 跟踪丢失期间暂停预测，输出 `risk=None / need_review=True`，不 emit 报警
  - 验收：构造「两人交替出现 / 目标短暂遮挡」测试视频，断言窗口内 `track_id` 恒定、切换时缓冲清空、无跨人帧混入

**产出**：`pose/rtmpose_detector.py` + 单测
**依赖**：阶段 0
**验收**：视频流连续帧稳定输出 21 点骨架；无人/遮挡时安全处理

---

## 阶段 2：归一化与滑窗
**目标**：把 21 点骨架转成 ST-GCN 输入 `(T,21,3)`。

- [ ] 2.1 新建 `features/skeleton_norm.py`：相对骨盆中心平移 + 躯干长度缩放（对齐设计 §5.2）
- [ ] 2.2 置信度通道保留（作遮挡掩码），**不做**手工速度/角度（先 C=3）
- [ ] 2.3 新建 `sequence/keypoint_buffer.py`：长度 T=60 环形缓冲，满窗返回 `(T,21,3)`（对齐 §5.3）
- [ ] 2.4 张量化：`(T,21,3)` → `(N=1, C=3, T, V=21, M=1)`

**产出**：`skeleton_norm.py` + `keypoint_buffer.py` + 单测
**依赖**：阶段 1
**验收**：喂 60 帧 → 输出形状 `(1,3,60,21,1)`；缓冲未满返回 None

---

## 阶段 3：ST-GCN 图拓扑（关键）
**目标**：定义与 RTMPose 21 点顺序严格对齐的骨骼邻接图。

- [ ] 3.1 新建 `model/stgcn/graph.py`：列出 21 点**有序索引**（去面部后的 HALPE-26 顺序）
- [ ] 3.2 定义骨骼边表（邻接矩阵 A）：颈—肩—肘—腕、颈—髋—膝—踝—脚
- [ ] 3.3 空间分区：root / centripetal / centrifugal 三子图（ST-GCN 原文策略）
- [ ] 3.4 **单测**：断言图节点数=21、边连接符合人体结构、索引与关键点序号一一对应

**产出**：`graph.py` + 单测
**依赖**：阶段 1（需已知点顺序）
**验收**：单测通过；图可视化后符合人体骨架（脚已纳入、面部无）

---

## 阶段 4：ST-GCN 模型与风险头
**目标**：实现时空图卷积网络 + 风险输出头。

- [ ] 4.1 `model/stgcn/stgcn_unit.py`：空间图卷积（按 A 分区）+ 时间卷积（kernel=9，stride 下采样）
- [ ] 4.2 `model/stgcn/stgcn_net.py`：堆叠 9 个单元 + 全局池化（时空）+ 输出特征
- [ ] 4.3 `model/stgcn/head.py`：
  - 回归头 `Linear(h,1)+sigmoid` → risk∈[0,1]（对齐 RiskRegressor）
  - 分类头 `Linear(h,3)+softmax` → Normal/Unstable/Fall（对齐 v5_w35）
  - 多任务头（推荐）：共享骨干 + 双头，`L=λ·BCE+(1−λ)·CE`
- [ ] 4.4 `model/stgcn/__init__.py`：统一构造器 `STGCNFallRisk(input_T, V=21, ...)`

**产出**：`model/stgcn/*` 全套 + 单测（随机输入跑通前向）
**依赖**：阶段 3
**验收**：随机 `(1,3,60,21,1)` 前向输出 risk 与/或 3 类 logits，形状正确

---

## 阶段 5：数据集重建
**目标**：用 RTMPose 重建原始骨架数据集（替代现有 (35,29) 特征）。

- [ ] 5.1 新建 `scripts/build_skeleton_dataset.py`：遍历源视频 → RTMPose 提点（**须开启阶段 1 的 track_id 关联**）→ 去面部 → 存 `(T,21,3)` 序列（落 `datasets/skeleton/*/X_raw.npy`）；同视频内只取 `primary_id`，避免标签错位
- [ ] 5.2 滑窗切样本：步幅 stride 可配（默认 1），窗口 T=60
- [ ] 5.3 标签对齐：沿用现有 Normal/Unstable/Fall 与连续 risk 标注（与 LSTM 数据集同源）
- [ ] 5.4 训练/验证/测试划分 + 元数据清单（视频→样本映射）

**产出**：`build_skeleton_dataset.py` + `datasets/skeleton/*`
**依赖**：阶段 1、2
**验收**：生成 `X_raw.npy` 形状 `(N,60,21,3)`；标签文件齐备

---

## 阶段 6：训练 Pipeline
**目标**：可训练 + 可微调 ST-GCN，输出可信指标。

- [ ] 6.1 新建 `scripts/train_stgcn.py`：DataLoader 加载 `(N,3,T,V,1)` + 标签
- [ ] 6.2 损失：多任务 `L=λ·BCE(risk)+(1−λ)·CE(cls)`；类别不平衡用 Focal/加权
- [ ] 6.3 指标：PR-AUC（不平衡下最诚实）、提前预警时间 lead time、每小时误报数
- [ ] 6.4 **微调入口**：`model.load_state_dict(torch.load(pretrained))` 继续训练（当前 LSTM 脚本缺此能力）
- [ ] 6.5 公开集预训练：NTU-RGB+D / 公开跌倒骨架集预训练 → 自有数据微调（数据不足时）
- [ ] 6.6 `build_config.py` 新增 `STGCN_*` 超参（T=60, V=21, hidden, dropout, lr, λ, 窗口 stride）

**产出**：`train_stgcn.py` + 配置项 + 首版训练曲线/测试报告
**依赖**：阶段 4、5
**验收**：能从头训练并在验证集出 PR-AUC/lead time；能加载预训练权重微调

---

## 阶段 7：融合接入
**目标**：把 ST-GCN 的 B 风险接入现有 B/E/P 融合，不动融合逻辑。

- [ ] 7.1 `scripts/fuse_risk.py` 复用：B 风险来源由 LSTM 改为 ST-GCN 输出，接口不变
- [ ] 7.2 验证 E/P 阈值调节范式不受影响（E/P 仍只调 θ_alarm/θ_warn）
- [ ] 7.3 `demo`/推理统一从 `build_config` 读 STGCN 超参

**产出**：融合联调通过
**依赖**：阶段 4、6
**验收**：ST-GCN 的 risk 经 fuse_risk 输出 WARNING/DANGER，与现有体系一致

---

## 阶段 8：实时 Demo
**目标**：端到端实时演示，替换原 demo 的 pose+model。

- [ ] 8.1 新建 `demo/demo_stgcn.py`：串联 `rtmpose_detector + skeleton_norm + keypoint_buffer + stgcn + fuse_risk`
- [ ] 8.2 保留交互：`cv2.WINDOW_NORMAL`（可缩放）、`--process_every_n` 跳帧
- [ ] 8.3 配置切换：pose backend（rtmpose / mediapipe fallback）、risk model（stgcn / lstm）可配，支持 A/B 与回滚

**产出**：`demo/demo_stgcn.py`
**依赖**：阶段 1–7
**验收**：摄像头/视频实时出风险与报警；帧率 ≥ 30fps（GPU）

---

## 阶段 9：部署
**目标**：导出轻量推理模型，落地边缘/GPU 服务。

- [ ] 9.1 RTMPose → MMDeploy → ONNX/TensorRT
- [ ] 9.2 ST-GCN → `torch.onnx.export` → ONNX → TensorRT / ONNXRuntime
- [ ] 9.3 `FastAPI` + ONNXRuntime(TensorRT EP) 推理服务串联全链路
- [ ] 9.4 边缘验证：Jetson 用 `rtmpose-t` + TensorRT，确认延迟达标

**产出**：导出模型 + 推理服务
**依赖**：阶段 4、8
**验收**：导出模型推理结果与 PyTorch 一致（误差 < 1e-3）；端到端延迟 < 25ms

---

## 阶段 10：评估对比
**目标**：量化 ST-GCN 相对现有 LSTM 方案的提升。

- [ ] 10.1 同源测试集：RTMPose+ST-GCN vs MediaPipe+LSTM 同指标对比
- [ ] 10.2 指标：PR-AUC、macro-F1、提前预警时间 lead time、每小时误报数
- [ ] 10.3 消融：V=21 vs V=26、C=3 vs C>3（加速度/角度）、T=60 vs 45/90
- [ ] 10.4 输出对比报告，给出是否替换线上 LSTM 的结论

**产出**：评估报告
**依赖**：阶段 6、8
**验收**：测试集给出双方案指标表 + 消融结论

---

## 依赖关系（简）

```
阶段0 → 阶段1 → 阶段2 ┐
                       ├→ 阶段3 → 阶段4 ┐
                       │                ├→ 阶段5 → 阶段6 ┐
                       │                │                ├→ 阶段7 → 阶段8 → 阶段9
                       │                │                └→ 阶段10
                       └────────────────┘
```

## 备注
- 全程遵循仓库约定：不提交 `frames/`、`env_labels/`、`.env`、`.claude/`、`.workbuddy/`。
- 实现以 PR 合入 `feature/rtmpose-stgcn-fall-risk`，`push` 由本机执行（凭证问题）。
- 关键点选择（body-26 去面部 → **V=21**，保留脚）见设计文档 §6.1.1。
- 输入通道 `C=3`（x,y,conf）先不加手工特征；后续实验需要速度/角度时扩通道（C>3），原始坐标保留。
