# 模型进化路线

> 目标：Unstable F1 从 0.53 → 0.70+，只优化姿态特征。（注：环境因子 HOME FAST 已于 2026-07-13 单独加入，见 `docs/跌倒风险预测_汇报.md` §3；本文件聚焦姿态基线。）

> ⚠️ 本文档（Phase-1 姿态回归）与 `risk_env_evolution.md` / `envrisk_v0_design.md` / `envrisk_r1_vlm_design.md` 不同——**本文件描述的 RiskRegressor 姿态主线仍然有效、是当前实现**，仅上方「加环境风险」一条已作废。

---

## 当前架构（固定）

```
Pose sequence (T frames)
    │
MediaPipe → 2D/3D Skeleton
    │
Feature extraction → (T, K)
    │
LSTM → Softmax → P(N) | P(U) | P(F)
    │
Risk = 0.1·P(N) + 0.6·P(U) + 0.9·P(F)   # 以 demo/demo.py 为准
# 状态阈值: SAFE(<0.3) / WARNING(0.3~0.6) / DANGER(≥0.6)
```

---

## 当前瓶颈（2026-07-11）

```
Normal  F1 ≈ 0.87  ✅  (5数据集 Pre-VFall 版)
Fall    F1 ≈ 0.76  ✅
Unstable F1 ≈ 0.66  ❌  ← 核心问题（当前最高记录, 来自 +Pre-VFall）
```

> 4数据集版: Normal 0.95 / Fall 0.88 / Unstable 0.605（macro 0.827）
> 5数据集版: Normal 0.87 / Fall 0.76 / Unstable 0.664（macro ≈0.77）
> 目标 Unstable F1 从 0.53 → 0.70+：+Pre-VFall 后已逼近，仍差约 0.04。

LSTM 能区分「正常」和「明显跌倒」，但**跌倒前的不稳定状态特征不足**。继续调参（epoch/batch/hidden size）收益有限，瓶颈在**标签定义与采样策略**，而非输入特征维度（手工运动特征 V2~V4 已被证无效）。

---

## 0. Pose 缓存格式

> `build_cache.py` 输出，后续所有特征迭代的数据源。

### 存储位置

```
pose_cache/
├── Le2i/        ← Home_01_video_(1).npz ...
├── UR-FALL/     ← Subject4Activity1Trial1.npz ...
├── CAUCAFall/   ← Subject.1_Fall_forward.npz ...
└── UP-FALL/     ← Subject4Activity1Trial1.npz ...
```

### 单文件内容 (`data = np.load("xxx.npz")`)

| 键 | shape | dtype | 坐标系 | 说明 |
|----|-------|-------|--------|------|
| `landmarks` | `(T, 33, 3)` | float32 | [0,1] 归一化 | 33关节 (x, y, z)，MediaPipe 原生 |
| `world_landmarks` | `(T, 33, 3)` | float32 | 米，原点髋中心 | 33关节 3D 世界坐标 |
| `visibility` | `(T, 33)` | float32 | [0,1] | 每关节置信度 |
| `timestamps` | `(T,)` | int64 | ms | 帧时间戳 |
| `frame_width` | int | — | px | 原始帧宽 |
| `frame_height` | int | — | px | 原始帧高 |
| `labels` | `(T,)` | int64 | {0,1,2} | 三分类标签，阶段 1 已映射 |
| `video_id` | str | — | — | 视频标识 |

### 可用的特征计算资源

从缓存可以重建以下数据用于任意特征工程：

```
data["landmarks"]          → 全部 33 关节 2D (x,y,z)  → 骨架坐标、bbox、宽高比
data["landmarks"][:, :13]  → CORE_JOINTS_13 子集       → 当前 skeleton_26
data["world_landmarks"]    → 全部 33 关节 3D (米)       → 躯干角、步速、逐关节 3D 速度
data["visibility"]         → 每关节置信度               → 质量加权、低置信帧过滤
data["timestamps"]         → 帧时间戳                   → Δt、帧率自适应
data["labels"]             → 三分类标签                 → 窗口标签
```

---

## V1 Baseline 实际输入（29 维）

由 `features/feature_builder.py` → `FrameFeatureBuilder.build(result, gait_speed)` 逐帧构建。

### 索引 0: `gait_speed` — 步行速度

| 属性 | 值 |
|------|-----|
| 来源 | `features/gait.py` → `GaitSpeedExtractor.compute_per_frame()` |
| 坐标系 | `world_landmarks` (米) |
| 算法 | `foot_center = (LEFT_ANKLE + RIGHT_ANKLE) / 2` |
| 公式 | `V = ‖foot_center(t) − foot_center(t−12)‖ / Δt` |
| 不足 12 帧 | 返回 0.0 |
| 维度 | 1 |

### 索引 1: `trunk_angle` — 躯干倾角

| 属性 | 值 |
|------|-----|
| 来源 | `features/torso.py` → `TorsoAngleExtractor.extract_frame()` |
| 坐标系 | `world_landmarks` → `clone_with_3d_normalized()` 归一化至 [−1,1] |
| 算法 | 三平面 arctan2 + df 阈值择优 |
| 三平面 | θ₁(x-y 俯仰) V1=(0,1)、θ₂(y-z 侧倾) V1=(0,1)、θ₃(x-z 扭转) V1=(0,1)=z⁺ |
| df | `‖m1.x − m2.x‖` 在原始米坐标上 < 0.15m → min(θ₁,θ₂,θ₃) ; ≥ 0.15m → max |
| 论文 | Lau et al. (2022) §2.2.1 |
| 维度 | 1 |

### 索引 2: `aspect_ratio` — 人体宽高比

| 属性 | 值 |
|------|-----|
| 来源 | `features/skeleton.py` → `SkeletonFeatureExtractor.extract_frame()` |
| 坐标系 | `landmarks` [0,1] 归一化 2D |
| 公式 | `Rf = bbox_width / bbox_height` |
| 论文 | Lau et al. (2022) §2.2.3，原文 W/H 阈值笔误已修正 |
| 维度 | 1 |

### 索引 3-28: `skeleton_26` — 13 核心关节坐标

| 属性 | 值 |
|------|-----|
| 来源 | `PoseResult.landmarks` — MediaPipe 原生 33 关键点 |
| 坐标系 | [0,1] 归一化 2D |
| 关节集 | `CORE_JOINTS_13`（鼻、双肩、双肘、双腕、双髋、双膝、双踝） |
| 每关节 | `(x, y)` — 不取 z、不取 visibility |
| 固定顺序 | NOSE → L_SHOULDER → R_SHOULDER → L_ELBOW → R_ELBOW → L_WRIST → R_WRIST → L_HIP → R_HIP → L_KNEE → R_KNEE → L_ANKLE → R_ANKLE |
| 维度 | 26 |

### 汇总

```
索引   维度   名称            代码来源                       坐标系
[0]    1      gait_speed      GaitSpeedExtractor             world_landmarks (米)
[1]    1      trunk_angle     TorsoAngleExtractor            world_landmarks → [−1,1]
[2]    1      aspect_ratio    SkeletonFeatureExtractor       landmarks [0,1]
[3-28] 26     skeleton_26     PoseResult.landmarks            landmarks [0,1]
──────────────────────────────────────────────────────────────────────────
共计   29
```

| 特征类型 | 已有 | 说明 |
|----------|------|------|
| 空间位置 (x,y) | ✅ 26维 | 13 核心关节，[0,1] 归一化 |
| 脚部整体速度 | ✅ 1维 | gait_speed（脚踝中心 12 帧位移速度） |
| 躯干 3D 角度 | ✅ 1维 | TrunkAngleExtractor 三平面 arctan2 + df 择优 |
| 身体比例 | ✅ 1维 | aspect_ratio = bbox_w / bbox_h |
| **逐关节速度** | ❌ | 每个关节独立的 vx, vy |
| **逐关节加速度** | ❌ | 每个关节独立的 ax, ay |
| **关节夹角** | ❌ | 肘、膝等关节弯曲角度 |

---

## 输入特征维度规划

### V2 — 逐关节速度（+26 维 → 55 维）

```
v_t = p_t - p_{t-1}   (13关节 × 2 = 26维, 首帧 v_0 = 0)
```

最终单帧向量：

```
[gait_speed, trunk_angle, aspect_ratio, x0,y0,...,x12,y12, vx0,vy0,...,vx12,vy12]
      1            1             1              26                      26
                                    ← 原始骨架 (索引 3-28) →  ← 逐关节速度 →
```

**为什么**：当前 gait_speed 只有一个标量（双脚中心整体速度）。跌倒时不同关节速度变化不同——手腕加速下撑、头部加速下落、膝盖弯曲减速。逐关节速度让 LSTM 看到**身体各部位的运动差异**。

**实现方案**：不改 `FrameFeatureBuilder`。在 `build_from_cache.py` 阶段 2 中，`(T, 29)` 构建完成后追加速度列。

```
改动位置:
  sequence/velocity.py          ← 新建: add_velocity(features) → (T, 55)
  scripts/build_from_cache.py   ← +1行: feats = add_velocity(feats)
  scripts/build_config.py       ← +1字段: USE_VELOCITY = True (开关)
  不需要重跑 MediaPipe（缓存已有）
```

### V2.2 — 逐关节加速度（+26 维 → 55 维，不加速度）

```
a_t = v_t - v_{t-1} = p_t - 2·p_{t-1} + p_{t-2}   (13关节 × 2 = 26维, 首两帧 a_0=a_1=0)
```

最终单帧向量：

```
[gait_speed, trunk_angle, aspect_ratio, x0,y0,...,x12,y12, ax0,ay0,...,ax12,ay12]
      1            1             1              26                      26
                                    ← 原始骨架 (索引 3-28) →  ← 逐关节加速度 →
```

**为什么**：V2 velocity 无效可能是因为 LSTM 已经能从位置序列学到速度。但加速度（运动变化率）是二阶信息——踉跄时加速度剧烈波动，正常行走时稳定。LSTM 可能更难从原始位置自行提取二阶差分。

**实现方案**：`sequence/velocity.py` 新增 `add_acceleration()`，`build_from_cache.py` 加 `--use_acceleration` 标志。

```
改动位置:
  sequence/velocity.py          ← 新增 add_acceleration(features) → (T, 55)
  scripts/build_from_cache.py   ← +--use_acceleration 参数
  不需要重跑 MediaPipe
```

### V3 — 逐关节加速度（+26 维 → 81 维，速度+加速度同时加）

> V2.2 单独验证加速度有效性后再定是否做 V3。

```
a_t = v_t - v_{t-1}   (13关节 × 2 = 26维)
```

加速度捕捉**运动变化率**——正常行走加速度平稳，踉跄时剧烈波动。

**实现方案**：在 `sequence/velocity.py` 扩展 `add_acceleration()`，`build_from_cache.py` 加一行。

```
改动位置:
  sequence/velocity.py          ← 扩展: add_acceleration(features)
  scripts/build_from_cache.py   ← +1行: feats = add_acceleration(feats)
  build_config.py               ← USE_ACCELERATION = True
  不需要重跑 MediaPipe
```

### V4 — 关节角度（+N 维）

| 特征 | 维度 | 说明 |
|------|------|------|
| 肘关节角 | 2 | 肩-肘-腕夹角 (左/右) |
| 膝关节角 | 2 | 髋-膝-踝夹角 (左/右) |
| 躯干-大腿角 | 2 | 髋关节屈伸 |
| 头-躯干角 | 1 | 颈部倾斜 |
| symmetry | 1 | 左右不对称度（已有 SkeletonFeatureExtractor） |
| **合计** | **~8** | |

**实现方案**：新增 `features/joint_angle.py`，在 `FrameFeatureBuilder.build()` 中追加。需重跑 `build_from_cache.py`（不需要重跑 MediaPipe）。

```
改动位置:
  features/joint_angle.py       ← 新建: 关节角度计算
  features/feature_builder.py   ← build() 追加角度维度
  scripts/build_from_cache.py   ← 自动适配新维度
```

### V5 — 窗口扩展

```
window_size: 35 → 50 → 75
```

35 帧 @~30fps ≈ 1.1 秒。跌倒前失衡持续 2-3 秒，更长窗口可能捕捉更完整的前兆。

**实现方案**：只改 `build_config.py` 的 `WINDOW_SIZE`，重跑 `build_from_cache.py` 即可（秒级）。

```
改动位置:
  scripts/build_config.py    ← WINDOW_SIZE = 50 或 75
  scripts/build_from_cache.py ← 自动读取
```

---

## 命令速查

```
datasets/experiment/
├── v1/          ← 29维
├── v2/          ← 55维 +velocity
├── v2_2/        ← 55维 +acceleration
├── v3/          ← 81维 +velocity+acceleration
├── v4/          ← 36维 +joint_angles

model/lstm/
├── v1/run_01/
├── v2/run_01/
├── v2_2/run_01/
├── v3/run_01/
└── v4/run_01/
```

```bash
# ===== V1 (29维) =====
python scripts/build_from_cache.py -v v1
python scripts/train_lstm.py -v v1

# ===== V2 (55维, +velocity) =====
python scripts/build_from_cache.py -v v2
python scripts/train_lstm.py -v v2

# ===== V2.2 (55维, +acceleration) =====
python scripts/build_from_cache.py -v v2.2
python scripts/train_lstm.py -v v2.2

# ===== V3 (81维, +velocity+acceleration) =====
python scripts/build_from_cache.py -v v3
python scripts/train_lstm.py -v v3

# ===== V4 (36维, +joint_angles) =====
python scripts/build_from_cache.py -v v4
python scripts/train_lstm.py -v v4

# ===== V5 窗口消融 (29维, 不同窗口) =====
python scripts/build_from_cache.py -v v1 -w 25 --output datasets/experiment/v5_w25
python scripts/build_from_cache.py -v v1 -w 30 --output datasets/experiment/v5_w30
python scripts/build_from_cache.py -v v1 -w 35 --output datasets/experiment/v5_w35
python scripts/build_from_cache.py -v v1 -w 50 --output datasets/experiment/v5_w50
python scripts/build_from_cache.py -v v1 -w 75 --output datasets/experiment/v5_w75
python scripts/build_from_cache.py -v v1 -w 100 --output datasets/experiment/v5_w100
# 训练
python scripts/train_lstm.py --data_dir datasets/experiment/v5_w25 --input_dim 29 --run_dir model/lstm/v5_w25
python scripts/train_lstm.py --data_dir datasets/experiment/v5_w30 --input_dim 29 --run_dir model/lstm/v5_w30
python scripts/train_lstm.py --data_dir datasets/experiment/v5_w35 --input_dim 29 --run_dir model/lstm/v5_w35
python scripts/train_lstm.py --data_dir datasets/experiment/v5_w50 --input_dim 29 --run_dir model/lstm/v5_w50
python scripts/train_lstm.py --data_dir datasets/experiment/v5_w75 --input_dim 29 --run_dir model/lstm/v5_w75
python scripts/train_lstm.py --data_dir datasets/experiment/v5_w100 --input_dim 29 --run_dir model/lstm/v5_w100

# 调权重（覆盖默认 1,5,3）
python scripts/train_lstm.py -v v3 --weight_scheme none
python scripts/train_lstm.py -v v3 --weight_scheme balanced
python scripts/train_lstm.py -v v3 --f1_weights 1,3,1
```

---

### V2.2 — +acceleration (55维, 不加速度)

| Run | loss权重 | U_P | U_R | U_F1 | Macro F1 |
|-----|---------|-----|-----|------|-----------|
| 02 | none [1,1,1] | 0.495 | 0.546 | 0.519 | 0.783 |
| 03 | 1,3,1 | 0.521 | 0.619 | **0.566** | 0.794 |
| 01 | 1,5,3 | 0.459 | 0.580 | 0.512 | 0.793 |

**结论**: 加速度比 velocity 更差。最佳 U-F1=0.566 (vs V2 的 0.605)。二阶差分噪声更大，LSTM 更难利用。

---

## 实验矩阵

| 版本 | 新增特征 | 维度 | 窗口 | 目标 Unstable F1 |
|------|----------|------|------|-----------------|
| V1 | 原始29维 | 29 | 35 | U-F1=0.55 (实测) |
| V2 | +velocity | 55 | 35 | U-F1=0.55 → 无效 |
| V2.2 | +acceleration | 55 | 35 | U-F1=0.52 → 更差 |
| V3 | +velocity+acceleration | 81 | 35 | U-F1=0.49~0.55 → 无效 |
| **V4** | **+关节角度** | **36** | **35** | **U-F1=0.44 → 严重退化** |
| V5 | 窗口消融 25/35/50/75/100 | 29 | 25~100 | **w=25 最优** |

### V5 — 窗口消融实验结果

| 窗口 | 时间 | N (样本) | U样本 | U_P | U_R | U_F1 | Macro F1 |
|------|------|---------|-------|-----|-----|------|----------|
| **25** | 1.0s | 16,731 | 892 | 0.539 | **0.688** | **0.605** | 0.783 |
| 35 | 1.4s | 15,981 | 654 | 0.526 | 0.624 | 0.571 | 0.788 |
| 50 | 2.0s | 14,856 | 387 | 0.304 | 0.525 | 0.385 | 0.725 |
| 75 | 3.0s | 13,018 | 290 | 0.363 | 0.445 | 0.400 | 0.730 |
| 100 | 4.0s | 11,256 | 230 | 0.361 | 0.448 | 0.400 | 0.735 |

**结论: 短窗口更优。** 与预期相反——Unstable 是短暂过渡（~1s），长窗口混入无关帧稀释信号。

- U-F1: w=25 → 0.605，w=100 → 0.400，**窗口越长越差**
- U样本: w=25 → 892，w=100 → 230，长窗口直接丢弃了更多 Unstable 窗口

---

## 总结

### 特征工程: 全部无效或有害

| 实验 | 新增 | U-F1 | Δ | 结论 |
|------|------|------|---|------|
| V1 | 原始29维 | 0.555 | — | baseline |
| V2 | +velocity (55维) | 0.555 | **0** | 无效 — LSTM 已学到速度 |
| V2.2 | +acceleration (55维) | 0.519 | **−0.04** | 更差 — 二阶噪声大 |
| V3 | +both (81维) | 0.491 | **−0.06** | 维度膨胀，信号被淹没 |
| **V4** | **+关节角度 (36维)** | **0.441** | **−0.11** | **严重退化** |

**V4 为什么最差**: 关节角度特征严重依赖 world_landmarks 的 3D 精度。MediaPipe 单目 3D 估计本身有误差，角度计算对误差敏感（小位移→大角度变化）。7 维噪声特征加入后 LSTM 更难收敛。

### 权重调优: 唯一有效手段

| 权重 | U-F1 | 备注 |
|------|------|------|
| none [1,1,1] | 0.555 | baseline |
| sqrt | 0.548 | 略降 |
| **1,5,3** | **0.605** | ★ 最佳 |
| balanced | 0.567 | 召回高误报多 |
| 1,6,4 (V3) | 0.552 | V3 下略好但仍不如 V2+1,5,3 |

### Pre-VFall — 新数据集

```
run: model/lstm/v1_pv/run_01
data: 5 数据集 (Le2i+UR-FALL+CAUCAFall+UP-FALL+Pre-VFall)
窗口: 25帧, 29维, weight=[1,5,3]

              precision    recall  f1-score   support
      Normal     0.964    0.793    0.870     11663
    Unstable     0.610    0.727    0.664      1339    ← **史上最高**
        Fall     0.647    0.931    0.763      4098

Confusion Matrix:
             Pred N   Pred U   Pred F
    True N     9252      523     1888
    True U      166      974      199
    True F      183       99     3816
```

| 指标 | 4数据集 | +Pre-VFall | Δ |
|------|--------|-----------|----|
| U-F1 | 0.60 | **0.66** | +10% |
| U-Recall | 0.68 | **0.73** | +7% |
| U样本 | 654 | 1339 | ×2 |
| N-Recall | 0.93 | 0.79 | −15% |

**结论**: Pre-VFall 的 Abnormal 类别（37.5% 占比，含意识混乱/肌无力/眩晕等）让模型学会了更广泛的"异常先兆"。U-F1 0.66 是目前最高记录。Normal recall 下降是可接受的——误报可用状态机过滤，漏报才致命。

### 下一步

Pre-VFall 是数据侧最大的突破。后续重心回到 P0（Unstable 标签扩展）+ 模型调优。

**当前最佳**: V1 29维 + `--weight_scheme 1,5,3` + `window_size=25` + Pre-VFall 五数据集, U-F1=0.664（run: model/lstm/v1_pv/run_01）。
4数据集版 U-F1=0.605（run: model/lstm/v5_w25/run_01）。

后续方向应转向模型架构（Attention、Transformer）或数据增强，而非继续手工特征工程。

---

## 实验结果记录

### V1 — Baseline (29维, sqrt weights)

```
run: model/lstm/run_02 (旧), 后续统一到 model/lstm/v1/
data: datasets/experiment/v1/
weight: sqrt  [1.34, 3.55, 1.66]

              precision    recall  f1-score
      Normal     0.98      0.93      0.95
    Unstable     0.50      0.61      0.55    ← 核心指标
        Fall     0.84      0.93      0.88
```

### V2 — +velocity (55维)

```
run: model/lstm/v2/run_01
data: datasets_v2/
weight: sqrt  [1.34, 3.55, 1.66]

结果与 V1 完全一致 (Macro F1 0.795, U-F1 0.55)。
结论: 逐关节 velocity 无增益，LSTM 从位置序列已学到运动模式。
```

### V2 权重消融实验（55维，同一数据）

| Run | loss权重 | F1权重 | U_P | U_R | U_F1 | Test Macro F1 | 备注 |
|-----|---------|--------|-----|-----|------|---------------|------|
| 02 | none [1,1,1] | 1,1,1 | 0.551 | 0.558 | 0.555 | 0.787 | 基线 |
| 01 | sqrt | 1,1,1 | 0.499 | 0.607 | 0.548 | 0.795 | V2 默认 |
| 03 | balanced | 1,1,1 | 0.479 | **0.694** | 0.567 | 0.789 | 召回最高但误报多 |
| 04 | **1,5,3** | 1,1,1 | 0.546 | 0.679 | **0.605** | **0.827** | ★ 综合最佳 |
| 05 | 2,4,3 | 1,1,1 | 0.512 | 0.609 | 0.556 | 0.789 | — |
| 06 | 1,5,3 | 1,3,1 | 0.535 | 0.636 | 0.581 | 0.800 | F1加权反降 |

### 权重实验结论

| 发现 | 说明 |
|------|------|
| **最佳配置** | `--weight_scheme 1,5,3` (run 04): U-F1=0.605, U-recall=0.679 |
| 无权重 | U-F1=0.555，模型倾向忽略 Unstable |
| 过重 (balanced) | U-recall 最高 0.694，但 precision 降到 0.479，大量误报 |
| F1 加权选模型 | 1,3,1 反而不如默认 1,1,1——loss 权重已经够强，F1 再加权导致过拟合 |
| **U-F1 从 0.55→0.605** | 纯调权重提升 +10%，比加 velocity 特征 (+0%) 有效得多 |

### P1 — Label Smoothing（已完成）

**数据**: w=35 (因 v5_w25 未构建，临时用 v5_w35)

| smoothing | U_P | U_R | U_F1 | Macro F1 |
|-----------|-----|-----|------|----------|
| 0 (baseline) | 0.418 | 0.592 | 0.490 | 0.755 |
| 0.03 | 0.427 | 0.595 | 0.497 | 0.774 |
| **0.05** | **0.507** | 0.575 | **0.539** | **0.778** |
| 0.1 | 0.471 | 0.573 | 0.517 | 0.771 |

**结论**: smoothing=0.05 最佳。U-P 从 0.42→0.51 (+21%)，模型不再把不确定帧全推给 Fall。但绝对值（0.54）低于 w=25 baseline（0.60），**窗口长度影响 > 标签软化**。该实验应在 w=25 数据上重跑以确认实际收益。

**判定**: 跑完 4 个实验，挑 Unstable F1 最高的。都 ≤0.60 则无效。

**成本**: 不改数据、不改模型，一行代码。不需要重建数据集。

```bash
python scripts/train_lstm.py --data_dir datasets/experiment/v5_w35 --input_dim 29 --run_dir model/lstm/p1_smooth/s0   --label_smoothing 0
python scripts/train_lstm.py --data_dir datasets/experiment/v5_w35 --input_dim 29 --run_dir model/lstm/p1_smooth/s003 --label_smoothing 0.03
python scripts/train_lstm.py --data_dir datasets/experiment/v5_w35 --input_dim 29 --run_dir model/lstm/p1_smooth/s005 --label_smoothing 0.05
python scripts/train_lstm.py --data_dir datasets/experiment/v5_w35 --input_dim 29 --run_dir model/lstm/p1_smooth/s01  --label_smoothing 0.1
```

存放: `model/lstm/p1_smooth/s{value}/run_01/`

### P0 — Unstable 标签扩展（待定）

---

## 训练输出规范

每个 run 目录必须包含以下 5 个文件。缺任意一个视为训练未完成。

```
run_XX/
├── config.txt           ← 超参，纯文本
├── best_model.pt        ← 最佳 checkpoint (val wF1)
├── model_info.json      ← demo 自动适配
├── test_report.txt      ← classification_report + 混淆矩阵
└── training_curve.png   ← Loss + Acc + CM 三合一图
```

### config.txt 模板

```
input_dim=29
epochs=100
batch_size=64
lr=0.001
val_ratio=0.15
patience=15
datasets=['Le2i', 'UR-FALL', 'CAUCAFall', 'UP-FALL']
weight_scheme=1,5,3
f1_weights=1,1,1
label_smoothing=0.05
class_weights=[1.0, 5.0, 3.0]
device=cuda
```

### model_info.json 模板

```json
{
  "input_dim": 29,
  "window_size": 25,
  "version": "v5_w25",
  "weight_scheme": "1,5,3",
  "label_smoothing": 0.05
}
```

### test_report.txt 模板

```
Test macro F1: 0.8271

              precision    recall  f1-score   support
      Normal     0.9787    0.9315    0.9545     11494
    Unstable     0.5455    0.6789    0.6049       654
        Fall     0.8410    0.9314    0.8839      3833
    accuracy                         0.9182     15981
   macro avg     0.7884    0.8473    0.8271     15981
weighted avg     0.9260    0.9182    0.9209     15981

Confusion Matrix:
             Pred N   Pred U   Pred F
    True N    10707      287      500
    True U       82      444      128
    True F      151      112     3570
```

### 实验规则

1. **一次只改一个变量**
2. **固定模型结构不变**（2 层 LSTM，两层 hidden 均为 64）
3. **用 Unstable F1 + Recall 评判**，不看 Accuracy
4. **每个 run 目录 5 个文件齐全**才算完成

---

## 不应该做的事（当前阶段）

- ❌ 加环境风险 R_env（光照、地面材质）→ **已作废（2026-07-13）**：环境因子 HOME FAST 已加入，见 `docs/跌倒风险预测_汇报.md` §3；此条不再适用。
- ❌ 加生理因素 R_phys（年龄、病史）
- ❌ 改模型架构（加 Attention/Transformer/更大 LSTM）
- ❌ 改风险公式
- ❌ 引入更多数据集

**原因**：如果 LSTM + 姿态特征已经把 Unstable 学到 0.70+，后续加什么都是增量。如果姿态基础不牢，加环境/生理只会掩盖问题。

---

## 实验记录模板

每次实验在 `model/lstm/` 下留记录：

```
run_XX/
├── config.txt          ← 超参 + 特征维度说明
├── test_report.txt     ← 分类报告
├── training_curve.png  ← 曲线图
└── notes.txt           ← 实验目的、结论
```

### notes.txt 模板

```
实验: V2 — xy + velocity
目的: 验证速度特征对 Unstable 的提升
变更: features/feature_builder.py 增加 velocity 维度
结果:
  Unstable F1: 0.53 → 0.58
  Unstable recall: 0.42 → 0.51
  Fall recall: 0.88 → 0.86 (轻微下降)
结论: velocity 有帮助，继续加 acceleration

---

## ST-GCN (S1) — 独立并行实验（2026-07-20）

> **性质**：独立于 LSTM 主线的并行架构分支。用 ST-GCN 替换 LSTM 作姿态风险 B 模型，
> 数据/标签/窗口与 LSTM 对齐（仅把「每帧 29 维手工特征」换成「每帧 17 关节 3D 世界坐标」），
> 输入 `(N,3,35,17)`，回归头 sigmoid→risk∈[0,1]。LSTM 路径**零侵入**，两条线 A/B 公平。
> 代码见 `model/stgcn/`（graph.py / stgcn.py / build_skeleton.py / train_stgcn.py / eval_stgcn.py），
> 推理切换由 `demo/demo_gate.py` 读 `model_info.json` 的 `model_type` 字段分支。

### 运行配置

```
run:      model/stgcn/stgcn_v1/run_01
data:     datasets/experiment/skeleton_w35  (5 数据集: Le2i+UR-FALL+CAUCAFall+UP-FALL+Pre-VFall)
input:    (T=35, V=17, C=3) 3D 世界坐标，髋中心归一化
joints:   FALL_JOINTS_17 = CORE_JOINTS_13 + 双脚跟(29/30) + 双脚尖(31/32)
layers:   [64, 64, 128, 128, 256]  ST-GCN blocks
loss:     class_weighted   params=1,190,209
epochs:   60 (early stopping, patience=12)   用户本地 cmd 运行
```

### ⚠️ 两个 baseline 必须分清（避免误读）

| Baseline | 来源 | 窗口 | 5 数据集 | U-F1 |
|----------|------|------|---------|------|
| **同数据公平基线** | `model/lstm/risk_v1/run_03/metrics.json` | 35 | ✅ | **0.5788** |
| LSTM 全时期最佳记录 | `model/lstm/v1_pv/run_01`（见上「当前最佳」） | 25 | ✅ | 0.664 |

> `eval_stgcn.py` 第 219 行打印的 `"(LSTM 旧最佳基线 0.664)"` 是**写死的 w=25 全时期记录**，
> 不是同数据对比。下面 A/B 一律用 **w=35 的 0.5788** 作公平基线（data/窗口一致）。
> 另注：`run_03/metrics.json` 实际 U-F1=0.5788，与之前对话里"run_03=0.664"的口误不一致，以文件为准。

### A/B 五维对比（S1 vs LSTM-w35 基线）

| 指标 | LSTM (run_03, w35) | ST-GCN (S1, run_01) | Δ | 胜方 |
|------|--------------------|---------------------|---|------|
| 全局 R² | 0.6830 | **0.8171** | +0.134 | ✅ ST-GCN |
| Test MAE | 0.0882 | **0.0679** | −0.020 | ✅ ST-GCN |
| Test MSE | 0.0580 | **0.0335** | −0.024 | ✅ ST-GCN |
| **Unstable F1** | **0.5788** | 0.4491 | −0.130 | ❌ LSTM |
| Macro F1 | **0.7933** | 0.7598 | −0.034 | ❌ LSTM |
| AUC-ROC | 0.9598 | **0.9770** | +0.017 | ✅ ST-GCN |
| AUC-PR | 0.8224 | **0.9261** | +0.104 | ✅ ST-GCN |
| DANGER 平均提前 | 0.186s | **0.223s** | +0.037s | ✅ ST-GCN |
| DANGER 中位提前 | 0.080s | **0.160s** | +0.080s | ✅ ST-GCN |
| WARNING 平均提前 | **0.993s** | 0.881s | −0.112s | ⚠️ LSTM |
| 跌倒前触发 DANGER 比例 | 99.64% (n=278) | **100.00%** (n=278) | +0.36% | ✅ ST-GCN |
| 平滑性 ‖Δrisk‖ 均值 | 0.0084 | **0.0068** | 更平稳 | ✅ ST-GCN |
| 参数量 | — | 1,190,209 | — | — |

### 分数据集 Unstable F1

| 数据集 | LSTM (w35) | ST-GCN (S1) | Δ |
|--------|-----------|-------------|---|
| Le2i | 0.366 | 0.227 | −0.139 |
| UR-FALL | 0.352 | 0.269 | −0.083 |
| CAUCAFall | 0.238 | **0.270** | +0.032 ✅ |
| UP-FALL | 0.694 | 0.505 | −0.189 |
| Pre-VFall | 0.785 | 0.591 | −0.194 |

### 分区间 MAE（诊断 U-F1 塌陷的根因）

| risk 区间 | LSTM MAE | ST-GCN MAE | 解读 |
|-----------|----------|------------|------|
| [0, 0.3) 正常带 | 0.090 | **0.055** | ST-GCN 更准 |
| **[0.3, 0.6) 不稳定带** | **0.168** | 0.234 | ❌ ST-GCN 在中间带更散 |
| [0.6, 1.0] 跌倒带 | 0.067 | 0.067 | 持平 |

### 结论与诊断

**ST-GCN 是一个更强的*回归器*，但派生 U-F1 坍缩。** 它全面胜出的是：
- 连续风险拟合（R² +0.13、MAE/MSE 大降）、
- 排序能力（AUC-ROC/PR 显著提升，PR +0.10）、
- 预判提前量（DANGER 中位提前翻倍 0.08→0.16s）、
- 时序平滑性（Δrisk 更小，报警更稳）、
- 跌倒召回覆盖（278 个跌倒视频 100% 在跌倒前触发 DANGER，LSTM 99.64%）。

**但 Unstable F1 从 0.579 跌到 0.449**，根因很明确——**中间不稳定带 [0.3,0.6) 的 MAE 高达 0.234**（LSTM 0.168）。
ST-GCN 在低/高端带都很准，唯独在过渡带方差过大，导致以 0.3/0.6 阈值派生三分类时，
大量 Unstable 窗口被甩到 Normal 或 Fall 带，U_F1 因此塌陷。
这个现象和 LSTM 早期一致：**瓶颈在中间过渡带的回归锐度，不在特征维度/架构**。

### 下一步（S2 候选，不破坏 S1）

1. **Midband loss（最高优先级）**：LSTM 侧已有 midband 损失，专门在 [0.3,0.6) 加压。
   给 `train_stgcn.py` 加 `--loss midband`，预期是 U-F1 的最大单点杠杆。
2. **阈值/校准扫描**：ST-GCN 的 risk 分布与 LSTM 不同，0.3/0.6 硬切未必最优，
   在 metrics 的 roc_pr.scan 上按 F1 重选每类阈值。
3. **class_weighted → focal/midband**：当前 class_weighted 对稀有 U 类的*方差*欠约束。
4. **Ensemble（B = α·B_lstm + (1−α)·B_stgcn）**：ST-GCN 强回归/AUC/lead，LSTM 强 U-F1，
   二者风险都是标量，门控 `θ` 直接吃融合值即可，成本低、互补强。

> 状态：S1 已完成训练+评估全流程，产物齐全（`best_model.pt`/`model_info.json`/`metrics.json`/
> `confusion.png`/`risk_curves_sample.png`/`lead_time_hist.png`/`training_curve.png`）。
> 未做集成、未接 demo 默认路径（demo 仍默认 LSTM，S1 需显式 `model_type=stgcn`）。

---

## ST-GCN (S2) — 窗口/权重调整实验（2026-07-20）

> **目的**：用户观察到 demo「会误报」（蹲/坐等过渡动作被 ST-GCN 误判 DANGER）。
> 建议下调 `w_unstable` 5.0→2.0、`w_fall` 3.0→2.0、窗口 35→60，"平衡模型对过渡动作的敏感度"。
> 配置改动见 `scripts/build_config.py`：`REG_W_UNSTABLE=2.0 / REG_W_FALL=2.0 / STGCN_WINDOW=60`。

### 运行配置

```
run:      model/stgcn/stgcn_v1/run_02   (run 目录自动递增，保留 S1=run_01)
data:     datasets/experiment/skeleton_w60  (5 数据集，统一窗口=60)
input:    (T=60, V=17, C=3) 3D 世界坐标
weights:  class_weighted = [1.0, 2.0, 2.0]   ← 旧 S1 = [1.0, 5.0, 3.0]
loss:     class_weighted   params=1,190,209
epochs:   60 (early stopping, patience=12)   用户本地 cmd 运行
```

### ⚠️ 关键混淆：S2 同时改了「窗口」和「权重」两个变量

S2 相对 S1 同时变了 `窗口 35→60` **和** `权重 [1,5,3]→[1,2,2]`。
因此**无法把任何指标变化单独归因于权重调整**——`窗口 60` 本身也可能贡献了回归/平滑改善。
要隔离需补 2 个 ablation：
- `run_03`: w60 + 旧权重 [1,5,3]（纯窗口贡献）
- `run_04`: w35 + 新权重 [1,2,2]（纯权重贡献，预期会复现 U-F1 塌陷）

### A/B 三方对比（S2 vs S1 vs LSTM-w35 公平基线）

| 指标 | LSTM (run_03, w35) | ST-GCN S1 (w35,[1,5,3]) | ST-GCN S2 (w60,[1,2,2]) | S2 最优项 |
|------|--------------------|--------------------------|--------------------------|-----------|
| 全局 R² | 0.6830 | 0.8171 | **0.8519** | ✅ S2 |
| Test MAE | 0.0882 | 0.0679 | **0.0653** | ✅ S2 |
| band[0,0.3) MAE | **0.090** | **0.0549** | 0.0659 | ⚠️ S1 更优 |
| **band[0.3,0.6) MAE** | **0.168** | 0.2337 | 0.2574 | ❌ S2 最差 |
| band[0.6,1.0] MAE | 0.067 | 0.0667 | **0.0378** | ✅ S2 |
| Normal F1 | 0.929 | 0.9490 | **0.9615** | ✅ S2 |
| **Unstable F1** | **0.5788** | 0.4491 | 0.4187 | ❌ S2 最差 |
| Fall F1 | 0.872 | 0.8812 | **0.9078** | ✅ S2 |
| AUC-ROC | 0.9598 | 0.9770 | **0.9888** | ✅ S2 |
| AUC-PR | 0.8224 | 0.9261 | **0.9613** | ✅ S2 |
| DANGER 中位提前 | 0.080s | **0.160s** | 0.120s | ⚠️ S1 更优 |
| Smoothness ‖Δrisk‖ | 0.0084 | 0.0068 | **0.0051** | ✅ S2 |
| 跌倒前触发 DANGER | 99.64% (n=278) | **100.00%** (n=278) | **100.00%** (n=257) | ✅ 持平 |

### 分数据集 Unstable F1（S2 触目惊心的塌方）

| 数据集 | LSTM (w35) | ST-GCN S1 | ST-GCN S2 | 备注 |
|--------|-----------|-------------|-------------|------|
| Le2i | 0.366 | 0.227 | 0.277 | S2 略升 |
| UR-FALL | 0.352 | 0.269 | **0.0000** | ❌ 完全塌方 |
| CAUCAFall | 0.238 | 0.270 | **0.0000** | ❌ 完全塌方 |
| UP-FALL | 0.694 | 0.505 | 0.450 | 降 |
| Pre-VFall | 0.785 | 0.591 | 0.667 | 升 |
| **汇总** | **0.5788** | 0.4491 | **0.4187** | S2 最差 |

> UR-FALL / CAUCAFall 的 Unstable F1 = **0.0000**（S1 时分别为 0.269/0.270）。
> 说明 S2 配置下模型**几乎不再预测这两个数据集里的任何 Unstable 窗口**——全都推到了 N 或 F 端。

### 结论与诊断（S2 未达成目标，且暴露问题）

**用户的假设被数据推翻。** 原假设："高 `w_unstable`(5.0) 让模型'宁可错杀'，蹲/坐等过渡动作触发高昂 unstable 惩罚 → 过度敏感 → 误报"。

实际机制：class_weighted 损失里 `w_unstable` 高 = 模型**答错**不稳定窗口时损失大 = 模型被**强迫去学中间带**。
把 `w_unstable` 5→2，**撤掉了这个强迫力**，模型于是"偷懒"把中间风险推到两端（Normal 或 Fall），导致：
- 中间带 [0.3,0.6) MAE 不降反升（0.234→0.257）；
- **Unstable F1 从 0.449 跌到 0.419**（与用户想减少误报的初衷相反）；
- UR-FALL / CAUCAFall 直接归零——这两个数据集的失稳模式被彻底丢弃。

**为什么 R²/AUC/Fall F1 反而全涨？** 因为`w_fall` 也同步 3→2（相对更均衡）+ `窗口 60` 给了更长时序上下文 →
整体回归更平滑、跌倒端更准（band[0.6,1] MAE 0.067→0.038，Fall F1 0.881→0.908）。
**S2 = 更好的跌倒检测器，但更差的不稳定检测器。**

**对 demo「会误报」的实际意义：**
- 真跌倒：S2 更干净（high band 更准、Fall F1 更高）→ 漏报更少 ✅
- 过渡动作（蹲/坐）：中间带更乱 + Unstable 几乎不预测 → 这些窗口大概率被推到 **Normal(<0.3)** 或偶尔 **Fall(≥0.6)**。
  - 若推到 Normal 端 → 风险被**低估**，误报减少但**真危险的过渡动作也预警不足**（漏警）；
  - 若推到 Fall 端 → **DANGER 误报反而增多**。
  - 从 U-F1 塌方看，更可能是"推到 Normal"，即**把过渡动作当成正常**——这不等于解决误报，是另一种失职。
- 由于 demo 已加 **EMA 平滑 + min-hold=3 去抖**，单帧毛刺被压住，但**持续型**过渡动作的走向仍需实跑 `run_02` 验证。

### 下一步（S3 候选，治本）

1. **Midband loss（优先级最高）**：在 [0.3,0.6) 单独加 sharp 惩罚（不是整体降权重），专门锐化中间带。
   这正是 S1 时标记的 S2 优先级①，但上次被"降权重"方案替代了。**降权重是错的方向，midband loss 才对。**
   给 `train_stgcn.py` 加 `--loss midband`，预期把中间带 MAE 和 U-F1 同时拉回。
2. **Ensemble（B = α·B_lstm + (1−α)·B_stgcn）**：LSTM 的 U-F1=0.579 明显高于 ST-GCN 任一变体（S1 0.449 / S2 0.419），
   ST-GCN 强回归/AUC/lead。二者风险皆标量，门控 θ 直接吃融合值，互补最强、成本低。
3. **回退一部分**：保留 `窗口=60`（有益），但把 `w_unstable` 提到 **3**（折中），即 [1,3,2]，看能否兼顾回归与失稳。
4. **补 ablation**：`run_03`(w60+[1,5,3]) / `run_04`(w35+[1,2,2]) 隔离窗口 vs 权重贡献，确认 S2 的回归增益来源。

> 状态：S2 已完成训练+评估，产物齐全（run_02 全套）。结论：**S2 配置方向错误（降权重反而害了失稳检测），
> 不应作为最终方案**；窗口=60 本身有益，应保留并转向 midband loss 或 ensemble。
```
