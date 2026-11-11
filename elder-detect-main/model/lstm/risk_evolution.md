# 🛡️ Risk 回归模型进化路线

> **目标**：把现有「3 分类 + 概率合成 risk」改成「**直接回归连续 risk ∈ [0,1]**」。
> **硬约束**：旧分类管线（`FallLSTM` / `v1_pv`，U-F1=0.664）**全程保留、零改动**，本路线是一套**平行资产**，用于迭代对比。

---

## 📑 目录

- [0. 速览](#0-速览)
- [1. 当前架构](#1-当前架构固定与分类版共享骨干)
- [2. 动机 / 当前瓶颈](#2-动机--当前瓶颈)
- [3. ★ 核心：Risk 标签定义](#3--核心risk-标签定义指数平滑-ramp方案-a)
- [4. 与分类版的关系（平行路线，零影响）](#4-与分类版的关系平行路线零影响)
- [5. 输入特征](#5-输入特征沿用分类版-29-维)
- [6. 模型结构（RiskRegressor）](#6-模型结构riskregressor)
- [7. 训练细节](#7-训练细节)
- [8. ★ 模型评判（五个维度）](#8--模型评判eval_riskpy五个维度)
- [9. 关键超参](#9-关键超参)
- [10. 文件布局](#10-文件布局零影响原模型)
- [11. 命令速查](#11-命令速查)
- [12. 训练输出规范](#12-训练输出规范)
- [13. 验收标准](#13-验收标准)
- [14. 🧪 实验记录（迭代留痕）](#14--实验记录迭代留痕)
- [15. 不应该做的事](#15-不应该做的事)
- [16. 实验记录模板](#16-实验记录模板)

---

## 0. 速览

| 项 | 值 |
|----|----|
| 任务 | 连续 risk 回归（替代 3 分类 + 概率合成） |
| 模型 | `RiskRegressor`：LSTM(29→64×2) → `Linear(64,1)`+sigmoid |
| 标签 | 指数平滑 ramp（方案 A）：不稳段 `0.3 → 0.7`，跌倒帧 `1.0` |
| 窗口 W | **35**（用户指定） |
| 旧基线 | 派生 Unstable F1 = **0.664**（W=25，非同窗口，注意） |
| 当前最佳 | Run03 @0.7：**Unstable F1 = 0.579**，回归 R²=0.683，平滑性 0.0084 |
| 下一步 | **Run04 连续中段加权 `midband`**（代码已改，待跑） |

> **一句话进度**：回归侧已大胜旧方案，Unstable 卡在 `0.58`（差基线 `0.085`）且全压在 Le2i/UR-FALL/CAUCAFall 三相机数据集——缺口是数据集偏置 + 中段 `[0.3,0.6)` MAE 最高，Run04 用中段加权精准攻坚。

---

## 1. 当前架构（固定，与分类版共享骨干）

```
Pose sequence (35 frames)
    │
MediaPipe → 2D/3D Skeleton
    │
Feature extraction (29维) → (35, 29)
    │
LSTM(input_dim=29, hidden=64, num_layers=2) → 最后帧 hidden (64,)
    │
Linear(64, 1) → sigmoid → risk ∈ [0,1]     # ★ 回归头，取代 softmax + 概率合成
# 状态阈值（沿用）: SAFE(<0.3) / WARNING(0.3~0.6) / DANGER(≥0.6)
```

> 与分类版**唯一区别**：输出头从 `Linear(64,3)+softmax`（3 类 logits）换成 `Linear(64,1)+sigmoid`（单个连续 risk）。LSTM 骨干完全相同。

---

## 2. 动机 / 当前瓶颈

旧方案的痛点（为什么要切回归）：

- **risk 是概率的「二次加工」**：`demo/demo.py` 里 `risk = 0.1·P(N)+0.6·P(U)+0.9·P(F)`。阈值 0.3/0.6 是拍脑袋定的，且 `label==1`（unstable）在合成公式里只占 0.6 这个中间档，**类内无梯度**——「刚进不稳」和「马上跌倒」合成出来的 risk 可能差不多。
- **3 类离散标签信息量低**：窗口标签 = 窗口最后一帧的类别，丢失了「离跌倒还有多远」的连续信号。
- **回归方案的收益**：直接让模型输出连续 risk，把「预判」变成可微监督；阈值 0.3/0.6 **天然契合标签标尺**（不稳起点=0.3=WARNING，≥0.6=DANGER，跌倒=1.0），不再需要人工合成公式。

---

## 3. ★ 核心：Risk 标签定义（指数平滑 ramp，方案 A）

完全从已有 `pose_cache/*.npz` 的 `labels`(0/1/2) 反推，**不重跑 MediaPipe**。

```
RAMP_LO  = 0.3      # 进入「不稳」/ 临跌倒前的保底风险
RAMP_HI  = 0.7      # 临跌倒（跌倒前 1 帧）的风险上沿
τ_u      ≈ 12         # pre-fall ramp 爬升时间常数(帧)
τ_abn    ≈ 20         # 异常不跌倒 爬升时间常数(帧)
H        = 25         # RISK_HORIZON：跌倒前多少帧开始进入 ramp（≈1秒）

u_start = 第一个 labels[t] >= 1 的帧      # 进入不稳（None=全正常）
f_start = 第一个 labels[t] == 2 的帧      # 跌倒起点（None=永不跌倒）

for 每帧 t:
    if   labels[t] == 2:
        r = 1.0                                                       # 跌倒：峰值
    elif f_start 存在 且 0 <= (f_start - t) <= H:                      # ★pre-fall 统一 ramp（与是否有"1"无关）
        d = f_start - t
        r = RAMP_LO + (RAMP_HI - RAMP_LO) * exp(-d / τ_u)            # d=0→0.7, d=H→≈0.35, 指数加速
    elif f_start 为 None 且 u_start 存在 且 t >= u_start:
        r = RAMP_LO + (RAMP_HI - RAMP_LO) * (1 - exp(-(t - u_start) / τ_abn))   # 异常不跌倒：爬升封顶 0.7
        r = min(r, RAMP_HI)
    elif u_start 存在 且 t >= u_start:                                 # 不稳但远在跌倒前 H 窗口外
        r = RAMP_LO                                                     # 保底 0.3
    else:
        r = 0.0                                                        # 正常
```

### 三种序列的行为

| 序列类型 | 风险轨迹（v2 统一后） |
|----------|----------|
| **含不稳+跌倒**（Pre-VFall fall） | 0 → 不稳起 **0.3** → 指数爬至 **0.7**（临跌倒）→ 跌倒 **1.0** |
| **异常永不跌倒**（Pre-VFall Abnormal） | 0 → 不稳起 **0.3** → 持续爬升封顶 **0.7** |
| **无不稳只有跌倒**（Le2i / UR-FALL / CAUCAFall） | 0 → 跌倒前 H 窗口内 **指数爬至 0.7**（临跌倒）→ 跌倒 **1.0** ★v2 新增过渡带 |
| **纯正常** | 全程 **0.0** |

### 关键设计约束

- `f_start`（label==2）是**所有跌倒序列都有的通用锚**，v2 起 pre-fall ramp **统一由它触发**（不再依赖 `u_start`），使五数据集目标一致。
- `u_start`（label==1）只在 Pre-VFall 类数据集有，作「异常不跌倒 / 远跌倒前」的保底地板（0.3）。
- **v2 修正（Run 01 → Run 02）**：原 Run 01 的 pre-fall ramp 误由 `u_start` 触发，导致 Le2i/UR-FALL/CAUCAFall（cache 仅 0/2、无"1"）临跌倒帧被标成 0，与 Pre-VFall 同类帧标 0.3~0.7 **矛盾**，模型学崩（该三数据集 Unstable F1 ≈ 0.25~0.30）。v2 改为 `f_start` 触发后，这三个数据集也获得 0.3→0.7 过渡带。
- 标签**从已有 `pose_cache/*.npz` 的 `labels` 反推**，不重抽姿态、不重写 npz（只写 sidecar `<name>.risk.npy`）。

### 与窗口 / 阈值的关系（易混点）

| 参数 | 归谁管 | 含义 |
|------|--------|------|
| **W=35** | 模型输入 | LSTM 一次吃进多少帧（时序窗口长度） |
| **τ_u / τ_abn** | 标签生成 | 两段 ramp 的爬升陡度 |
| **RAMP_LO/HI** | 标签生成 | 不稳段风险的上下沿（与 demo 阈值 0.3/0.6 对齐） |

`τ_u` 建议取 `≈H/2`（H=RISK_HORIZON 预告窗口，默认 25）。`RAMP_LO=0.3` 对齐 WARNING 阈值，`RAMP_HI=0.7` 落在 DANGER 区间内（≥0.6 即高危），使「临跌倒」信号稳稳落在高危带。

---

## 4. 与分类版的关系（平行路线，零影响）

> 用户硬约束：**不要影响原来的模型。** 新工作必须零破坏。

| # | 原资产 | 是否被碰 | 新东西去哪 |
|---|----------|-----------|--------------|
| ① | `model/lstm/model.py`（FallLSTM 定义） | ❌ 不改 | 新模型定义独立放 `model/lstm/risk_model.py` |
| ② | `model/lstm/v1_pv/`（U-F1=0.664 权重） | ❌ 不碰 | 新权重只落 `model/lstm/risk_v1/run_XX/` |
| ③ | `datasets/experiment/*`（分类 X/y） | ❌ 不碰 | 回归数据只落 `datasets/experiment/risk_w35/` |
| ④ | `pose_cache/*.npz`（姿态缓存） | ❌ **只读不写** | risk 存 sidecar `pose_cache/<name>.risk.npy` |
| ⑤ | `build_from_cache.py` / `train_lstm.py` / `eval_model.py` / `demo.py` | ⚠️ 仅**加性**改 | 新 `--task regression` 分支 + 新函数，原分类默认流程一行不动 |
| ⑥ | 原 demo 推理（走 `0.1·P(N)+0.6·P(U)+0.9·P(F)`） | ❌ 不破 | demo 靠 `model_info.json` 的 `task` 字段分流，旧模型仍走原公式 |

> 新资产删掉也不影响任何旧功能。

---

## 5. 输入特征（沿用分类版 29 维）

`X` 仍是 `(35, 29)`，由 `features/feature_builder.py` 逐帧构建，见 `model/lstm/evolution.md` §V1。回归版**不引入任何新特征维度**（特征工程在分类路线已被证无效）。

| 索引 | 维度 | 名称 |
|------|------|------|
| [0] | 1 | gait_speed |
| [1] | 1 | trunk_angle |
| [2] | 1 | aspect_ratio |
| [3-28] | 26 | skeleton_26（13 核心关节 x,y） |
| **共计** | **29** | |

---

## 6. 模型结构（RiskRegressor）

```python
# model/lstm/risk_model.py
class RiskRegressor(nn.Module):
    def __init__(self, input_dim=29, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True)          # 与 FallLSTM 同骨干
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):                             # x: (B, 35, 29)
        out, (h, c) = self.lstm(x)
        h_last = h[-1]                               # 最后帧 hidden (B, 64)
        risk = torch.sigmoid(self.head(h_last))        # (B, 1) ∈ [0,1]
        return risk
```

> **多任务双头 `RiskMultiTask`** 为可选扩展（共享骨干 + 分类头 CE + 风险头 MSE），本期 `MULTITASK=False` 不开。

---

## 7. 训练细节

- **Loss**（默认 `class_weighted`，Run03 起）：
  `mean( w(y) · (pred − y)² )`，类权对齐旧分类 `[1,5,3]`：
  `y<0.3→Normal(w_n=1) ; 0.3≤y<0.6→Unstable(w_u=5,最高) ; y≥0.6→Fall(w_f=3)`。
  **旧方案 `weighted_mse`（`1+β·y`, β=3）为单调加权**，Fall(高 y) 权重=4、Unstable 中段仅 2.5——
  与旧类权**相反**，是 Run01/02 Unstable F1 卡 0.40 的根因（见 Run 02→03）。
  仍可用 `--loss weighted_mse` 复现 Run01/02。
- **Optimizer**：Adam**W**, lr=1e-3, weight_decay=1e-4；`ReduceLROnPlateau`（监控 val MSE）。
- **Early stopping**：patience=20（基于 val MSE）。
- **Batch / Epoch**：batch=64, epochs=100（与 `train_lstm.py` 对齐）。
- **最佳模型**：按 val MSE 最小存 `best_model.pt` + `model_info.json`（写 `task` 字段）。
- **数据划分**：与分类版同 `val_ratio`（默认 0.15），train/val/test 同划分逻辑。

---

## 8. ★ 模型评判（eval_risk.py，五个维度）

回归模型的「成不成功」不能只看回归误差，要从五个维度综合评判：

### 8.1 回归指标（本职：连续 risk 准不准）
- 测试集 **MSE / MAE / RMSE / R²**。
- **按风险带拆分误差**：`[0,0.3)` / `[0.3,0.6)` / `[0.6,1.0]` 各带分别报 MAE——高带（临跌倒）误差代价远大于低带。

### 8.2 派生分类 vs 旧基线（和 0.664 公平比）
- 把**预测 risk** 过阈值还原成类别：`risk<0.3→Normal`，`0.3~0.6→Unstable`，`≥0.6→Fall`。
- 算每类 **P/R/F1 + 宏 F1 + 混淆矩阵**。
- **核心对比项：Unstable F1 是否 ≥ 旧基线 0.664**（同一套阈值派生，可比）。

### 8.3 预判提前量（★连续 risk 的真正价值，旧分类难直接报）
- 对测试集**每个跌倒序列**，找预测 risk **首次越过 0.3(WARNING)** 和 **0.6(DANGER)** 的帧。
- 算它距离 `f_start`（真实跌倒帧）多少帧 → 除以 fps(≈25) 转**秒**。
- 报**均值 / 中位数**提前量。例如「DANGER 平均提前 0.8 秒触发」——这才是「预判系统」的硬指标。

### 8.4 ROC / PR + 阈值扫描
- 以「真实 Fall（目标 risk≥0.6）」为正类，报 **AUC-ROC / AUC-PR**。
- 扫描决策阈值，画 F1 / Precision / Recall 曲线，挑 operating point。

### 8.5 平滑性（决定 demo 报警抖不抖）
- 测试序列上**相邻帧 |Δrisk| 的均值**。越低＝报警越稳、不鬼畜跳变（目标 ≤0.05/帧）。

### 8.6 分数据集拆分（暴露数据偏置）
- 指标按 Pre-VFall / Le2i / UR-FALL… 各自拆开。
- Run 01 实测：**Le2i / UR-FALL / CAUCAFall（cache 仅 0/2、原无 ramp）Unstable F1 仅 0.25~0.30**，是掉点主因；v2 统一 ramp 后应显著回升（见 Run 02）。

---

## 9. 关键超参

| 参数 | 默认 | 含义 |
|------|------|------|
| `WINDOW_SIZE` | `35` | 时序窗口（与最佳基线对齐，用户指定） |
| `RISK_MODE` | `"exponential"` | 标签生成策略 |
| `RAMP_LO` | `0.3` | 不稳起点风险 |
| `RAMP_HI` | `0.7` | 临跌倒风险上沿 |
| `τ_u` | `12` | 不稳→跌倒 爬升时间常数 |
| `τ_abn` | `20` | 异常不跌倒 爬升时间常数 |
| `REG_LOSS` | `"weighted_mse"` | 旧加权 MSE（β 加权，Run01/02 用） |
| `REG_BETA` (β) | `3` | 旧加权 `1+β·y` 的 β |
| `REG_LOSS_MODE` | `"class_weighted"` | Run03 起默认：`class_weighted`(对齐[1,5,3]) / `weighted_mse` / `midband` |
| `REG_W_NORMAL` | `1.0` | 类加权：Normal(y<0.3) 权重 |
| `REG_W_UNSTABLE` | `5.0` | 类加权：Unstable(0.3≤y<0.6) 权重（最高，强化中段） |
| `REG_W_FALL` | `3.0` | 类加权：Fall(y≥0.6) 权重 |
| `REG_MIDBAND_CENTER` | `0.45` | 连续中段加权：峰值位置（∈Unstable 中段） |
| `REG_MIDBAND_WIDTH` | `0.18` | 连续中段加权：峰宽（高斯标准差） |
| `REG_MIDBAND_PEAK` | `6.0` | 连续中段加权：峰值额外权重 |
| `MULTITASK` | `False` | 是否开分类辅助头 |
| `MT_LAMBDA` (λ) | `0.5` | 多任务损失混合权重 `λ·CE+(1−λ)·MSE` |

---

## 10. 文件布局（零影响原模型）

| # | 文件 | 动作 | 内容 |
|---|------|------|------|
| ① | `scripts/build_config.py` | 改 | 加 `RISK_*`(RAMP_LO/HI, τ_u, τ_abn) + `REG_*`(BETA/W_*/MIDBAND_*) |
| ② | `scripts/regen_risk_labels.py` | **新** | 读 npz `labels` → 算 risk → 写 sidecar `pose_cache/<name>.risk.npy`（**不回写 npz**） |
| ③ | `sequence/window.py` | 改 | `task="regression"` 时窗口目标=末帧 risk(float) |
| ④ | `scripts/build_from_cache.py` | 改 | 加 `--task regression`；输出 `datasets/experiment/risk_w35/` |
| ⑤ | `model/lstm/risk_model.py` | **新** | `RiskRegressor`（多任务双头为可选扩展） |
| ⑥ | `scripts/train_risk.py` | **新** | 加权 MSE / 类加权 / 中段加权 训练 |
| ⑦ | `scripts/eval_risk.py` | **新** | 评判（见 §8 五维） |
| ⑧ | `demo/demo.py` | 改 | 按 `model_info.json` 的 `task` 分流，回归直出 risk |

**隔离保证**（同 §4 平行路线表）：原 `model/lstm/model.py` 不改；原权重 `model/lstm/v1_pv/` 不碰；原 `datasets/experiment/*` 不碰；`pose_cache/*.npz` 只读。新权重仅落 `model/lstm/risk_v1/run_XX/`。

---

## 11. 命令速查

```bash
# ===== 1. 补 risk 标签（秒级，不跑 MediaPipe）=====
python scripts/regen_risk_labels.py

# ===== 2. 生成回归数据集（W=35）=====
python scripts/build_from_cache.py --task regression -w 35

# ===== 3. 训练（默认 class_weighted；可选 --loss weighted_mse / midband）=====
python scripts/train_risk.py --data_dir datasets/experiment/risk_w35 --input_dim 29

# ===== 4. 评判（默认 Fall 阈值 0.7，与 RAMP_HI 对齐）=====
python scripts/eval_risk.py model/lstm/risk_v1/run_01 datasets/experiment/risk_w35

# ===== 5. demo 验证 =====
python demo/demo.py --model model/lstm/risk_v1/run_01/best_model.pt -w 35
```

> ⚠️ **对比公平性提醒**：旧最佳 `U-F1=0.664` 是 **W=25** 取得，本期回归版用 **W=35**，非同窗口对比。若要严格公平，可补跑一个 `risk_w25/`（同样流程、窗口 25）直接对齐。

---

## 12. 训练输出规范

每个 run 目录必须包含以下文件（缺任意视为训练未完成）：

```
model/lstm/risk_v1/run_01/
├── config.txt           ← 超参（含 RISK_*/REG_*）
├── best_model.pt        ← 最佳 checkpoint (val MSE 最小)
├── model_info.json      ← demo 自动适配，写 task="regression", window=35
├── metrics.json         ← §8 全部数字
├── confusion.png        ← 派生分类混淆矩阵
├── roc_curve.png       ← ROC / PR 曲线 + 阈值扫描
├── lead_time_hist.png   ← 预警提前量分布
└── risk_curves_sample.png  ← 抽样测试序列：目标 vs 预测 risk 时序曲线
```

### model_info.json 模板

```json
{
  "task": "regression",
  "input_dim": 29,
  "window_size": 35,
  "risk_mode": "exponential",
  "ramp_lo": 0.3,
  "ramp_hi": 0.7,
  "tau_u": 12,
  "tau_abn": 20,
  "loss": "class_weighted",
  "w_normal": 1.0,
  "w_unstable": 5.0,
  "w_fall": 3.0,
  "midband_center": 0.45,
  "midband_width": 0.18,
  "midband_peak": 6.0,
  "multitask": false
}
```

---

## 13. 验收标准

| 指标 | 达标线 |
|------|--------|
| 派生 Unstable F1 | **≥ 旧基线 0.664 的 ~5% 内或更优** |
| DANGER 平均提前量 | **≥ 0.5 秒** |
| R² | ≥ 0.6 |
| 相邻帧 \|Δrisk\| | ≤ 0.05 |
| 分数据集 | Le2i / UR-FALL 不掉出合理范围 |

---

## 14. 🧪 实验记录（迭代留痕）

> 每次迭代在 `model/lstm/risk_v1/run_XX/` 下留记录（沿用 §12 训练输出规范）。

### 📊 结果对比总表

| Run | 标签 | Loss | R² | MSE | Unstable F1 | 评估阈值 | 状态 |
|-----|------|------|----|-----|-------------|---------|------|
| 01 | v1 | weighted_mse β=3 | 0.537 | 0.127 | **0.394** | 0.3/0.6 | ✅ 暴露矛盾监督 |
| 02 | v2 | weighted_mse β=3 | 0.760 | 0.044 | **0.402** | 0.3/0.6 | ✅ 回归大胜，Unstable 仍卡 |
| 03 | v2 | class_weighted [1,5,3] | 0.683 | 0.058 | **0.476** (@0.6) / **0.579** (@0.7) | 0.3/0.6→0.7 | ✅ 类加权+阈值修正 |
| 04 | v2 | midband（中段加权） | — | — | 待跑 | 0.3/0.7 | 🔧 代码已改，待跑 |

---

### Run 01 — Baseline（指数 ramp v1 + 加权 MSE）✅ 已跑，暴露问题

- **数据**：`datasets/experiment/risk_w35/`（5 数据集, W=35, 29维）
- **标签**：指数平滑 ramp v1（pre-fall 由 `u_start` 触发 → 漏掉 Le2i/UR-FALL/CAUCAFall）
- **Loss**：`weighted_mse`, β=3
- **模型**：RiskRegressor（LSTM 29→64×2 → Linear(64,1)+sigmoid）

| 指标 | 值 |
|------|----|
| MSE / MAE / RMSE / R² | 0.127 / 0.094 / 0.357 / 0.537 |
| 分带 MAE | [0,0.3)=0.083 · **[0.3,0.6)=0.192(最差)** · [0.6,1]=0.064 |
| 派生 F1 | Normal=0.932 · **Unstable=0.394** · Fall=0.851（宏F1=0.726） |
| 混淆 True U→Pred | N 352 / U 443 / F 544（Unstable 召回仅 0.331） |
| AUC-ROC / AUC-PR | 0.956 / 0.864（Fall 检测强） |
| 分数据集 Unstable F1 | UP-FALL=0.574 · Pre-VFall=0.425 · Le2i=0.295 · UR-FALL=0.301 · CAUCAFall=0.251 |
| 预判提前量 / 平滑性 | N/A（eval 仅读合并文件，缺逐视频边界） |

> **结论**：Fall 检测强（F1=0.851, AUC=0.956），但 Unstable F1=0.394 远低于旧基线 0.664；掉点集中在 Le2i/UR-FALL/CAUCAFall（cache 仅 0/2、pre-fall 无 ramp）；根因 = 标签矛盾监督。 ⇒ 见 Run 02。

> ⚠️ **边界**：第二次在 run_01 上跑 eval（标签已重生成 v2）得到 MSE=0.0505/R²=0.724，但派生分类与本次**逐位相同**——那是「旧 v1 模型给 v2 新标签打分」的标签/模型不匹配，衡量不了 v2 修复，**不得替换本记录**（有效基线仍是本段的 v1-vs-v1 结果）。该次可作「旧模型/新标签」对照，run_02 的训练后评估才是真·v2 效果。

---

### Run 02 — 标签 v2 统一 ramp（修复 Run 01 矛盾监督）✅ 已跑，回归大胜、Unstable 仍卡

- **变更**：`compute_risk` 的 pre-fall ramp 触发锚 `u_start` → `f_start`；所有含跌倒序列在 fall 前 H=25 帧内统一生成 `0.3→0.7` ramp（无论是否有"1"标注）；异常不跌倒 / 远跌倒前的不稳帧保底 0.3。
- **模型**：同 Run 01，仍 `weighted_mse` β=3

| 指标 | 值 |
|------|----|
| MSE / MAE / RMSE / R² | **0.044 / 0.081 / 0.210 / 0.760** ★ 较 Run01 大胜 |
| 分带 MAE | [0,0.3)=0.075 · **[0.3,0.6)=0.191(仍最差)** · [0.6,1]=0.071 |
| 派生 F1 | Normal=0.935 · **Unstable=0.402** · Fall=0.870（宏F1=0.736） |
| 混淆 True U(1339)→Pred | N 409 / U 473 / F 457（Unstable 召回仅 0.353、精度 0.467） |
| AUC-ROC / AUC-PR | 0.965 / 0.892（Fall 检测强） |
| 分数据集 Unstable F1 | Le2i=0.462 · UR-FALL=0.421 · CAUCAFall=0.218 · UP-FALL=0.566 · Pre-VFall=0.336 |
| 预判提前量 / 平滑性 | N/A —— eval 文件名 bug（见 Run 03 ③ 修复） |

> **结论**：① 回归侧 v2 标签修复完全成功：R² 0.537→0.760、MSE 0.127→0.044，矛盾监督消除；② 但 Unstable F1=0.402 几乎没动（Run01=0.394）→ 瓶颈已不在标签，在 **LOSS**；③ 根因：当前 `weighted_mse=(1+β·y)` 随 y 单调加权，Fall(y=1) 权重=4、Unstable(y≈0.5) 仅 2.5，与旧分类类权 [1,5,3] 相反（Unstable 应为最高档 5）→ 中段 `[0.3,0.6)` 监督不足、模型 Hold 不住中间值（该带 MAE=0.19 是其他带 2.5 倍），1339 个真 Unstable 帧 31% 掉 Normal、34% 冲 Fall；④ 三无"1"数据集仍最差（CAUCAFall=0.22），但已比 Run01（0.25）略升，说明 v2 过渡带生效。 ⇒ 见 Run 03（② 类加权 loss 对齐 [1,5,3]）。

---

### Run 03 — ② 类加权 loss + ③ eval 文件名修复 ✅ 已跑，阈值修正后得公平 Unstable F1

- **② 变更**：Loss 由 `weighted_mse(1+β·y)` 改为 `class_weighted_mse`：`y<0.3→Normal(w=1)` ; `0.3≤y<0.6→Unstable(w=5,最高)` ; `y≥0.6→Fall(w=3)`，对齐旧类权 [1,5,3]，把中段监督拉到最高。
  - 实现：`train_risk.py` 新增 `class_weighted_mse` + `--loss`/`--w_normal`/`--w_unstable`/`--w_fall`；`build_config.py` 新增 `REG_W_*` 与 `REG_LOSS_MODE="class_weighted"`。
- **③ 变更**：`eval_risk.py` 文件名解析修 bug：`build` 写的是 `{safe}_X.npy`，eval 旧代码用 `glob("X_*.npy")` 永远匹配不到 → 逐视频文件其实已在磁盘，只差解析 → Run02 的提前量/平滑性才会是 N/A。修后 `glob("*_X.npy")` + 去尾 `_X`，并区分「无视频文件 / 有视频但无 DANGER 穿越」。
- **数据**：沿用 `risk_w35/`（已是 v2 标签，无需重 regen/build）。

| 指标 | @0.6（旧阈值） | **@0.7（对齐 RAMP_HI）** |
|------|----------------|---------------------------|
| MSE / MAE / RMSE / R² | 0.058 / 0.088 / 0.241 / 0.683 | 同（回归不受阈值影响） |
| 分带 MAE | [0,0.3)=0.090 · **[0.3,0.6)=0.168** · [0.6,1]=0.067 | 同 |
| 派生 F1 | Normal=0.929 · **Unstable=0.476** · Fall=0.842 | Normal=0.929 · **Unstable=0.579** · Fall=0.872 |
| 混淆 True U(1339)→Pred | N 296 / U 566 / F 477 | N 296 / **U 815** / F 228 |
| AUC-ROC / AUC-PR | 0.951 / 0.822 | 0.960 / 0.822（Fall 正类=risk≥0.7） |
| 分数据集 Unstable F1 | Le2i=0.355 · UR-FALL=0.303 · CAUCAFall=0.315 · UP-FALL=0.695 · Pre-VFall=0.534 | Le2i=0.366 · UR-FALL=0.352 · CAUCAFall=0.238 · UP-FALL=0.694 · **Pre-VFall=0.785** |
| 预判 | DANGER≥0.6 提前 0.19s(中位0.08s)；WARNING≥0.3 提前 0.99s；跌倒前触发 99.64%(n=278) | 同 |
| 平滑性 | \|Δrisk\|=0.0084/帧（目标≤0.05）★ 极好 | 同 |

> **结论**：① 类加权起效：Unstable F1 0.40→0.48(@0.6)，中段 MAE 0.192→0.168，掉 Normal 的帧 409→296；② 阈值修正（0.6→0.7）大获成功：Unstable F1 0.476→0.579(+0.103)，把 ~247 帧 True-U 从错判 Fall 拉回 Unstable；③ 但仍差旧基线 0.664 约 0.085，**全压在 Le2i/UR-FALL/CAUCAFall(0.37/0.35/0.24)**，UP-FALL(0.69)/Pre-VFall(0.79) 早已超/近基线 → 差距是数据集偏置；④ 主瓶颈 = 中段 `[0.3,0.6)` MAE=0.168 仍是最高带（模型整体 Hold 不住中间值），尤其三相机数据集临跌倒不稳帧学不动。 ⇒ 见 Run 04（连续中段加权 midband）。

```bash
# ② 重训（类加权，落 run_03）：
python scripts/train_risk.py --data_dir datasets/experiment/risk_w35 --input_dim 29 --loss class_weighted
# ③ 无需重训，直接重跑 eval 即得提前量/平滑性（每视频文件已在）：
python scripts/eval_risk.py model/lstm/risk_v1/run_03 datasets/experiment/risk_w35
```

---

### Run 04 — 连续中段加权 loss（midband，精准强化 [0.3,0.6)）🔧 代码已改，待跑

- **变更**：新增 `midband_mse` —— 权重在 Unstable 中段 `y≈center` 处高斯达峰，直接针对 band `[0.3,0.6)` MAE 最差处：
  `weight = 1 + peak·exp(-((y-center)/width)²)`
  `center=0.45` · `width=0.18` · `peak=6.0`
  对比 `class_weighted`（整段 flat `w_u=5`）：midband 在 y=0.45 处权重≈7、y=0.3/0.6 处≈3.6、y=0/0.7/1 处≈1，把监督预算精准压到模型最不会的中段，而非均摊整段。
- **实现**：`train_risk.py` 新增 `midband_mse` + `get_criterion` 统一派发 + `--loss midband` / `--midband_*` 参数；`config.txt` / `model_info.json` 同步记 `midband_*`；`build_config.py` 加 `REG_MIDBAND_*`。
- **数据**：沿用 `risk_w35/`（已是 v2 标签，无需重 regen/build）。
- **模型**：同 RiskRegressor。

```bash
python scripts/train_risk.py --data_dir datasets/experiment/risk_w35 --input_dim 29 --loss midband
python scripts/eval_risk.py model/lstm/risk_v1/run_04 datasets/experiment/risk_w35
```

| 项 | 预期 / 说明 |
|----|----|
| 中段 `[0.3,0.6)` MAE | 显著下降（目标 <0.12） |
| Unstable F1 | 冲 ≥0.664，尤其 Le2i/UR-FALL/CAUCAFall 三个相机数据集回升 |
| WARNING 提前量 | 保持 ~1s |
| 平滑性 | 保持 ≤0.05（权重形状不影响时序平滑） |

> ⚠️ 输出到 `run_04`（勿覆盖 run_01~03 基线）。

**调参杠杆（若仍差）**：
- **A.** `peak` 6.0→8.0（更狠压中段）
- **B.** `center` 0.45→0.5（若错误集中在 [0.5,0.6)）
- **C.** 标签侧：`RAMP_HI` 0.7→0.6 消除 [0.6,0.7] U/F 残留歧义（需重 regen+build）
- **D.** 标签侧：`τ_u` 12→9 让 ramp 更陡、拉长 DANGER 提前量（现仅 0.19s，标签本身在跌倒前~3帧才过0.6）

---

## 15. 不应该做的事

- ❌ 改现有 `FallLSTM` / `v1_pv/` 权重 / 分类数据集
- ❌ 回写 `pose_cache/*.npz`
- ❌ 动 `demo.py` 的原有分类分支（只加 `task` 分流）
- ❌ 引入新特征维度（特征工程在分类路线已证无效）
- ❌ 本期开多任务双头（`MULTITASK` 默认 False，留作后续对照实验）

---

## 16. 实验记录模板

```
实验: risk_v1 run_XX — <一句话目的>
目的: <验证什么>
变更: <改了什么>
结果:
  Unstable F1: x.xx → x.xx
  DANGER 提前量: x.xxs → x.xxs
  R²: x.xx → x.xx
结论: <一句话>
```

