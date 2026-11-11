> ⚠️ **本文档已作废（superseded）**：其所描述的 `RuleEnvRisk`（房间危险度表 `ENV_ROOM_BASE` / `RISK_ADD_LIGHT` / `floor_hazard` 三查表）＋「`E` 加性进 `risk_short` 分数」均已被 **2026-07-13 的 HOME FAST 重构** 取代。当前环境模型是 `model/env/env_risk_model.py` 的 `HomeFastModel`（HOME FAST 25 题，单一 E，阈值调节），v0 的房间查表思路已被统一规则表替代。本文件仅作**历史留痕**，切勿据此实现。最新设计以 `docs/跌倒风险预测_汇报.md` §3 为准。

# 🌍 EnvRisk v0（R0 规则映射）设计方案

> **隶属**：Phase-2 多分量复合风险（`model/lstm/risk_env_evolution.md` 的 §4 落地版）。
> **本文件范围**：仅 EnvRisk 的 **v0 / R0 规则映射** 这一最小可跑版本，目标是**立刻打通双输出融合 pipeline**，不引入新训练模型、不碰 Phase-1 任何资产。
> **已锁定决策**（来自前序对话）：
> 1. **感知输入 = 数据集元标签**（房间类型由已知布景给出，零推理成本、零新标签）。
> 2. **作用范围 = 仅推理/融合**（env 只在 `fuse_risk.py` 与 `risk_姿态` 加性融合，**不写进 RiskRegressor 训练标签**）。
> 3. **双模型原则**：姿态一个模型（`RiskRegressor`）、环境一个模型（`EnvRisk`）。

---

## 📑 目录

- [0. 速览](#0-速览)
- [0.5 与《跌倒前预测可行性报告》的关系](#05-与跌倒前预测可行性报告的关系)
- [1. 定位与边界](#1-定位与边界)
- [2. 整体数据流](#2-整体数据流)
- [3. 感知层（v0 = 元标签，零模型）](#3-感知层v0--元标签零模型)
- [4. 风险映射层（R0 规则表）](#4-风险映射层r0-规则表)
- [5. 存储与对齐（隔离）](#5-存储与对齐隔离)
- [6. 融合层（fuse_risk.py）](#6-融合层fuse_riskpy)
- [7. 文件布局](#7-文件布局)
- [8. 配置项（build_config.py 新增）](#8-配置项build_configpy-新增)
- [9. 验证方案（v0 无 env 标签）](#9-验证方案v0-无-env-标签)
- [10. 验收标准](#10-验收标准)
- [11. 落地顺序（实施步骤）](#11-落地顺序实施步骤)
- [12. 与 R1/R2 的衔接](#12-与-r1r2-的衔接)
- [13. 待确认 / 开放点](#13-待确认--开放点)

---

## 0. 速览

| 项 | 值 |
|----|----|
| 版本 | EnvRisk **v0** = R0 规则映射 |
| 感知输入 | **数据集元标签**（房间类型）+ 可选帧亮度（需 `--video_root`） |
| 新训练模型 | **无**（纯查表 + 线性危险度增量；`EnvRisk` 以**模型接口**形态存在，v0 为规则实现） |
| 输出 | 每帧 `risk_环境 ∈ [0,1]`，分即时 / 静态两路 |
| 作用范围 | **仅推理 / 融合**，不动 `RiskRegressor` 训练 |
| 存储 | sidecar `pose_cache/<stem>.env_risk.npy` + `.<stem>.env_meta.json` |
| 依赖 | 仅 `numpy` + `PoseCache` 扫描；帧亮度可选 `opencv` |
| 状态 | 🔧 设计完成（已对齐可行性报告），待写代码 |

---

## 0.5 与《跌倒前预测可行性报告》的关系

> 参考文件：`fall-pre-event-prediction-feasibility-report.md`（用户 2026-07-12 提供）。

本报告把"老年人跌倒前预测"拆成**四类风险源**，并给出"规则 baseline → 传统 ML → 深度时序"三阶段路线。它**直接验证并丰富了本设计方案**，对应关系如下：

| 报告的风险源 | 本方案对应的模型 / 组件 | 实现形态（v0） |
|---|---|---|
| ① 行为/姿态风险 B | **姿态模型** = `RiskRegressor`（Phase-1 已训） | 训练 LSTM，独立出分 |
| ② 功能/生理**代理**风险 F | 功能代理提取器（骨架派生，规则） | 规则，从骨架特征算 |
| ③ 环境风险 E | **环境模型** = `EnvRisk`（本文件） | v0 = R0 规则，挂模型接口 |
| ④ 个体基础风险 P | 档案/问卷规则器 | 规则，从元数据算 |

**报告对本设计的 6 点关键支撑 / 修正：**

1. **✅ v0 规则版正是报告推荐的 Phase-1**：§7.1"第一阶段：规则 + 可解释加权评分，适合当前项目快速落地，优先推荐"；§5.3"环境建议先用用户选择，比赛演示更稳定"；§11.4"环境识别自动化难…第一版使用用户输入"。→ **彻底解决了"环境一个模型能不能做"的纠结**：v0 的 `EnvRisk` 以**规则实现挂模型接口**存在，正是报告主张的第一步，不阻塞、零数据。

2. **🔧 术语严谨化（重要）**：报告 §3.3 / §11.4 强调——骨骼点**不能直接测量**心率/血压等真实生理值，应称为**"功能/生理代理指标"**（平衡、步态、起立能力、疲劳趋势）。→ 本方案原 `risk_生理` 一律改为 **`risk_功能代理(F)`**，避免评审时被质疑"用骨骼点测生理"。

3. **⚠️ 环境是放大器，不是独立触发器**（报告 §5.4）：环境风险**不应单独触发 critical**，而应作为行为风险的**放大器**。→ v0 融合层新增"环境联动放大"规则（§6.5）。

4. **📊 四源加权默认权重**（报告 §7.1）：`final = 0.45·B + 0.25·F + 0.15·E + 0.15·P`。→ 作为融合层权重初值（§6）。

5. **🏷️ 四级输出 + 分期**（报告 §2.2）：`safe / watch / warning / critical` + `pre_event / in_event / post_event`。→ 融合层新增四级分层与风险分期（§6.6）。

6. **🔬 场景分层验证**（报告 §10.3）：按场景（卫生间/暗光…）分别统计召回。→ 验证方案新增按场景分层指标（§9）。

---

## 1. 定位与边界

**v0 在 Phase-2 里做什么**

```
Phase-2 复合风险 (四源)
 ├─ B  risk_行为姿态  (Phase-1 RiskRegressor, 已训 run_03/04)   ← 不动，训练 LSTM
 ├─ F  risk_功能代理  (骨架派生: 步态/平衡/起立/疲劳, 规则)        ← v0 规则提取
 ├─ E  risk_环境      (本文件 EnvRisk v0 R0 规则)                ← 本次新增
 └─ P  risk_个体基础  (问卷/档案元数据, 规则)                     ← v0 规则，远期可接入
```

> **★ 双模型原则（用户 2026-07-12 明确）**：**姿态一个模型、环境一个模型**。
> - **姿态模型** = `RiskRegressor`（Phase-1，已训 run_03/04），独立出 `risk_行为姿态`，本期不碰。
> - **环境模型** = `EnvRisk`（本文件本期落地对象），独立出 `risk_环境`。
> - 二者是**两个解耦的独立模型**，各自有输入/权重/产物，**仅在融合层晚期加权相加**。任一模型可单独重训 / 替换 / 升级（如 EnvRisk 从 v0 规则升 R1 训练头），不影响另一个。
> - v0 的 `EnvRisk` 虽是 **R0 规则实现**，但仍以**模型接口**形态存在（如 `EnvRiskModel` 类，`RuleEnvRisk` 为 v0 实现、`TrainedEnvRisk` 为 R1 实现），对外统一 `forward(感知属性) → risk_环境`，满足"环境一个模型"的结构要求。
> - **F（功能代理）/ P（个体基础）是规则组件**，非训练模型；它们与两个模型解耦，可单独调参与替换。

**硬约束（继承 Phase-1，v0 同样遵守）**
- 原 `FallLSTM` / `v1_pv` / 分类数据集 / `pose_cache/*.npz` **零改动**；`*.npz` 只读。
- 新资产独立：只写 `*.env_risk.npy` / `*.env_meta.json` sidecar，删掉不影响旧功能。
- **env 不进训练标签**：`regen_risk_labels.py` / `build_from_cache.py` / `train_risk.py` 一律不碰。env 仅在融合阶段与已训 `risk_行为姿态` 加性合并。

**v0 为什么"粗但够用"**
- 现有 5 数据集无环境标注 → R0 规则映射**不需要标签**，立刻可跑（且正是报告推荐的 Phase-1）。
- 先验证"双输出融合 pipeline 是否通畅 + env 是否真能拉开报警/监护分"，再上 R1 数据驱动。

---

## 2. 整体数据流

```
                         pose_cache/<stem>.npz  (只读, 已有)
                                     │ 扫描得到 stem 列表
                                     ▼
        ┌──────────────── EnvPerceive (scripts/env_perceive.py) ──────────────┐
        │  ① room_type  ← VIDEO_ROOM_MAP[stem] 或 DATASET_ROOM_DEFAULT[ds]      │
        │  ② light_level(t) ← [可选] 读视频帧亮度  (无 --video_root 则跳过,权重0) │
        │  ③ floor_hazard(t) ← [v0 关闭] 预留接口, ENV_FLOOR_ENABLED=False      │
        │  ④ risk_环境(t) = clamp( BASE[room_type]                              │
        │                        + ENV_LIGHT_W · RISK_ADD_LIGHT(light_level)      │
        │                        + ENV_FLOOR_W · floor_hazard , 0, 1 )           │
        └──────► 写 sidecar: <stem>.env_risk.npy (每帧) + <stem>.env_meta.json ┘
                                     │
                                     ▼
        ┌──────────────── FuseRisk (scripts/fuse_risk.py) ───────────────────┐
        │  risk_行为姿态(t) ← 已训 RiskRegressor(run_03/04) 滑窗预测→逐帧对齐   │
        │  risk_功能代理(t)   ← 骨架特征规则提取 (步态/平衡/起立/疲劳)           │
        │  risk_个体基础      ← 元数据规则 (年龄/跌倒史/独居...)  [可选]          │
        │  risk_short(t)  = clamp( 0.45·B + 0.25·F + 0.15·E , 0, 1 )        │
        │      + 环境联动放大 (见 §6.5)                                          │
        │  risk_long      = clamp( 0.15·E_static + 0.25·F_trend + 0.15·P ,0,1)│
        │  分级: safe/watch/warning/critical ; 分期: pre/in/post_event           │
        │  报警: risk_short ≥ FUSE_DANGER(0.6) → WARNING(0.3) 提醒            │
        └──────► metrics.json + 抽样曲线 PNG + 合成注入对比 + 场景分层 ──────────┘
```

---

## 3. 感知层（v0 = 元标签，零模型）

### 3.1 属性定义

| 属性 | 来源 | 范围 | 喂入 | v0 是否启用 |
|------|------|------|------|-------------|
| `room_type` | **元标签映射**（数据集/视频已知布景） | 离散枚举 | 即时 + 静态 | ✅ 必用 |
| `light_level(t)` | [可选] 视频帧平均亮度归一 | [0,1]，1=亮 0=暗 | 即时 | ⚠️ 需 `--video_root` |
| `floor_hazard(t)` | [预留] 近场湿滑/杂物启发式 | [0,1] | 即时 | ❌ v0 关闭 |

> **为什么 v0 用元标签而非预训练分类器**：用户已选"数据集元标签（推荐）"——零推理成本、零新标签、立跑；房间类型是静态属性，逐视频赋一次即可，无需逐帧模型。预训练场景分类器定为 **R0.5 可选升级**（见 §12）。

### 3.2 房间类型映射（默认假设，可逐视频覆盖）

`DATASET_ROOM_DEFAULT`（按数据集名，已知布景假设）：

| 数据集 | 默认 room_type | 理由 |
|--------|----------------|------|
| Le2i | `home` | 居家跌倒场景 |
| UR-FALL | `clinical` | 实验室/床边监护布景 |
| CAUCAFall | `home` | 多房间居家 |
| UP-FALL | `home` | 居家 ADL 活动 |
| Pre-VFall | `home` | 日常活动 |

`VIDEO_ROOM_MAP`（逐 stem 覆盖，v0 默认空；后续人工校正或 R1 反标时填）：

```python
VIDEO_ROOM_MAP = {
    # "Coffee_room_01_video_(1)": "kitchen",
    # "Stairs_03_video_(2)": "stairs",
}
```

### 3.3 帧亮度（可选，⚠️ 受源视频可用性限制）

仅当传入 `--video_root` 时启用：
- 逐帧读图 → 转灰度 → 均值亮度 `L ∈ [0,255]` → `light_level = L/255`。
- 无视频源时 `light_level` 视为充足（危险度增量=0），`risk_环境` 退化为纯 `BASE[room_type]`（每帧常数，仍可融合）。

> **⚠️ 项目现状（已核实）**：`pose_cache/` 有全部 5 数据集的 `.npz` 姿态缓存，但**这 5 个数据集的源 RGB 视频不在仓库内**（仅 `test/video/` 下有少量 demo 片段）。因此 v0 **实际只能跑纯元标签模式**（`risk_环境 = BASE[room_type]`，逐帧常数）。`light_level` / `floor_hazard` / R1 的 VLM 打标都**依赖帧图像**，需先重新取得源视频（见 §13-③）。
> 开发期可用 `test/video/*`（真实视频有帧）验证 `--video_root` 亮度分支；5 数据集则走纯元标签。
>
> 注：这与可行性报告 §11.4 的判断一致——"环境识别自动化难，第一版使用用户输入"，故 v0 纯元标签是合理且被报告支持的设计。

---

## 4. 风险映射层（R0 规则表）

### 4.1 公式

```
risk_环境(t) = clamp( BASE[room_type]
                    + ENV_LIGHT_W · RISK_ADD_LIGHT(light_level(t))
                    + ENV_FLOOR_W · floor_hazard(t) ,  0, 1 )
```

其中：
- `BASE[room_type]`：**静态分量**，喂 `risk_long` / 融合里的 `E_static`。
- 含 `light_level`/`floor_hazard` 的全值：**即时分量**，喂 `risk_short` 里的 `E`。

### 4.2 房间危险度表 `BASE`（R0 核心）

与可行性报告 §5.3 的"场景类型风险"方向一致（卫生间/厨房/楼梯最高），本表用 `[0,1]` 归一，便于与 `[0,1]` 的姿态风险直接加权融合：

| room_type | BASE | 报告对应(additive点) | 临床/常识依据 |
|-----------|------|---------------------|---------------|
| `stairs` | **0.70** | （报告未列，阶梯风险最高） | 台阶跌落风险最高 |
| `bathroom` | **0.60** | bathroom +25（最高） | 湿滑 + 硬地 + 独处 |
| `kitchen` | **0.40** | kitchen +15 | 水渍 + 利器 + 瓷砖 |
| `bedroom` | **0.30** | bedroom +10 | 夜间起床、地毯 |
| `living` | **0.20** | living_room +5 | 相对开阔 |
| `hallway` | **0.30** | corridor +10 | 门槛/杂物 |
| `clinical` | **0.25** | （监护环境） | 监护环境但设备多 |
| `home`(默认) | **0.30** | — | 居家综合基线 |
| `outdoor` | **0.15** | — | 本项目暂不涉及 |

> 排序与报告完全一致：卫生间 ≥ 厨房 ≥ 卧室/走廊 ≥ 客厅，验证了危险度先验的合理性。这些值既是 v0 的"专家知识"，也是 **R1 弱标签的先验种子**（见 §12）。

### 4.3 光照危险度增量 `RISK_ADD_LIGHT`

对齐可行性报告 §5.2 的离散分级（`brightness = mean(gray)/255`）：

```
分级（与报告一致）:
    L > 0.45        → normal  (危险度增量 0)
    0.25 ≤ L ≤ 0.45 → dim     (轻度危险度增量)
    L < 0.25        → dark    (重度危险度增量)

RISK_ADD_LIGHT(L):
    if L < ENV_LIGHT_DARK_TH (0.25):   return (ENV_LIGHT_DARK_TH - L) * ENV_LIGHT_W
    if L > ENV_LIGHT_GLARE_TH (0.85): return (L - ENV_LIGHT_GLARE_TH) * ENV_LIGHT_W * 0.5
    return 0
```
- 过暗（如夜间卧室）显著抬升即时风险；眩光小幅危险度增量。
- `ENV_LIGHT_W=0.6` → 全黑最多 +0.15（约对应报告 dark +20 的量级）。

### 4.4 地面危险 `floor_hazard`（v0 关闭）

预留接口：`ENV_FLOOR_ENABLED=False` 时权重恒 0，不影响输出；R0.5/R1 可打开（近场湿滑/杂物启发式或分类器）。

---

## 5. 存储与对齐（隔离）

| 文件 | 内容 | 说明 |
|------|------|------|
| `pose_cache/<stem>.env_risk.npy` | `float32` 数组，长度 = 该视频帧数 | 每帧 `risk_环境` |
| `pose_cache/<stem>.env_meta.json` | `{room_type, base, light_mean, n_frames, fps, has_video}` | 可解释追溯 |

- **只读 `*.npz`**：`env_perceive.py` 仅扫描 `pose_cache/*.npz` 拿 stem 列表，**不读取/不修改** npz 内部。
- **帧对齐**：`env_risk.npy` 长度与同视频 `labels` 帧数一致，融合时按索引直接对齐。
- **删除安全**：sidecar 全删，Phase-1 / 分类功能毫发无损。

---

## 6. 融合层（fuse_risk.py）

### 6.1 四源加权（对齐可行性报告 §7.1）

> 报告给出可解释加权评分默认权重：`final = 0.45·B + 0.25·F + 0.15·E + 0.15·P`（总和为 1.0）。

**短期报警（秒级，喂即时分量）**
```
risk_short(t) = clamp( W_B · B(t) + W_F · F(t) + W_E_IM · E(t) , 0, 1 )
```
- `B(t) = risk_行为姿态(t)`：已训 `RiskRegressor` 滑窗(W=35, stride=1)预测 → 逐帧对齐。
- `F(t) = risk_功能代理(t)`：骨架特征规则提取（步态稳定性/平衡/起立能力/疲劳趋势，见 §6.2）。
- `E(t) = risk_环境即时(t)`：含光照/地面危险度增量的全值。
- 默认 `W_B=0.45, W_F=0.25, W_E_IM=0.15`（来自报告）；总和 0.85，余量留给放大项（§6.5）。

**长期监护分（天/周级）**
```
risk_long = clamp( W_E_ST · E_static + W_F_LT · F_trend + W_P · P , 0, 1 )
```
- `E_static = BASE[room_type]`：静态环境基线。
- `F_trend`：步态/平衡的长窗退化趋势（v0 可先用逐视频均值代理）。
- `P`：个体基础风险（元数据规则，见 §6.3），v0 可选接入。
- 默认 `W_E_ST=0.15, W_F_LT=0.25, W_P=0.15`。

> 加性融合 = 各分量独立出分再加权，**模块解耦**：EnvRisk 从 R0 升 R1 不影响 PostureRisk；F/P 规则调整也不影响两个模型。

### 6.2 `risk_功能代理(F)` 规则提取（骨架派生，v0 规则版）

> 严格遵循报告 §3.3 / §4：**不称"生理"**，统称"功能代理指标"。v0 用规则从已有骨架特征算，不需新标注。

| F 子指标 | 骨架算法（报告 §4 对应） | v0 规则映射 |
|---|---|---|
| 步态稳定性 | `gait_variability = std(step_interval)/mean` (§4.3.4) | 变异性高 → F↑ |
| 平衡能力 | `sway_index` 归一 (§4.4.1) | 摆动大 → F↑ |
| 起立能力 | `sit_to_stand_time` (§4.5.1) | 耗时长 → F↑ |
| 疲劳趋势 | `fatigue_proxy_score` (§4.7) | 后段变差 → F↑ |

- v0 实现：复用 `FeatureEngine` 已有 `sway_index` / `com_velocity` 等，做 min-max 归一 + 加权求和 → `F(t) ∈ [0,1]`。
- 远期：F 可升级为单独的小型回归头（仍属"姿态/功能模型"一侧，不另立训练模型）。

### 6.3 `risk_个体基础(P)` 规则（元数据，v0 可选）

对齐报告 §6.2 的最小问卷字段（`age / fall_history_12m / uses_walking_aid / living_alone / …`）：
```
P = min( 10·[age≥75] + 10·[age≥85] + 25·[fall_history_12m≥1]
       + 10·[uses_walking_aid] + 10·[living_alone] + 15·[dizziness] , 100 ) / 100
```
- v0 默认 `W_P=0.15` 但 **P 输入为空时权重自动归零**（只融合 B/F/E），保证无元数据也不报错。
- 个体基础风险用于**调阈值**而非替代视觉（报告 §6.3）：高 P 时 `FUSE_WARN`/`FUSE_DANGER` 各下调 0.05，更早预警。

### 6.4 复用 RiskRegressor（零重训）

- `fuse_risk.py` 读 `run_dir/model_info.json` 拿 `input_dim` / `window` / `task=regression`。
- 对每个视频：`features`(来自 npz) 滑窗 → `RiskRegressor` 前向 → 每窗口 1 个 risk → 映射到窗口中心帧 → 逐帧 `B(t)`。
- **不加载训练代码、不更新权重**，纯推理。

### 6.5 ⚠️ 环境联动放大（报告 §5.4，关键修正）

> 环境风险**不应单独触发 critical**，只能作为行为/功能风险的放大器。

```
# 在 §6.1 加权求和之后追加：
if E(t) >= ENV_AMP_TH (0.5) and B(t) >= BEH_WARN_TH (0.4):
    risk_short = clamp( risk_short + ENV_AMP (0.08) , 0, 1 )   # 环境高危 × 行为不稳 → 显著抬升
if room_type == "bathroom" and F(t) >= F_UNSTABLE_TH (0.5):
    risk_short = clamp( risk_short + ENV_AMP_BATH (0.10) , 0, 1 )
if light_level == "dark" and F_gait_variability_high:
    risk_short = clamp( risk_short + ENV_AMP_DARK (0.08) , 0, 1 )
```
- 这保证：纯环境高危（如老人不在场、仅场景是浴室）**不会**误报 critical；只有"环境高危 + 行为/功能不稳"才显著推高。
- 放大项幅度 `ENV_AMP*` 为可配置超参（§8）。

### 6.6 🏷️ 四级输出 + 风险分期（报告 §2.2）

将 `[0,1]` 的 `risk_short` 映射为可解释等级（对齐报告 `0–30/30–50/50–75/75–100` 分段）：

| risk_short | 等级 | 含义 | 系统动作 |
|---|---|---|---|
| `[0, 0.30)` | **safe** | 稳定 | 无 |
| `[0.30, 0.50)` | **watch** | 轻度风险 | 记录趋势 |
| `[0.50, 0.75)` | **warning** | 高风险前兆 | 提醒（家属远程确认 / 开灯清障） |
| `[0.75, 1.0]` | **critical** | 即将/正在跌倒 | 立即报警 |

**风险分期**：
- `pre_event`：`risk_short` 处于 watch/warning 且呈上升趋势、行为不稳但未跌倒。
- `in_event`：`B(t)` 骤升（高 com 速度 + 身高骤降）→ 进入 critical。
- `post_event`：检测到已跌倒（复用 Phase-1 Fall 检测）后进入，转事后响应。

**可解释输出**（对齐报告 §2.2 JSON）：输出 `trigger_factors`（如 `bathroom_scene` / `lateral_sway_high` / `gait_variability_high`）与 `recommended_actions`，满足比赛"解释为什么预警"的要求。

### 6.7 对照组（验证 env 真有用）

- `risk_short_pose_only(t) = 0.45·B + 0.25·F`（无 env）用作基线对比。
- 输出两类 risk_short 的报警触发帧、平均提前量，证明 env 带来增益/更早报警。

---

## 7. 文件布局

| # | 文件 | 动作 | 内容 |
|---|------|------|------|
| ① | `scripts/env_perceive.py` | **新** | 扫描 npz → 元标签 room_type + [可选]亮度 → 调 `EnvRiskModel.forward` → `env_risk.npy` + `env_meta.json` |
| ①b | `model/env/env_risk_model.py` | **新** | **环境模型类**（满足"环境一个模型"）：`EnvRiskModel`(接口) / `RuleEnvRisk`(v0 规则实现) / `TrainedEnvRisk`(R1 训练头)；`env_perceive.py` 仅调用接口，不关心实现 |
| ② | `scripts/fuse_risk.py` | **新** | 加载 RiskRegressor(run_03/04) + env sidecar + F/P 规则 → `risk_short`/`risk_long` → 四级分级 + 分期 + 联动放大 → metrics + 曲线 + 注入对比 + 场景分层 |
| ③ | `scripts/build_config.py` | **改** | 加 `ENV_*`(房间表/光照/地面/放大) + `FUSE_*`(权重/阈值/等级) + `P_*`(个体基础) + `VIDEO_ROOM_MAP`/`DATASET_ROOM_DEFAULT` |
| ④ | `demo/demo.py` | 改(可选) | 加 `task=fusion` 分流：Posture+Env+F+P 融合实时推理 |
| ⑤ | `model/lstm/envrisk_v0_design.md` | **新** | 本文件 |
| ⑥ | `model/lstm/risk_env_evolution.md` | 改 | §4/§7/§9 增加 v0 决策链接与状态标记 |

---

## 8. 配置项（build_config.py 新增）

```python
# ===== EnvRisk v0 (R0 规则) =====
ENV_ROOM_BASE = {            # BASE[room_type]，静态环境危险度 (对齐报告 §5.3 方向)
    "stairs": 0.70, "bathroom": 0.60, "kitchen": 0.40,
    "bedroom": 0.30, "hallway": 0.30, "clinical": 0.25,
    "home": 0.30, "outdoor": 0.15, "default": 0.30,
}
ENV_LIGHT_W       = 0.6     # 光照危险度增量权重
ENV_LIGHT_DARK_TH = 0.25    # 低于此=过暗(dark)，开始危险度增量 (对齐报告 §5.2)
ENV_LIGHT_GLARE_TH= 0.85    # 高于此=眩光，小幅危险度增量
ENV_FLOOR_ENABLED = False    # v0 关闭地面危险
ENV_FLOOR_W       = 0.0     # 预留

# 环境联动放大 (报告 §5.4)
ENV_AMP_TH        = 0.50    # 环境高危阈值
BEH_WARN_TH       = 0.40    # 行为不稳阈值
ENV_AMP           = 0.08    # 通用放大
ENV_AMP_BATH      = 0.10    # 浴室 × 功能不稳
ENV_AMP_DARK      = 0.08    # 暗光 × 步态变异

DATASET_ROOM_DEFAULT = {     # 数据集 → 默认房间类型
    "Le2i": "home", "UR-FALL": "clinical", "CAUCAFall": "home",
    "UP-FALL": "home", "Pre-VFall": "home",
}
VIDEO_ROOM_MAP = {}          # 逐 stem 覆盖（人工校正/R1 反标时填）

# ===== 融合 FuseRisk (对齐报告 §7.1 权重) =====
FUSE_W_B      = 0.45       # 行为姿态权重
FUSE_W_F      = 0.25       # 功能代理权重
FUSE_W_E_IM   = 0.15       # 即时环境权重
FUSE_W_E_ST   = 0.15       # 静态环境权重 (长期)
FUSE_W_F_LT   = 0.25       # 功能趋势权重 (长期)
FUSE_W_P      = 0.15       # 个体基础权重 (无元数据则自动归零)

# 四级等级阈值 (基于 risk_short ∈ [0,1])
FUSE_LV_WATCH   = 0.30
FUSE_LV_WARNING = 0.50
FUSE_LV_CRITICAL= 0.75
FUSE_DANGER     = 0.60      # 报警阈值（沿用 demo 约定，介于 warning/critical）
FUSE_WARN       = 0.30      # 预警阈值

# ===== 个体基础风险 P (报告 §6.2) =====
P_AGE_75   = 10
P_AGE_85   = 10
P_FALL_HIST= 25
P_AID      = 10
P_ALONE    = 10
P_DIZZY    = 15
```

---

## 9. 验证方案（v0 无 env 标签）

因 5 数据集无环境标注，**v0 不做"env 准确率"评估**，改为（对齐报告 §10 指标）：

1. **分布合理性**：各数据集 `risk_long`（=BASE）均值应符合房间假设（如 UR-FALL=clinical 0.25 低于 home 0.30；若人工把某 Le2i 标 stairs 应显著升高）。
2. **可解释追溯**：打印每视频 `env_meta.json` 的规则拆解（room base + light risk_add）+ 融合输出的 `trigger_factors`。
3. **合成注入对比（核心）**：取一段正常视频，在 `VIDEO_ROOM_MAP` 强制标为 `bathroom`/`stairs`，重跑融合 → 验证 `risk_short` 报警**更早触发 / 触发率上升**，证明 env 分量真实贡献（且必通过 §6.5 放大规则，而非单独误报）。
4. **报警不退化**：`risk_short`（含 env）相对纯 `B+F` 的 DANGER 提前量 ≥ 0.5s（不比 Phase-1 差）。
5. **🔬 场景分层指标（报告 §10.3）**：按 `room_type` 与 `light_level` 分层统计 warning/critical 召回与误报率（如"卫生间场景 warning 召回""暗光条件准确率"），证明融合在各场景均稳健。
6. **四级分布**：统计 safe/watch/warning/critical 的帧占比，确认未见 critical 暴涨（环境单独误报被放大规则抑制）。

---

## 10. 验收标准

| 指标 | 达标线 |
|------|--------|
| 融合 pipeline 通畅 | `env_perceive.py` + `fuse_risk.py` 在 5 数据集全跑通，产出 sidecar + metrics |
| 隔离 | 原 `RiskRegressor` / 分类管线零改动；仅新增 sidecar |
| env 可解释 | 每视频规则拆解 + `trigger_factors` 可逐项追溯 |
| 合成注入增益 | 标 stairs/bathroom 后报警明显更早/更多，且不单独误报 critical |
| 报警不退化 | `risk_short` 提前量 ≥ Phase-1 纯姿态 |
| 场景分层稳健 | 各 room_type / light_level 下 warning 召回与误报率合理 |
| 四级输出 | 能输出 safe/watch/warning/critical + pre/in/post_event 分期 |

---

## 11. 落地顺序（实施步骤）

1. **`build_config.py`**：加 §8 全部 `ENV_*` / `FUSE_*` / `P_*` / 映射字典。
2. **`model/env/env_risk_model.py`**：写 `EnvRiskModel` 接口 + `RuleEnvRisk`(v0)。
3. **`scripts/env_perceive.py`**：扫描 `pose_cache/*.npz` → room_type → R0 规则 → 写 sidecar（先支持无视频源纯元标签；`--video_root` 可选开亮度）。
4. **`scripts/fuse_risk.py`**：加载 run_03/04 → 滑窗出 `B` → 算 `F`(规则) → 读 env sidecar → 四源加权 + 联动放大 + 四级分级 + 分期 → metrics + 抽样曲线 + 合成注入对比 + 场景分层。
5. **`demo/demo.py`**（可选）：加 `task=fusion` 实时分支。
6. 跑通后回填 `risk_env_evolution.md` §4/§7/§9 状态为 ✅，并把 v0 实测记入实验模板。

---

## 12. 与 R1/R2 的衔接

- **R0 → R1 种子**：本文件的 `ENV_ROOM_BASE` 危险度表，可直接作为 R1 用 VLM 打弱标签时的**先验/校验锚**；R1 训出的 `EnvRisk` 头输出分布应与该表量级一致。
- **R0.5（可选）**：打开 `--video_root` 后用帧亮度已是第一步；进一步可加预训练场景分类器（torchvision Places365/ResNet）替代/校验元标签 `room_type`。
- **R1**：VLM 对帧零样本打 hazard 分(0~1)+理由 → 攒弱标签 → 训小 `EnvRisk` 回归头（存 `model/env/env_risk_model.py` 的 `TrainedEnvRisk`），`env_perceive.py` 切换为模型推理。F(功能代理) 也可同步升级为小回归头。
- **R2**：每帧直接问 VLM（非实时，仅调试），不在主链路。

---

## 13. 待确认 / 开放点

1. **房间默认假设**：§3.2 的 `DATASET_ROOM_DEFAULT` 是按布景推的，是否认可？尤其 UR-FALL 归 `clinical` 还是 `bedroom`？
2. **逐视频校正**：要不要现在就人工把已知特殊视频（如明显楼梯/浴室）填进 `VIDEO_ROOM_MAP`？还是先全默认、等 R1 反标？
3. **帧亮度**：⚠️ 已核实 5 数据集**源视频不在仓库**（仅 `test/video/` 有 demo 片段）。故 v0 **默认只能纯元标签**（`light_level` 危险度增量=0）。若要启用亮度/地面/VLM，需先**重新取得 5 数据集源视频**（见下方「如何获得数据」），再传 `--video_root`。你手上是否还存着这些源视频？
   > 注：可行性报告 §11.4 明确"第一版使用用户输入"，故纯元标签 v0 是被报告支持的正解，不阻塞。
4. **长期监护分**：v0 的 `risk_long` 暂 = `0.15·E_static + 0.25·F_trend + 0.15·P`，其中 F_trend/P 需要长窗或元数据；v0 可先让 P 留空（权重自动归零）、F_trend 用逐视频均值代理。是否够用？还是想把 `gait_trend` 一并接入？

> 以上任一项确认后，即可按 §11 开工写代码。
