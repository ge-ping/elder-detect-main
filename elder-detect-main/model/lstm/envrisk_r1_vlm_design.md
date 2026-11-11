> ⚠️ **本文档已作废（superseded）**：其所描述的「`room_type`/`lighting`/`floor_hazard` → 旧 `RuleEnvRisk.compute()` 三查表」、默认 API `gpt-4o`/`qwen-vl`/`gemini`、以及「`E` 加性进 `risk_short` 分数」均已被 **2026-07-13 的 HOME FAST 重构** 取代。当前 R1 链路是 `scripts/env_label_vlm.py`（provider 默认 `ark` 火山方舟 + `Doubao-Seed-1.6-vision`，key 从 `.env` 读）打 **HOME FAST 细标签** → `HomeFastModel.from_labels()` 换算 25 题 0/1（`model/env/env_risk_model.py`）。本文件仅作**历史留痕**，切勿据此实现。最新设计以 `docs/跌倒风险预测_汇报.md` §3 为准。

# 🛰️ EnvRisk R1（VLM API 直接产出环境风险）设计方案

> **隶属**：Phase-2 多分量复合风险（`risk_env_evolution.md`）的 **E 环境源** 升级路线；承接 `envrisk_v0_design.md` 的 R0 规则版。
> **本文件范围**：用 **VLM API 零样本直接标注**，把"环境"从 v0 的"查表常数"升级为**数据驱动的环境风险**。用户已锁定 **不训 ViT**——环境风险由 VLM API 直接产出，不走本地模型训练。
> **已锁定决策**：
> 1. **获取方式 = VLM API**（GPT-4o / Qwen-VL-Max / Gemini，非本地部署）。
> 2. **标签性质 = 弱标签**（VLM 零样本产出，无人工标注）。
> 3. **不训本地模型**：环境风险 = VLM API 直接输出的 `fall_risk` / 结构化属性 → 映射成 `risk_环境`，**没有 ViT 训练步骤**。
> 4. **离线批标 + 缓存**：环境是静态的，对每段视频抽 1~3 帧用 API 标注一次、存 sidecar；**线上推理只读 sidecar，不实时调 API**（零延迟、零 API 费）。
> 5. **隔离原则继承 v0**：不碰 `RiskRegressor` / `pose_cache/*.npz` / 训练标签，env 仍只在融合层加性合并。

---

## 📑 目录

- [0. 速览](#0-速览)
- [1. 定位：R1 在整条路线里的位置](#1-定位r1-在整条路线里的位置)
- [2. 前置条件与硬约束](#2-前置条件与硬约束)
- [3. 整体数据流（两段）](#3-整体数据流两段)
- [4. 第一段：抽帧](#4-第一段抽帧)
- [5. 第二段：VLM API 打标](#5-第二段vlm-api-打标)
- [6. Prompt 设计（核心）](#6-prompt-设计核心)
- [7. 输出解析与质控](#7-输出解析与质控)
- [8. 弱标签 → risk_环境 映射](#8-弱标签--risk_环境-映射)
- [9. 为什么不用 ViT / 线上怎么跑](#9-为什么不用-vit--线上怎么跑)
- [10. 成本与耗时估算](#10-成本与耗时估算)
- [11. 存储与文件布局](#11-存储与文件布局)
- [12. 配置项（build_config.py 新增）](#12-配置项build_configpy-新增)
- [13. 验证方案](#13-验证方案)
- [14. 落地顺序](#14-落地顺序)
- [15. 风险与回退](#15-风险与回退)
- [16. 待确认 / 开放点](#16-待确认--开放点)

---

## 0. 速览

| 项 | 值 |
|----|----|
| 版本 | EnvRisk **R1** = VLM API 零样本直接标注（**不训 ViT**） |
| 打标器 | **VLM API**（默认 GPT-4o-mini；可切 Qwen-VL-Max / Gemini-1.5） |
| 标签来源 | 零样本推理产出弱标签（`room_type / lighting / floor_hazard / fall_risk`） |
| 环境模型 | **无本地训练模型**：环境风险 = VLM API 直接产出 → 映射 → `risk_环境` |
| 线上推理 | **离线批标一次 + 缓存 sidecar**，推理只读，零 API 延迟/费用 |
| 作用范围 | 仍**仅推理/融合**，不写 `RiskRegressor` 训练标签 |
| 存储 | 弱标签 `env_labels/<ds>/<stem>.json` + sidecar `pose_cache/<stem>.env_risk.npy` |
| 前置卡点 | ⚠️ **5 数据集源 RGB 视频需先取回**（v0 已核实不在仓库） |
| 状态 | ✅ 代码已写（API 留空：provider/key 走环境变量，含 `--mock` 无网验证） |

---

## 1. 定位：R1 在整条路线里的位置

```
环境源 E 的演进路线
  R0  规则查表（v0，已设计）      : room_type 硬编码 → BASE 表 → 常数 risk_环境   [零数据, 不阻塞]
  R0.5 帧亮度（可选）            : 有视频后加 light_level 危险度增量
  R1  VLM API 直接标注（本文件） : 用现成 VLM 对帧零样本打 hazard 分 → 直接映射 risk_环境  [需源视频 + API，不训 ViT]
  R2  在线直接问 VLM            : 非实时，仅调试对照，不进主链路
```

**R1 与 R0 的关系**：R0 的 `ENV_ROOM_BASE` 危险度表**不作废**——它变成 R1 的**先验种子 / 校验锚**：VLM 打出的 `fall_risk` 分布应与该表量级一致（如 bathroom 显著高于 living_room），否则说明打标 prompt 或采样有问题（见 §7-④）。

**为什么"直接用 API"而不是训 ViT**：
- 用户已明确：**不用 ViT**。环境风险直接由 VLM（现成大模型）产出，省去训练一个本地视觉模型。
- VLM 本身就是"一个模型"——我们只是**调用它、不微调它**。这满足"环境一个模型"的精神（一个统一的环境感知组件），只是该组件是 API 服务而非本地权重。
- 环境是**静态**的（房间/光照/布局在一段视频里几乎不变），只需**离线标注一次**缓存下来，线上推理完全不碰 API → 没有延迟、没有费用、没有联网依赖。

---

## 2. 前置条件与硬约束

**前置（缺一不可）**
1. **源 RGB 视频**：5 数据集（Le2i / UR-FALL / CAUCAFall / UP-FALL / Pre-VFall）的原始视频。⚠️ v0 已核实**不在仓库**，需重新下载或用 `--video_root` 指回本地存储。
2. **VLM API Key**：至少一家（OpenAI / 阿里云 DashScope / Google）。写入环境变量，不硬编码进代码。

**硬约束（继承 v0）**
- 原 `FallLSTM` / `RiskRegressor` / `pose_cache/*.npz` / 分类与回归训练标签 **零改动**；`*.npz` 只读。
- env 弱标签、env_risk sidecar 全部独立存放，删除不影响 Phase-1。
- **env 不进 `RiskRegressor` 训练**：只在 `fuse_risk.py` 与 `risk_姿态` 加性融合。

---

## 3. 整体数据流（两段）

```
┌─ 第一段 抽帧 (scripts/env_extract_frames.py) ──────────────────────┐
│  源视频 <ds>/<stem>.mp4  ──每视频抽 K 帧──►  frames/<ds>/<stem>/f0.jpg… │
└───────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─ 第二段 VLM API 打标 (scripts/env_label_vlm.py) ──────────────────┐
│  frame.jpg + prompt  ──HTTP──►  VLM API  ──►  结构化 JSON            │
│  {room_type, lighting, floor_hazard, fall_risk, reason, confidence}  │
│  → 视频级聚合（K 帧投票/取中位）→ env_labels/<ds>/<stem>.json      │
└───────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
            §8 映射 → 写 sidecar pose_cache/<stem>.env_risk.npy
            （离线一次性；线上推理只读此 sidecar，不再调 API）
```

> **关键点**：VLM 打标是**离线、一次性**的。打完即把 `risk_环境` 固化进 sidecar，`fuse_risk.py` / `demo.py` 推理时直接读 sidecar，与 v0 规则版**完全同格式**——所以"换环境实现"对融合层零侵入。

---

## 4. 第一段：抽帧

- **采样数 K**：环境语义在单段视频里近似静止，`K=3` 足够（首/中/尾各 1 帧）；地面危险想更稳可 `K=5`。
- **去冗余**：同视频多帧差异极小，K 帧已能覆盖光照/遮挡波动，避免浪费 API 调用。
- **分辨率**：长边 resize 到 **512~768px**（多数 VLM 建议区间），压 JPEG q=85，显著降 token/费用。
- **产物**：`frames/<ds>/<stem>/f{0..K-1}.jpg` + 一个 `index.csv`（stem, frame_paths, n_frames_video, fps）。

**规模估算**：458 视频 × 3 帧 ≈ **1374 张图**。

---

## 5. 第二段：VLM API 打标

### 5.1 API 选型

| API | 模型 | 优势 | 单图成本(约) | 备注 |
|-----|------|------|-------------|------|
| **OpenAI** | `gpt-4o` / `gpt-4o-mini` | 结构化输出稳、JSON mode | 4o ≈ $0.005–0.01；mini ≈ $0.001 | **默认推荐**，mini 打粗标已够 |
| **阿里云 DashScope** | `qwen-vl-max` / `qwen-vl-plus` | 中文理解好、国内直连快 | ≈ ¥0.008–0.02/图 | 国内无需代理，首选备份 |
| **Google** | `gemini-1.5-flash/pro` | flash 便宜、上下文大 | flash 极低 | 可批量兜底 |

- **默认**：`gpt-4o-mini` 打粗标（room/light）+ `gpt-4o` 复核地面危险这种细语义；预算敏感可全程 `qwen-vl-plus`。
- Key 走环境变量：`OPENAI_API_KEY` / `DASHSCOPE_API_KEY` / `GOOGLE_API_KEY`，代码只读 `os.environ`。

### 5.2 调用要点
- **强制 JSON**：OpenAI 用 `response_format={"type":"json_object"}`；其他家在 prompt 里硬约束"只输出 JSON"。
- **并发 + 限流**：线程池 `ENV_VLM_CONCURRENCY=4`，配合指数退避重试（429/超时最多 `ENV_VLM_RETRY=3` 次）。
- **断点续跑**：每张图打完即写盘（或 append JSONL），已存在结果跳过 → 中断可续、不重复烧钱。
- **缓存**：以 `sha1(图片字节 + prompt 版本)` 为键缓存响应，重跑同图零成本。

---

## 6. Prompt 设计（核心）

打标质量 80% 取决于 prompt。用**结构化 + few-shot + 枚举约束**三件套。

### 6.1 System / 指令

```
你是老年人居家跌倒风险的环境评估专家。给你一张室内监控画面，
请只评估「环境本身」对老人跌倒的危险度，不要评估画面里人的姿态或动作。
严格输出 JSON，不要输出任何多余文字。
```

### 6.2 User（含枚举与字段约束）

```
分析这张图的环境跌倒风险，严格按以下 JSON 模式输出：
{
  "room_type":   "从 [bathroom, kitchen, stairs, bedroom, living_room, hallway, clinical, outdoor, unknown] 选一个",
  "lighting":    "从 [bright, dim, dark] 选一个",
  "floor_hazard":"从 [none, wet, obstacle, uneven, stairs] 选一个",
  "fall_risk":   "0.0~1.0 的浮点数，仅代表环境对老人的跌倒危险度",
  "confidence":  "0.0~1.0，你对本次判断的把握",
  "reason":      "一句中文，说明主要危险因素"
}
评分参考：楼梯/湿滑浴室最高(0.6~0.8)，厨房中(0.4)，卧室走廊(0.3)，明亮客厅最低(0.2)；
光线昏暗、地面有障碍会进一步抬高。
```

### 6.3 Few-shot（可选，显著提升一致性）
附 1~2 个"图 + 期望 JSON"范例（如一张暗浴室 → `{room_type:bathroom, lighting:dark, fall_risk:0.75, …}`），能大幅降低格式漂移与打分方差。

> **反作弊约束**：明确"只评估环境、不看人"，避免 VLM 把"有人躺地上"当成环境危险，污染标签。

---

## 7. 输出解析与质控

1. **JSON 解析**：`json.loads`；失败则正则抽 `{...}` 再试；仍失败标 `parse_error` 并重试。
2. **枚举校验**：`room_type` 等不在白名单 → 归 `unknown`，`fall_risk` 裁剪到 `[0,1]`。
3. **视频级聚合**（K 帧 → 1 个视频标签）：
   - `room_type` / `lighting` / `floor_hazard`：**众数投票**（并列取危险度更高者）。
   - `fall_risk`：**中位数**（抗离群），并记 `std` 反映帧间一致性。
   - `confidence`：均值；低于 `ENV_VLM_MIN_CONF(0.4)` 的视频标记 `need_review`。
4. **先验一致性检查**：聚合 `fall_risk` 与 R0 `ENV_ROOM_BASE[room_type]` 偏差 > 0.3 的视频进 `outliers.csv`，人工抽查（这是防 prompt 跑偏的关键闸门）。

---

## 8. 弱标签 → risk_环境 映射（纯口径 B）

**R0 与 R1 共用同一套规则表，唯一区别仅在于标签来源**：

- **R0**：标签来自数据集元标签 / 硬编码映射（room_type 由已知布景给，零模型、立跑）；
- **R1**：标签来自 VLM 看画面打标（room_type / lighting / floor_hazard）。

两者都把 `(room_type, lighting, floor_hazard)` 喂入 **同一个** `RuleEnvRisk.compute()`：

```
risk_环境 = clamp( ENV_ROOM_BASE[room_type]
                   + ENV_LIGHT_RISK_ADD[lighting]
                   + ENV_FLOOR_RISK_ADD[floor_hazard] , 0, 1 )
```

即「VLM 出标签 → 规则匹配」。VLM 聚合标签里的 `fall_risk` 仅作**观测记录**（写进 sidecar meta 便于复核），**不参与 risk 计算**——避免 VLM 既当"标签机"又当"打分机"导致职责不清，且 `fall_risk` 无标尺、难解释。

产出后写 sidecar `pose_cache/<stem>.env_risk.npy`（每帧常数或按 K 帧线性插值），**与 v0 完全同格式** → `fuse_risk.py` 无需改动即可消费。

---

## 9. 为什么不用 ViT / 线上怎么跑

**用户决策：环境风险直接由 VLM API 产出，不训练本地 ViT。** 这带来三个直接后果：

1. **没有训练步骤**：删去原 R1 的 ViT 微调 / `TrainedEnvRisk` / `train_env.py`。环境风险 = VLM API 直出的结构化标签经 §8 映射得到，**整个环境模块 = "API 标注 + 规则映射"**，没有本地权重需要训练或部署。
2. **"环境一个模型"的重新表述**：环境感知组件 = 调用一个现成的 VLM（它本身就是大模型）。我们**调用它、不微调它**——这仍是"一个统一的环境模型（服务）"，只是以 API 形式存在，而非 `model/env/*.pt` 本地权重。
3. **线上零 API 依赖（关键）**：因为环境是静态的，标注**只需离线做一次**。流程：
   - 开发期：对每段视频抽 K 帧 → 调一次 VLM API → 聚合 → 映射 → 写 `env_risk.npy` sidecar。
   - 部署期：`fuse_risk.py` / `demo.py` **直接读 sidecar**，毫秒级、不联网、不烧钱。
   - 只有当视频源本身变化（新房间/新摄像头）才需要重新跑一次标注。

> 对比：若走 ViT，则要多维护一个本地训练+推理模型；现在省掉这部分复杂度，代价是"环境理解能力上限 = 所选 VLM 的能力"，对房间/光照/地面这类**粗语义**完全够用（见 §13 验证目标）。

---

## 10. 成本与耗时估算

| 项 | 量 | 估算 |
|----|----|----|
| 抽帧 | 458 视频 × 3 | ~1374 张，本地几分钟 |
| VLM 打标 (gpt-4o-mini) | 1374 图 × ~$0.001 | **≈ $1.5** 一次性 |
| VLM 打标 (gpt-4o 全量) | 1374 图 × ~$0.008 | ≈ $11 |
| VLM 打标 (qwen-vl-plus) | 1374 图 × ~¥0.01 | ≈ ¥14 |
| 线上推理 | 读 sidecar | **零 API 费、零延迟** |

> 打标是**一次性离线**开销；线上零 API 依赖。gpt-4o-mini 打粗标 + gpt-4o 复核细项是性价比最优组合。

---

## 11. 存储与文件布局

| # | 路径 | 动作 | 内容 |
|---|------|------|------|
| ① | `scripts/env_extract_frames.py` | **新** | 源视频 → 抽 K 帧 → `frames/<ds>/<stem>/*.jpg` + index.csv |
| ② | `scripts/env_label_vlm.py` | **新** | 帧 + prompt → VLM API → 逐帧 JSONL → 视频级聚合 → `env_labels/<ds>/<stem>.json` |
| ③ | `scripts/env_labels_to_risk.py` | **新** | §8 映射 → 写 `pose_cache/<stem>.env_risk.npy`（复用 v0 sidecar 格式） |
| ④ | `scripts/build_config.py` | 改 | 加 `ENV_VLM_*`（见 §12） |
| ⑤ | `model/lstm/risk_env_evolution.md` | 改 | §4/§12 回链 R1 状态 |

**目录（新增，均可删不影响 Phase-1）**
```
frames/                         # 抽帧（大，可 .gitignore）
env_labels/<ds>/<stem>.json     # VLM 弱标签（视频级）
env_labels/_raw/<ds>/<stem>.jsonl  # 逐帧原始响应（审计用）
pose_cache/<stem>.env_risk.npy  # 最终 risk_环境 sidecar（与 v0 同）
```

> **注意**：不再有 `model/env/vit_run_XX/` —— 因为不训 ViT，没有本地环境模型权重目录。

---

## 12. 配置项（build_config.py 新增）

```python
# ===== EnvRisk R1: VLM API 直接标注（不训 ViT） =====
ENV_VLM_PROVIDER   = "openai"        # openai | dashscope | google
ENV_VLM_MODEL      = "gpt-4o-mini"   # 粗标；细项可切 gpt-4o
ENV_VLM_FRAMES_PER_VIDEO = 3         # 抽帧数 K
ENV_VLM_IMG_LONG_SIDE    = 640       # resize 长边
ENV_VLM_CONCURRENCY = 4              # 并发线程
ENV_VLM_RETRY       = 3              # 失败重试
ENV_VLM_MIN_CONF    = 0.4            # 低于此标 need_review
ENV_VLM_PROMPT_VER  = "v1"           # prompt 版本（进缓存键）
ENV_VLM_ROOM_ENUM   = ["bathroom","kitchen","stairs","bedroom",
                        "living_room","hallway","clinical","outdoor","unknown"]
ENV_VLM_LIGHT_ENUM  = ["bright","dim","dark"]
ENV_VLM_FLOOR_ENUM  = ["none","wet","obstacle","uneven","stairs"]

# 弱标签 → risk 映射口径（纯口径 B：VLM 出标签 → 同 R0 规则表）
ENV_LIGHT_RISK_ADD   = {"bright":0.0,"dim":0.08,"dark":0.15}
ENV_FLOOR_RISK_ADD   = {"none":0.0,"wet":0.25,"obstacle":0.20,"uneven":0.15,"stairs":0.25}
```

---

## 13. 验证方案

1. **打标质量抽检**：随机抽 30~50 张图，人工核对 VLM 的 `room_type`/`lighting` 命中率（目标 room ≥ 85%、light ≥ 80%）。
2. **先验一致性**：聚合 `fall_risk` 均值按数据集/房间排序，应与 R0 `ENV_ROOM_BASE` 同序（bathroom>kitchen>bedroom>living）。
3. **融合不退化（核心）**：把 R1 的 `risk_环境` sidecar 接入 `fuse_risk.py`，对比 v0 规则版在**合成注入 / 场景分层**上的报警提前量与召回，证明 API 标注带来增益（或至少不退化）。
4. **端到端不退化**：融合后 `risk_short` 提前量 ≥ Phase-1 纯姿态；critical 不因 env 单独暴涨（v0 文档 §6.5 放大规则仍生效）。
5. **推理零依赖**：确认 `demo.py` / `fuse_risk.py` 在**无 API Key、无网络**环境下仅靠 sidecar 即可跑通（验证离线缓存生效）。

---

## 14. 落地顺序

1. **取回源视频**（用户侧）→ 确定 `--video_root` 路径。
2. **配 API Key**（环境变量）→ 选 provider/model。
3. `build_config.py` 加 §12 配置。
4. 写 `env_extract_frames.py` → 抽帧、出 index。
5. 写 `env_label_vlm.py` → 小批量（如 20 图）试跑校 prompt → 全量打标 → 聚合。
6. 质控：抽检 + 先验一致性 + outliers 复查。
7. 写 `env_labels_to_risk.py` → 落 `env_risk.npy` sidecar（**此时已可直接融合，R1 完成**）。
8. 跑 §13 验证 → 回填 `risk_env_evolution.md` 状态。

> **里程碑**：第 7 步即拿到"VLM 数据驱动的环境风险"，无 ViT 训练步骤。第 8 步验证离线缓存下的端到端融合。

---

## 15. 风险与回退

| 风险 | 影响 | 应对 |
|------|------|------|
| 源视频始终拿不到 | R1 全程阻塞 | 回退 v0 规则版（已可跑），不影响主线 |
| VLM 打标噪声大 | 弱标签质量差 | few-shot + 先验闸门 + 纯口径 B（规则表）+ 低置信降权 |
| 类别极不均衡（多数 home/bright） | 单一 VLM 分方差大 | 纯口径 B（规则表锚定）+ 低置信标记复核 |
| API 涨价/限流/断网 | 打标中断 | 断点续跑 + 缓存 + 多 provider 切换（仅影响离线打标，不影响线上） |
| 帧里有人干扰环境判断 | 标签污染 | prompt 硬约束"只评估环境不看人" + 抽检 |

---

## 16. 待确认 / 开放点

1. **源视频**：你手上是否还存有 5 数据集原始视频？还是需要我先列各数据集**官方下载源 + 步骤**？（这是 R1 唯一硬前置）
2. **API provider**：默认 `gpt-4o-mini`（便宜稳），还是走国内 `qwen-vl-max`（免代理）？你有哪家的 Key？
3. **打标口径**：已定 —— **纯口径 B**（VLM 出标签 room/light/floor → 同 R0 规则表 `RuleEnvRisk.compute`），不混合 VLM 直出 `fall_risk` 分。
4. **抽帧数 K**：默认 3；若某些视频环境波动大（如有人走动带门）可设 5。

> 任一项确认（尤其 §16-1 源视频、§16-2 API Key），即可按 §14 开工。

---

## 17. 落地记录（代码已写，2026-07-12）

> 用户拍板「直接写代码，API 留空」。按 §14 路线，代码已全部落地，`py_compile` 全通过。

### 17.1 新增 / 改动文件

| 文件 | 动作 | 说明 |
|------|------|------|
| `scripts/build_config.py` | 改 | 末尾追加 §12 全部 `ENV_VLM_*` / `ENV_ROOM_BASE` / `ENV_LIGHT_RISK_ADD` / `ENV_FLOOR_RISK_ADD` |
| `scripts/env_extract_frames.py` | **新** | 第一段抽帧：`--video_root` 递归扫视频 → 每视频均匀抽 K 帧 → `frames/<ds>/<stem>/f*.jpg` + `frames/index.csv` |
| `scripts/env_label_vlm.py` | **新** | 第二段打标：`frames/index.csv` → VLM API（openai/dashscope/google）→ 逐帧 JSONL + 视频级聚合 → `env_labels/<ds>/<stem>.json` |
| `scripts/env_labels_to_risk.py` | **新** | 第三段映射：聚合标签 → §8 纯口径 B（同 R0 规则表）→ 写 `pose_cache/<ds>/<stem>.env_risk.npy`（与姿态 risk 同形，融合零侵入） |

### 17.2 API 留空约定（固化到代码）

- **provider**：`--provider` 或环境变量 `ENV_VLM_PROVIDER`，默认 `openai`；代码内不写死。
- **model**：`--model` 或 `ENV_VLM_MODEL`，默认 `gpt-4o-mini`。
- **key**：**仅从环境变量读**，绝不硬编码 —— `OPENAI_API_KEY` / `DASHSCOPE_API_KEY` / `GOOGLE_API_KEY`；缺失时 `_api_key()` 抛清晰报错。
- **无网验证**：`--mock` 走占位标签，零网络即可跑通整条链路（抽帧→打标→映射→sidecar）。

### 17.3 sidecar 约定（与 Phase-1 对齐）

- 路径：`pose_cache/<ds>/<stem>.env_risk.npy`（与 `pose_cache/<ds>/<stem>.npz` 同目录同 stem）。
- 形态：若对应 `.npz` 存在，把标量 risk **broadcast 成 (T,) 常数数组**（T=该视频帧数），与姿态 risk sidecar 完全同形；npz 缺失则存 (1,) 标量并 `[WARN]`（命名未对齐）。
- 融合层 `fuse_risk.py`（待写）直接读 `.env_risk.npy`，与 `.risk.npy`（姿态）加性合并，零侵入。

### 17.4 真实前置仍待解（非代码）

1. **源视频**：5 数据集原始 RGB 视频仍不在仓库（仅 `test/video/` 有 demo 片段）。`env_extract_frames.py` 需 `--video_root` 指向实际视频位置才能跑全量；可用 `test/video/` 先验证 `--mock` 链路。
2. **API Key**：真实打标需任一家的 key 写入环境变量（非 mock 模式）。

### 17.5 端到端跑通命令（用户环境）

```bash
# 0) 取回源视频后，指向它（或用 test/video/ 先验证）
# 1) 抽帧
python scripts/env_extract_frames.py --video_root /path/to/videos
# 2) 打标（设好 OPENAI_API_KEY 后；或 --mock 无网验证）
python scripts/env_label_vlm.py --index frames/index.csv
# 3) 映射 → sidecar（此时已拿到数据驱动的环境风险）
python scripts/env_labels_to_risk.py
```

> 第 3 步完成即达成 R1 里程碑：环境风险由 VLM 直接产出、数据驱动，无 ViT 训练。
