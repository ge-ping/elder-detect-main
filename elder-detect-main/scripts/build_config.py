"""
数据集构建共享配置 — 改一处，所有 build_*.py 生效
"""
from pathlib import Path

# ============================================================
# 窗口参数
# ============================================================
WINDOW_SIZE = 35          # 滑动窗口帧数
STRIDE     = 1            # 窗口步长（1=稠密采样）

# ============================================================
# 数据集划分
# ============================================================
TRAIN_RATIO = 0.8         # 视频级训练集比例
SEED        = 42          # 随机种子（确保可复现）

# ============================================================
# Pose 配置
# ============================================================
POSE_CONFIG = "config/pose_config.yaml"

# ============================================================
# 输出根目录
# ============================================================
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "datasets"

# ============================================================
# Risk 回归标签 / 训练（model/lstm/risk_evolution.md）
#   - 标签从 pose_cache/*.npz 的 labels(0/1/2) 反推，不重跑 MediaPipe
#   - 指数平滑 ramp 方案 A：不稳段 0.3→0.7 指数爬升，跌倒帧=1.0
# ============================================================
RISK_MODE     = "exponential"   # 标签生成策略
# 有效预告窗口(帧)：跌倒前多长区间内 ramp 0.3→0.7；窗口外风险=0.3 保底。
# 取值对齐各数据集「预冲击标注窗」(如 CAUCAFall PRE_WINDOW=35)，使 risk 目标曲线覆盖
# 与类标签 Unstable 同一段预冲击区，监督一致；≈35/25fps≈1.4s 预警。
# ⚠️ 非硬编码死值：可通过 regen_risk_labels.py --horizon 临时覆盖，无需改本文件。
RISK_HORIZON = 35
RAMP_LO       = 0.3               # 不稳起点风险（WARNING 阈值）
RAMP_HI       = 0.7               # 临跌倒风险上沿（落入 DANGER 区间）
TAU_U         = 12                 # 不稳→跌倒 爬升时间常数(帧)
TAU_ABN       = 20                 # 异常不跌倒 爬升时间常数(帧)
REG_LOSS      = "weighted_mse"      # 加权 MSE（β 加权，旧方案，Run01/02 用）
REG_BETA      = 3                  # 高 risk 样本权重 (1+β·y)，抑制「全预测0」
# 类加权（对齐旧分类类权 [1,5,3]）：Run02 诊断发现单调 (1+β·y) 把 Fall(高y) 拉到 4x，
# 与旧类权相反（Unstable 应为最高），导致中段 [0.3,0.6) 监督不足、Unstable F1 卡 0.40。
# Run03 起改用 class_weighted：Unstable 权重最高，强化中段。
REG_W_NORMAL   = 1.0              # Normal   (y < 0.3)
REG_W_UNSTABLE = 2.0              # Unstable (0.3 ≤ y < 0.6) —— S2: 由5.0下调，平衡对过渡动作敏感度
REG_W_FALL     = 2.0              # Fall     (y ≥ 0.6)        —— S2: 由3.0下调，与unstable对齐
# 连续中段加权（Run04 起）：权重在 Unstable 中段 y≈center 达峰，
# 直接针对 band [0.3,0.6) MAE 最差处，比 flat w_u=5 更精准地强化中段。
REG_MIDBAND_CENTER = 0.45          # 峰值位置（∈Unstable 中段）
REG_MIDBAND_WIDTH  = 0.18         # 峰宽（高斯标准差）
REG_MIDBAND_PEAK   = 6.0          # 峰值额外权重（weight=1+peak·exp(...)）
REG_LOSS_MODE  = "class_weighted" # class_weighted|weighted_mse|midband
MULTITASK     = False              # 是否开分类辅助头（本期单头回归）
MT_LAMBDA     = 0.5               # 多任务损失混合权重 λ·CE+(1−λ)·MSE

# ============================================================
# Risk 回归训练默认
# ============================================================
RISK_EPOCHS     = 100
RISK_BATCH_SIZE = 64
RISK_LR         = 1e-3
RISK_VAL_RATIO  = 0.15
RISK_PATIENCE   = 20
RISK_RUN_ROOT    = "model/lstm/risk_v1"

# ============================================================
# EnvRisk R1: VLM API 直接标注（不训 ViT）
#   见 model/lstm/envrisk_r1_vlm_design.md
#   ⚠️ API provider / key 一律不硬编码：key 仅从环境变量读
#      OPENAI_API_KEY / DASHSCOPE_API_KEY / GOOGLE_API_KEY
# ============================================================
ENV_VLM_PROVIDER   = "ark"            # ark(火山方舟 Doubao) | openai | dashscope | google
ENV_VLM_MODEL      = "Doubao-Seed-1.6-vision"  # 方舟视觉模型（R1 真实标注默认）
ENV_VLM_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"  # 火山方舟 OpenAI 兼容端点
ENV_VLM_FRAMES_PER_VIDEO = 3         # 抽帧数 K
ENV_VLM_IMG_LONG_SIDE    = 640       # resize 长边
ENV_VLM_CONCURRENCY = 4              # 并发线程（并行"视频"，视频内部帧仍串行）
ENV_VLM_RETRY       = 3              # 失败重试
ENV_VLM_TIMEOUT     = 60            # 单次 VLM 调用超时(秒)；超时即抛错→触发 retry，避免永久卡死
ENV_VLM_MIN_CONF    = 0.4            # 低于此标 need_review
ENV_VLM_PROMPT_VER  = "v1"           # prompt 版本（进缓存键）

# 枚举白名单（与 prompt 约束一致）
#   room_type / lighting 仍由 VLM 输出（定位房间、判光照）；
#   floor_hazard 枚举已废弃 —— HOME FAST 改用细标签（见下 HOME_FAST_VLM_TAGS）
ENV_VLM_ROOM_ENUM   = ["bathroom", "kitchen", "stairs", "bedroom",
                         "living_room", "hallway", "clinical", "outdoor", "unknown"]
ENV_VLM_LIGHT_ENUM  = ["bright", "dim", "dark"]

# ============================================================
# 环境风险量表 = HOME FAST（居家跌倒意外筛查工具，25 题）
#   取代旧版 room/light/floor 三查表（RuleEnvRisk 已废弃）。
#   规则依据（用户 2026-07-13 指定「用这个，不要原来的」）：
#     每题「否(安全隐患)」= 1 分；「是 / 不适用(N/A)」= 0 分；
#     总分 T(0~25) 越高 → 居家跌倒环境隐患越多、风险越高；
#     分级：0~4 低 / 5~7 中 / >=8 高。
#   视觉增强版（用户 2026-07-13 选定）：
#     . 22 题视觉可判项由 R1(VLM 看画面) 自动打细标签；
#     . 仅 5/6/17 三题纯体验走问卷（VLM 不判，标记 N/A）；
#     . 单视频只覆盖所见房间，未见的题标记 N/A（不计入分母）。
#   VLM 细标签 → 题号 映射见 HOME_FAST_VLM_TAGS；
#   归一化环境分 E = T / N_obs（N_obs = 非 N/A 题数）∈ [0,1]，进融合。
# ============================================================
HOME_FAST_N = 25
HOME_FAST_COVERAGE_FLOOR = 0.6  # 覆盖度守卫：N_obs/25 低于此 → need_review（E 不可全信）
HOME_FAST_LEVEL_LOW = 4    # 0~4 低
HOME_FAST_LEVEL_MID = 7    # 5~7 中；>=8 高

# 25 题定义：(题号, 模块, 问题, 视觉可判性, NA场景)
#   visual: vqa = 纯视觉(VLM直接判) | weak = 弱视觉(VLM判设施+问卷补体验) | qa = 纯问卷(必备)
#   na_when: 仅在该场景房间才计分，否则 N/A（单视频只看到部分房间）
HOME_FAST_ITEMS = [
    (1,  "地面区域",   "走道无电线、纸箱等杂物遮挡",           "vqa",  None),
    (2,  "地面区域",   "全屋地面、地毯铺装完好",               "vqa",  None),
    (3,  "地面区域",   "厨房/卫浴地面具备防滑功能",             "weak", None),
    (4,  "地面区域",   "零散地垫完全固定",                     "vqa",  None),
    (5,  "家具起身适配","可轻松安全上下床",                     "qa",   None),
    (6,  "家具起身适配","可轻松从沙发起身",                     "qa",   None),
    (7,  "照明系统",   "全屋照明亮度充足",                     "weak", None),
    (8,  "照明系统",   "床上可直接开灯",                       "vqa",  None),
    (9,  "照明系统",   "户外台阶、入户楼道夜间有灯",           "weak", "outdoor"),
    (10, "卫浴区域",   "马桶起身安全省力",                     "weak", "bathroom"),
    (11, "卫浴区域",   "独立安全进出浴缸",                     "weak", "bathroom"),
    (12, "卫浴区域",   "自由进出淋浴区",                       "weak", "bathroom"),
    (13, "卫浴区域",   "浴缸/淋浴配有牢固承重扶手",             "vqa",  "bathroom"),
    (14, "卫浴区域",   "浴区内铺设防滑垫",                     "vqa",  "bathroom"),
    (15, "卫浴区域",   "卧室距离卫生间很近",                   "weak", None),
    (16, "储物区域",   "常用物品无需登高弯腰取放",             "weak", None),
    (17, "储物区域",   "餐食可平稳端送到用餐区",               "qa",   None),
    (18, "楼梯台阶",   "室内台阶全长扶手",                     "vqa",  "stairs"),
    (19, "楼梯台阶",   "室外连续台阶全长扶手",                 "vqa",  "outdoor"),
    (20, "楼梯台阶",   "上下台阶轻松无疲劳",                   "weak", "stairs"),
    (21, "楼梯台阶",   "台阶边缘清晰易分辨",                   "vqa",  "stairs"),
    (22, "楼梯台阶",   "入户门进出平衡安全",                   "weak", None),
    (23, "户外庭院",   "庭院步道完好无障碍物",                 "vqa",  "outdoor"),
    (24, "穿戴宠物",   "日常鞋/拖鞋防滑合脚",                 "weak", None),
    (25, "穿戴宠物",   "宠物不会绊倒老人",                     "weak", None),
]

# R1(VLM) 细标签 → HOME FAST 题号 映射
#   key = VLM 输出字段；value = (题号, 取值语义)
#   bool 字段语义：True = 安全条件满足（该题 0 分）；False = 隐患（该题 1 分）
#   na_when 场景外的题不计入分母（单视频只看到部分房间）
HOME_FAST_VLM_TAGS = {
    "corridor_clear":     (1,  "bool_false_is_risk"),
    "floor_intact":       (2,  "bool_false_is_risk"),
    "floor_antislip":     (3,  "bool_false_is_risk"),
    "mat_fixed":          (4,  "bool_false_is_risk"),
    "bedside_light":      (8,  "bool_false_is_risk"),
    "outdoor_light":      (9,  "bool_false_is_risk"),
    "toilet_grab_bar":    (10, "bool_false_is_risk"),
    "bath_accessible":    (11, "bool_false_is_risk"),  # 11、12 共用（浴缸/淋浴进出）
    "bath_grab_bar":     (13, "bool_false_is_risk"),
    "bath_antislip_mat": (14, "bool_false_is_risk"),
    "near_toilet":        (15, "bool_false_is_risk"),
    "storage_reachable":  (16, "bool_false_is_risk"),
    "indoor_stair_rail": (18, "bool_false_is_risk"),
    "outdoor_stair_rail":(19, "bool_false_is_risk"),
    "stair_regular":      (20, "bool_false_is_risk"),
    "stair_edge_clear":   (21, "bool_false_is_risk"),
    "entry_safe":         (22, "bool_false_is_risk"),
    "yard_path_intact":   (23, "bool_false_is_risk"),
    "shoe_antislip":     (24, "bool_false_is_risk"),
    "pet_present":        (25, "pet_present"),
    "pet_safe":           (25, "pet_safe"),
    "lighting":           (7,  "light_not_bright_is_risk"),
}
# 题 12 复用题 11 的 bath_accessible 标签
HOME_FAST_SHARED_TAG = {12: "bath_accessible"}

# 场景词 → 用于 NA 判定（room_type 命中该场景才计分）
HOME_FAST_NA_SCENE = {qid: scene for qid, _, _, _, scene in HOME_FAST_ITEMS if scene}


# ============================================================
# EnvRisk v0 → 融合 FuseRisk（对齐《跌倒前预测可行性报告》）
#   三因子加权（行为姿态B + 环境E + 个体基础P）：
#     short = B（B 独占，E/P 只调阈值不进分）
#     long  = 0.15·E + 0.40·P        （E = HOME FAST 单分，P = 问卷个体基础）
# ============================================================
# 感知层：房间类型映射（数据集元标签，v0 零模型）
DATASET_ROOM_DEFAULT = {       # 数据集 → 默认房间类型（按已知布景假设）
    "Le2i": "home", "UR-FALL": "clinical", "CAUCAFall": "home",
    "UP-FALL": "home", "Pre-VFall": "home",
}
VIDEO_ROOM_MAP = {}            # 逐 stem 覆盖（人工校正 / R1 反标时填）

# ============================================================
# 阈值调节（上下文灵敏度）—— 替代旧版「加性放大器」
#   设计哲学（回应设计评审 #2/#3）：
#     B 是唯一「此刻正在摔」信号；环境 E 与 个体 P 绝不进 risk_short 分数
#     （避免「场景凭空把良性弯腰推过报警线」→ 高频误报、养老院拔电源）。
#     二者只**下调报警/预警阈值**（越危险越敏感），且阈值有**硬下限**，
#     保证任何上下文都救不回 B<下限 的良性动作。
#   时延安全（#1）：阈值只是一次比较，E 标签来自预存 sidecar（VLM 离线批标），
#     不参与在线 API 调用，不击穿 B 的 0.19s DANGER 窗。
# ============================================================
THR_ALARM_BASE  = 0.60       # 基准报警阈值（= 旧 0.60 报警线）
THR_ALARM_FLOOR = 0.45       # 报警阈值硬下限（B<0.45 永不报警，无论场景多危）
THR_WARN_BASE   = 0.40       # 基准预警阈值（同室声光喝止线）
THR_WARN_FLOOR  = 0.30       # 预警阈值硬下限
CTX_W_E         = 0.60        # 上下文·环境权重（HOME FAST 单分 E）
CTX_W_P         = 0.40        # 上下文·个体基础权重（P）
CTX_GAIN        = 0.20        # 上下文对阈值的下调幅度（×ctx，封顶不超 BASE−FLOOR）
CTX_MAX         = 1.00        # ctx 上限（双重保险，配合 FLOOR）
# E 上下文权重：HOME FAST 改为单一环境权重（不再区分静态/即时，统一 E=T/N_obs）
#   （原 E_CTX_W_STATIC / E_CTX_W_IM 已废弃，见上 HOME FAST 块）

# 三因子融合权重（行为姿态B + 环境E + 个体基础P，见汇报 §6）
#   短期：risk_short = B（B 独占，已含 gait；E/P 只调阈值不进分）
#   长期：risk_long = w_E_st·E + w_P·P（E=HOME FAST 单分，监护分，分档）
FUSE_W_B    = 1.00           # 行为姿态 B 权重（短期唯一分量，= B 本身）
FUSE_W_E_ST = 0.15           # 环境 E(HOME FAST) 权重（长期 risk_long）
FUSE_W_P    = 0.40           # 个体基础 P 权重（问卷，长期）

# 等级阈值（仅用于 risk_long / 长期分档显示）
FUSE_LV_WATCH    = 0.30      # safe → watch
FUSE_LV_WARNING  = 0.50      # watch → warning
FUSE_LV_CRITICAL = 0.75      # warning → critical
# 注：risk_short 的「有效报警/预警阈值」由 THR_ALARM_*/THR_WARN_*
#     （随上下文 E/P 下调，见上）动态给出，不再用固定 FUSE_DANGER/FUSE_WARN。

# 个体基础风险 P（报告 §6.2 最小问卷字段，v0 可选，缺则权重归零）
P_AGE_75   = 10
P_AGE_85   = 10
P_FALL_HIST= 25
P_AID      = 10
P_ALONE    = 10
P_DIZZY    = 15
P_MAX      = 100               # P 封顶分母（min(Σ,100)/100）

# ============================================================
# 环境交互风险预测（I 门控）—— Phase-1 v0
#   E_effective(t) = E_semantic × I(t)
#   I(t) = 预测区域 Zone(t) 落在环境掩码 M(t) 内的占比 ∈ [0,1]
#   设计见 docs/环境交互风险预测_设计.md
#   ⚠️ 非硬编码死值：均可通过 CLI（fuse_risk.py --intent-horizon 等）覆盖
# ============================================================
# --- 运动意图投影（features/intent.py · MotionIntentProjector）---
INTENT_HORIZON            = 35      # 预测未来帧数（对齐 RISK_HORIZON ≈ 1.5s @23fps）
INTENT_VEL_WINDOW         = 5       # 速度矢量滑窗帧数（k 帧位移 / k）
INTENT_ZONE_LONG_AXIS     = 0.15    # 预测椭圆长轴（归一化坐标，≈画面宽 15%）
INTENT_ZONE_SHORT_AXIS    = 0.06    # 预测椭圆短轴（横向容差）
INTENT_OPTICAL_VEL_THRESH = 0.005   # |v|₂D 低于此 → 疑似沿光轴运动（像素速度近 0）
INTENT_OPTICAL_Z_THRESH   = -0.01   # world_z 帧间变化小于此(米) → 沿光轴接近（z 持续减小）

# --- 环境掩码（features/env_mask.py · EnvMaskBuilder）---
ENV_MASK_MODE           = "vlm_coarse"  # vlm_coarse(v0 粗掩码) | bg_diff(v1 背景差分)
ENV_MASK_H              = 64      # 掩码降采样高（边缘端友好，逐帧 64×64 而非原分辨率）
ENV_MASK_W              = 64      # 掩码降采样宽
ENV_MASK_GROUND_RATIO   = 0.33    # 地面 ROI = 画面下方 1/3（家居机位假设：地面在下）
ENV_MASK_FOOT_RADIUS    = 0.04    # 人脚挖孔半径（归一化坐标，排除人脚自身污染）

# --- 掩码误报抑制（2026-07-17 新增，解决"把老人/反光/噪声当危险"）---
ENV_MASK_POKE_PERSON     = True   # 挖掉整个人 bbox（不只是脚），避免 MOG2 把移动中的老人当前景
ENV_MASK_PERSON_MARGIN  = 0.15    # 人 bbox 外扩比例（归一化），防手臂/衣服边缘漏标
ENV_MASK_USE_SPECULAR   = False   # 是否启用 HSV 反光(水渍)检测；默认关（误报最重：白墙/亮地/幕布都中标）
ENV_MASK_SPEC_MIN_AREA  = 0.01    # 反光连通域最小面积占比，小于则丢弃（仅 use_specular 时生效）
ENV_MASK_MOG2_VAR       = 40      # MOG2 varThreshold（原 25 偏低→噪声多），调高抑制小噪声
ENV_MASK_MIN_AREA_RATIO = 0.005   # 前景连通域最小面积占比，小于则丢弃（去灯光闪烁/尘埃/小噪声）
ENV_MASK_MAX_AREA_RATIO = 0.5     # 前景总占比超此值（光照突变/切场景）→ 整帧清空，防整帧误报
ENV_MASK_CLOSE_KERNEL   = 5       # 闭运算核（填充小洞），0 表示不闭运算

# --- 帧差(absdiff)掩码（2026-07-17 新增，替代 MOG2，完全不用 VLM）---
# 思路：部署时存一张"干净空房基准图"，实时帧与基准图 absdiff → 二值化 →
#       超过阈值即"地上多了一块东西"=潜在障碍物。比 MOG2 确定性高、几乎不误报。
ENV_MASK_ABS_THRESH    = 30      # absdiff 后二值化阈值（0-255，越大越严格）
ENV_MASK_ABS_BLUR      = 5       # 差分前高斯模糊核（去传感器/压缩噪声，0=不模糊，须为奇数）
ENV_MASK_ABS_USE_COLOR = True    # True=取三通道 absdiff 最大值（能抓彩色异物如红毛巾）；False=转灰度
ENV_MASK_GROUND_ROI    = True    # 是否只保留地面区域（默认下方 1-ground_ratio，匹配"地上多了一块东西"）

# --- 门控开关（scripts/fuse_risk.py）---
GATE_ENABLED_DEFAULT    = False   # 默认关，关闭时行为完全等价现有 pipeline（向后兼容）

# ============================================================
# --- ST-GCN（model/stgcn · 替代 LSTM 的 B 模型，独立模块，不影响旧路径）---
#   输入：每帧 17 关节 3D 世界坐标(髋中心归一化 [-1,1]) → (N,3,T,17)
#   输出：(N,1) ∈ [0,1] 连续跌倒风险（与 RiskRegressor 同接口，demo 门控一行不改）
#   训练数据：scripts/build_skeleton.py 从 pose_cache/*.npz 直接取 world 坐标生成
#   与 LSTM 路径完全并行：仅当模型 model_info.json 带 model_type=stgcn 时 demo 才走此分支
# ============================================================
STGCN_JOINT_SET         = "fall17"          # 关节集名（对应 graph.FALL_JOINTS_17）
STGCN_NUM_JOINTS       = 17                 # 关节数 V
STGCN_INPUT_CHANNEL     = 3                 # C = (x,y,z)；改 2 即退化为 2D 坐标
STGCN_WINDOW           = 60                 # 窗口帧数（S2: 由35上调至60，更长时序上下文）
STGCN_WINDOW_CMP       = 25                 # 公平对比窗口（LSTM 最优）
STGCN_LAYERS           = [64, 64, 128, 128, 256]  # 各 STGCNBlock 输出通道
STGCN_TEMP_KERNEL      = 9                  # 时序卷积核（奇数）
STGCN_DROPOUT          = 0.3                # dropout
STGCN_RUN_ROOT         = "model/stgcn/stgcn_v1"     # 训练 run 目录（自动递增 run_xx）
STGCN_DATA_ROOT        = "datasets/experiment"      # 训练数据根（skeleton_w35 / skeleton_w25）
STGCN_EPOCHS           = 60
STGCN_BATCH_SIZE       = 64
STGCN_LR               = 1e-3
STGCN_VAL_RATIO        = 0.1
STGCN_PATIENCE         = 12

