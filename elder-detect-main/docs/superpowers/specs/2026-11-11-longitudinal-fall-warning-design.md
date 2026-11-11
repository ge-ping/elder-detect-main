# 老年人纵向跌倒预警设计

## 目标与边界

在现有实时跌倒检测链路之外，增加基于个人长期步态和姿态画像的未来跌倒预警。系统先建立个人正常基线，再计算每日观测相对基线的偏离程度；有真实纵向跌倒标签并完成模型校准后，输出未来7天和30天累计跌倒概率。

当前仓库没有包含个人长期观测、随访截止时间和真实跌倒日期的数据，因此首版必须区分两类结果：

- `deviation_index` 是个人步态和姿态偏离指数，可在完成基线后使用，不是跌倒概率。
- `risk_7d` 和 `risk_30d` 只有在加载已校准 Landmark Cox 模型后才允许返回数值，否则必须为 `null`。

心理因素仅作为可选辅助变量。未训练状态下，心理量表不得通过人工权重改变偏离指数；训练状态下，只允许由纵向数据学习出的模型系数参与概率计算。

## 与现有项目的关系

现有 B/E/P/I 实时链路继续负责秒级跌倒态势检测、环境风险和告警阈值。新增纵向模块负责天到月级的趋势监护，两者不共用风险数值，也不改变现有模型、阈值和默认命令行为。

新增结果可通过 `scripts/fuse_risk.py --future-risk-json` 选择性附加到汇总 JSON。未传参数时，现有行为完全保持不变。

## 证据与设计选择

| 设计选择 | 依据 | 项目中的用法 |
|---|---|---|
| 14个有效日建立个人基线 | Mehdizadeh 等对自然步态进行两周基线观测 | `min_baseline_days=14` |
| 未来7天、30天预测窗口 | Mehdizadeh 等使用 Cox 模型预测7天和30天无跌倒生存概率；7天 AUROC 0.80，30天 AUROC 0.67 | 主窗口7天，辅助窗口30天 |
| 持续步态监测预测未来4周 | Adeli 等使用环境视觉步态、临床量表和用药信息预测未来4周跌倒，AUROC 0.762 | 佐证持续更新和约30天窗口 |
| Cox 比例风险模型 | Cox 1972 | 从 Landmark 样本估计条件风险 |
| Landmark 动态预测 | van Houwelingen 2007 | 每个有效观测日作为预测时点，构造7天和30天训练样本 |
| 稳健标准化 | Iglewicz 与 Hoaglin 的 MAD 稳健离群值方法 | 以中位数和 MAD 计算个人标准化偏离 |
| 统计过程异常提示 | NIST/SEMATECH 统计过程控制手册 | 未校准状态仅提示持续统计异常，不声称临床概率 |

主要论文：

1. Mehdizadeh S, et al. Predicting Short-Term Risk of Falls in a High-Risk Group With Dementia. Journal of the American Medical Directors Association. DOI: `10.1016/j.jamda.2020.07.030`。
2. Adeli V, et al. Ambient Monitoring of Gait and Machine Learning Models for Dynamic and Short-Term Falls Risk Assessment in People With Dementia. IEEE Journal of Biomedical and Health Informatics. DOI: `10.1109/JBHI.2023.3267039`。
3. Cox DR. Regression Models and Life-Tables. DOI: `10.1111/j.2517-6161.1972.tb00899.x`。
4. van Houwelingen HC. Dynamic Prediction by Landmarking in Event History Analysis. DOI: `10.1111/j.1467-9469.2007.00529.x`。

## 数据模型

### 每日观测

`DailyObservation` 包含：

- `person_id`：个人标识。
- `observed_at`：ISO 8601 日期时间。
- `features`：步态和姿态指标字典。
- `quality`：有效帧数、有效步数、平均置信度和有效步行时长。
- `psychological`：可选 FES-I 等心理辅助字段。
- `source`：视频、姿态缓存或外部设备来源。

首版支持的核心指标：

- `gait_speed_proxy`：同一摄像机条件下的归一化步态速度代理量。
- `cadence_spm`：每分钟步数。
- `step_time_mean_s`、`step_time_cv`：平均步时和变异系数。
- `stance_width_norm`：双踝距离除以肩宽的归一化站距。
- `gait_asymmetry`：左右步时不对称性。
- `trunk_tilt_mean_deg`、`trunk_sway_rms_deg`：躯干倾斜均值与摆动均方根。
- `aspect_ratio_mean`、`aspect_ratio_cv`：人体框宽高比及变异。
- `emos_proxy`：质心投影相对双脚支撑边界的归一化稳定裕度代理量。

名称中带 `proxy` 的指标必须在输出中保留该后缀，因为单目姿态无法等价替代实验室测力台或完整生物力学测量。

### 个人画像

`PersonalBaseline` 保存：

- 14个以上有效日的每项特征中位数、MAD、有效样本数。
- 初始锚点基线，不随时间自动覆盖。
- 最近低偏离观测形成的慢速参考线，只用于识别摄像机漂移和短期比较。
- 画像版本、创建时间、最近更新时间和配置快照。

### 模型产物

每个预测窗口对应一个 JSON 产物，包含系数、特征顺序、训练时间、训练样本摘要、基线累积风险、校准状态和证据标识。缺少 `calibrated=true` 时，推理层拒绝生成概率。

## 数据质量门控

每日观测只有同时满足以下条件才可进入基线或风险计算：

- 平均姿态置信度不低于0.5。
- 有效帧数不少于150。
- 有效步数不少于20。
- 有效步行时长不少于10秒。
- 核心特征至少80%为有限数值。

这些阈值是工程数据质量默认值，不是临床诊断阈值，全部集中在配置文件中并写入输出元数据。低质量观测返回原因列表，不静默填零。

## 基线和偏离计算

每项特征使用中位数作为中心，以 `1.4826 × MAD` 估计稳健尺度。尺度为零时退化为 IQR/1.349；仍为零时将该特征标记为不可用于偏离计算。

单项偏离为：

`z_j = (x_j - median_j) / robust_scale_j`

综合偏离为有效特征稳健 z 分数的均方根，并截断单项绝对值到5，避免一个坏点完全支配结果：

`distance = sqrt(mean(clip(z_j, -5, 5)^2))`

对外 `deviation_index = min(distance / 5, 1)`。同时返回完整 `feature_deviations`，不使用不可解释的临床权重。

持续异常定义为连续3个有效观测日 `distance >= 3`。这只产生 `statistical_deviation` 提示，不直接等同于未来跌倒。

## 动态生存模型

训练数据按 Landmark 时点构造。对每个个人的每个有效观测日，使用当日偏离、近7天趋势、持续异常天数和可选辅助变量作为协变量，并分别构造7天与30天结局：

- `duration_days = min(距下一次跌倒天数, 预测窗口)`。
- `event = 1` 当且仅当预测窗口内发生跌倒。
- 无跌倒时按随访截止时间删失。

训练器使用 Cox 比例风险模型，并按 `person_id` 聚类计算稳健方差。产物只有在训练、验证和概率校准流程显式标记后才能设为 `calibrated=true`。

推理时根据 Cox 公式计算：

`risk_h = 1 - S0(h) ^ exp(beta · x)`

其中 `h` 为7天或30天，`S0(h)` 和 `beta` 必须来自对应模型产物。项目不内置虚构系数或示例百分比。

## 输出契约

统一输出 `FutureRiskResult`：

```json
{
  "person_id": "elder_001",
  "status": "calibration_required",
  "baseline_valid_days": 14,
  "observation_date": "2026-11-11",
  "deviation_index": 0.63,
  "distance": 3.15,
  "persistent_deviation_days": 3,
  "feature_deviations": {"step_time_cv": 3.6},
  "risk_7d": null,
  "risk_30d": null,
  "quality_reasons": [],
  "message": "检测到持续个人基线偏离；当前结果未经跌倒概率校准"
}
```

状态枚举：

- `insufficient_quality`：本次数据质量不足。
- `collecting`：有效基线不足14天。
- `baseline_ready`：基线可用，当前未达到持续异常。
- `calibration_required`：检测到持续统计偏离，但没有校准模型。
- `calibrated`：已加载校准模型并输出7天和30天概率。

## 存储与命令

个人数据默认存放在用户指定的 `--profile-dir`，不提交仓库：

```text
profiles/<person_id>/observations.jsonl
profiles/<person_id>/baseline.json
profiles/<person_id>/latest_risk.json
```

提供四个中文帮助命令：

- `scripts/collect_longitudinal.py`：从姿态缓存或特征 JSON 追加每日观测。
- `scripts/build_personal_baseline.py`：质量筛选并建立个人基线。
- `scripts/predict_future_fall.py`：计算偏离并按模型可用性输出概率。
- `scripts/train_landmark_cox.py`：从纵向 CSV 构建 Landmark 样本并训练模型。

## 错误处理

- 日期、字段或数值无效时返回明确中文错误并以非零状态退出。
- 重复日期默认拒绝；显式 `--replace` 才允许替换。
- 画像配置与当前配置不一致时拒绝推理，避免量纲混用。
- 模型特征顺序、窗口或校准标志不一致时拒绝输出概率。
- 原始心理字段缺失时保持缺失，不填充成低风险。

## 验证策略

使用标准库 `unittest`，不要求额外测试框架。测试覆盖：

- 数据质量通过与拒绝原因。
- 14日基线边界、MAD/IQR退化和无效特征。
- 单项与多项偏离、异常值截断、连续异常计数。
- 未校准模型不输出概率。
- 校准产物的 Cox 概率公式与窗口校验。
- JSONL 重复日期和替换语义。
- CLI 端到端：观测导入、基线构建、偏离预测。
- 现有 `scripts/fuse_risk.py` 未传新参数时保持原行为。

## 范围外事项

- 不生成或宣称临床验证结果。
- 不内置虚构跌倒概率、系数或阈值。
- 不修改现有 ST-GCN、LSTM、环境模型和实时告警阈值。
- 不增加前端页面、数据库服务或账号系统。
