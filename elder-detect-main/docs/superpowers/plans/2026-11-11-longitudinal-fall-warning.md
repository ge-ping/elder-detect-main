# 老年人纵向跌倒预警 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 增加个人14日步态姿态基线、稳健偏离监测和可加载校准 Landmark Cox 模型的7天/30天未来跌倒预警。

**Architecture:** 新建独立 `longitudinal` 包，按数据契约、质量门控、存储、基线、偏离和生存推理分层。四个 CLI 负责观测导入、画像建立、风险预测和模型训练；现有融合脚本只增加可选结果附加入口，默认行为不变。

**Tech Stack:** Python 3.10+、numpy、标准库 dataclasses/json/csv/unittest；训练命令使用新增依赖 lifelines 和 pandas，运行时 Cox 推理只依赖 numpy。

## Global Constraints

- 所有说明、CLI 帮助、错误信息和文档使用中文。
- `risk_7d`、`risk_30d` 仅在模型产物 `calibrated=true` 时输出数值。
- 心理因素仅作为可选训练协变量，不通过人工权重修改未校准偏离指数。
- 不修改现有 ST-GCN、LSTM、环境模型、实时风险阈值和默认命令行为。
- 预测窗口固定为7天和30天，分别由 Mehdizadeh 2021 的 Cox 研究直接支持。
- 名称中带 `proxy` 的单目姿态指标在接口和文档中保留该后缀。

---

### Task 1: 数据契约与质量门控

**Files:**
- Create: `longitudinal/__init__.py`
- Create: `longitudinal/config.py`
- Create: `longitudinal/schema.py`
- Create: `longitudinal/quality.py`
- Test: `tests/test_longitudinal_schema_quality.py`

**Interfaces:**
- Consumes: 每日特征、质量指标和可选心理字段字典。
- Produces: `LongitudinalConfig`、`DailyObservation.from_dict()`、`QualityAssessment`、`assess_quality()`。

- [ ] **Step 1: 写失败测试**：验证日期规范化、非有限数值拒绝、默认质量阈值和中文拒绝原因。
- [ ] **Step 2: 运行测试确认失败**：`python -m unittest tests.test_longitudinal_schema_quality -v`，预期因模块不存在失败。
- [ ] **Step 3: 实现最小数据类和质量判断**：配置包含14日基线、150帧、20步、10秒、0.5置信度和80%核心特征覆盖率；数据类提供 `to_dict/from_dict`。
- [ ] **Step 4: 运行测试确认通过**：同 Step 2，预期全部通过。

### Task 2: JSONL 个人存储

**Files:**
- Create: `longitudinal/storage.py`
- Test: `tests/test_longitudinal_storage.py`

**Interfaces:**
- Consumes: `DailyObservation`、`PersonalBaseline` 和 `FutureRiskResult`。
- Produces: `ProfileStore.append_observation(observation, replace=False)`、`load_observations()`、`save_baseline()`、`load_baseline()`、`save_latest_result()`。

- [ ] **Step 1: 写失败测试**：覆盖目录创建、日期排序、重复日期拒绝、`replace=True` 替换和原子 JSON 写入。
- [ ] **Step 2: 运行测试确认失败**：`python -m unittest tests.test_longitudinal_storage -v`。
- [ ] **Step 3: 实现存储**：路径限定在传入的 profile 根目录内，JSONL 使用 UTF-8，单文件写入通过临时文件后原子替换。
- [ ] **Step 4: 运行测试确认通过**。

### Task 3: 个人基线与稳健偏离

**Files:**
- Create: `longitudinal/baseline.py`
- Create: `longitudinal/deviation.py`
- Test: `tests/test_longitudinal_baseline_deviation.py`

**Interfaces:**
- Consumes: 质量合格的 `list[DailyObservation]`。
- Produces: `BaselineFeatureStats`、`PersonalBaseline`、`build_baseline()`、`compute_deviation()`、`count_persistent_deviation()`。

- [ ] **Step 1: 写失败测试**：覆盖不足14日、刚好14日、MAD尺度、IQR退化、常量特征排除、z分数截断、RMS距离和连续3日异常。
- [ ] **Step 2: 运行测试确认失败**：`python -m unittest tests.test_longitudinal_baseline_deviation -v`。
- [ ] **Step 3: 实现基线**：中位数、`1.4826*MAD`、`IQR/1.349`，保存配置指纹和有效日期。
- [ ] **Step 4: 实现偏离**：返回距离、0到1偏离指数、各特征 z 分数和有效特征数。
- [ ] **Step 5: 运行测试确认通过**。

### Task 4: 姿态日摘要特征

**Files:**
- Create: `longitudinal/pose_features.py`
- Test: `tests/test_longitudinal_pose_features.py`

**Interfaces:**
- Consumes: `(T, J, 2|3)` 关键点数组、置信度数组、时间戳和关节索引映射。
- Produces: `summarize_pose_sequence()` 返回核心特征、有效帧数、步数、步行秒数和平均置信度。

- [ ] **Step 1: 写失败测试**：用合成周期踝轨迹验证步事件、步频、步时均值/CV、站距、左右不对称、躯干倾角/摆动、宽高比和 `emos_proxy` 的有限输出。
- [ ] **Step 2: 运行测试确认失败**：`python -m unittest tests.test_longitudinal_pose_features -v`。
- [ ] **Step 3: 实现特征提取**：对低置信度帧置无效，使用踝部相对轨迹峰值检测步事件；所有空间量按肩宽或身高代理归一化。
- [ ] **Step 4: 运行测试确认通过**。

### Task 5: Cox 模型产物与安全推理

**Files:**
- Create: `longitudinal/survival.py`
- Test: `tests/test_longitudinal_survival.py`

**Interfaces:**
- Consumes: 模型 JSON、推理特征字典。
- Produces: `CoxModelArtifact.from_dict()`、`predict_cumulative_risk()`、`load_model_bundle()`。

- [ ] **Step 1: 写失败测试**：验证 Cox 公式、缺特征、错误窗口、未校准产物、概率边界和7/30模型成对加载。
- [ ] **Step 2: 运行测试确认失败**：`python -m unittest tests.test_longitudinal_survival -v`。
- [ ] **Step 3: 实现推理**：严格校验 `calibrated`、窗口、特征顺序、有限系数和 `baseline_survival` 范围，不依赖 lifelines。
- [ ] **Step 4: 运行测试确认通过**。

### Task 6: 纵向风险服务

**Files:**
- Create: `longitudinal/service.py`
- Test: `tests/test_longitudinal_service.py`

**Interfaces:**
- Consumes: `ProfileStore`、`LongitudinalConfig` 和可选7/30天模型包。
- Produces: `FutureRiskResult`、`LongitudinalRiskService.evaluate(person_id, observation)`。

- [ ] **Step 1: 写失败测试**：覆盖五种状态、未校准概率为 null、持续偏离提示、配置指纹不匹配和校准模型概率输出。
- [ ] **Step 2: 运行测试确认失败**：`python -m unittest tests.test_longitudinal_service -v`。
- [ ] **Step 3: 实现服务编排**：质量不合格不写基线；基线不足返回 collecting；模型缺失时只返回偏离；模型存在时返回7天和30天概率。
- [ ] **Step 4: 运行测试确认通过**。

### Task 7: 四个中文 CLI

**Files:**
- Create: `scripts/collect_longitudinal.py`
- Create: `scripts/build_personal_baseline.py`
- Create: `scripts/predict_future_fall.py`
- Create: `scripts/train_landmark_cox.py`
- Modify: `requirements.txt`
- Test: `tests/test_longitudinal_cli.py`

**Interfaces:**
- Consumes: 特征 JSON、姿态 NPZ、纵向 CSV 和 profile 目录。
- Produces: 每日 JSONL、`baseline.json`、`latest_risk.json` 和7/30天模型 JSON。

- [ ] **Step 1: 写失败测试**：在临时目录运行特征 JSON 导入、14日基线建立和预测，验证中文 JSON 输出及 `risk_7d/risk_30d=null`。
- [ ] **Step 2: 运行测试确认失败**：`python -m unittest tests.test_longitudinal_cli -v`。
- [ ] **Step 3: 实现观测导入 CLI**：支持标准特征 JSON 和包含 `landmarks/timestamps/visibility` 的 NPZ，重复日期需 `--replace`。
- [ ] **Step 4: 实现基线和预测 CLI**：所有输出 UTF-8，错误以中文写入 stderr 并返回非零状态。
- [ ] **Step 5: 实现 Landmark Cox 训练 CLI**：用 pandas/lifelines 构造7天、30天 Landmark 数据，按 person 聚类拟合，默认导出 `calibrated=false`；只有显式提供独立验证指标和 `--mark-calibrated` 才标记为 true。
- [ ] **Step 6: 增加依赖**：在 `requirements.txt` 增加 `pandas>=2.0.0`、`lifelines>=0.30.0`。
- [ ] **Step 7: 运行测试确认通过**。

### Task 8: 现有融合输出的可选接入

**Files:**
- Modify: `scripts/fuse_risk.py`
- Test: `tests/test_fuse_future_risk.py`

**Interfaces:**
- Consumes: `--future-risk-json` 指向 `latest_risk.json`。
- Produces: 汇总 JSON 中可选 `future_risk` 字段；未传参数时不增加字段。

- [ ] **Step 1: 写失败测试**：直接测试纯函数 `load_future_risk(path)`，并验证未传参数时汇总结构不变。
- [ ] **Step 2: 运行测试确认失败**：`python -m unittest tests.test_fuse_future_risk -v`。
- [ ] **Step 3: 实现最小接入**：只读取并校验已生成结果，不重新计算、不改变现有短期或长期风险。
- [ ] **Step 4: 运行测试确认通过**。

### Task 9: 中文使用与证据文档

**Files:**
- Create: `docs/纵向个人基线跌倒预警.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: 最终 CLI 和 JSON 契约。
- Produces: 安装、采集、基线、预测、训练、解释和文献映射说明。

- [ ] **Step 1: 编写文档**：明确区分偏离指数和概率，列出 DOI、时间窗依据、数据格式和完整命令示例。
- [ ] **Step 2: 检查命令帮助**：运行四个脚本 `--help`，确保参数名和文档一致。
- [ ] **Step 3: 运行文档示例的无模型流程**：确认结果状态为 `baseline_ready` 或 `calibration_required`，概率字段为 null。

### Task 10: 全量验证

**Files:**
- Modify: `docs/superpowers/plans/2026-11-11-longitudinal-fall-warning.md`

**Interfaces:**
- Consumes: Tasks 1-9 的实现。
- Produces: 完成勾选和验证证据。

- [ ] **Step 1: 运行纵向模块全量测试**：`python -m unittest discover -s tests -p 'test_longitudinal*.py' -v`，预期全部通过。
- [ ] **Step 2: 运行融合接入测试**：`python -m unittest tests.test_fuse_future_risk -v`，预期全部通过。
- [ ] **Step 3: 运行现有轻量烟雾测试**：`python smoke_test_body26.py`，预期退出码0。
- [ ] **Step 4: 编译检查**：`python -m compileall longitudinal scripts tests`，预期退出码0。
- [ ] **Step 5: 扫描虚假概率**：`rg -n 'risk_7d.*[0-9]\.|risk_30d.*[0-9]\.' longitudinal scripts docs/纵向个人基线跌倒预警.md`，确认没有硬编码示例概率。
