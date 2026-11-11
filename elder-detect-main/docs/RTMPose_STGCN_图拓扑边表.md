# RTMPose + ST-GCN 图拓扑边表（V=21）

> 配套：`docs/RTMPose_STGCN_跌倒风险设计.md` §6.3、`docs/RTMPose_STGCN_开发任务清单.md` 阶段 3
> 分支：`feature/rtmpose-stgcn-fall-risk`
> 状态：**设计规格（边表数据 + 分区规则），非实现代码**
> 用途：`model/stgcn/graph.py` 实现时的唯一事实源——邻接矩阵 A 必须与本表严格一致，否则图卷积失效。

---

## 1. 前提：HALPE-26 原始关键点顺序

RTMPose body+foot 输出 **HALPE-26** 标准，26 点原始索引（0-based）如下：

| 原索引 | 名称 | 类别 |
|:---:|------|------|
| 0 | nose 鼻 | 面部（**剔除**） |
| 1 | left_eye 左眼 | 面部（**剔除**） |
| 2 | right_eye 右眼 | 面部（**剔除**） |
| 3 | left_ear 左耳 | 面部（**剔除**） |
| 4 | right_ear 右耳 | 面部（**剔除**） |
| 5 | left_shoulder 左肩 | 上肢 |
| 6 | right_shoulder 右肩 | 上肢 |
| 7 | left_elbow 左肘 | 上肢 |
| 8 | right_elbow 右肘 | 上肢 |
| 9 | left_wrist 左腕 | 上肢 |
| 10 | right_wrist 右腕 | 上肢 |
| 11 | left_hip 左髋 | 下肢 |
| 12 | right_hip 右髋 | 下肢 |
| 13 | left_knee 左膝 | 下肢 |
| 14 | right_knee 右膝 | 下肢 |
| 15 | left_ankle 左踝 | 下肢 |
| 16 | right_ankle 右踝 | 下肢 |
| 17 | head 头顶 | 中轴（保留） |
| 18 | neck 颈 | 中轴（保留） |
| 19 | hip 骨盆中心 | 中轴（保留，重心参考点） |
| 20 | left_big_toe 左脚尖 | 脚 |
| 21 | right_big_toe 右脚尖 | 脚 |
| 22 | left_small_toe 左小趾 | 脚 |
| 23 | right_small_toe 右小趾 | 脚 |
| 24 | left_heel 左脚跟 | 脚 |
| 25 | right_heel 右脚跟 | 脚 |

> ⚠️ 若实际部署的 RTMPose 权重关键点顺序与上表不同（不同 config 可能有出入），**必须以推理输出实测为准**重新核对，再套用本重映射。

---

## 2. 去面部 5 点后的 21 点索引（本构建正式索引）

剔除原索引 `0,1,2,3,4`（鼻/眼/耳），其余 21 点**重新编号为 0~20**（这是 ST-GCN 图节点的正式索引）：

| 新索引 V | 原 HALPE 索引 | 名称 | 左右 |
|:---:|:---:|------|:---:|
| 0 | 5 | left_shoulder 左肩 | L |
| 1 | 6 | right_shoulder 右肩 | R |
| 2 | 7 | left_elbow 左肘 | L |
| 3 | 8 | right_elbow 右肘 | R |
| 4 | 9 | left_wrist 左腕 | L |
| 5 | 10 | right_wrist 右腕 | R |
| 6 | 11 | left_hip 左髋 | L |
| 7 | 12 | right_hip 右髋 | R |
| 8 | 13 | left_knee 左膝 | L |
| 9 | 14 | right_knee 右膝 | R |
| 10 | 15 | left_ankle 左踝 | L |
| 11 | 16 | right_ankle 右踝 | R |
| 12 | 17 | head 头顶 | 中 |
| 13 | 18 | neck 颈 | 中 |
| 14 | 19 | **hip 骨盆中心（重心）** | 中 |
| 15 | 20 | left_big_toe 左脚尖 | L |
| 16 | 21 | right_big_toe 右脚尖 | R |
| 17 | 22 | left_small_toe 左小趾 | L |
| 18 | 23 | right_small_toe 右小趾 | R |
| 19 | 24 | left_heel 左脚跟 | L |
| 20 | 25 | right_heel 右脚跟 | R |

> **重映射规则**：`new_idx = original_idx - 5`（因剔除的恰是最前 5 个连续面部点，故整体前移 5）。实现时务必写死此映射并加单测，避免顺序错位。

---

## 3. 骨骼边表（无向边，用新索引 0~20）

以人体真实骨骼连接定义。共 **22 条边**（含脚部小环）。

### 3.1 中轴（2 条）
| 边 | 新索引 | 含义 |
|----|:---:|------|
| head — neck | (12, 13) | 头—颈 |
| neck — pelvis | (13, 14) | 颈—骨盆（躯干主干） |

### 3.2 上肢（6 条）
| 边 | 新索引 | 含义 |
|----|:---:|------|
| neck — L_shoulder | (13, 0) | 颈—左肩 |
| neck — R_shoulder | (13, 1) | 颈—右肩 |
| L_shoulder — L_elbow | (0, 2) | 左肩—左肘 |
| L_elbow — L_wrist | (2, 4) | 左肘—左腕 |
| R_shoulder — R_elbow | (1, 3) | 右肩—右肘 |
| R_elbow — R_wrist | (3, 5) | 右肘—右腕 |

### 3.3 下肢（6 条）
| 边 | 新索引 | 含义 |
|----|:---:|------|
| pelvis — L_hip | (14, 6) | 骨盆—左髋 |
| pelvis — R_hip | (14, 7) | 骨盆—右髋 |
| L_hip — L_knee | (6, 8) | 左髋—左膝 |
| L_knee — L_ankle | (8, 10) | 左膝—左踝 |
| R_hip — R_knee | (7, 9) | 右髋—右膝 |
| R_knee — R_ankle | (9, 11) | 右膝—右踝 |

### 3.4 脚部（8 条，含小环）
| 边 | 新索引 | 含义 |
|----|:---:|------|
| L_ankle — L_heel | (10, 19) | 左踝—左脚跟 |
| L_ankle — L_big_toe | (10, 15) | 左踝—左脚尖 |
| L_heel — L_big_toe | (19, 15) | 左脚跟—左脚尖（成三角，增稳） |
| L_big_toe — L_small_toe | (15, 17) | 左脚尖—左小趾 |
| R_ankle — R_heel | (11, 20) | 右踝—右脚跟 |
| R_ankle — R_big_toe | (11, 16) | 右踝—右脚尖 |
| R_heel — R_big_toe | (20, 16) | 右脚跟—右脚尖（成三角，增稳） |
| R_big_toe — R_small_toe | (16, 18) | 右脚尖—右小趾 |

### 3.5 可选加强边（默认不启用，消融实验用）
| 边 | 新索引 | 用途 |
|----|:---:|------|
| L_shoulder — L_hip | (0, 6) | 强化躯干刚性（左侧闭环） |
| R_shoulder — R_hip | (1, 7) | 强化躯干刚性（右侧闭环） |
| L_shoulder — R_shoulder | (0, 1) | 肩带横向约束 |
| L_hip — R_hip | (6, 7) | 髋带横向约束 |

> 主边表为树+脚部小环结构；可选边形成躯干闭环，实现阶段作消融对比（§开发清单阶段 10.3）。

### 3.6 边表数据（供实现直接引用）
```
# 主骨骼边（22 条，新索引，无向）
EDGES = [
    (12,13),(13,14),                                  # 中轴
    (13,0),(13,1),(0,2),(2,4),(1,3),(3,5),            # 上肢
    (14,6),(14,7),(6,8),(8,10),(7,9),(9,11),          # 下肢
    (10,19),(10,15),(19,15),(15,17),                  # 左脚
    (11,20),(11,16),(20,16),(16,18),                  # 右脚
]
# 自环（ST-GCN 需给每节点加 self-loop）
SELF_LOOPS = [(i, i) for i in range(21)]
# 重心参考节点（向心/离心分区用）
CENTER = 14   # pelvis 骨盆中心
```

---

## 4. ST-GCN 空间分区（root / centripetal / centrifugal）

ST-GCN 对每个节点的邻居按「到重心的距离」分三类子图，分别用独立卷积核再求和。重心取 **pelvis（新索引 14）**。

### 4.1 各节点到重心的 hop 距离（沿骨骼边最短路）
| hop | 节点（新索引） |
|:---:|------|
| 0 | 14(pelvis) |
| 1 | 6(L_hip), 7(R_hip), 13(neck) |
| 2 | 8(L_knee), 9(R_knee), 12(head), 0(L_shoulder), 1(R_shoulder) |
| 3 | 10(L_ankle), 11(R_ankle), 2(L_elbow), 3(R_elbow) |
| 4 | 4(L_wrist), 5(R_wrist), 15(L_big_toe), 19(L_heel), 16(R_big_toe), 20(R_heel) |
| 5 | 17(L_small_toe), 18(R_small_toe) |

### 4.2 分区规则（对每条边的两端）
对节点 `i` 及其邻居 `j`：
- **root**：`j == i`（自身，self-loop）
- **centripetal（向心）**：`hop(j) < hop(i)`——邻居更靠近重心
- **centrifugal（离心）**：`hop(j) > hop(i)`——邻居更远离重心

据此把邻接矩阵 A 拆成 `A_root + A_centripetal + A_centrifugal` 三个子矩阵，每个配一组卷积权重（ST-GCN 原文 partition strategy = "spatial configuration"）。

> 同 hop 的横向边（如可选边 0—1 肩—肩）归入哪类需单独约定：一般并入 root 或单列，实现时在 `graph.py` 注释明确。

---

## 5. 实现与验收（对应开发清单阶段 3）

- `model/stgcn/graph.py` 依本表构建 `A`（21×21），拆三分区。
- **单测必须断言**：
  1. 节点数 == 21；
  2. `EDGES` 中所有索引 ∈ [0,20]；
  3. 无面部点残留（原 0~4 已全部剔除）；
  4. 每条边两端 hop 差 ≤ 1（骨骼相邻性自检）；
  5. 重映射 `new = original - 5` 与关键点名称表逐一对应。
- 可视化：把 21 点按典型坐标画出 + 连边，肉眼确认是完整人形（含脚、无脸）。

---

## 6. 备注
- 本表基于 HALPE-26 通用定义；**上线前用实际 RTMPose 权重实测关键点顺序复核**（§1 警告）。
- 后续若改用全 26 点（保留面部）或换 body-17，边表与分区须重做，本文件仅对应 **V=21** 方案。
- 通道仍 `C=3`（x,y,conf）；扩通道（速度/角度）不影响图拓扑，只改输入 C，不改本边表。
