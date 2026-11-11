# 跌倒风险检测系统 · 接口文档（修订版 v2）

> **通用规范（全局约定，所有模块生效）**
>
> 1. **鉴权**：除登录类接口外，所有请求须带请求头 `Authorization: Bearer <JWT>`。服务端从 token 解析当前用户身份，`processedBy` 等用户字段**由服务端填充，前端不传**。
> 2. **错误码体系（一码一义，按模块分段）**：`401xx` 通用 / `41xxx` 任务 / `42xxx` 事件 / `43xxx` 设备 / `44xxx` 系统。完整映射见文末《错误码总表》。
> 3. **数据类型约定**：百分比一律返回 `number`（0~1，如 `0.0897`），前端自行格式化；内存/显存一律返回整数 `MB`（如 `gpuMemoryUsedMB: 19800`）；时间戳统一 ISO 8601 字符串。
> 4. **枚举命名去歧义**：任务状态用 `taskStatus`（`running`/`connecting`/`failed`/`stopped`）；事件状态用 `eventStatus`（`unprocessed`/`handled`/`false_positive`）；设备状态用 `cameraStatus`（`online`/`offline`）；系统状态用 `systemStatus`（`healthy`/`warning`/`critical`）。
> 5. **优先级约定**：`sensitivity` 取值优先级为 **任务级 > 摄像头级 > 全局默认**；建任务时 `streamUrl` 可选，未传则取 `cameraId` 在模块三注册的地址，传了则任务级覆盖设备级。
> 6. **bbox 坐标系**：统一为**归一化** `[x1, y1, x2, y2]`（0.00~1.00，相对 snapshot 原图宽高），不再混合像素坐标。

---

# 模块一：视频流与分析任务管理接口文档

该模块是整个跌倒风险检测系统的核心引擎入口，主要负责调度算力、拉取视频流并执行 AI 推理。

## 1.1 创建/启动视频流分析任务

| 属性 | 说明 |
| --- | --- |
| **接口描述** | 提交摄像头视频流地址，分配 GPU 算力并启动跌倒风险分析任务，返回 WebSocket 告警通道地址。 |
| **请求路径** | `/api/v1/fall-risk/tasks` |
| **请求方式** | `POST` |

### 请求参数 (Body)

| 参数名 | 必填 | 类型 | 说明 | 示例 |
| --- | --- | --- | --- | --- |
| `cameraId` | 是 | String | 摄像头的唯一标识符（须已在模块三注册） | `cam_ward_A1_001` |
| `streamUrl` | 否 | String | 实时流地址 (RTSP/RTMP/HLS)。<br>不传则取 `cameraId` 在模块三注册的地址；传了则以任务级覆盖设备级。 | `rtsp://10.0.0.15:554/live/stream` |
| `analyzeMode` | 是 | Integer | 算力分配模式。<br>`1`: 实时高帧率 (延迟 < 0.5s，高算力)<br>`2`: 抽帧低功耗 (每秒2帧，低算力) | `1` |
| `sensitivity` | 否 | String | 模型灵敏度，默认 `medium`。<br>`low`: 仅严重跌倒<br>`medium`: 标准报警<br>`high`: 疑似不稳即报警。<br>优先级低于模块三的摄像头级配置。 | `high` |

### 响应数据 (Success 200)

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "taskId": "task_fw29dj391x",
    "status": "connecting",
    "wsUrl": "wss://ai.domain.com/ws/fall-risk/task_fw29dj391x"
  }
}
```

### 业务异常说明 (Exceptions)

| 错误码 | 错误信息 | 触发场景 | 前端处理建议 |
| --- | --- | --- | --- |
| `41001` | Invalid Stream URL | `streamUrl` 格式错误或协议不支持。 | 提示“请输入有效的 RTSP/RTMP/HLS 流地址”。 |
| `41002` | Camera Already Analyzing | 该摄像头当前已有正在运行的分析任务。 | 提示“该设备已在分析中，请勿重复创建”。 |
| `43001` | Camera Not Found | `cameraId` 在模块三中不存在。 | 提示“摄像头未注册，请先在设备管理中添加”。 |
| `44001` | GPU Resource Exhausted | 服务器当前 GPU 显存耗尽。 | 提示“当前系统算力已满，请停止其他任务或稍后再试”。 |
| `40103` | Invalid Parameter | `analyzeMode` 非 1/2，或必填字段缺失。 | 表单校验拦截，提示“参数有误”。 |

---

## 1.2 停止视频流分析任务

| 属性 | 说明 |
| --- | --- |
| **接口描述** | 停止指定的视频流分析任务，主动断开视频拉流并释放底层的 GPU 计算资源。 |
| **请求路径** | `/api/v1/fall-risk/tasks/{taskId}` |
| **请求方式** | `DELETE` |

### 请求参数 (Path)

| 参数名 | 必填 | 类型 | 说明 | 示例 |
| --- | --- | --- | --- | --- |
| `taskId` | 是 | String | 分析任务的全局唯一标识 ID | `task_fw29dj391x` |

### 响应数据 (Success 200)

```json
{
  "code": 200,
  "message": "Task stopped and resources released.",
  "data": null
}
```

### 业务异常说明 (Exceptions)

| 错误码 | 错误信息 | 触发场景 | 前端处理建议 |
| --- | --- | --- | --- |
| `41003` | Task Not Found | 传入的 `taskId` 不存在或已过期。 | 刷新任务列表，该任务可能已被其他终端停止。 |
| `41004` | Task Already Stopped | 任务处于非运行状态（如已因为断流自动停止）。 | 忽略该错误，直接在前端列表中将该任务移除。 |

---

## 1.3 获取当前运行任务列表

| 属性 | 说明 |
| --- | --- |
| **接口描述** | 分页查询当前系统中正在运行或连接中的分析任务，用于前端监控面板展示大盘状态。 |
| **请求路径** | `/api/v1/fall-risk/tasks` |
| **请求方式** | `GET` |

### 请求参数 (Query)

| 参数名 | 必填 | 类型 | 说明 | 示例 |
| --- | --- | --- | --- | --- |
| `taskStatus` | 否 | String | 任务状态筛选。<br>`running`: 推理中<br>`connecting`: 拉流重连中<br>`failed`: 拉流失败中断<br>`stopped`: 已停止（原文档遗漏，此处补回） | `running` |
| `cameraId` | 否 | String | 指定摄像头搜索 | `cam_ward_A1_001` |
| `page` | 否 | Integer | 当前页码，默认 1 | `1` |
| `pageSize` | 否 | Integer | 每页数据量，默认 20，最大 50 | `20` |

### 响应数据 (Success 200)

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "total": 12,
    "page": 1,
    "pageSize": 20,
    "list": [
      {
        "taskId": "task_fw29dj391x",
        "cameraId": "cam_ward_A1_001",
        "streamUrl": "rtsp://10.0.0.15:554/live/stream",
        "analyzeMode": 1,
        "taskStatus": "running",
        "fps": 24.5,
        "uptimeSeconds": 3600,
        "createdAt": "2026-07-16T17:00:00Z"
      },
      {
        "taskId": "task_xp83md922a",
        "cameraId": "cam_hallway_B2",
        "streamUrl": "rtsp://10.0.0.22:554/live/stream",
        "analyzeMode": 2,
        "taskStatus": "connecting",
        "fps": 0,
        "uptimeSeconds": 0,
        "createdAt": "2026-07-16T18:15:00Z"
      }
    ]
  }
}
```

*字段说明：`fps` 表示当前模型实际处理的帧率（可用于判断视频流是否卡顿）；`uptimeSeconds` 表示任务已持续运行的秒数。*

### 业务异常说明 (Exceptions)

| 错误码 | 错误信息 | 触发场景 | 前端处理建议 |
| --- | --- | --- | --- |
| `40102` | Invalid Pagination | `pageSize` 超过最大限制 50。 | 前端修正传参，避免一次请求过多数据导致数据库和网络压力。 |

---

## 1.4 获取单个任务详情（新增接口）

| 属性 | 说明 |
| --- | --- |
| **接口描述** | 查询指定任务的实时详情，避免前端为看单个任务而拉取全量列表过滤。 |
| **请求路径** | `/api/v1/fall-risk/tasks/{taskId}` |
| **请求方式** | `GET` |

### 请求参数 (Path)

| 参数名 | 必填 | 类型 | 说明 | 示例 |
| --- | --- | --- | --- | --- |
| `taskId` | 是 | String | 分析任务的全局唯一标识 ID | `task_fw29dj391x` |

### 响应数据 (Success 200)

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "taskId": "task_fw29dj391x",
    "cameraId": "cam_ward_A1_001",
    "streamUrl": "rtsp://10.0.0.15:554/live/stream",
    "analyzeMode": 1,
    "taskStatus": "running",
    "fps": 24.5,
    "uptimeSeconds": 3600,
    "sensitivity": "high",
    "createdAt": "2026-07-16T17:00:00Z"
  }
}
```

### 业务异常说明 (Exceptions)

| 错误码 | 错误信息 | 触发场景 | 前端处理建议 |
| --- | --- | --- | --- |
| `41003` | Task Not Found | 传入的 `taskId` 不存在或已过期。 | 提示“任务不存在或已停止”。 |

---

## 1.5 建任务前预检（新增接口）

| 属性 | 说明 |
| --- | --- |
| **接口描述** | 创建任务前预检：摄像头是否已注册、是否已在分析、GPU 是否还有余量，避免先 POST 再吃错误码。 |
| **请求路径** | `/api/v1/fall-risk/tasks/precheck` |
| **请求方式** | `GET` |

### 请求参数 (Query)

| 参数名 | 必填 | 类型 | 说明 | 示例 |
| --- | --- | --- | --- | --- |
| `cameraId` | 是 | String | 待创建任务的摄像头 ID | `cam_ward_A1_001` |

### 响应数据 (Success 200)

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "cameraRegistered": true,
    "alreadyAnalyzing": false,
    "gpuAvailable": true,
    "suggestStreamUrl": "rtsp://10.0.0.15:554/live/stream"
  }
}
```

*字段说明：`alreadyAnalyzing=true` 时前端应提示“该设备已在分析中”；`gpuAvailable=false` 时提示“算力已满”；`suggestStreamUrl` 为模块三注册的流地址，前端可预填。*

### 业务异常说明 (Exceptions)

| 错误码 | 错误信息 | 触发场景 | 前端处理建议 |
| --- | --- | --- | --- |
| `43001` | Camera Not Found | `cameraId` 未注册。 | 提示“摄像头未注册，请先添加设备”。 |

---

## 1.6 WebSocket 告警通道协议（新增）

| 属性 | 说明 |
| --- | --- |
| **接口描述** | 1.1 返回的 `wsUrl` 即告警推送通道。建立连接后须先发送订阅帧，之后服务端实时推送告警，并周期性心跳保活。 |
| **连接地址** | `wss://ai.domain.com/ws/fall-risk/{taskId}`（来自 1.1 的 `wsUrl`） |
| **协议** | 文本帧，JSON 编码 |

### 客户端 → 服务端 消息

| 消息类型 `type` | 字段 | 说明 |
| --- | --- | --- |
| `subscribe` | `taskId` | 订阅指定任务告警（连接后首帧必发） |
| `ping` | — | 客户端心跳，服务端回 `pong` |

### 服务端 → 客户端 消息

| 消息类型 `type` | 字段 | 说明 |
| --- | --- | --- |
| `subscribed` | `taskId` | 订阅成功确认 |
| `alert` | 见下方 AlertPayload | 实时告警（与模块二 `EventItem` 同源，靠 `eventId` 关联） |
| `pong` | — | 心跳回应 |
| `error` | `code`, `message` | 通道级错误（见错误码总表） |

### AlertPayload（与模块二 `EventItem` 字段对齐）

```json
{
  "type": "alert",
  "eventId": "evt_883920194a",
  "taskId": "task_fw29dj391x",
  "cameraId": "cam_ward_A1_001",
  "timestamp": "2026-07-16T18:05:12Z",
  "riskLevel": 3,
  "confidence": 0.96,
  "bbox": [0.15, 0.32, 0.45, 0.68],
  "snapshotUrl": "https://storage.hospital.com/falls/img/evt_883920194a.jpg",
  "videoClipUrl": "https://storage.hospital.com/falls/video/evt_883920194a.mp4"
}
```

*说明：`alert` 推送的每条告警，后端同时落库为模块二的 `EventItem`，二者通过 `eventId` 一一对应，前端可将实时弹窗与历史列表/误报标注对上号。*

---

# 模块二：告警事件闭环管理接口文档

该模块不仅提供历史告警的查询能力，更重要的是引入了“人工确认”机制，这对于过滤误报、优化迭代底层 AI 模型起着关键作用。

## 2.1 查询告警记录列表

| 属性 | 说明 |
| --- | --- |
| **接口描述** | 多维度筛选查询系统记录的跌倒风险事件。支持按时间、摄像头、风险等级以及处理状态进行检索。 |
| **请求路径** | `/api/v1/fall-risk/events` |
| **请求方式** | `GET` |

### 请求参数 (Query)

| 参数名 | 必填 | 类型 | 说明 | 示例 |
| --- | --- | --- | --- | --- |
| `cameraId` | 否 | String | 指定特定摄像头检索 | `cam_ward_A1_001` |
| `riskLevel` | 否 | Integer | 风险等级。<br>`1`: 低风险<br>`2`: 中风险<br>`3`: 高风险 | `3` |
| `eventStatus` | 否 | String | 处理状态。<br>`unprocessed`: 未处理<br>`handled`: 已确认跌倒<br>`false_positive`: 误报 | `unprocessed` |
| `startTime` | 是 | String | 查询起始时间，ISO 8601 格式 | `2026-07-16T00:00:00Z` |
| `endTime` | 是 | String | 查询结束时间，ISO 8601 格式 | `2026-07-16T23:59:59Z` |
| `page` | 否 | Integer | 当前页码，默认 1 | `1` |
| `pageSize` | 否 | Integer | 每页数据量，默认 20，最大 100 | `20` |

### 响应数据 (Success 200)

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "total": 45,
    "page": 1,
    "pageSize": 20,
    "list": [
      {
        "eventId": "evt_883920194a",
        "taskId": "task_fw29dj391x",
        "cameraId": "cam_ward_A1_001",
        "timestamp": "2026-07-16T18:05:12Z",
        "riskLevel": 3,
        "confidence": 0.96,
        "bbox": [0.15, 0.32, 0.45, 0.68],
        "snapshotUrl": "https://storage.hospital.com/falls/img/evt_883920194a.jpg",
        "videoClipUrl": "https://storage.hospital.com/falls/video/evt_883920194a.mp4",
        "eventStatus": "unprocessed"
      }
    ]
  }
}
```

*字段说明：`bbox` 为归一化 `[x1, y1, x2, y2]`（0~1，相对 snapshot 原图）；`taskId` 关联模块一任务，便于溯源。*

### 业务异常说明 (Exceptions)

| 错误码 | 错误信息 | 触发场景 | 前端处理建议 |
| --- | --- | --- | --- |
| `42001` | Invalid Time Range | `startTime` 晚于 `endTime`，或跨度超过 30 天。 | 提示“时间范围有误，最大支持跨度30天”。 |
| `40102` | Invalid Pagination | `pageSize` 超过最大限制 100。 | 前端限制每页条数，传参不超过 100。 |

---

## 2.2 告警状态人工确认 (消除误报)

| 属性 | 说明 |
| --- | --- |
| **接口描述** | 医护或安保人员在前端查看告警图片/视频后，对该告警进行人工标记。标记为“误报”的数据将被自动归档，用于后续模型优化。 |
| **请求路径** | `/api/v1/fall-risk/events/{eventId}/status` |
| **请求方式** | `PUT` |

### 请求参数 (Path)

| 参数名 | 必填 | 类型 | 说明 | 示例 |
| --- | --- | --- | --- | --- |
| `eventId` | 是 | String | 告警事件的唯一标识 ID | `evt_883920194a` |

### 请求参数 (Body)

| 参数名 | 必填 | 类型 | 说明 | 示例 |
| --- | --- | --- | --- | --- |
| `eventStatus` | 是 | String | 目标状态枚举值。<br>`handled`: 确认是跌倒且已处理<br>`false_positive`: 确认为模型误报 | `false_positive` |
| `remark` | 否 | String | 人工处理备注说明，限 200 字以内 | `患者在做俯卧撑，非跌倒` |

*注：`processedBy`（处理人）与 `processedAt`（处理时间）由服务端从 JWT 与当前时间填充，前端不传。*

### 响应数据 (Success 200)

```json
{
  "code": 200,
  "message": "Event status updated successfully.",
  "data": {
    "eventId": "evt_883920194a",
    "updatedStatus": "false_positive",
    "processedBy": "user_nurse_li",
    "processedAt": "2026-07-16T18:10:05Z"
  }
}
```

### 业务异常说明 (Exceptions)

| 错误码 | 错误信息 | 触发场景 | 前端处理建议 |
| --- | --- | --- | --- |
| `42002` | Event Not Found | 传入的 `eventId` 不存在。 | 提示“该告警记录不存在或已被删除”。 |
| `42003` | Invalid Status Value | 传入了不支持的 `eventStatus` 字段值。 | 检查传参，仅支持 `handled` 或 `false_positive`。 |
| `42004` | Event Already Processed | 该事件已经被其他人员处理过了。 | 提示“该事件已被同事处理”，并刷新当前记录状态。 |

---

## 2.3 告警数据聚合统计面板

| 属性 | 说明 |
| --- | --- |
| **接口描述** | 按照指定的时间范围和维度，聚合统计跌倒告警数据。专为大屏展示、报表生成和误报率分析设计。 |
| **请求路径** | `/api/v1/fall-risk/events/stats` |
| **请求方式** | `GET` |

### 请求参数 (Query)

| 参数名 | 必填 | 类型 | 说明 | 示例 |
| --- | --- | --- | --- | --- |
| `startTime` | 是 | String | 统计起始时间，ISO 8601 格式 | `2026-07-09T00:00:00Z` |
| `endTime` | 是 | String | 统计结束时间，ISO 8601 格式 | `2026-07-16T23:59:59Z` |
| `dimension` | 是 | String | 聚合维度。<br>`by_day`: 按天聚合趋势<br>`by_camera`: 按摄像头分组统计 | `by_day` |

### 响应数据 (Success 200)

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "summary": {
      "totalAlerts": 156,
      "trueFalls": 142,
      "falsePositives": 14,
      "unprocessed": 0,
      "falsePositiveRate": 0.0897
    },
    "trends": [
      {
        "label": "2026-07-15",
        "total": 22,
        "handled": 20,
        "falsePositive": 2
      },
      {
        "label": "2026-07-16",
        "total": 18,
        "handled": 18,
        "falsePositive": 0
      }
    ]
  }
}
```

*注：`falsePositiveRate` 现为 `number`（0~1，示例 0.0897 = 8.97%），前端格式化展示；`summary` 额外补回 `unprocessed` 计数，避免大屏数字对不上。`by_camera` 维度下 `trends[].label` 变更为摄像头 ID。*

### 业务异常说明 (Exceptions)

| 错误码 | 错误信息 | 触发场景 | 前端处理建议 |
| --- | --- | --- | --- |
| `42001` | Invalid Time Range | 起止时间格式错误或 `startTime` 晚于 `endTime`。 | 提示“统计时间范围无效，请重新选择”。 |
| `42005` | Invalid Dimension | 传入了不支持的聚合维度。 | 检查传参是否为 `by_day` 或 `by_camera`。 |

---

# 模块三：设备与场景配置接口文档

在实际的 AI 视觉落地场景中，由于摄像头安装角度、背景干扰（如走廊外的行人、电视机画面）等因素，ROI (感兴趣区域) 和独立灵敏度配置是降低误报率最核心的手段。

## 3.1 获取摄像头设备列表

| 属性 | 说明 |
| --- | --- |
| **接口描述** | 分页获取接入当前 AI 分析系统的摄像头设备基础信息列表。 |
| **请求路径** | `/api/v1/cameras` |
| **请求方式** | `GET` |

### 请求参数 (Query)

| 参数名 | 必填 | 类型 | 说明 | 示例 |
| --- | --- | --- | --- | --- |
| `keyword` | 否 | String | 根据摄像头名称或位置进行模糊搜索 | `病房` |
| `cameraStatus` | 否 | String | 设备在线状态。<br>`online`: 在线可拉流<br>`offline`: 离线 | `online` |
| `page` | 否 | Integer | 当前页码，默认 1 | `1` |
| `pageSize` | 否 | Integer | 每页数据量，默认 20，最大 100 | `20` |

### 响应数据 (Success 200)

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "total": 128,
    "page": 1,
    "pageSize": 20,
    "list": [
      {
        "cameraId": "cam_ward_A1_001",
        "name": "A1栋-骨科101病房",
        "location": "A1-F1-Room101",
        "streamUrl": "rtsp://10.0.0.15:554/live/stream",
        "cameraStatus": "online",
        "resolution": "1920x1080",
        "hasRoiConfig": true
      }
    ]
  }
}
```

### 业务异常说明 (Exceptions)

| 错误码 | 错误信息 | 触发场景 | 前端处理建议 |
| --- | --- | --- | --- |
| `40102` | Invalid Pagination | `pageSize` 超过最大限制 100。 | 前端限制每页条数，避免全量拉取卡顿。 |

---

## 3.2 设置感兴趣区域 (ROI) 与屏蔽区

| 属性 | 说明 |
| --- | --- |
| **接口描述** | 为指定摄像头配置 AI 分析区域。包含 `roiPolygons` (只在该区域内检测) 和 `maskPolygons` (绝对不检测的屏蔽区)。 |
| **请求路径** | `/api/v1/cameras/{cameraId}/roi` |
| **请求方式** | `PUT` |

### 请求参数 (Path)

| 参数名 | 必填 | 类型 | 说明 | 示例 |
| --- | --- | --- | --- | --- |
| `cameraId` | 是 | String | 摄像头的唯一标识符 | `cam_ward_A1_001` |

### 请求参数 (Body)

*坐标点**必须**使用归一化数值 (0.00 ~ 1.00)，即 `坐标值 / 画面宽或高`。分辨率切换时 ROI 依然准确生效。*

| 参数名 | 必填 | 类型 | 说明 | 示例 |
| --- | --- | --- | --- | --- |
| `roiPolygons` | 否 | Array | 感兴趣区域（多边形）列表。每个元素是一个坐标点数组 `[[x1,y1], [x2,y2]...]`。不传或为空则全图检测。 | `[[[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]]` |
| `maskPolygons` | 否 | Array | 屏蔽区域（多边形）列表，优先级高于 ROI。用于遮蔽电视机或窗外走动人群。 | `[[[0.7, 0.1], [0.9, 0.1], [0.9, 0.3], [0.7, 0.3]]]` |

### 响应数据 (Success 200)

```json
{
  "code": 200,
  "message": "ROI configuration updated successfully.",
  "data": {
    "cameraId": "cam_ward_A1_001",
    "updatedAt": "2026-07-16T18:30:00Z"
  }
}
```

### 业务异常说明 (Exceptions)

| 错误码 | 错误信息 | 触发场景 | 前端处理建议 |
| --- | --- | --- | --- |
| `43001` | Camera Not Found | 传入的 `cameraId` 系统中不存在。 | 提示“摄像头不存在”，刷新设备树。 |
| `43002` | Invalid Polygon Format | 坐标组无法构成闭合多边形（单组点数少于3个），或包含超出 `0.0~1.0` 范围的数值。 | 弹窗提示前端画图工具：“无效的绘制区域，请确保绘制完整的闭合图形”。 |

---

## 3.3 摄像头独立算法阈值配置

| 属性 | 说明 |
| --- | --- |
| **接口描述** | 针对特定复杂场景（如夜间红外模式噪点多、或者特定视角容易误报），单独下发模型灵敏度和过滤策略，覆盖全局默认配置。 |
| **请求路径** | `/api/v1/cameras/{cameraId}/config` |
| **请求方式** | `PUT` |

### 请求参数 (Path)

| 参数名 | 必填 | 类型 | 说明 | 示例 |
| --- | --- | --- | --- | --- |
| `cameraId` | 是 | String | 摄像头的唯一标识符 | `cam_ward_A1_001` |

### 请求参数 (Body)

| 参数名 | 必填 | 类型 | 说明 | 示例 |
| --- | --- | --- | --- | --- |
| `sensitivity` | 否 | String | 模型检测灵敏度（摄像头级，优先级高于任务级与全局）。<br>`low`: 低，防误报<br>`medium`: 中，均衡<br>`high`: 高，防漏报 | `medium` |
| `minPersonBoxArea` | 否 | Float | 最小人体框面积占比 (0.0~1.0)。识别到的人体框面积小于此值直接丢弃，用于过滤远处无关人员。 | `0.05` |
| `triggerDuration` | 否 | Integer | 持续触发时间(毫秒)。模型连续判断跌倒超过该时长后才上报，过滤弯腰捡东西的瞬间误判。<br>注：与 `analyzeMode` 帧率耦合——模式1(24fps)下≈36帧、模式2(2fps)下≈3帧，二者语义不同，配置时需注意。 | `1500` |

### 响应数据 (Success 200)

```json
{
  "code": 200,
  "message": "Algorithm configuration updated.",
  "data": {
    "cameraId": "cam_ward_A1_001",
    "sensitivity": "medium",
    "minPersonBoxArea": 0.05,
    "triggerDuration": 1500,
    "updatedAt": "2026-07-16T18:35:10Z"
  }
}
```

### 业务异常说明 (Exceptions)

| 错误码 | 错误信息 | 触发场景 | 前端处理建议 |
| --- | --- | --- | --- |
| `43001` | Camera Not Found | 传入的 `cameraId` 系统中不存在。 | 提示设备不存在。 |
| `43003` | Invalid Config Parameter | `triggerDuration` 为负数，或 `minPersonBoxArea` 超出 0~1 合理范围。 | 表单校验拦截，提示“请输入有效的阈值参数”。 |

---

# 模块四：系统与模型状态接口文档

在依赖 GPU 算力的 AI 监控项目中，系统层面的监控接口对于前端判断资源余量、限制新任务下发以及排查线上问题至关重要。

## 4.1 获取系统算力与资源状态

| 属性 | 说明 |
| --- | --- |
| **接口描述** | 实时获取底层 AI 分析服务器的硬件资源占用情况，包括 GPU 显存使用率和当前并发解析的视频流路数。前端据此判断是否允许用户新建分析任务。 |
| **请求路径** | `/api/v1/system/status` |
| **请求方式** | `GET` |

### 请求参数

*(GET 请求，无路径和请求体参数)*

### 响应数据 (Success 200)

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "systemStatus": "healthy",
    "activeStreams": 24,
    "maxStreamsCapacity": 30,
    "hardware": {
      "gpuModel": "NVIDIA RTX 4090",
      "gpuUsageRate": 0.825,
      "gpuMemoryUsedMB": 19800,
      "gpuMemoryTotalMB": 24576,
      "cpuUsageRate": 0.452,
      "ramUsageRate": 0.601
    },
    "uptimeSeconds": 1209600
  }
}
```

*字段说明：`systemStatus` 为系统整体健康度枚举值（`healthy` 正常可分配 / `warning` 算力接近极限 / `critical` 满载或异常）；百分比改为 `number`（0~1），内存改为整数 `MB`。*

### 业务异常说明 (Exceptions)

| 错误码 | 错误信息 | 触发场景 | 前端处理建议 |
| --- | --- | --- | --- |
| `44002` | Stats Collection Failed | 后端无法读取显卡驱动或系统监控进程超时。 | 在系统状态看板显示“-”，并提示“获取硬件状态失败”。 |

---

## 4.2 获取当前加载的 AI 模型信息

| 属性 | 说明 |
| --- | --- |
| **接口描述** | 获取当前系统在显存中实际加载的跌倒检测模型版本及推理引擎信息。常用于系统诊断页面和给技术支持人员提单时的附加信息。 |
| **请求路径** | `/api/v1/model/version` |
| **请求方式** | `GET` |

### 请求参数

*(GET 请求，无路径和请求体参数)*

### 响应数据 (Success 200)

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "modelName": "FallDetect-Pose-Net",
    "version": "v2.3.1-trt",
    "inferenceEngine": "TensorRT 8.6",
    "releaseDate": "2026-05-20T00:00:00Z",
    "supportedInputResolutions": ["640x640", "1280x720", "1920x1080"],
    "features": [
      "fall_detection",
      "stumble_warning",
      "bed_exit_warning"
    ],
    "isLoaded": true
  }
}
```

*注：模型未就绪时**仍返回 200**，仅 `isLoaded: false`（不再使用 `50021` 冲突错误码）。前端据此拦截新建任务并展示“AI 模型未就绪，请联系管理员”。*

### 业务异常说明 (Exceptions)

*(本接口不再定义 `50021`；模型未加载通过 `isLoaded:false` 表达，与 200 共存，语义唯一。)*

---

# 错误码总表（修订后，一码一义）

| 段 | 错误码 | 含义 | 归属模块 |
| --- | --- | --- | --- |
| 通用 | `40101` | Invalid Authentication（鉴权失败） | 全部 |
| 通用 | `40102` | Invalid Pagination（分页超限，统一阈值按接口注明） | 全部 |
| 通用 | `40103` | Invalid Parameter（参数缺失/非法） | 全部 |
| 任务 | `41001` | Invalid Stream URL | 模块一 |
| 任务 | `41002` | Camera Already Analyzing | 模块一 |
| 任务 | `41003` | Task Not Found | 模块一 |
| 任务 | `41004` | Task Already Stopped | 模块一 |
| 事件 | `42001` | Invalid Time Range | 模块二 |
| 事件 | `42002` | Event Not Found | 模块二 |
| 事件 | `42003` | Invalid Status Value | 模块二 |
| 事件 | `42004` | Event Already Processed | 模块二 |
| 事件 | `42005` | Invalid Dimension | 模块二 |
| 设备 | `43001` | Camera Not Found | 模块三 |
| 设备 | `43002` | Invalid Polygon Format | 模块三 |
| 设备 | `43003` | Invalid Config Parameter | 模块三 |
| 系统 | `44001` | GPU Resource Exhausted | 模块一/四 |
| 系统 | `44002` | Stats Collection Failed | 模块四 |

---

# 主要修订点对照（相对原文档）

| # | 原问题 | 修订动作 |
| --- | --- | --- |
| 1 | `40410` 同时表示“任务不存在”和“摄像头不存在” | 拆为 `41003` / `43001` |
| 2 | 分页错误码三接口三种（`40021`/`40021`/`40030`） | 统一为 `40102`，各接口在说明中标注自身最大限制 |
| 3 | 无鉴权，`processedBy` 来源不明 | 全局 Bearer JWT，`processedBy` 由服务端填充 |
| 4 | WebSocket 只给 `wsUrl` 无协议 | 新增 1.6 节定义订阅/心跳/alert 消息 schema |
| 5 | WS 告警与 REST 事件无桥接 | `EventItem` 增 `taskId`，alert 与 event 靠 `eventId` 关联 |
| 6 | 百分比/内存混用字符串 | 统一 `number`（0~1）与整数 `MB` |
| 7 | `sensitivity` 三处优先级不清 | 明确 任务 > 摄像头 > 全局 |
| 8 | `status` 枚举自相矛盾（缺 `stopped`） | 补 `stopped`，按模块改名 `taskStatus`/`eventStatus`/`cameraStatus`/`systemStatus` |
| 9 | `streamUrl` 重复且冲突 | 建任务改为可选，任务级覆盖设备级 |
| 10 | `cameraId` 是否需注册未约束 | 建任务/预检均返回 `43001 Camera Not Found` |
| 11 | `triggerDuration` 与帧率耦合未解释 | 3.3 注明模式1/模式2 下“连续”语义差异 |
| 12 | `bbox` 格式未定义 | 统一归一化 `[x1,y1,x2,y2]`（0~1） |
| 13 | `riskLevel` 与 `confidence` 映射缺失 | 文档注明二者并列字段，映射规则由模型配置页统一说明（建议在模型文档补充） |
| 14 | 分页最大限制不统一（50/100/100） | 保留各接口自身上限，但错误码统一 `40102` |
| 15 | ROI 坐标“推荐”归一化但校验 0~1 矛盾 | 改为“必须”归一化，校验拒绝越界 |
| 16 | 2.3 缺 `unprocessed` 计数 | summary 补 `unprocessed` |
| 17 | 4.2 `isLoaded` 与 `50021` 冲突 | 删除 `50021`，统一 `200 + isLoaded:false` |
| 18 | 缺单任务查询 | 新增 1.4 `GET /tasks/{taskId}` |
| 19 | 缺建任务预检 | 新增 1.5 `GET /tasks/precheck` |
| 20 | 失败任务无重建机制 | 前端可依 `failed` 状态引导用户重新调用 1.1（后端建议补充重试接口，本文档标注待定） |
