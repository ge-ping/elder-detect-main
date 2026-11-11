# 跌倒风险检测系统 API 修订说明（v1 → v2）

配套文件：`fall_risk_api_v2.yaml`（OpenAPI 3.0，可直接导入 Swagger UI / Apifox / Postman）。
本说明按"问题 → 修订"逐条对应，方便后端照改、前端核对。

---

## 一、严重级（必改，否则联调必踩坑）

### 1. 错误码冲突 → 全量重排，一码一义，按模块分段
v1 里 `40410` 同时表示「Task Not Found」和「Camera Not Found」；`40021` 分页错误在不同接口最大限制不同；`40030` 又是另一个分页码。v2 统一为：

| 段 | 归属 |
|----|------|
| `401xx` | 通用（鉴权/分页等跨模块） |
| `41xxx` | 任务模块 |
| `42xxx` | 事件模块 |
| `43xxx` | 设备模块 |
| `44xxx` | 系统/模型模块 |

新错误码表：

| code | message | 触发场景 | HTTP |
|------|---------|----------|------|
| `40101` | Unauthorized | Token 缺失/失效 | 401 |
| `40102` | Invalid Pagination | pageSize 超上限（各接口上限见 yaml） | 400 |
| `41001` | Invalid Stream URL | 流地址格式/协议不支持 | 400 |
| `41002` | Camera Already Analyzing | 该摄像头已有运行任务 | 409 |
| `41003` | Task Not Found | taskId 不存在/过期 | 404 |
| `41004` | Task Already Stopped | 任务已停止 | 409 |
| `42001` | Invalid Time Range | 起止时间非法或跨度>30天 | 400 |
| `42002` | Event Not Found | eventId 不存在 | 404 |
| `42003` | Invalid Status Value | status 非 handled/false_positive | 400 |
| `42004` | Event Already Processed | 已被他人处理 | 409 |
| `42005` | Invalid Dimension | dimension 非 by_day/by_camera | 422 |
| `43001` | Camera Not Found | cameraId 不存在 | 404 |
| `43002` | Invalid Polygon Format | 多边形<3点或坐标越界 | 400 |
| `43003` | Invalid Config Parameter | 阈值参数非法 | 400 |
| `44001` | GPU Resource Exhausted | 显存耗尽 | 503 |
| `44002` | Stats Collection Failed | 读硬件监控失败 | 500 |

> 关键：原 `40410` 拆成 `41003`(任务) 与 `43001`(设备)，前端不再需要靠上下文猜含义。

### 2. 补全鉴权
新增全局 `BearerAuth`（JWT）。所有接口需带 `Authorization: Bearer <token>`。
2.2 的 `processedBy` / `processedAt` **由服务端从 token 解析**，前端不再传（v1 里这个字段无处可来）。

### 3. 补全 WebSocket 告警协议
v1 只给了 `wsUrl` 却没定义收什么。v2 在 yaml 里加了 `WsAlertMessage` schema：
- 消息 `type`：`alert` / `ping` / `pong` / `subscribed` / `error`
- 告警帧结构与 `EventItem` 对齐（同字段、同 `eventId`）
- 建任务响应新增 `wsProtocol` 字段标注协议版本

### 4. WS 告警与 REST 事件桥接
每条 WS `alert` **落库为一条 EventItem，eventId 一致**；`EventItem` 新增 `taskId` 字段。
这样实时弹窗（WS）→ 历史列表/误报标注（REST）能靠 `eventId` 对上号。

---

## 二、重要级

### 5. 数据类型统一为数字
`falsePositiveRate` / `gpuUsageRate` / `cpuUsageRate` / `ramUsageRate` 一律用 `0~1` 的数字；
`gpuMemoryUsed`/`Total` 改为整数字段 `gpuMemoryUsedMB` / `gpuMemoryTotalMB`。单位交前端格式化，不再返回 `"82.5%"` / `"19800MB"` 这类字符串。

### 6. sensitivity 优先级明确
`任务级(1.1) > 摄像头级(3.3) > 全局默认(medium)`，写进两处接口描述。

### 7. status 枚举补齐并改名去歧义
- 任务列表 `status` 补 `stopped`（对应 `41004`）
- 四个模块的 `status` 改名区分：事件用 `eventStatus`、设备用 `deviceStatus`、任务保留 `status`（值域已明确）

### 8. streamUrl 语义明确
1.1 的 `streamUrl` 改为**可选**：不传取设备注册地址，传则任务级覆盖。消除与 3.1 的重复冲突。

### 9. cameraId 约束
1.1 建任务的 `cameraId` **必须是 3.1 已注册设备**，否则 `43001`。

### 10. triggerDuration 与帧率解耦
明确后端按"持续时长(ms)"判定，而非帧数；模式1(24fps)/模式2(2fps) 下 1500ms 语义一致。

### 11. bbox 坐标系定义
`bbox` 改为**归一化 `[x1,y1,x2,y2]`（0~1）**，明确相对 `snapshotUrl` 的原始帧。分辨率切换不受影响。

### 12. riskLevel 映射说明
`riskLevel` 由 `confidence` 结合该摄像头 `sensitivity` 阈值分档映射，写进字段描述，打通模型输出与接口字段。

---

## 三、轻微级

- **13. 分页上限统一策略**：仍保留业务差异（任务列表 50、事件/设备 100），但都用同一错误码 `40102`，并在 yaml 参数里显式声明 `maximum`。
- **14. ROI 坐标**：措辞从"推荐归一化"改为"必须归一化 0~1"，与校验一致。
- **15. 统计 summary**：新增 `unprocessed` 字段，明确 `totalAlerts` 口径，避免大屏数字对不上。
- **16. 模型未加载**：统一返回 `200 + isLoaded:false + loadError`，不再与 `50021` 冲突（该码删除）。
- **17. 新增单任务查询**：`GET /fall-risk/tasks/{taskId}`。
- **18/19. 新增建任务预检**：`GET /fall-risk/tasks/precheck?cameraId=`，POST 前即可知是否已分析/算力是否足够，不必靠吃 `41002` 报错。

---

## 使用建议
1. 把 `fall_risk_api_v2.yaml` 导入 Swagger UI / Apifox 生成可交互文档与 Mock。
2. 后端优先落地：错误码重排 + 鉴权 + WS 协议 + event 桥接（这四项影响面最大）。
3. 前端据 v2 重写错误码映射表与 WS 消息处理。
