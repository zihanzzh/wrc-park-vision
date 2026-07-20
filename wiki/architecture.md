# Architecture

本文件记录当前共享多模型 Runtime 架构、训练路线和 Thor 交付方向。比赛背景见 [[competition-rules]]，类别定义见 [[class-list]]，设备角色见 [[hardware-notes]]。

## 当前架构结论

当前主线是“多个独立视觉模型 + 共享 Runtime Pipeline + 条件式 Qwen / VLM”：

```text
Robot sends one image
  -> Thor runtime receives image
  -> Run prohibited_items detector
  -> Run garbage detector
  -> Run behavior detector or behavior pipeline
  -> Normalize and merge detections
  -> Add task_group metadata
  -> Resolve duplicate/conflicting detections if needed
  -> High-confidence results return directly
  -> Low-confidence / occluded / ambiguous cases trigger Qwen / VLM
  -> Fuse results
  -> Return within 10 seconds or use fallback response
```

已确认：

- 当前不是一个统一 YOLO 模型。
- 禁带品和垃圾分别训练独立 YOLO11m；不文明行为使用后续独立模型或视觉方案。
- 多个独立模型共享同一个 Runtime Pipeline。
- 机器人只发送图片，不提供 `taskId`、`taskType`、`mode` 或 `category`。
- Pipeline 根据模型来源写入 `task_group`，不依赖机器人提供任务类型。
- 高置信度结果直接返回；Qwen / VLM 只处理低置信度、遮挡或类别歧义样本。
- 10 秒是完整链路目标；是否满足必须在 Thor 上实测，当前不作未经验证的性能承诺。

该路线是比赛时间限制下的风险控制方案。数据正确性和可交付性优先于单模型架构的简洁性。

## Runtime v1 已实现范围

正式 Runtime 代码位于 `src/wrc_park_vision/runtime/`，当前实现链路为：

```text
image path
  -> 配置、模型路径与 expected_class_names 启动校验
  -> 图片解码、尺寸校验和 request_id
  -> sequential 运行全部 enabled task modules
  -> 单模块异常隔离
  -> backend 输出转为统一 Observation
  -> 稳定排序并分配 observation id
  -> 跨 task_group IoU 冲突标记
  -> ReviewPolicy 生成 pending / not_required
  -> PipelineResponse
  -> result.json
  -> 使用同一个 PipelineResponse 绘制 preview.jpg
```

实现边界：

- 当前通过配置注册 `prohibited_items` 和 `garbage` 两个通用 `DetectionModule`，主 Pipeline 不写死模块数量或业务类别。
- Ultralytics backend 在 Pipeline 初始化时加载一次模型，并立即把 Ultralytics result 转成内部普通对象。
- enabled detection module 必须配置有序 `expected_class_names`。权重加载后严格比较 class ID 连续性、类别数量、名称和顺序，校验发生在任何图片处理之前。
- `bbox_xyxy` 是 canonical 像素坐标；`bbox_normalized_xyxy` 从同一个 geometry 计算。
- Fusion 失败时使用原始 normalized observations 的稳定排序深拷贝作为 fallback；Review 失败时保留 Fusion 结果。只要至少一个模块成功，后处理失败返回 `partial_success`。
- `Observation.track_id` 已作为可空字段预留，`RequestContext` 支持 ISO 8601 timestamp 和非负 frame index；当前没有实现 Tracking 或多帧融合。
- schema 已为 `mask`、`pose`、`region` 和 `relation` 预留 observation geometry。
- TensorRT backend 和 behavior module 当前明确返回未实现错误，不伪造能力。
- ReviewPolicy 当前只决定是否需要复核，不运行 VLM。
- 当前只支持单张图片路径 CLI、顺序执行和耗时记录。
- 当前没有实现 API、ROS2、tracking、并行执行、强制 timeout 或 Thor 部署。
- Runtime 要求 Python 3.10 或更高版本。

## 路线变更原因

原计划曾创建 `unified_detection`，用于将禁带品和垃圾合并为 14 类并训练一个统一 YOLO11m。人工检查其 train / val / test previews 后发现大量 bbox 显示错误，尤其是 `spray_can`。

当前处理：

- `unified_detection` 不得用于训练。
- 不继续投入主要比赛准备时间修复该合并产物。
- 原始 `datasets_final/prohibited_items/` 和 `datasets_final/garbage/` 保持为正式、相对可信的数据入口。
- `unified_detection` 若仍存在，只能标记为 `deprecated` / `investigation`；其在训练机上的物理存在状态待确认。

## Runtime Architecture

### 1. 图片输入层

机器人向 Thor Runtime 发送一张图片。图片编码、分辨率、传输协议、请求 ID 和时间戳仍待接口联调确认，但请求不包含业务 task type。

Runtime v1 当前负责：

- 图片解码与输入校验。
- 将同一图片交给启用的视觉模块。
- 隔离单模块异常，并返回 `success`、`partial_success` 或 `failure`。
- 输出统一 JSON 和直接复用最终 observation 的 Preview。

请求级 10 秒预算、模块 timeout 和降级中断仍是 Thor 阶段待实现能力。

### 2. Prohibited Items Detector

- 数据入口：`datasets_final/prohibited_items/data.yaml`。
- 基础权重：`yolo11m.pt`。
- 训练产物：独立的 prohibited_items YOLO11m。
- Pipeline 补充：`task_group: prohibited_items`。

`roller_skates` 和 `barbecue_grill` 可能仍为 0 样本或待补充，训练前必须核对 3090 manifest / README。

### 3. Garbage Detector

- 数据入口：`datasets_final/garbage/data.yaml`。
- 基础权重：`yolo11m.pt`。
- 训练产物：独立的 garbage YOLO11m。
- Pipeline 补充：`task_group: garbage`。

垃圾 detector 保持最终 Roboflow `data.yaml` 的 6 类 class id，不重新映射。

### 4. Behavior Module

Runtime 预留独立 behavior module 接口，并补充 `task_group: uncivilized_behavior`。

该模块尚未定型，可能包含：

- 独立 YOLO detector。
- YOLO segmentation。
- pose / action 线索。
- 人与 bench、草坪、消防通道等区域的位置关系。
- tracking、多帧时序、规则层和 VLM。

不能预设五类不文明行为都能作为普通 object detection 类别直接解决。

### 5. 多模型调度

同一图片当前顺序运行 prohibited detector 和 garbage detector。模块由配置列表注册，后续 behavior module 可按相同接口加入，不需要改写主 Pipeline。

待 Thor benchmark 后决定：

- 是否从当前 sequential 改为并行。
- 是否按资源情况限制并发。
- 是否需要模型预热和常驻 engine。
- behavior module 是否总是运行或采用内部触发条件。

多模型预计会增加计算量，但不能仅凭模型数量推断能否满足 10 秒。

### 6. 结果规范化与 Task Group

每个模型的原始 class id 只在其自身类别空间内有效。共享 Runtime 当前将不同模型结果规范为 snake_case 字段：

- `source.model_id`
- `task_group`
- `class_id`
- `class_name`
- `geometry.bbox_xyxy`
- `geometry.bbox_normalized_xyxy`
- `mask` / `pose` / `region` / `relation`（未来 geometry 类型）
- `confidence`
- `review.status`
- `metadata`

`taskGroup` 来源映射：

- prohibited_items model -> `prohibited_items`
- garbage model -> `garbage`
- behavior model / pipeline -> `uncivilized_behavior`

不要求不同模型共享同一全局 class id，也不修改原数据集标签。

### 7. 重复与冲突处理

Runtime v1 采用保守规则：

- 同模型 NMS 由 Ultralytics backend 完成。
- 不同 `task_group` 的 bbox IoU 达到配置阈值时，两个 observation 均保留。
- 双方 `conflicts` 记录对方 observation id，类型为 `cross_model_overlap`。
- 冲突可触发 `review.status: pending`。
- 不实施类别覆盖、业务优先级或跨模型删除。

后续是否需要置信度校准或业务规则，必须依据真实冲突样本决定。

### 8. 置信度与 Qwen / VLM

- 高置信且无冲突：当前标为 `review.status: not_required`，不是 `confirmed`。
- 低于 review 阈值或存在跨任务冲突：标为 `review.status: pending`。
- 模块失败：触发顶层 `review.reasons: [module_failure]`，不伪造 observation。
- 真正的目标裁剪、VLM 调用、复核回写和超时降级尚未实现。

VLM 不处理所有图片或所有帧。YOLO 推理预计不是主要时间瓶颈，VLM 更可能成为主要延迟来源，但必须通过实际 profiling 验证。

### 9. 10 秒预算与降级

10 秒覆盖图片接收、多个视觉模块、结果规范化、冲突处理、可选 VLM、融合和序列化。

Runtime 应支持：

- 请求级 deadline。
- 子模型和 VLM 的独立超时。
- 取消或忽略超时结果。
- 保留可用的高置信检测。
- 对未确认结果返回 `suspected` / `uncertain`。
- 记录各阶段耗时和降级原因。

三模型是否能在 10 秒内完成尚未验证。最终依据 Thor benchmark 决定串行/并行、模型尺寸和触发策略。

### 10. 日志与数据回流

记录：

- 每个模型版本、engine 和耗时。
- 模型来源与 `task_group`。
- 误报、漏报、低置信和冲突样本。
- VLM 请求、超时和复核结果。
- 完整请求耗时和降级状态。

失败样本回到 3090 进行复盘、补标和模型迭代，不提交 GitHub。

## 训练计划

3090 单卡按顺序训练，不同时并发占用同一张 RTX 3090：

1. prohibited_items YOLO11m。
2. 第一项成功完成后自动开始 garbage YOLO11m。

共同计划：

- `model: yolo11m.pt`
- `epochs: 200`（最多）
- `patience: 50`
- 使用 early stopping
- `batch`、`workers`、`imgsz`、`device` 在训练前按 3090 环境确认

建议输出：

- `runs/detect/wrc_prohibited_yolo11m/`
- `runs/detect/wrc_garbage_yolo11m/`

建议最终命名：

- `prohibited_items_yolo11m_best.pt`
- `garbage_yolo11m_best.pt`

Ultralytics 训练的候选最佳权重通常位于各运行目录的 `weights/best.pt`。`yolo11m.pt` 是预训练起点，不是最终自定义权重；`yolo26n.pt` 是旧测试或备用预训练权重，暂不删除。

## Thor 部署与交付

目标流程：

```text
3090 训练生成各自 best.pt
  -> 将模型带到 Thor
  -> 在 Thor 实际 JetPack / TensorRT 环境导出或构建 engine
  -> Runtime 加载多个 engine
  -> Thor 实机 benchmark 与 10 秒链路验证
  -> 机器人接口联调
  -> 形成可交付部署包
```

最终交付不能只有普通 `.pt`，至少应包含：

- 模型或 TensorRT engine。
- class names。
- model -> `task_group` mapping。
- Runtime code。
- configuration。
- run command。
- sample request / response。
- environment notes。

TensorRT engine 应在 Thor 实际环境中构建或验证，避免脱离目标 JetPack / TensorRT 环境假设兼容性。

## Development Roadmap

- 已完成：禁带品和垃圾原始最终数据的人工检查与整理。
- 已暂停：`unified_detection` 统一 14 类训练路线。
- 已完成：两个独立 YOLO11m 在外部训练机完成训练，权重尚待交付当前 Mac。
- 已完成：共享多模型 Runtime v1，包括配置、Ultralytics backend、模块调度、统一 schema、冲突标记、review decision、JSON、Preview、CLI 和 FakeBackend 测试。
- 下一步：放置两个正式权重，用真实照片完成双模型 smoke test，并检查 JSON / Preview 一致性。
- 下一步：评估两个模型的误报、漏报、冲突和各类表现。
- 下一步：在 Thor 构建多个 TensorRT engine，并 benchmark 串行/并行策略。
- 后续：接入 behavior module、机器人接口和 VLM 复核。
- 后续：失败样本回流和模型迭代。

## 架构待确认

- `unified_detection` 在 3090 上是否已经物理删除。
- 禁带品 `roller_skates` / `barbecue_grill` 的实际样本数。
- 两个训练任务的 `batch`、`workers`、`imgsz` 和 `device`。
- Thor 上多模型串行/并行策略与实际延迟。
- 当前 `schema_version: 1.0` 与机器人侧最终协议如何封装。
- 跨模型冲突在真实照片中的频率，以及是否需要置信度校准或业务规则。
- behavior module 的模型与数据方案。
- 机器人输入输出协议和 `suggestedAction` 职责边界。
- VLM 现场设备、模型版本、联网条件和时间预算。
