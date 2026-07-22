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
  -> Normalize detections and build Detection Summary
  -> Optional Qwen2.5-VL independently reviews the full image
  -> Fuse YOLO observations, VLM decisions and VLM-only findings
  -> Return PipelineResponse, JSON and Preview
```

已确认：

- 当前不是一个统一 YOLO 模型。
- 禁带品和垃圾分别训练独立 YOLO11m；不文明行为使用后续独立模型或视觉方案。
- YOLO-World 作为可选 object detector backend 接入，不替换现有 YOLO11m；它可以按配置检测多个任务组的基础物体。
- 多个独立模型共享同一个 Runtime Pipeline。
- 机器人只发送图片，不提供 `taskId`、`taskType`、`mode` 或 `category`。
- Pipeline 根据模型来源写入 `task_group`，不依赖机器人提供任务类型。
- Qwen / VLM 启用时必须检查完整原图；Detection Summary 只是上下文，不是候选范围。
- VLM 只负责语义理解，不负责 bbox、mask 或其他定位。
- VLM 可以报告 YOLO 完全漏检的目标；这类 finding 可以没有 bbox。
- 原始 YOLO observations、VLM findings 和最终 fusion decisions 必须同时保留。
- 10 秒是完整链路目标；是否满足必须在 Thor 上实测，当前不作未经验证的性能承诺。

该路线是比赛时间限制下的风险控制方案。数据正确性和可交付性优先于单模型架构的简洁性。

## Runtime 已实现范围

正式 Runtime 代码位于 `src/wrc_park_vision/runtime/`，当前实现链路为：

```text
image path
  -> 配置、模型路径与 backend 类别映射启动校验
  -> 图片解码、尺寸校验和 request_id
  -> sequential 运行全部 enabled task modules
  -> 单模块异常隔离
  -> backend 输出转为统一 Observation
  -> 稳定排序并分配 observation id
  -> 跨 task_group IoU 冲突标记
  -> 生成 Detection Summary
  -> 可选 Qwen2.5-VL 全图 Review
  -> Final Fusion 生成显式决策
  -> PipelineResponse
  -> result.json
  -> 使用同一个 PipelineResponse 绘制 preview.jpg
```

实现边界：

- 当前通过配置注册 `prohibited_items` 和 `garbage` 两个通用 `DetectionModule`，主 Pipeline 不写死模块数量或业务类别。
- Ultralytics backend 在 Pipeline 初始化时加载一次模型，并立即把 Ultralytics result 转成内部普通对象。
- 固定类别 Ultralytics module 必须配置有序 `expected_class_names`。权重加载后严格比较 class ID 连续性、类别数量、名称和顺序，校验发生在任何图片处理之前。
- YOLO-World module 使用分组的 `open_vocabulary_classes`，显式配置每个类别的 `task_group`、组内 `class_id`、canonical `class_name` 和 prompts；模型加载后只调用一次 `set_classes()`。
- YOLO-World backend 将 prompt 级检测映射成 canonical 类别，并在 backend 输出中携带 `task_group`；`DetectionModule` 优先使用该值构造统一 `Observation`。
- `bbox_xyxy` 是 canonical 像素坐标；`bbox_normalized_xyxy` 从同一个 geometry 计算。
- Review 或 Final Fusion 失败时保留已有 YOLO observations；只要至少一个模块成功，后处理失败返回 `partial_success`。
- `Observation.track_id` 已作为可空字段预留，`RequestContext` 支持 ISO 8601 timestamp 和非负 frame index；当前没有实现 Tracking 或多帧融合。
- schema 已为 `mask`、`pose`、`region` 和 `relation` 预留 observation geometry。
- TensorRT backend 和 behavior module 当前明确返回未实现错误，不伪造能力。
- Review provider 默认关闭；启用后通过 OpenAI-compatible endpoint 调用 Qwen2.5-VL，发送完整原图和 Detection Summary。
- Response Parser 要求逐条复核 Detection Summary 中的 YOLO detection，并拒绝 VLM 输出定位字段。
- Final Fusion 不修改原始 YOLO 类别和 bbox；纠正、拒绝和 VLM-only finding 通过独立决策记录表达。
- 当前只支持单张图片路径 CLI、顺序执行和耗时记录。
- 当前没有实现 API、ROS2、tracking、并行执行、请求级 deadline 或 TensorRT backend；provider HTTP timeout 已配置为 10 秒。
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

### 4.1 YOLO-World Object Backend

YOLO-World 是现有 detector 集合中的可选 backend，不替代两个已训练 YOLO11m。它只负责 object-level detection，可在同一个模型实例中覆盖：

- `prohibited_items`：正式 8 类禁带品。
- `garbage`：正式 6 类垃圾。
- `uncivilized_behavior`：`person`、`bench`、`grass`、`cigarette`、`vehicle` 等行为判断需要的基础物体。

`trampling_grass`、`smoking`、`blocking_fire_lane`、`standing_on_bench`、`lying_on_bench` 等行为不得作为 YOLO-World class。这些语义需要由后续 Behavior Pipeline 结合基础对象、区域、姿态、关系、tracking 和 VLM 判断。

开放词汇配置可以为一个 canonical 类别提供有限同义 prompts。backend 输出会把命中的 prompt 规范化为组内 class ID 和 canonical class name；Qwen Review 继续只接收统一 Detection Summary，不感知 backend 差异。

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

`task_group` 来源映射：

- prohibited_items model -> `prohibited_items`
- garbage model -> `garbage`
- behavior model / pipeline -> `uncivilized_behavior`
- YOLO-World -> 按 `open_vocabulary_classes` 中每个 canonical 类别的显式映射确定，可由同一 backend 输出多个 task group

不要求不同模型共享同一全局 class id，也不修改原数据集标签。

### 7. 重复与冲突处理

Runtime v1 采用保守规则：

- 同模型 NMS 由 Ultralytics backend 完成。
- 不同 `task_group` 的 bbox IoU 达到配置阈值时，两个 observation 均保留。
- 双方 `conflicts` 记录对方 observation id，类型为 `cross_model_overlap`。
- 冲突可触发 `review.status: pending`。
- 不实施类别覆盖、业务优先级或跨模型删除。

后续是否需要置信度校准或业务规则，必须依据真实冲突样本决定。

### 8. Detection Summary 与 Qwen / VLM

- Detection Summary 包含 observation ID、`task_group`、类别、置信度、YOLO bbox、冲突和 review 原因。
- Summary 只提供上下文；Qwen2.5-VL 接收完整图片并独立观察全图。
- VLM 对每条 YOLO detection 返回 `confirmed`、`rejected`、`corrected` 或 `uncertain`。
- VLM 可以通过 `new_findings` 报告 YOLO 漏检的项目类别目标；finding 不包含 bbox。
- parser 严格校验 observation 覆盖、重复 ID、任务类别目录和返回结构，并拒绝 bbox / mask / polygon 等额外定位字段。
- provider 默认关闭，因此现有 detector-only Pipeline 继续工作；启用 provider 后每张输入图片执行一次全图 Review。
- VLM 真实延迟与效果尚未验证，不能用 mock 测试推断比赛性能。

### 9. Final Fusion 与输出

- `observations` 保留原始 YOLO 结果及其 geometry，不因 VLM 拒绝或纠正而删除或改写。
- `review.decisions` 与 `review.findings` 保留 VLM 原始语义结论。
- `fusion.decisions` 明确记录 `keep_yolo`、`reject_yolo`、`correct_yolo` 或 `add_vlm_finding`。
- `geometry_source: yolo` 表示坐标来自 YOLO；VLM-only finding 使用 `geometry_source: none`。
- Preview 从最终 `PipelineResponse` 读取同一份 observation 和 fusion decision。VLM-only finding 只显示文字，不创建框。

### 10. 10 秒预算与降级

10 秒覆盖图片接收、多个视觉模块、结果规范化、冲突处理、可选 VLM、融合和序列化。

Runtime 应支持：

- 请求级 deadline。
- 子模型和 VLM 的独立超时。
- 取消或忽略超时结果。
- 保留可用的高置信检测。
- 对未确认结果返回 `suspected` / `uncertain`。
- 记录各阶段耗时和降级原因。

当前 Qwen provider 已有 10 秒 HTTP timeout；完整请求级 deadline、跨阶段预算和主动取消仍待实现。Review 请求、响应解析或 Fusion 失败时，Runtime 保留 YOLO observations 并返回 `partial_success`。

三模型是否能在 10 秒内完成尚未验证。最终依据 Thor benchmark 决定串行/并行、模型尺寸和触发策略。

### 11. 日志与数据回流

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
- 已完成：共享多模型 Runtime detector 链路，并在 macOS 与 Thor 跑通两个真实模型。
- 已完成：Detection Summary、Qwen2.5-VL provider interface、全图 Prompt Builder、严格 Response Parser、Final Fusion 和更新后的 Preview。
- 已完成：可选 YOLO-World object detector backend、分组开放词汇配置、prompt 到 canonical 类别映射和 mock 自动测试；尚未完成真实权重 Runtime smoke test。
- 下一步：连接真实 Qwen2.5-VL endpoint，检查 JSON / Preview 与人工判断的一致性。
- 下一步：评估两个模型的误报、漏报、冲突和各类表现。
- 下一步：在 Thor 构建多个 TensorRT engine，并 benchmark 串行/并行策略。
- 后续：接入 behavior module、机器人接口和 TensorRT backend。
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
- Qwen2.5-VL endpoint、具体模型版本、认证方式、现场设备、联网条件和时间预算。
- 两个 detector 全部失败但 VLM 返回 finding 时，顶层状态应为 `failure` 还是 `partial_success`。
