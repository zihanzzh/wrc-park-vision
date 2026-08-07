# Architecture

本文件记录当前共享多模型 Runtime 架构、训练路线和 Thor 交付方向。比赛背景见 [[competition-rules]]，类别定义见 [[class-list]]，设备角色见 [[hardware-notes]]。

## 当前架构结论

当前主线是“多个独立视觉模型 + 共享 Runtime Pipeline + 单次多图 Qwen / VLM”：

```text
Robot sends one image
  -> Thor runtime receives image
  -> Run YOLO-World prohibited_items / behavior-object module
  -> Run Ultralytics YOLO11m garbage module
  -> Run behavior pipeline
  -> Normalize detections and build Detection Summary
  -> Select low-confidence, conflict, small-object and behavior candidates
  -> Generate at most five merged important crops
  -> Qwen2.5-VL-32B reviews original image and crops in one request
  -> Fuse/deduplicate original-image-coordinate findings
  -> Return PipelineResponse, JSON and Preview
```

已确认：

- 当前不是一个统一 YOLO 模型。
- 已训练的两个 YOLO11m 权重继续保留；当前 Runtime detection 分工使用 YOLO-World 负责禁带品和行为辅助对象，独立 YOLO11m 负责垃圾。
- YOLO-World 不负责垃圾，配置和 backend 都拒绝其产生 `task_group: garbage`。
- 多个独立模型共享同一个 Runtime Pipeline。
- 机器人只发送图片，不提供 `taskId`、`taskType`、`mode` 或 `category`。
- Pipeline 根据模型来源写入 `task_group`，不依赖机器人提供任务类型。
- Qwen / VLM 启用时每张图片只执行一次 multi-image 请求；Detection Summary 只是上下文，不是观察范围。
- VLM 不修改 YOLO geometry；corrected 复用 YOLO bbox。VLM 新 finding 必须返回相对原图的 normalized bbox。
- crops 只围绕重要候选生成、合并和限量，不使用固定网格；原图和全部 crops 在同一次 HTTP 请求中发送。
- 原始 YOLO observations、VLM findings 和最终 fusion decisions 必须同时保留。
- 正式优先级为物体准确率、冲突裁决、四类行为和输出协议稳定性；10 秒性能目标已取消。
- 300 秒总预算和 180 秒 VLM timeout 只负责异常保护；Thor 验证要求 VLM 实际完成且结果非 degraded。
- 内部 `PipelineResponse` 与 Competition SDK Response V1 通过独立 adapter 隔离。

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
  -> 选择低置信、冲突、小目标和行为候选
  -> 生成并合并最多 5 个重点 crops，跳过面积达到原图 90% 的重复 crop
  -> Qwen2.5-VL-32B 单次 Multi-Image Review
  -> 将 finding 原图 normalized bbox 转为 canonical geometry
  -> Final Fusion 生成显式决策
  -> PipelineResponse
  -> result.json / competition_result.json
  -> 使用同一个 PipelineResponse 绘制 preview.jpg
```

实现边界：

- 当前通过配置注册 YOLO-World object module 和独立 garbage YOLO11m module，二者都是通用 `DetectionModule`；主 Pipeline 不写死模块数量或业务类别。
- Ultralytics backend 在 Pipeline 初始化时加载一次模型，并立即把 Ultralytics result 转成内部普通对象。
- 固定类别 Ultralytics module 必须配置有序 `expected_class_names`。权重加载后严格比较 class ID 连续性、类别数量、名称和顺序，校验发生在任何图片处理之前。
- YOLO-World module 使用分组的 `open_vocabulary_classes`，仅配置禁带品和行为辅助对象，并显式声明每个类别的 `task_group`、组内 `class_id`、canonical `class_name` 和 prompts；模型加载后只调用一次 `set_classes()`。
- YOLO-World backend 将 prompt 级检测映射成 canonical 类别，并在 backend 输出中携带 `task_group`；`DetectionModule` 优先使用该值构造统一 `Observation`。
- `bbox_xyxy` 是 canonical 像素坐标；`bbox_normalized_xyxy` 从同一个 geometry 计算。
- Review 或 Final Fusion 失败时保留已有 YOLO observations；只要至少一个模块成功，后处理失败返回 `partial_success`。
- `Observation.track_id` 已作为可空字段预留，`RequestContext` 支持 ISO 8601 timestamp 和非负 frame index；当前没有实现 Tracking 或多帧融合。
- schema 已为 `mask`、`pose`、`region` 和 `relation` 预留 observation geometry。
- TensorRT backend 和未来独立 behavior model module 仍明确返回未实现错误；当前单图 Behavior Pipeline 已作为检测后的语义阶段实现。
- Review provider 默认关闭；启用后通过 OpenAI-compatible endpoint 对每张图片执行一次 multi-image 请求。
- 统一 Prompt 和逐项容错 Parser 使用 `yolo_reviews`、`new_findings`、`behavior_reviews` 三个顶层数组。
- 新 finding 的原图 normalized bbox 经过合法性检查和 `[0,1]` 裁剪；单条无效 finding 只产生 `ReviewIssue`。
- Final Fusion 对最终 observations 应用 confirmed / corrected / rejected / uncertain 语义；corrected 复用 YOLO bbox 和 confidence，原始检测信息保留在 Detection Summary、Review 和 FusionDecision 中。
- Fusion 对同类结果按配置化 IoU 去重，并保留 merged source trace；不同类别重叠结果双方保留并标记 conflict。
- 当前只支持单张图片路径 CLI；detector 执行支持配置化 sequential / parallel，默认 sequential。
- 已实现请求级 300 秒安全预算、VLM 剩余 timeout、详细 VLM timing、Competition adapter 和降级状态；当前没有实现 API、ROS2、tracking 或 TensorRT backend。
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
- 隔离单模块异常，并返回 `success`、`partial_success` 或 `failed`；schema 仍接受历史值 `failure`。
- 输出统一 JSON 和直接复用最终 observation 的 Preview。

请求级 300 秒安全预算已实现；VLM 请求失败或剩余时间不足时按 `review_failure_policy` 降级并返回 Competition Response。操作系统调度、冷启动和外部传输无法由进程内 deadline 绝对保证。

### 2. Prohibited Items Detector

- 数据入口：`datasets_final/prohibited_items/data.yaml`。
- 已有训练产物：独立的 prohibited_items YOLO11m，继续保留但不被本阶段删除或覆盖。
- 当前 Runtime module：YOLO-World 开放词汇检测。
- Pipeline 补充：`task_group: prohibited_items`。

example 与 gitignored local 配置均展示正式 8 类；YOLO-World vocabulary 不代表已有专用训练数据，真实能力仍需 Thor 图片验收。

### 3. Garbage Detector

- 数据入口：`datasets_final/garbage/data.yaml`。
- 基础权重：`yolo11m.pt`。
- 训练产物：独立的 garbage YOLO11m。
- Pipeline 补充：`task_group: garbage`。

垃圾 detector 复用现有 Ultralytics backend。模型路径、`expected_class_names`、confidence、IoU、imgsz 和 device 全部由 YAML 提供；权重加载时严格校验类别数量、名称和顺序。模型缺失或映射不一致必须明确失败，不允许回退到 YOLO-World。

垃圾 detector 保持最终 Roboflow `data.yaml` 和 `garbage_best.pt` 的 6 类 class id，不重新映射：

1. `crumpled_paper_ball`
2. `disposable_food_container`
3. `empty_cigarette_box`
4. `plastic_drink_bottle`
5. `plastic_food_wrapper`
6. `rigid_takeout_bag`

### 4. Behavior Pipeline

当前已实现单图、配置驱动的 Behavior Pipeline，输出 `task_group: uncivilized_behavior`。正式四类为：

- `trampling_grass`
- `smoking`
- `blocking_fire_lane`
- `standing_or_lying_on_bench`

执行流程：

```text
YOLO-World 基础对象 observations
  -> 配置化 candidate rules
  -> Detection Summary 携带 behavior classes/candidates
  -> 同一次 Multi-Image Qwen Review 验证 candidate 并主动扫描四类行为
  -> 仅 confirmed 结果生成 kind: behavior observation
  -> Fusion / JSON / Preview
```

`person + grass`、`person + cigarette`、`vehicle`、`person + bench` 只生成 candidate，不能直接判定行为。即使 YOLO-World 没有检出 `grass`、`cigarette` 或任何 behavior candidate，只要 Behavior Pipeline 与 Qwen provider 启用，同一次请求仍主动扫描四类行为。

四类行为的关系判据由配置中的 `decision_rules` 提供。confirmed behavior 可携带相对原图的 normalized bbox；当前正式 example 配置要求 bbox，Pipeline 将其映射为 `BBoxGeometry`，供 Runtime JSON 和 Preview 共用。旧配置不声明 `require_bbox` 时仍兼容无框行为。Competition adapter 暂不增加 behavior bbox 字段。

当前没有实现多帧、tracking、pose、segmentation 或消防通道区域模型；这些能力后续可在 Behavior Pipeline 内增加，不需要重写主 Pipeline。

### 4.1 YOLO-World Object Backend

YOLO-World 是现有 detector 集合中的 object-level backend，不删除已有 YOLO11m。当前同一个模型实例只覆盖：

- `prohibited_items`：正式 8 类禁带品。
- `uncivilized_behavior`：`person`、`bench`、`grass`、`cigarette`、`vehicle` 五个 canonical 基础对象；每个对象可配置有限同义 prompts，例如 people/human/pedestrian、lawn/turf、smoker/smoking/smoke、car/truck/bus/parking 和 park bench/seat。

`garbage` 不得出现在 YOLO-World `open_vocabulary_classes` 中；六类垃圾由独立 Ultralytics YOLO11m module 输出。

`trampling_grass`、`smoking`、`blocking_fire_lane`、`standing_or_lying_on_bench` 不得作为 YOLO-World class。这些语义由 Behavior Pipeline 结合基础对象和全图 VLM 判断，后续可继续加入区域、姿态、关系和 tracking。

开放词汇配置可以为一个 canonical 类别提供有限同义 prompts。backend 输出会把命中的 prompt 规范化为组内 class ID 和 canonical class name；Qwen Review 继续只接收统一 Detection Summary，不感知 backend 差异。

### 5. 多模型调度

同一图片当前顺序运行 YOLO-World object module 和 garbage YOLO11m module。两个模块由配置列表注册，其合法结果统一转成 observations 并进入同一个 Detection Summary，不需要改写主 Pipeline。

待 Thor benchmark 后决定：

- 是否从当前 sequential 改为并行。
- 是否按资源情况限制并发。
- 是否需要模型预热和常驻 engine。
- behavior module 是否总是运行或采用内部触发条件。

多模型预计会增加计算量；当前先以 Thor accuracy-first 验收结果决定后续性能优化，不设置固定延迟门槛。

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

- YOLO-World prohibited class -> `prohibited_items`
- YOLO-World behavior helper class -> `uncivilized_behavior`
- garbage YOLO11m module -> `garbage`
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

### 8. Detection Summary 与 Qwen / VLM

- Detection Summary 包含 observation ID、`task_group`、类别、置信度、YOLO bbox、冲突和 review 原因。
- Detection Summary 同时包含正式行为类别和由基础对象生成的 behavior candidates。
- Summary 只提供上下文；Qwen2.5-VL 始终接收完整原图，并额外接收少量重点 crops。
- VLM 对每条 YOLO detection 返回 `confirmed`、`rejected`、`corrected` 或 `uncertain`。
- Candidate Selector 从低置信、跨模型冲突、小目标和 behavior candidates 中选择重点区域。
- Crop Generator 围绕候选扩展上下文、合并高重叠区域并按优先级保留最多 5 个 crops；面积达到原图 90% 的 crop 不重复发送，没有候选时不虚构固定网格。
- 单次 multi-image 响应通过 `new_findings` 报告漏检目标，并通过 `behavior_reviews` 确认或否定行为；无 candidate 时仍可从原图发现明显行为。
- 所有 finding bbox 都相对完整原图归一化。`crop_id` 只表示帮助判断的 crop，不改变坐标系。
- parser 分别解析 observation review、finding 和 behavior review。非法项、缺失项、重复 ID 或非法类别写入 `ReviewIssue`，合法项继续进入 Fusion；顶层响应无法解析时才整体失败。
- provider 默认关闭，因此现有 detector-only Pipeline 继续工作；启用 provider 时每张输入图片只执行一次 HTTP 请求。
- Thor 已验证旧单图 Review 链路；Runtime V3 的多图请求延迟、准确率和漏检收益仍需实测。

### 9. Final Fusion 与输出

- 最终 `observations` 应用 VLM verdict：confirmed 保留；corrected 复用原 YOLO geometry/confidence 并更新 task/class；rejected 移除；uncertain 按配置处理，默认保留并标记 pending。
- 原始 YOLO detection 仍可通过 Detection Summary、ReviewDecision 和 FusionDecision 审计；FusionDecision 同时记录 YOLO confidence 与 VLM confidence。
- VLM `confirmed` 的行为以 `kind: behavior` observation 加入结果；同一行为类别单图只保留一条，不确认则不添加。
- `review.decisions` 与 `review.findings` 保留 VLM 原始语义结论。
- `fusion.decisions` 明确记录保留、拒绝、纠正、新增 finding、同类去重和异类冲突。
- `geometry_source: yolo` 表示 confirmed/corrected 坐标来自 YOLO；Runtime V3 新 finding 使用 `vlm_multi_image`。
- Preview 从最终 `PipelineResponse.observations` 读取同一份 geometry，绘制 detector、corrected、Multi-Image finding 和 flagged 状态，不重新计算 bbox。
- Competition adapter 从同一份最终 observations 生成精简对象/行为响应；rejected 不进入 SDK objects。

### 10. Accuracy-first 安全预算与降级

300 秒总预算覆盖图片接收、多个视觉模块、结果规范化、冲突处理、VLM、融合和序列化；180 秒 VLM timeout 是安全保护，不是性能目标。

Runtime 应支持：

- 请求级 deadline。
- 子模型和 VLM 的独立超时。
- 取消或忽略超时结果。
- 保留可用的高置信检测。
- 对未确认结果返回 `suspected` / `uncertain`。
- 记录各阶段耗时和降级原因。

Runtime 从图片处理开始计算总 deadline，并在调用 VLM 前扣除已用时间与 Fusion/输出预留。VLM timeout 取 multi-image、provider 与 remaining 的最小值；remaining 不足时跳过请求，required observations 按失败策略处理。请求、响应解析或 Fusion 失败时，Runtime 保留可用 detector 结果并返回 `partial_success`。

Thor 验证必须预热并运行多次，且每次 VLM 实际完成、非 degraded；timing 继续记录用于后续优化，但当前通过条件不含固定延迟阈值。Qwen2.5-VL-32B 真实 accuracy-first 验收尚未执行。

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
  -> Thor Qwen2.5-VL-32B accuracy-first 实机验证
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
- 已完成：请求级 deadline、VLM 详细 timing、Competition SDK Response V1 adapter 和状态化 Preview。
- 下一步：在 Thor 预热真实 Qwen2.5-VL endpoint，执行至少 5 次非降级 warm-run benchmark，并检查 JSON / Preview 与人工判断的一致性。
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
