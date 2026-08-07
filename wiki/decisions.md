# Decisions

本文件记录当前有效决策和被替代的历史方案。未确认事项写入 [[open-questions]]。

## 当前有效决策

### D001：项目默认使用中文文档

- 日期：2026-06-22
- 决策：沟通、项目文档和 Obsidian/wiki 默认使用中文；技术标识可以使用英文。

### D002：大型数据、模型和实验输出不进入 Git

- 日期：2026-06-22，2026-07-17 更新
- 决策：GitHub 保存代码、配置、wiki、Runtime 和部署脚本；数据集、Roboflow 导出、runs、权重和大型缓存不提交。

### D003：正式数据以两个原始 datasets_final 为准

- 日期：2026-07-18
- 决策：正式训练入口为 `datasets_final/prohibited_items/data.yaml` 和 `datasets_final/garbage/data.yaml`。
- 说明：两个数据集已经过人工检查，当前比 `unified_detection` 合并产物更可信。

### D004：Unified Detection 暂停并禁止训练

- 日期：2026-07-18
- 决策：`unified_detection` 因 train / val / test previews 出现大量 bbox 错误，尤其是 `spray_can`，当前不得用于训练。
- 说明：不继续投入主要比赛准备时间修复；若目录仍存在，只能标记为 `deprecated` / `investigation`。

### D005：分别训练两个 YOLO11m

- 日期：2026-07-18
- 决策：基于 `yolo11m.pt` 分别训练 prohibited_items YOLO11m 和 garbage YOLO11m。
- 说明：这是比赛约剩 12 天时的数据风险控制选择，正确性和可交付性优先于单模型简洁性。

### D006：3090 单卡顺序训练

- 日期：2026-07-18
- 决策：同一张 RTX 3090 不并发执行两个训练任务，先 prohibited_items，成功后再 garbage。
- 当前计划：最多 200 epochs、`patience=50`、early stopping；其余资源参数训练前确认。

### D007：多个独立模型共享一个 Runtime Pipeline

- 日期：2026-07-18
- 决策：Runtime 对同一图片运行 prohibited detector、garbage detector，并预留 behavior module，然后规范化、合并和融合结果。
- 说明：当前不是统一 YOLO 模型。

### D008：机器人仍只发送图片

- 日期：2026-07-17，2026-07-18 重申
- 决策：机器人不发送 `taskId`、`taskType`、`mode` 或 `category`。
- 说明：Pipeline 根据模型来源写入 `task_group`。

### D009：Task Group 由 detector 配置映射确定

- 日期：2026-07-18，2026-07-21 更新
- 决策：固定类别 detector 按模块来源映射 task group：prohibited model -> `prohibited_items`；garbage model -> `garbage`；behavior module -> `uncivilized_behavior`。允许多任务开放词汇 backend 为每个 canonical 类别显式配置 `task_group`。
- 说明：不同模型保留各自 class id，不创建统一全局 class id，也不修改最终 labels。

### D010：VLM 使用可配置启用方式

- 日期：2026-07-18，2026-07-20 更新
- 决策：Qwen / VLM provider 默认关闭，detector-only Pipeline 必须保持可用；provider 启用后对每张输入图片执行一次完整原图 Review。
- 说明：旧的“只复核低置信候选 crop”假设已被全图 Review 决策替代。

### D011：全链路目标为 10 秒，但必须实测

- 日期：2026-07-18
- 决策：完整 Pipeline 应在 10 秒内返回或降级。
- 说明：不声称多模型天然满足 10 秒；必须在 Thor 上 benchmark，最终决定串行/并行、模型尺寸和触发策略。

### D012：不文明行为独立设计

- 日期：2026-06-22，2026-07-18 重申
- 决策：不文明行为不强行作为普通 detection 类别混入当前两个 YOLO 数据集。
- 说明：Runtime 预留 behavior module，可采用 detector、segmentation、pose、关系规则、tracking、VLM 或组合方案。

### D013：垃圾 class id 保持最终 Roboflow 顺序

- 日期：2026-07-17
- 决策：垃圾顺序固定为 `crumpled_paper_ball`、`disposable_food_container`、`empty_cigarette_box`、`plastic_drink_bottle`、`plastic_food_wrapper`、`rigid_takeout_bag`，ID 为 0 至 5。
- 说明：不重新映射最终垃圾 labels。

### D014：禁带品 8 类定义保留，样本状态以训练机为准

- 日期：2026-07-18
- 决策：比赛类别定义保持 8 类；`roller_skates` 和 `barbecue_grill` 可能为 0 样本或待补充，训练前必须核对，不虚构已有数据。

### D015：Thor 最终交付多个实机验证 engine

- 日期：2026-07-18
- 决策：3090 生成各自 best.pt，在 Thor 实际 JetPack / TensorRT 环境构建或导出 engine，并由 Runtime 加载多个 engine。
- 交付至少包含模型/engine、class names、task group mapping、Runtime、配置、运行命令、请求响应示例和环境说明。

### D016：权重文件暂不清理

- 日期：2026-07-18
- 决策：训练机上的 `yolo11m.pt` 作为训练起点保留；`yolo26n.pt` 作为旧测试/备用权重暂不删除。
- 说明：确认训练产物和用途后再单独整理；两者都不提交 GitHub。

### D017：正式 Runtime 使用配置驱动的模块注册

- 日期：2026-07-18
- 决策：主 Pipeline 遍历配置中的 enabled modules，不写死模型数量、业务类别、class names 或模型路径。
- 说明：当前 `prohibited_items` 和 `garbage` 都通过通用 `DetectionModule` 接入，后续 behavior 通过新增模块接入。

### D018：Runtime v1 使用稳定统一 schema

- 日期：2026-07-18
- 决策：输出使用 `schema_version: 1.0`；每条 observation 包含 `task_group`、类别、置信度、来源模型、geometry、review、conflicts 和 metadata。
- 说明：像素 `bbox_xyxy` 是 canonical 坐标，归一化 xyxy 从同一个 geometry 计算；schema 同时预留 mask、pose、region 和 relation。

### D019：Runtime v1 顺序执行并隔离模块失败

- 日期：2026-07-18
- 决策：当前只支持 sequential；所有模块成功为 `success`，部分成功为 `partial_success`，输入无效或全部模块失败现输出 `failed`。schema 继续接受历史值 `failure`。
- 说明：单个模块失败不得丢弃其他成功模块的 observation；当前只记录耗时，不实现强制 timeout。

### D020：跨任务冲突保留双方结果

- 日期：2026-07-18
- 决策：不同 `task_group` 的 bbox 达到 IoU 阈值时，两条 observation 都保留并互相标记 `cross_model_overlap`。
- 说明：当前不实施业务类别覆盖优先级，也不进行跨模型强制去重；低置信和冲突只触发 review pending。

### D021：JSON 与 Preview 共享最终 PipelineResponse

- 日期：2026-07-18
- 决策：Preview 只能读取最终 response 中的 geometry，不重新推理、不读取 labels、不重建 bbox。
- 说明：JSON 先原子写出；Preview 失败会记录 output error，不丢失已成功的推理结果。

### D022：推理 backend 与业务模块隔离

- 日期：2026-07-18
- 决策：Ultralytics 与未来 TensorRT 通过 `InferenceBackend` 隔离，backend-specific result 不得泄露到 Pipeline。
- 说明：TensorRT、behavior 和 VLM 当前只保留明确扩展接口或 schema，不伪造已实现能力。

### D023：权重类别映射必须在启动时严格校验

- 日期：2026-07-19，2026-07-21 更新
- 决策：固定类别 detection module 必须提供有序 `expected_class_names`；Ultralytics 权重加载后校验 class ID 从 0 连续、数量一致、名称和顺序完全一致。YOLO-World module 改用分组 `open_vocabulary_classes`，校验 task group、组内连续 class ID、canonical name 和 prompts。
- 说明：两种 backend 的类别映射都必须在处理图片前完成严格校验；不能把开放词汇类别平铺后丢失 task group。

### D024：后处理失败不得删除成功推理结果

- 日期：2026-07-19
- 决策：Fusion 与 Review 分阶段隔离。任一阶段失败时保留已有 observations 并记录对应阶段错误；只要至少一个模块成功，顶层状态为 `partial_success`。
- 说明：Fusion fallback 继续使用统一稳定排序和 observation ID 分配逻辑，不复制另一套规则。

### D025：Schema 预留 track_id，但不代表 Tracking 已实现

- 日期：2026-07-19
- 决策：`Observation.track_id` 为可空字段；`RequestContext.timestamp` 使用 datetime，`frame_index` 必须非负；`schema_version` 仍为 `1.0`。
- 说明：当前单图流程不生成 track ID，也没有 Tracking、视频或多帧融合。

### D026：Runtime 要求 Python 3.10 或更高

- 日期：2026-07-19
- 决策：`pyproject.toml` 的 `requires-python` 为 `>=3.10`，与当前类型注解语法保持一致。

### D027：Review 必须独立检查完整原图

- 日期：2026-07-20
- 决策：VLM 接收完整原始图片和 Detection Summary；Summary 只作为 YOLO 上下文，不能限制 VLM 的观察范围。
- 说明：VLM 应能确认、拒绝或纠正 YOLO，也能发现 YOLO 完全漏掉的项目类别目标。

### D028：定位始终由 YOLO 负责

- 日期：2026-07-20
- 决策：VLM 只提供语义理解，不输出或修正 bbox、mask、polygon、pose 等定位信息。
- 说明：本决策中“VLM 不输出新 finding bbox”的部分已被 D036 替代；corrected 继续严格复用 YOLO bbox。

### D029：Fusion 必须保留所有来源的可审计信息

- 日期：2026-07-20，2026-07-23 Phase 1 更新
- 决策：最终 observations 应用 VLM verdict；原始 YOLO 信息、VLM review 和 final fusion decisions 仍必须可审计，不得静默丢失。
- 说明：confirmed 保留；corrected 复用原 YOLO bbox/confidence 并更新最终 task/class；rejected 从最终 observations 移除；uncertain 和 review failure 按配置处理。原检测仍保存在 Detection Summary 与 FusionDecision。

### D030：Qwen2.5-VL 通过独立 provider 接入

- 日期：2026-07-20
- 决策：当前实现使用 OpenAI-compatible HTTP provider，通过配置管理 endpoint、model ID、认证环境变量、timeout 和生成参数。
- 说明：真实 Qwen2.5-VL 服务参数待确认；provider、parser 或 Fusion 失败时保留 detector 结果并降级为 `partial_success`。

### D031：YOLO-World 作为可选的分组物体检测 backend

- 日期：2026-07-21，2026-07-23 Phase 2 更新
- 决策：新增 `yolo_world` backend，但不删除现有 YOLO11m。当前只用它检测禁带品 8 类和行为判断需要的基础物体，不再检测垃圾；每条结果规范化为显式 `task_group`、组内 class ID 和 canonical class name。
- 说明：YOLO-World 只负责 object-level detection。踩踏草坪、吸烟、占用消防通道、站立/躺在长椅等行为不作为其 class，后续由独立 Behavior Pipeline 处理。Qwen Review 继续接收统一 Detection Summary，无需 backend-specific 分支。

### D032：单图行为复用现有一次全图 VLM Review

- 日期：2026-07-23
- 决策：正式行为类别固定为 `trampling_grass`、`smoking`、`blocking_fire_lane`、`standing_or_lying_on_bench`。YOLO-World 基础对象组合只生成 candidate，最终行为必须由现有同一次全图 VLM 请求确认。
- 说明：行为与 object review 共用一次 VLM 请求；该单请求原则已由 D039 的 Multi-Image Review 恢复。没有基础对象时仍允许从原图发现明显行为。

### D033：VLM Response 使用逐项容错解析

- 日期：2026-07-23
- 决策：`yolo_reviews`、`new_findings`、`behavior_reviews` 分别逐项解析。单条非法结果记录结构化 `ReviewIssue` 并跳过，不能使其他合法条目丢失。
- 说明：只有顶层响应无法解析为 JSON object 时 Review 才整体失败；缺失审核按 review failure policy 处理。没有确认行为时 `behavior_reviews` 使用空数组。

### D034：Phase 1 Review 降级默认保留并标记

- 日期：2026-07-23
- 决策：`uncertain_policy` 和 `review_failure_policy` 默认均为 `keep_flagged`。
- 说明：不确定或未完成审核的 YOLO 结果不会被静默视为 confirmed；最终 observation 保持 pending 并记录原因，FusionDecision 显式记录对应动作。

### D035：垃圾检测从 YOLO-World 分离

- 日期：2026-07-23
- 决策：六类垃圾只由独立 Ultralytics YOLO11m detection module 负责；YOLO-World 配置和 backend 都拒绝 `task_group: garbage`。
- 说明：garbage 模型路径、`expected_class_names`、confidence、IoU、imgsz 和 device 均由 YAML 配置。启动时使用真实权重元数据严格校验数量、名称与顺序，不允许缺失或不匹配时回退到 YOLO-World。
- 类别真源：当前 `garbage_best.pt` 已核对为 `crumpled_paper_ball`、`disposable_food_container`、`empty_cigarette_box`、`plastic_drink_bottle`、`plastic_food_wrapper`、`rigid_takeout_bag`，与 [[class-list]] 一致。

### D036：单帧 Review 固定使用 Full Image 与 Crop Scan 双 Pass（已被 D039 替代）

- 日期：2026-07-23
- 决策：Qwen provider 启用且两个 Pass 均开启时，每张图片固定执行两次请求。Pass 1 接收完整原图，负责 YOLO review、明显漏检和行为；Pass 2 在一次请求中接收全部重叠 crops，只做独立漏检扫描。
- 说明：真实测试显示顺序双请求无法稳定满足比赛时间预算，因此该编排不再使用。

### D037：VLM 新 finding 必须携带 normalized bbox（坐标规则已由 D039 更新）

- 日期：2026-07-23
- 决策：Full Image finding 返回原图 normalized bbox；Crop finding 返回 `crop_id` 与 crop 内 normalized bbox。Pipeline 负责校验、裁剪和映射为完整原图 geometry。
- 说明：crop-local 坐标已停止使用。corrected 仍复用 YOLO 原框；Runtime V3 finding 坐标统一遵循 D039。

### D038：跨来源 finding 使用配置化 IoU 去重

- 日期：2026-07-23
- 决策：YOLO、Full Image finding 和 Crop finding 的同 task/class 高 IoU 结果合并，默认阈值 0.65；保留经过确认或置信度更高的结果及来源追踪。
- 说明：不同类别高 IoU 结果不能静默删除，双方保留并记录 conflict / fusion decision。Preview 只绘制最终 observations 中的同一份 geometry。

### D039：Runtime V3 使用单次 Multi-Image Review

- 日期：2026-07-24
- 决策：每张图片最多调用一次 Qwen / VLM。请求同时包含原始图片、Detection Summary、behavior candidates 和配置限量的候选驱动重点 crops；当前默认最多 3 个。
- 说明：重点 crops 只围绕低置信、跨模型冲突、小目标和行为候选生成，并在发送前合并重叠区域；不再使用固定网格或第二次 VLM 请求。
- 坐标：所有 VLM `new_findings.bbox_normalized_xyxy` 必须相对原始完整图片。`crop_id` 只记录视觉参考，不参与坐标转换。
- 兼容：detector-only Pipeline、现有 Parser 逐项容错、Fusion verdict 语义、Behavior 类别和输出 schema 主体保持不变。V2 配置仅迁移到 V3 参数，不恢复双请求。

### D040：仅 required observations 需要 VLM decision

- 日期：2026-07-25
- 决策：Candidate Selector 未选中的 detector observation 直接保留，使用 `review.required=false` 和 Fusion action `keep_detector_result`；只有 required observation 缺项或 Review 失败才标记 `keep_review_failed`。
- 说明：Runtime 内部保留完整 Detection Summary；为降低 Thor token 与延迟，Prompt 中只发送 required detection。原图仍始终发送，用于漏检和主动行为判断。

### D041：Competition SDK Response 使用独立 V1 adapter

- 日期：2026-07-25
- 决策：完整内部 `PipelineResponse` 保持不变；独立 adapter 只映射 Fusion 后有效 objects / behaviors，并输出暂定 `competition_result.json`。
- 说明：当前字段根据已知的类别、bbox、confidence 和帧上下文要求设计，不声称是官方协议。拿到正式 SDK 文档后通过 adapter 映射字段名、坐标和 transport。

### D042：10 秒 deadline 与性能验收分开判断

- 日期：2026-07-25
- 决策：Runtime 使用 10 秒总预算限制 VLM 剩余 timeout，并在失败时按策略降级；该机制只负责异常保护。
- 验收：只有 Thor warm-run 中 VLM 实际完成、结果非 degraded，且至少 5 次 warm runs 的 median total 小于 10 秒，才判定性能达标。VLM 被跳过、timeout 或失败后的快速返回不计为通过。
- 性能输入：默认最多 3 个重点 crops，面积达到原图 80% 的 crop 不重复发送；VLM 图像默认最长边 640、JPEG quality 85，详细 timing 和 token 使用写入 Review metrics。

### D043：不文明行为使用 object candidate + 主动全图扫描

- 日期：2026-07-25
- 决策：四类行为继续使用配置化 object candidate，但 candidate 不是调用 VLM 的前置条件。Behavior Pipeline 启用时，同一次 Multi-Image VLM 请求必须主动检查 `trampling_grass`、`smoking`、`blocking_fire_lane`、`standing_or_lying_on_bench`。
- 定位：当前 example 配置要求 confirmed behavior 返回相对原图的 normalized bbox；Pipeline 校验并映射为最终 behavior observation geometry。该字段是 Runtime schema 的向后兼容扩展，Competition Response V1 不新增字段。
- 融合：VLM 对必审 detector observation 的明确 `rejected` / `corrected` 优先于 detector 语义，原检测仍保留在 Detection Summary 与 FusionDecision 中供审计。VLM behavior finding 只在 `confirmed` 时进入最终 observations。

### D044：Behavior candidate 使用完整 decision 覆盖协议

- 日期：2026-08-03
- 决策：存在 behavior candidates 时，`behavior_reviews` 必须为每个 candidate 恰好返回一条 `confirmed` / `rejected` / `uncertain` decision。缺 section、空数组、部分覆盖和重复 candidate 均记录结构化 issue。
- 定位：confirmed behavior 必须提供相对原图的 normalized bbox；缺失或非法 geometry 不进入最终 behavior observation。无 candidate 时仍在同一次请求中主动扫描四类行为。

### D045：正式 VLM 切换为 Qwen2.5-VL-32B accuracy-first

- 日期：2026-08-03
- 决策：正式 alias 为可配置的 `qwen-vl-32b`，每张图片保持一次 Multi-Image 请求。10 秒正式目标取消，D011/D042 的固定性能验收被本决策替代。
- 配置：总安全上限 300 秒，VLM timeout 180 秒，最长边 1024、JPEG quality 90、最多 5 crops、max tokens 1200；耗时继续记录但不作为固定阈值验收。

## 被替代的历史方案

- 早期 YOLO11n 环境验证：已完成，仅保留历史意义。
- 3 类 Mac `prohibited_items_3cls`：流程验证历史，已从 Mac 清理。
- 统一 YOLO11m + `unified_detection`：已被两个独立 YOLO11m + 共享 Runtime 方案替代。
- 统一全局 class id：当前不再需要；两个 detector 保持各自类别空间，以 `task_group` 区分。
- 基于机器人 `taskType` 切换模型：未采用；机器人仍只发送图片。
- Orange Pi / RK3588 主线部署：未采用；当前主线为 Thor。
