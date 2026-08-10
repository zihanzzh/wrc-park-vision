# Project Context

## 项目背景

`wrc-park-vision` 是 WRC 园区管理岗视觉识别项目，同时作为 Obsidian vault 维护项目记忆。视觉部分覆盖：

1. 禁带品检查。
2. 垃圾识别与拾取分类。
3. 不文明行为识别与提醒。

比赛规则见 [[wiki/competition-rules]]，类别定义见 [[wiki/class-list]]。

## 当前阶段

禁带品和垃圾数据准备已基本完成，两个独立 YOLO11m 已在外部训练机完成训练。项目当前进入：

- 两个已训练 detector 已在 macOS 和 NVIDIA Thor 上完成 Runtime 实际运行验证。
- 当前共享 Runtime V3 已形成 Detection -> Detection Summary -> 重点候选与 crops -> 单次多图 Review -> Fusion -> Output 链路。
- 准备 NVIDIA Jetson AGX Thor Developer Kit 的多模型部署和 TensorRT 验证。
- Thor 已使用 Qwen2.5-VL-32B 跑通 person + grass、`trampling_grass` candidate/confirmation、behavior bbox、Parser、Fusion、Preview 和 Competition output。
- 当前 accuracy-first 重点是让所有 garbage / prohibited detector observations 经过同一次 32B Review，降低高置信 detector false positive 直接进入最终输出的风险。

不文明行为尚未形成独立训练数据集和专用模型；当前已实现单图 Behavior Pipeline，使用 YOLO-World 基础对象、配置化候选规则和同一次多图 Qwen Review 共同判断四类行为。

## 最新模型决策

原计划曾将禁带品和垃圾合并为 `unified_detection`，训练一个统一 YOLO11m。人工检查该合并数据集的 train / val / test previews 后发现大量 bbox 显示错误，尤其是 `spray_can`，因此当前决定：

- 暂停并废弃 `unified_detection` 训练路线。
- 不使用 `unified_detection` 训练，也不把主要比赛准备时间投入其修复。
- 保留两个经过人工检查、相对可靠的正式数据入口。
- 基于 `yolo11m.pt` 分别训练禁带品和垃圾两个 YOLO11m。
- Runtime 接收同一张图片后运行多个独立视觉模块，再统一规范化和融合结果。

这是比赛时间限制下的风险控制选择：数据正确性和可交付性优先于单模型架构的简洁性。

## Runtime 实现状态

- 机器人只发送图片，不发送 `taskId`、`taskType`、`mode` 或 `category`。
- 正式 Runtime v1 已实现配置加载、输入校验、模块注册、顺序执行、模块故障隔离、统一 schema、冲突标记、review decision、JSON、Preview 和 CLI。
- YOLO-World 与独立 garbage YOLO11m 通过配置注册为通用 `DetectionModule`，主 Pipeline 不写死模型数量、类别或路径。
- YOLO-World 只输出禁带品和行为相关基础对象；六类垃圾由独立 Ultralytics YOLO11m 模块负责，两者都保留每条 detection 的 `task_group`。
- 模型启动时加载一次；enabled 模型路径缺失时明确失败，不允许自动下载。
- enabled detection module 必须声明有序 `expected_class_names`；Ultralytics 权重加载后、图片处理前严格校验类别 ID、数量、名称和顺序。
- Pipeline 根据模型来源写入 `task_group`，不同模型保留各自 class id 空间。
- 跨 task group 高 IoU 结果全部保留，并互相标记 `cross_model_overlap`，不实施业务优先级删除。
- 已实现 Detection Summary、候选驱动重点裁剪、Qwen2.5-VL 单次多图 provider、统一 Prompt、逐项容错 Parser 和最终 Fusion。
- 已实现配置驱动的单图 Behavior Pipeline：基础对象只生成候选，最终行为必须由同一次 VLM Review 确认；无候选时仍允许从原图发现明显行为。
- 当前正式行为类别为 `trampling_grass`、`smoking`、`blocking_fire_lane`、`standing_or_lying_on_bench`。
- 单次 Multi-Image Review 同时接收完整原图、紧凑 Detection Summary 和少量候选驱动 crops，完成确认、拒绝、纠正、漏检和行为判断。
- `garbage` 与 `prohibited_items` 的每条 detector observation 均由配置要求进入该次 Review；behavior 基础对象仍使用低置信、小目标、冲突和行为证据候选策略。
- corrected 始终复用 YOLO bbox；VLM 新 finding 必须提供相对完整原图的 normalized bbox。
- Fusion 对同类高 IoU 结果去重，并保留来源追踪；不同类别高 IoU 结果不静默删除。
- JSON 与 Preview 使用同一个最终 `PipelineResponse`，Preview 不重新推理或重算 bbox。
- Fusion 或 Review 失败不会删除成功模块的 observations；结果保留并以 `partial_success` 和阶段错误返回。
- `Observation.track_id` 已预留且单图流程默认为 `null`；Tracking 和多帧融合尚未实现。
- detector 执行支持 sequential / parallel 配置，默认 sequential；Runtime 分别记录模型初始化、detection wall time、候选、crop、VLM 请求链路、Fusion、Competition adapter、序列化和 Preview 耗时。
- 正式配置采用 300 秒请求安全上限与 180 秒 VLM timeout；10 秒性能目标已取消，失败时 required observations 仍按现有策略降级。
- 已新增独立 Competition SDK Response V1 adapter，保留完整内部 RuntimeResult；官方机器人 SDK 字段和坐标协议仍待确认。
- Behavior 的单图语义链路已实现；多帧、tracking、pose/区域关系增强和 TensorRT 仍是后续扩展。Qwen 旧单图 Review 已在 Thor 跑通，Runtime V3 单次多图链路尚待真实 VLM 服务复测。

核心自动测试使用 FakeBackend 和 mock HTTP，不调用真实 YOLO 或 VLM；detector 实际运行已由 macOS 与 Thor 验证。

Runtime Python 版本要求为 3.10 或更高。

架构细节见 [[wiki/architecture]]。

## 正式数据入口

最终训练数据只保存在 3090：

- `datasets_final/prohibited_items/data.yaml`
- `datasets_final/garbage/data.yaml`

`unified_detection` 不再是正式入口。其在训练机上的物理目录是否已经删除尚待确认；如果仍存在，只能视为 `deprecated` / `investigation`，不得训练。

Mac 已清理早期 `data/`、`datasets_raw/`、`datasets_stage/`、`datasets_clean/`、测试 `runs/` 和 `yolo11n.pt`，不复制 `datasets_final/`。

## 设备分工

- Mac：主开发机，负责 Codex、代码、共享 Runtime Pipeline、GitHub、Obsidian/wiki 和轻量调试。
- 3090 Linux 工作站：保存最终数据，按顺序训练两个 YOLO11m，保存 runs / best.pt，并在需要时运行 Qwen / VLM。
- NVIDIA Jetson AGX Thor Developer Kit：最终边缘部署目标，负责加载多个 TensorRT engine、Qwen2.5-VL-32B、accuracy-first 验证和机器人联调。
- Orange Pi / RK3588：历史备选或测试路线，不是当前主线。

## 当前边界

尚需确认：

- 两个正式权重的最终版本、哈希和部署产物对应关系。
- 两个 detector 在 Thor 上继续串行还是改为并行运行。
- 机器人图片传输和结构化输出协议。
- Thor 实际 JetPack / TensorRT / CUDA / ROS2 / Docker 环境。
- behavior module 的模型、数据和推理形式。
- Qwen2.5-VL 的 endpoint、准确模型版本、运行设备、认证方式和实测超时降级行为。
