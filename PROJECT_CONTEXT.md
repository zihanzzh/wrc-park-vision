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
- 当前共享 Runtime 已形成 Detection -> Detection Summary -> Full Image Review -> Crop Scan Review -> Fusion -> Output 链路。
- 准备 NVIDIA Jetson AGX Thor Developer Kit 的多模型部署和 TensorRT 验证。
- 接入真实 Qwen2.5-VL 服务并验证 10 秒超时与降级行为。

不文明行为尚未形成独立训练数据集和专用模型；当前已实现单图 Behavior Pipeline，使用 YOLO-World 基础对象、配置化候选规则和 Full Image Qwen Review 共同判断四类行为。

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
- 已实现 Detection Summary、Qwen2.5-VL 双 Pass provider、两套互补 Prompt、共享逐项容错 Parser 和最终 Fusion。
- 已实现配置驱动的单图 Behavior Pipeline：基础对象只生成候选，最终行为必须由 Full Image Pass 确认；无候选时仍允许全图发现明显行为。
- 当前正式行为类别为 `trampling_grass`、`smoking`、`blocking_fire_lane`、`standing_or_lying_on_bench`。
- Pass 1 接收完整原图和 Detection Summary，负责确认、拒绝、纠正 YOLO、明显漏检和行为；Pass 2 在一次请求中接收全部重叠 crops，独立扫描小目标。
- corrected 始终复用 YOLO bbox；VLM 新 finding 必须提供 full-image 或 crop-relative normalized bbox，由 Pipeline 转为完整原图 geometry。
- Fusion 对同类高 IoU 结果去重，并保留来源追踪；不同类别高 IoU 结果不静默删除。
- JSON 与 Preview 使用同一个最终 `PipelineResponse`，Preview 不重新推理或重算 bbox。
- Fusion 或 Review 失败不会删除成功模块的 observations；结果保留并以 `partial_success` 和阶段错误返回。
- `Observation.track_id` 已预留且单图流程默认为 `null`；Tracking 和多帧融合尚未实现。
- 当前执行策略固定为 sequential，分别记录 detection、Full Image Review、crop 生成、Crop Scan Review、Fusion 和 Preview 耗时；每个 Review Pass 有独立 HTTP timeout，完整请求级 10 秒 deadline 尚未实现。
- Behavior 的单图语义链路已实现；多帧、tracking、pose/区域关系增强和 TensorRT 仍是后续扩展。Qwen 单次全图 Review 已在 Thor 跑通，双 Pass 链路尚待真实 VLM 服务复测。

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
- NVIDIA Jetson AGX Thor Developer Kit：最终边缘部署目标，负责加载多个 TensorRT engine、性能 benchmark、10 秒预算验证和机器人联调。
- Orange Pi / RK3588：历史备选或测试路线，不是当前主线。

## 当前边界

尚需确认：

- 两个正式权重的最终版本、哈希和部署产物对应关系。
- 两个 detector 在 Thor 上继续串行还是改为并行运行。
- 机器人图片传输和结构化输出协议。
- Thor 实际 JetPack / TensorRT / CUDA / ROS2 / Docker 环境。
- behavior module 的模型、数据和推理形式。
- Qwen2.5-VL 的 endpoint、准确模型版本、运行设备、认证方式和实测超时降级行为。
