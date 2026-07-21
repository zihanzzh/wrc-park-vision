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
- 当前扩展共享 Runtime，形成 Detection -> Detection Summary -> 全图 Review -> Fusion -> Output 链路。
- 准备 NVIDIA Jetson AGX Thor Developer Kit 的多模型部署和 TensorRT 验证。
- 接入真实 Qwen2.5-VL 服务并验证 10 秒超时与降级行为。

不文明行为尚未形成最终数据集和模型，需要独立设计并通过 behavior module 接入同一 Runtime Pipeline。

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
- `prohibited_items` 与 `garbage` 通过配置注册为通用 `DetectionModule`，主 Pipeline 不写死模型数量、类别或路径。
- 模型启动时加载一次；enabled 模型路径缺失时明确失败，不允许自动下载。
- enabled detection module 必须声明有序 `expected_class_names`；Ultralytics 权重加载后、图片处理前严格校验类别 ID、数量、名称和顺序。
- Pipeline 根据模型来源写入 `task_group`，不同模型保留各自 class id 空间。
- 跨 task group 高 IoU 结果全部保留，并互相标记 `cross_model_overlap`，不实施业务优先级删除。
- 已实现 Detection Summary、Qwen2.5-VL provider 接口、全图 Prompt Builder、严格 Response Parser 和最终 Fusion。
- VLM 接收完整原图，Detection Summary 仅作为上下文；VLM 可以确认、拒绝、纠正 YOLO，并报告没有 bbox 的漏检 finding。
- bbox 始终由 YOLO observation 提供；VLM 不输出或修正坐标。
- JSON 与 Preview 使用同一个最终 `PipelineResponse`，Preview 不重新推理或重算 bbox。
- Fusion 或 Review 失败不会删除成功模块的 observations；结果保留并以 `partial_success` 和阶段错误返回。
- `Observation.track_id` 已预留且单图流程默认为 `null`；Tracking 和多帧融合尚未实现。
- 当前执行策略固定为 sequential，只记录耗时，尚未实现强制 timeout 或 10 秒 deadline。
- behavior 和 TensorRT 仍保留接口或 schema 扩展点；Qwen provider 已实现，但尚未连接真实 VLM 服务。

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
