# Project Context

## 项目状态

`wrc-park-vision` 是 WRC 园区管理岗视觉识别项目。当前状态为 **feature complete / integration ready**。

NVIDIA Jetson AGX Thor 真实验收已经覆盖：

- 8 类 `prohibited_items`。
- 6 类 `garbage`。
- 4 类 `uncivilized_behavior`。
- YOLO / YOLO-World → Review → Qwen2.5-VL-32B → Fusion → Runtime JSON → Competition JSON → Preview 完整链路。
- Qwen2.5-VL-32B bounded output 与单次 truncation fallback。

三组任务整体准确率已达到当前项目交付要求。本阶段不继续调模型准确率、不更换模型、不重构 Pipeline。

## 最终技术栈

- Garbage：独立 YOLO11m detector。
- Prohibited / behavior auxiliary objects：YOLOv8s-WorldV2 / YOLO-World。
- Semantic verification / behavior reasoning：Qwen2.5-VL-32B-Instruct-AWQ。
- Runtime / Fusion：Python `RuntimePipeline`。
- 部署目标：NVIDIA Jetson AGX Thor Developer Kit，Docker + vLLM。
- vLLM served alias：`qwen-vl`。
- 正式策略：accuracy-first；不再使用 10 秒最终验收目标。

核心设计：Detector 提供 proposal 与定位，32B VLM 负责语义复核和行为推理，Fusion 以确定性规则生成最终结果。

## 正式 Runtime

```text
image / PIL frame
  -> input validation
  -> YOLO-World + garbage YOLO
  -> behavior candidates
  -> all garbage/prohibited proposals required for review
  -> important crops
  -> bounded Multi-Image Qwen2.5-VL-32B
  -> deterministic Fusion
  -> PipelineResponse
  -> CompetitionResponse
```

- `RuntimePipeline` 初始化时加载模型，实例应长驻并连续复用。
- `process()` 支持图片路径、PIL image 或 `ValidatedImage`。
- `RequestContext` 支持 `camera_id`、timestamp、`session_id`、`frame_index`。
- CLI 与产品 Python API 共用同一 Pipeline。
- 正常每帧一次 VLM 请求；只有明确截断时允许一次 compact fallback。
- 全部 garbage / prohibited detector proposals 必须经过 32B review。
- 四类 behavior 保持完整主动扫描。
- Runtime internal output 与 Competition/Product output 通过 adapter 分离。

## 配置与模型边界

- Tracked 配置只保存 deployable template。
- `configs/runtime.yolo-world.local.yaml`、endpoint、API key、机器路径和设备参数保持 gitignored。
- 正式 32B 模型宿主机目录为 `models/Qwen2.5-VL-32B-Instruct-AWQ`，容器中为 `/models/Qwen2.5-VL-32B-Instruct-AWQ`。
- 数据集、权重、模型、runs、runtime outputs 和 previews 不进入 Git。
- 正式数据与训练资产仍由外部训练/部署环境保存；仓库保存代码、schema、配置模板、测试和项目记忆。

## 已知扩展边界

以下不是当前交付 blocker：

- 机器人最终 transport 仍可按产品环境封装为 ROS2、HTTP、gRPC 或其他形式；不得复制 Runtime 逻辑。
- Tracking、多帧关系、pose、区域规则和 TensorRT 专用 backend 可作为后续版本能力。
- 正式部署包仍应记录 detector 权重哈希、Thor Docker/vLLM/JetPack 版本和启动参数。

架构见 [[wiki/architecture]]，产品对接见 [[wiki/integration-guide]]，恢复部署见 [[wiki/deployment-checklist]]。
