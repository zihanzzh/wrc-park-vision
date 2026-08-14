# Current Status

## 2026-08-11 Final engineering closure

项目状态：**feature complete / integration ready**。

### 真实验收

- NVIDIA Jetson AGX Thor + Docker + vLLM 已正常运行 `Qwen2.5-VL-32B-Instruct-AWQ`，served alias 为 `qwen-vl`。
- `prohibited_items`、`garbage`、`uncivilized_behavior` 已完成真实测试，整体准确率达到当前交付要求。
- YOLO / YOLO-World → Behavior Candidates → Accuracy-first Review → 32B → Fusion → Runtime JSON → Competition JSON → Preview 全链路已跑通。
- 所有 garbage / prohibited detector proposals 均经过 32B review；四类 behavior 保持完整。
- Bounded output protocol 与单次 truncation fallback 已完成并通过真实链路验证。
- 正式策略为 accuracy-first，不再以 10 秒作为最终要求。

### 产品集成状态

- `RuntimePipeline` 是唯一推理入口；初始化时加载模型并支持连续复用。
- `process()` 接受图片路径、PIL frame 或内部 `ValidatedImage`。
- `RequestContext` 支持 camera、timestamp、session、frame index。
- `build_competition_response()` 提供稳定的 product-facing V1 adapter。
- CLI 与 Python API 共用同一 Pipeline，没有两套逻辑。
- `wiki/integration-guide.md` 与 `wiki/deployment-checklist.md` 已作为交付和恢复入口。

### 正式模型栈

- YOLO11m garbage detector。
- YOLOv8s-WorldV2 / YOLO-World object proposal。
- Qwen2.5-VL-32B-Instruct-AWQ semantic review。
- NVIDIA Jetson AGX Thor、Docker、vLLM。

### 配置与协议

- Tracked example 的 alias 已统一为真实 `qwen-vl`。
- Machine-specific local YAML、endpoint、API key、路径继续 gitignored。
- Runtime 总预算 600 秒；VLM timeout 300 秒；primary/fallback token 为 3000/1800。
- Runtime contract 用于 debug/audit；Competition contract 用于产品集成。
- Fallback 成功时 Review completed，但 Competition `degraded=true`，明确表示 missed-object scan 未完成。

### 当前非 blocker 扩展

- 机器人最终 transport 封装（ROS2/HTTP/gRPC）。
- Tracking、多帧 behavior、pose、区域关系增强。
- TensorRT 专用 backend 与进一步资源优化。
- 正式部署包补录权重 SHA256、Thor 软件版本和已验收 vLLM 启动参数。

历史阶段、7B 实测、旧 10 秒策略和被替代路线见 [[codex-log]] 与 [[decisions]]，不再作为当前状态。
