# Integration Guide

## Product Integration

### 系统职责

WRC Park Vision 接收单张图片或相机帧，运行 detector、32B semantic review 与 deterministic Fusion，返回完整 `PipelineResponse`；产品侧通常消费精简的 `CompetitionResponse`。

Runtime 不负责机器人 transport、动作决策、语音文案或跨帧 tracking。

### 输入

`RuntimePipeline.process()` 支持：

- 图片路径：`str` / `Path`。
- 内存帧：`PIL.Image.Image`。
- 内部 `ValidatedImage`。

可选 `RequestContext`：`camera_id`、timestamp、`session_id`、`frame_index`。

### 最小 Python API

```python
from pathlib import Path

from wrc_park_vision.runtime import (
    RequestContext,
    RuntimePipeline,
    build_competition_response,
    load_runtime_config,
)

config = load_runtime_config(Path("configs/runtime.yolo-world.local.yaml"))
runtime = RuntimePipeline(config)  # 初始化一次，模型常驻

try:
    result = runtime.process(
        Path("/path/to/frame.jpg"),
        context=RequestContext(
            camera_id="camera-01",
            session_id="robot-session-01",
            frame_index=1,
        ),
    )
    product = build_competition_response(result)
    payload = product.model_dump(mode="json")
finally:
    runtime.close()
```

持续帧循环必须复用同一个 `runtime`。`RuntimePipeline` 也支持 context manager。不要为每帧重新加载配置、权重或 VLM provider。

### 输出

- `PipelineResponse`：完整模块状态、observations、Detection Summary、VLM Review、Fusion、issues/errors、timing 和 metrics。
- `CompetitionResponse`：`frame`、`status`、`objects`、`behaviors`、`processing_time_ms`、`degraded`。

产品 transport 层应直接序列化 `CompetitionResponse.model_dump(mode="json")`，不要重新解释 detector/VLM verdict 或再次实现 Fusion。

## Thor Runtime

### Prerequisites

- NVIDIA Jetson AGX Thor 上已验收的 Docker、vLLM 和 Python 环境。
- YOLOv8s-WorldV2、garbage YOLO11m 权重可用。
- 宿主机 `models/Qwen2.5-VL-32B-Instruct-AWQ` 已映射为容器内 `/models/Qwen2.5-VL-32B-Instruct-AWQ`。
- vLLM served alias 为 `qwen-vl`。
- `configs/runtime.yolo-world.local.yaml` 已配置机器路径、endpoint 和 API key environment name。

仓库未记录已验收环境的完整 Docker/vLLM 启动命令，不应从文档猜测参数；恢复时从部署记录补回版本、挂载、显存和视觉输入参数。

### Health check

先启动 vLLM，再检查：

```bash
curl http://127.0.0.1:8000/v1/models
```

返回模型列表必须包含 `qwen-vl`，并与 `WRC_VLM_MODEL_ID` 一致。

### CLI smoke test

```bash
python -m wrc_park_vision.runtime.cli \
  --config configs/runtime.yolo-world.local.yaml \
  --image test_images/example.jpg
```

检查 `status`、`result.json`、`competition_result.json`、`preview.jpg` 和 CLI exit code。CLI 与产品 API 使用同一个 `RuntimePipeline`。

## Failure behavior

| 情况 | Runtime / Product 语义 | 产品处理建议 |
|---|---|---|
| 单 YOLO module 失败、其他模块成功 | `partial_success`，`degraded=true`，保留其他 detector 结果 | 可继续使用明确标记的 fallback 结果；记录并报警 |
| VLM request / parse 失败 | required proposals 按 `review_failure_policy` 保留并标记；通常 `partial_success` | 不把 `review_failed` 当成 32B confirmed；报警并保留原图/日志 |
| Primary response 截断、fallback 成功 | Review completed；object/behavior 已 Fusion；`new_findings=[]`；`degraded=true` | 结果可用，但记录 missed-object scan 未完成 |
| Fallback 再失败 | 不再 retry；进入 review failure policy | 按 VLM failure 处理并报警 |
| 输入非法或所有 detector 失败 | `failed` | 不作为有效识别结果使用 |
| Preview 失败 | 结构化推理结果仍保留，附 output error | 产品结果可继续；修复 artifact/路径问题 |

需要记录/报警的字段：`status`、`degraded`、`errors`、object `review_status`、Review issues、fallback metrics。异常 VLM 原文仅保存在 `vlm_raw_response.txt`，不得转发为 Competition payload。
