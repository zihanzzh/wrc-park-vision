# WRC Park Vision

WRC 园区管理岗视觉识别 Runtime。当前开发状态为 **feature complete / integration ready**：禁带品、垃圾和四类不文明行为已在 NVIDIA Jetson AGX Thor 上完成真实 accuracy-first 验收。

## 最终数据流

```text
image / camera frame
  -> RuntimePipeline
  -> YOLOv8s-WorldV2 + YOLO11m garbage detector
  -> Behavior Candidate Generation
  -> Accuracy-first Review Policy
  -> Multi-Image Qwen2.5-VL-32B
  -> deterministic Fusion
  -> PipelineResponse
  -> Competition Response Adapter
  -> competition_result.json
```

核心职责分层：Detector 是 proposal / localization layer，32B VLM 是 semantic verification + behavior reasoning layer，Fusion 是 deterministic final decision layer。正常帧只发送一次 VLM 请求；明确截断时最多追加一次 compact fallback。

## 产品 Python API

产品代码应长驻一个 `RuntimePipeline`，不能每帧重新初始化模型。现有 Pipeline 就是正式集成 API，没有第二套推理逻辑：

```python
from datetime import datetime, timezone
from pathlib import Path

from wrc_park_vision.runtime import (
    RequestContext,
    RuntimePipeline,
    build_competition_response,
    load_runtime_config,
)

config = load_runtime_config(Path("configs/runtime.yolo-world.local.yaml"))

with RuntimePipeline(config) as runtime:
    result = runtime.process(
        Path("/path/to/frame.jpg"),
        context=RequestContext(
            camera_id="park-camera-01",
            timestamp=datetime.now(timezone.utc),
            session_id="robot-session-01",
            frame_index=42,
        ),
    )
    product_result = build_competition_response(result)
    product_json = product_result.model_dump(mode="json")
```

`process()` 也接受 `PIL.Image.Image` 或内部 `ValidatedImage`，适合相机帧内存调用。内存帧不会自动写临时图片；如需生成 `preview.jpg`，应传入可读取的图片路径。

详细对接说明见 [Integration Guide](wiki/integration-guide.md)。

## Thor / 本地 CLI

CLI 与 Python API 共用同一个 `RuntimePipeline`：

```bash
python -m wrc_park_vision.runtime.cli \
  --config configs/runtime.yolo-world.local.yaml \
  --image test_images/example.jpg
```

默认生成 `result.json`、`competition_result.json` 和 `preview.jpg`。可使用 `--no-preview`、`--output-dir` 或 `--output-mode both|internal|competition`。退出码：`0` 成功、`2` 部分成功、`1` 失败。

## 正式配置与模型

Tracked deployable templates：

- `configs/runtime.example.yaml`
- `configs/runtime.yolo-world.example.yaml`

Machine-specific config：

- `configs/runtime.yolo-world.local.yaml`，由 `.gitignore` 排除。
- 权重路径、endpoint、API key 和设备参数不得写入 tracked template。

正式环境：

- Garbage detector：YOLO11m。
- Object proposals：YOLOv8s-WorldV2 / YOLO-World。
- Semantic review：`Qwen2.5-VL-32B-Instruct-AWQ`。
- vLLM served model alias：`qwen-vl`。
- 目标硬件：NVIDIA Jetson AGX Thor Developer Kit，Docker + vLLM。
- 宿主机模型目录：`models/Qwen2.5-VL-32B-Instruct-AWQ`。
- 容器内模型目录：`/models/Qwen2.5-VL-32B-Instruct-AWQ`。仓库没有固化具体 Docker 启动命令；部署时必须使用已验收环境的挂载参数，使上述容器路径成立。
- 策略：accuracy-first。600 秒 Runtime 总预算和 300 秒 VLM timeout 是安全上限，不是 `<10 seconds` 验收目标。
- Token：primary 3000，compact fallback 1800。

启动 Runtime 前检查 vLLM：

```bash
curl http://127.0.0.1:8000/v1/models
```

响应必须包含 `qwen-vl`，且 local YAML 的 `WRC_VLM_MODEL_ID` 必须一致。

## 正式类别

类别真源为 `configs/runtime.yolo-world.example.yaml` 与 detector 权重 metadata。

Prohibited items（8 类）：

1. `spray_can`
2. `portable_gas_stove`
3. `megaphone`
4. `skateboard`
5. `kick_scooter`
6. `speaker`
7. `roller_skates`
8. `barbecue_grill`

Garbage（6 类）：

1. `crumpled_paper_ball`
2. `disposable_food_container`
3. `empty_cigarette_box`
4. `plastic_drink_bottle`
5. `plastic_food_wrapper`
6. `rigid_takeout_bag`

Behavior auxiliary objects：`person`、`bench`、`grass`、`cigarette`、`vehicle`。

正式 behaviors：`trampling_grass`、`smoking`、`blocking_fire_lane`、`standing_or_lying_on_bench`。

## Output contracts

### Runtime `PipelineResponse`

用于 debug、audit 和 pipeline inspection，包含 module status、最终 observations、Detection Summary、VLM review、Fusion decisions、issues/errors、timings 和 request metrics。CLI 的 `result.json` 是该结构的 JSON 形式。

### Competition / Product `CompetitionResponse`

用于机器人和产品集成，固定包含：

- `frame`：`frame_id`、timestamp、width、height。
- `status`：`success`、`partial_success` 或 `failed`。
- `objects`：最终 object ID、task group、class、pixel/normalized bbox、confidence、review status、source。
- `behaviors`：四类正式行为、confidence、evidence object IDs。
- `processing_time_ms`。
- `degraded`。

`objects` 当前可能包含 `prohibited_items`、`garbage` 和 `uncivilized_behavior` 辅助对象。只有 Fusion 后仍有效的 observations 会进入 product response；rejected proposal 不会输出。

## Failure semantics

- 单个 YOLO module 失败：隔离该模块；若其他模块成功，返回 `partial_success`，结果仍可作为 detector fallback 使用，但产品应记录并报警。
- VLM 请求或解析失败：按 `review_failure_policy` 保留并标记 required detector proposals，返回 `partial_success` / `degraded=true`。
- Primary 截断：自动进行一次 compact fallback；成功时 object/behavior decisions 继续 Fusion，`new_findings=[]`，Review 保持 completed，但 product response 为 `degraded=true`。
- Fallback 再失败：不重试，进入既有 review failure policy。
- `failed`：输入无效或没有成功 detector module，产品不应把结果当作有效识别结果。
- Preview 失败不删除已生成的结构化推理结果，但应记录 output error。

## 安装与验证

Python 要求 3.10+：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[ultralytics,yolo-world,dev]"
python -m unittest discover -s tests -t . -v
```

不要把数据集、权重、32B 模型、runtime outputs、preview 或 raw VLM response 提交到 Git。

## 文档入口

- [Project Context](PROJECT_CONTEXT.md)
- [Integration Guide](wiki/integration-guide.md)
- [Deployment Checklist](wiki/deployment-checklist.md)
- [Architecture](wiki/architecture.md)
- [Current Status](wiki/current-status.md)
- [Decisions](wiki/decisions.md)
- [Hardware Notes](wiki/hardware-notes.md)
