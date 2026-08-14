# WRC Park Vision

WRC Park Vision is an accuracy-first computer vision runtime for intelligent park inspection and management. It combines YOLO-based object proposals, a dedicated garbage detector, Qwen2.5-VL-32B semantic verification and behavior reasoning, and deterministic fusion. The system has been deployed and validated on NVIDIA Jetson AGX Thor and exposes an integration-ready Python API and competition response.

- **Status:** Feature Complete / Integration Ready
- **Release:** v1.0.0
- **Target:** NVIDIA Jetson AGX Thor
- **VLM:** Qwen2.5-VL-32B-Instruct-AWQ

## Highlights

- YOLOv8s-WorldV2 open-vocabulary object proposals
- Dedicated YOLO11m garbage detector
- Qwen2.5-VL-32B semantic verification and behavior reasoning
- Four supported uncivilized behavior classes
- Accuracy-first review of every garbage and prohibited-item proposal
- Multi-image review using the original frame plus focused crops
- Bounded JSON output with one-shot compact truncation fallback
- Deterministic review fusion and cross-source deduplication
- Reusable Python integration API and shared CLI pipeline
- NVIDIA Thor deployment with Docker and vLLM

## Architecture

```text
Image / Camera Frame
        ↓
YOLO-World + Garbage YOLO11m
        ↓
Behavior Candidate Generation
        ↓
Accuracy-first Qwen2.5-VL-32B Review
        ↓
Bounded JSON / Truncation Fallback
        ↓
Deterministic Fusion
        ↓
PipelineResponse
        ↓
CompetitionResponse
```

**Detector** = proposal and localization layer · **32B VLM** = semantic verification and behavior reasoning layer · **Fusion** = deterministic final decision layer.

Normal frames use one unified VLM request. A second, compact request is allowed only when the primary response is clearly truncated.

## Supported Tasks

The runtime class catalog is sourced from `configs/runtime.yolo-world.example.yaml` and detector weight metadata.

| Task | Classes |
|---|---|
| Prohibited Items | `spray_can`, `portable_gas_stove`, `megaphone`, `skateboard`, `kick_scooter`, `speaker`, `roller_skates`, `barbecue_grill` |
| Garbage | `crumpled_paper_ball`, `disposable_food_container`, `empty_cigarette_box`, `plastic_drink_bottle`, `plastic_food_wrapper`, `rigid_takeout_bag` |
| Behaviors | `trampling_grass`, `smoking`, `blocking_fire_lane`, `standing_or_lying_on_bench` |

Auxiliary behavior objects: `person`, `bench`, `grass`, `cigarette`, `vehicle`.

## Quick Start

### Python Integration API

Keep one `RuntimePipeline` instance alive so models are initialized once and reused across frames:

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

`process()` also accepts `PIL.Image.Image` and the internal `ValidatedImage` abstraction. In-memory frames are not written to temporary files; provide a readable image path when `preview.jpg` is required.

### CLI

The CLI and product API use the same `RuntimePipeline`:

```bash
python -m wrc_park_vision.runtime.cli \
  --config configs/runtime.yolo-world.local.yaml \
  --image test_images/example.jpg
```

By default, the CLI generates `result.json`, `competition_result.json`, and `preview.jpg`. Use `--no-preview`, `--output-dir`, or `--output-mode both|internal|competition` as needed. Exit codes are `0` for success, `2` for partial success, and `1` for failure.

See the [Integration Guide](wiki/integration-guide.md) for the full product handoff contract.

## Deployment

- **Hardware:** NVIDIA Jetson AGX Thor Developer Kit
- **Runtime:** Docker + vLLM
- **VLM:** Qwen2.5-VL-32B-Instruct-AWQ
- **Served model alias:** `qwen-vl`
- **Host model directory:** `models/Qwen2.5-VL-32B-Instruct-AWQ`
- **Container model directory:** `/models/Qwen2.5-VL-32B-Instruct-AWQ`
- **Policy:** accuracy-first; fixed sub-10-second latency is not a release requirement

The repository does not prescribe an unverified Docker command. Deployment must reuse the validated environment and mount configuration so the container path above remains correct.

Before starting the runtime, verify the vLLM service:

```bash
curl http://127.0.0.1:8000/v1/models
```

The response must expose `qwen-vl`, and `WRC_VLM_MODEL_ID` in the local configuration must match that alias.

## Output Contracts

### `PipelineResponse` — internal, debug, and audit

The complete runtime result includes module status, fused observations, Detection Summary, VLM review decisions, Fusion decisions, issues and errors, timings, and request metrics. The CLI serializes this contract as `result.json`.

### `CompetitionResponse` — robot and product integration

The product-facing response contains:

- `frame`: frame ID, timestamp, width, and height
- `status`: `success`, `partial_success`, or `failed`
- `objects`: final class, task group, pixel/normalized bbox, confidence, review status, and source
- `behaviors`: behavior class, confidence, and evidence object IDs
- `processing_time_ms`
- `degraded`

`objects` may include `prohibited_items`, `garbage`, and `uncivilized_behavior` auxiliary objects. Only observations that remain after Fusion are exported; rejected proposals are omitted.

## Reliability & Failure Handling

- **Detector isolation:** one detector may fail without discarding results from successful modules; the response becomes `partial_success` and should be logged or alerted.
- **VLM failure fallback:** request or parse failures retain and mark required detector proposals according to `review_failure_policy`, with `degraded=true`.
- **Truncation fallback:** a truncated primary response triggers one compact retry. If it succeeds, object and behavior decisions still reach Fusion, `new_findings=[]`, and the product response remains explicitly degraded.
- **No retry loop:** if the compact fallback also fails, the runtime applies the existing review failure policy without another retry.
- **Failed result:** invalid input or no successful detector module produces `failed`; product code must not treat it as a valid recognition result. Preview failure does not delete already generated structured results.

## Runtime Configuration

Tracked deployable templates:

- `configs/runtime.example.yaml`
- `configs/runtime.yolo-world.example.yaml`

Machine-specific configuration:

- `configs/runtime.yolo-world.local.yaml` is excluded by `.gitignore`.
- Weight paths, endpoints, API keys, and device-specific settings must not be written into tracked templates.

Current accuracy-first safety limits:

- Runtime total timeout: 600 seconds
- VLM timeout: 300 seconds
- Primary output budget: 3000 tokens
- Compact fallback budget: 1800 tokens

These are safety limits, not performance claims. Datasets, weights, the 32B model, runtime outputs, previews, and raw VLM responses must not be committed to Git.

## Installation & Verification

Python 3.10 or newer is required:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[ultralytics,yolo-world,dev]"
python -m unittest discover -s tests -t . -v
```

## Documentation

Detailed engineering documentation is maintained under `wiki/`:

- [Project Context](PROJECT_CONTEXT.md)
- [Architecture](wiki/architecture.md)
- [Integration Guide](wiki/integration-guide.md)
- [Deployment Checklist](wiki/deployment-checklist.md)
- [Current Status](wiki/current-status.md)
- [Decisions](wiki/decisions.md)
- [Hardware Notes](wiki/hardware-notes.md)

## Development Status

- **Development Status:** Feature Complete / Integration Ready
- **Release:** v1.0.0

The project is now in a stable delivery state. Future feature work should start from the release branch or a dedicated new branch rather than changing the validated runtime in place.
