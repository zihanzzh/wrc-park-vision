# wrc-park-vision

WRC 园区管理岗视觉识别项目。仓库同时作为 Obsidian vault，维护正式 Runtime、配置、自动测试、架构决策和项目记忆。

## 当前主线

- 当前单帧 Runtime 使用 YOLO-World 检测禁带品和行为辅助对象，使用独立 YOLO11m 检测垃圾。
- 同一张图片由共享 Runtime 顺序交给所有 enabled task modules。
- YOLO-World 不负责垃圾；配置和 backend 都会拒绝其输出 `task_group: garbage`。
- 模型结果统一为稳定 schema，并保留各自 `task_group`、类别、置信度、坐标和来源模型。
- 跨任务高 IoU 结果不会被擅自删除，只会保留并标记冲突。
- 可选 Qwen2.5-VL Review 默认固定执行两次互补请求：完整原图审核，以及一次携带全部重叠 crops 的独立漏检扫描。
- corrected 继续复用 YOLO bbox；VLM 新 finding 必须返回 normalized bbox，由 Pipeline 映射为完整原图 geometry。
- Fusion 跨 YOLO、全图 finding 和 crop finding 做同类 IoU 去重；不同类别高 IoU 结果保留并标记冲突。
- 单图 Behavior Pipeline 根据基础对象生成候选，并由 Full Image Pass 确认四类不文明行为；没有基础对象时仍允许全图发现明显行为。
- TensorRT backend、正式 Thor 部署包、API、ROS2、tracking、多帧行为判断和请求级强制超时尚未实现。

正式数据和训练产物只保存在 3090 Linux 工作站，不进入本仓库。两个已训练 detector 已完成 macOS 与 Thor 实际运行验证；Qwen2.5-VL 单次全图 Review 已在 Thor 跑通，新的双 Pass 链路目前只有 mock 自动测试，尚待真实服务复测。

## Runtime 结构

- `configs/runtime.example.yaml`：可提交的运行配置示例。
- `src/wrc_park_vision/runtime/`：Pipeline、schema、backend、task module、行为候选、融合、review、输出和 CLI。
- `src/wrc_park_vision/runtime/crops.py`：配置驱动的重叠 crop 生成和坐标映射。
- `src/wrc_park_vision/runtime/vlm/`：双 Pass Review provider、Prompt Builder 和共享 Response Parser。
- `tests/runtime/`：不依赖真实权重的自动测试。
- `runtime_outputs/<request_id>/result.json`：结构化结果。
- `runtime_outputs/<request_id>/preview.jpg`：直接基于同一份最终结果绘制的预览图。

## 环境准备

Runtime 要求 Python 3.10 或更高版本。本轮实现没有安装或升级依赖。需要正式建立环境时，在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[ultralytics,yolo-world,dev]"
```

本地绝对权重路径写入环境变量或 gitignored 的 `configs/runtime.local.yaml`，不要写进 `runtime.example.yaml`。建议将权重放在本地 `models/`：

```text
models/yolov8s-worldv2.pt
models/garbage_yolo11m_best.pt
```

`.pt` 已被 `.gitignore` 忽略。

## 单图运行

```bash
cp configs/runtime.example.yaml configs/runtime.local.yaml
export WRC_YOLO_WORLD_MODEL_PATH="$PWD/models/yolov8s-worldv2.pt"
export WRC_GARBAGE_MODEL_PATH="$PWD/models/garbage_yolo11m_best.pt"

python -m wrc_park_vision.runtime.cli \
  --config configs/runtime.local.yaml \
  --image /absolute/path/to/test-image.jpg
```

可用 `--no-preview` 禁用预览，或用 `--output-dir` 覆盖输出目录。退出码为：`0` 表示全部模块成功，`2` 表示部分成功，`1` 表示失败。

Runtime 在启动时校验 enabled 模块的模型路径，模型不存在会明确失败，不会触发自动下载。`device: auto` 优先选择 CUDA，其次 MPS，最后 CPU。

固定类别的 enabled detection module 必须配置有序 `expected_class_names`。垃圾 YOLO11m 权重加载后会在处理图片前严格校验 class ID 连续性、类别数量、名称和顺序，避免错误权重静默进入 Runtime。YOLO-World 使用分组 `open_vocabulary_classes`，当前只允许 `prohibited_items` 与 `uncivilized_behavior` 基础对象。

Review provider 默认关闭。启用时需在本地配置提供 OpenAI-compatible chat completions endpoint 和 Qwen2.5-VL `model_id`。默认每张图片调用两次同一 provider：Pass 1 接收完整原图并处理物体审核、明显漏检和行为；Pass 2 在一次 HTTP 请求中接收全部配置化重叠 crops，只扫描遗漏对象。两个 Pass 分别配置 timeout 和 `max_tokens`，一个 Pass 失败不阻止另一个 Pass 的合法结果进入 Fusion，顶层状态降级为 `partial_success`。

Preview 只读取最终 `PipelineResponse`：YOLO、VLM corrected、完整原图 finding 和 crop finding 都使用最终 observation 中的同一份 bbox，不重新计算或推理；无 geometry 的行为结果显示在预览底部。`Observation.track_id` 已作为可空字段写入 schema，但当前没有实现 Tracking 或多帧融合。

## 自动测试

安装 dev extra 后运行：

```bash
pytest
```

测试使用 FakeBackend，不需要权重，也不会运行 YOLO。也可以使用标准库入口：

```bash
python -m unittest discover -s tests -t . -v
```

## 文档入口

- `PROJECT_CONTEXT.md`：项目背景、当前阶段和职责边界。
- `AGENTS.md`：Codex / agent 长期工作规则。
- `wiki/content-map.md`：Obsidian/wiki 导航。
- `wiki/current-status.md`：最新状态和下一步。
- `wiki/architecture.md`：正式多模型 Runtime 架构和后续扩展方向。
- `wiki/decisions.md`：已确认决策。
- `wiki/open-questions.md`：仍待确认的问题。

默认使用中文沟通和维护文档，不自动 commit、push 或 deploy。
