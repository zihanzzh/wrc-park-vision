# wrc-park-vision

WRC 园区管理岗视觉识别项目。仓库同时作为 Obsidian vault，维护正式 Runtime、配置、自动测试、架构决策和项目记忆。

## 当前主线

- `prohibited_items` 和 `garbage` 使用两个独立 YOLO11m 模型。
- 同一张图片由共享 Runtime 顺序交给所有 enabled task modules。
- 可选 YOLO-World backend 检测禁带品、垃圾和行为判断所需的基础对象。
- 模型结果统一为稳定 schema，并保留各自 `task_group`、类别、置信度、坐标和来源模型。
- 跨任务高 IoU 结果不会被擅自删除，只会保留并标记冲突。
- 可选 Qwen2.5-VL Review 接收完整原图和 Detection Summary，独立完成语义复核和漏检补充。
- VLM 不生成 bbox；Fusion 保留原始 YOLO observations、VLM findings 和最终决策，不静默删除结果。
- 单图 Behavior Pipeline 根据基础对象生成候选，并由同一次全图 VLM 请求确认四类不文明行为；没有基础对象时仍允许全图发现明显行为。
- TensorRT backend、正式 Thor 部署包、API、ROS2、tracking、多帧行为判断和请求级强制超时尚未实现。

正式数据和训练产物只保存在 3090 Linux 工作站，不进入本仓库。两个已训练 detector 已完成 macOS 与 Thor 实际运行验证；Qwen2.5-VL provider 当前只有 mock 自动测试，尚未完成真实服务联调。

## Runtime 结构

- `configs/runtime.example.yaml`：可提交的运行配置示例。
- `src/wrc_park_vision/runtime/`：Pipeline、schema、backend、task module、行为候选、融合、review、输出和 CLI。
- `src/wrc_park_vision/runtime/vlm/`：全图 Review provider、Prompt Builder 和 Response Parser。
- `tests/runtime/`：不依赖真实权重的自动测试。
- `runtime_outputs/<request_id>/result.json`：结构化结果。
- `runtime_outputs/<request_id>/preview.jpg`：直接基于同一份最终结果绘制的预览图。

## 环境准备

Runtime 要求 Python 3.10 或更高版本。本轮实现没有安装或升级依赖。需要正式建立环境时，在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[ultralytics,dev]"
```

本地绝对权重路径写入环境变量或 gitignored 的 `configs/runtime.local.yaml`，不要写进 `runtime.example.yaml`。建议将权重放在本地 `models/`：

```text
models/prohibited_items_yolo11m_best.pt
models/garbage_yolo11m_best.pt
```

`.pt` 已被 `.gitignore` 忽略。

## 单图运行

```bash
cp configs/runtime.example.yaml configs/runtime.local.yaml
export WRC_PROHIBITED_MODEL_PATH="$PWD/models/prohibited_items_yolo11m_best.pt"
export WRC_GARBAGE_MODEL_PATH="$PWD/models/garbage_yolo11m_best.pt"

python -m wrc_park_vision.runtime.cli \
  --config configs/runtime.local.yaml \
  --image /absolute/path/to/test-image.jpg
```

可用 `--no-preview` 禁用预览，或用 `--output-dir` 覆盖输出目录。退出码为：`0` 表示全部模块成功，`2` 表示部分成功，`1` 表示失败。

Runtime 在启动时校验 enabled 模块的模型路径，模型不存在会明确失败，不会触发自动下载。`device: auto` 优先选择 CUDA，其次 MPS，最后 CPU。

每个 enabled detection module 还必须配置有序的 `expected_class_names`。Ultralytics 权重加载后会在处理图片前严格校验 class ID 连续性、类别数量、名称和顺序，避免错误权重静默进入 Runtime。

Review provider 默认关闭。启用时需在本地配置提供 OpenAI-compatible chat completions endpoint 和 Qwen2.5-VL `model_id`；请求超时默认 10 秒。每张图片最多调用一次 provider，该请求同时复核物体、发现漏检物体、验证行为候选并执行四类行为全图扫描。Fusion 或 Review 后处理失败时，Runtime 会保留已成功产生的 observations、记录对应错误，并将顶层状态降级为 `partial_success`。

Preview 只读取最终 `PipelineResponse`：YOLO bbox 直接来自 `observations`，VLM-only finding 因没有坐标只显示在预览底部，不会生成推测框。`Observation.track_id` 已作为可空字段写入 schema，但当前没有实现 Tracking 或多帧融合。

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
