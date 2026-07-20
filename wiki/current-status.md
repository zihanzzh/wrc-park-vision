# Current Status

## 当前阶段

禁带品和垃圾数据准备已基本完成，两个独立 YOLO11m 已在外部训练机完成训练。Mac 仓库已完成正式 Runtime Pipeline v1 实现，但尚未获得两个正式权重，因此还没有执行真实模型 smoke test。

当前没有训练、Thor 部署、TensorRT 推理、behavior 推理或 VLM 推理在本轮发生。

## Runtime v1

已实现：

- `pyproject.toml` 与 `src/` Python package。
- 可提交的 `configs/runtime.example.yaml` 和 gitignored 的本地配置约定。
- 配置环境变量展开、disabled module 支持、enabled 模型路径启动校验。
- enabled detection module 的 `expected_class_names` 配置校验，以及 Ultralytics 权重 class ID / 数量 / 名称 / 顺序启动校验。
- 通用 `TaskModule`、`DetectionModule` 和 `InferenceBackend` 接口。
- Ultralytics backend，模型在 Pipeline 初始化时加载一次，禁止缺失路径触发自动下载。
- 通过配置启用 `prohibited_items` 和 `garbage`，主 Pipeline 不写死模块数量、类别或模型路径。
- 单图输入校验、sequential 多模块执行和单模块故障隔离。
- 初始化 load 失败清理此前已加载模块，`close()` 对所有模块执行 best-effort 清理。
- 顶层 `success`、`partial_success`、`failure` 状态。
- 统一 `PipelineResponse` / `Observation` schema，bbox 同时提供像素和归一化 xyxy。
- 跨 task group 重叠结果全部保留并标记 `cross_model_overlap`。
- `low_confidence`、`cross_model_overlap`、`module_failure` review decision。
- Fusion / Review 独立故障隔离，后处理失败时保留正常 observations 并返回 `partial_success`。
- `Observation.track_id` 可空字段、datetime timestamp 和非负 frame index；Tracking 本身尚未实现。
- 原子 JSON 写出，以及直接复用最终 observation 的 Preview。
- CLI 和不依赖真实权重的 FakeBackend 自动测试。

当前明确未实现：

- behavior 模型内部逻辑。
- 真正的 Qwen / VLM 调用与结果回写。
- TensorRT backend 和 Thor 部署。
- API、ROS2、stream、tracking、并行执行和强制 timeout。

## 验证状态

- 自动测试：35 项通过。
- Python 语法编译检查：通过。
- 测试不加载真实权重，不运行 YOLO。
- 当前 `.venv` 尚未按新 `pyproject.toml` 完整安装依赖，本轮没有安装或升级任何包。
- 真实权重 smoke test：待执行。
- 真实权重的 `expected_class_names` 启动校验：逻辑和纯函数测试已完成，真实权重尚未验证。
- Thor benchmark 与 10 秒完整链路验证：待执行。

## 数据与设备边界

- 正式数据继续只保存在 3090 的 `datasets_final/prohibited_items/` 和 `datasets_final/garbage/`。
- Mac 不保存正式数据集，不修改 labels。
- 建议将两个本地权重放到 gitignored 的 `models/`，并通过环境变量配置路径。
- Thor 仍是最终边缘部署目标；Orange Pi / RK3588 不是当前主线。

## 下一步

1. 提供 `prohibited_items_yolo11m_best.pt` 与 `garbage_yolo11m_best.pt`，同时确认各自 class names、训练版本和文件哈希。
2. 按 `pyproject.toml` 建立可复现环境，不做全局安装。
3. 使用一张真实园区或模拟比赛照片运行两个 enabled modules。
4. 人工核对 `result.json` 与 `preview.jpg` 是否对应同一 observation、坐标和模型来源。
5. 记录 Mac 的真实 smoke test 耗时，再进入 Thor TensorRT backend 和 benchmark 阶段。
