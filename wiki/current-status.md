# Current Status

## 当前阶段

禁带品与垃圾两个独立 detector 已完成训练，并已在 macOS 和 NVIDIA Thor 上跑通现有 Runtime 检测链路。当前进入 VLM Review Pipeline 实现与联调阶段。

本轮已完成代码实现与 mock 自动测试，没有运行真实 Qwen2.5-VL、训练模型、修改数据集、安装依赖或 push。

## Runtime 当前链路

```text
完整图片
  -> prohibited_items / garbage detection modules
  -> Detection Summary
  -> 可选 Qwen2.5-VL 全图 Review
  -> Final Fusion
  -> PipelineResponse
  -> result.json / preview.jpg
```

已实现：

- 配置驱动的通用 task modules，当前接入 `prohibited_items` 和 `garbage`。
- Ultralytics backend 启动时加载一次模型，并严格校验 `expected_class_names`。
- 单模块故障隔离、稳定 observation ID、跨 task group 冲突保留与标记。
- Detection Summary，向 VLM 提供 YOLO 语义上下文，但不限制 VLM 的全图观察范围。
- Qwen2.5-VL provider interface、OpenAI-compatible HTTP provider、Prompt Builder 和严格 Response Parser。
- VLM 可以确认、拒绝或纠正 YOLO 类别，也可以报告 YOLO 完全漏检的目标。
- VLM 不承担定位，不允许返回 bbox；VLM-only finding 的 `geometry` 为 `null`。
- Final Fusion 同时保留原始 YOLO observations、VLM decisions/findings 和显式 fusion decisions，不静默删除结果。
- Review 或 Fusion 失败时保留 detector 结果，并返回阶段错误和 `partial_success`。
- JSON 与 Preview 使用同一个最终 `PipelineResponse`；VLM-only finding 只在预览信息区显示，不绘制推测框。
- Qwen provider 默认关闭，detector-only 配置继续保持可用。

当前明确未完成：

- 真实 Qwen2.5-VL endpoint 联调与效果验证。
- 请求级 10 秒 deadline；当前 provider 只有单次 HTTP 10 秒 timeout。
- behavior 模型内部逻辑。
- TensorRT backend、正式 Thor engine 部署与 benchmark。
- API、ROS2、stream、tracking 和并行执行。

## 验证状态

- 自动测试：46 项通过。
- Python `compileall`：通过。
- `git diff --check`：通过。
- Qwen 请求测试使用 mock HTTP，确认发送完整图片 data URL 和 Detection Summary prompt，没有访问真实服务。
- detector 实际运行：macOS 与 NVIDIA Thor 已跑通。
- VLM 实际运行：尚未执行。
- 10 秒完整链路：尚未实测。

## 数据与设备边界

- 正式数据继续只保存在 3090 的 `datasets_final/prohibited_items/` 和 `datasets_final/garbage/`。
- Mac 不保存正式数据集，不修改 labels。
- 本地权重和 `configs/runtime.local.yaml` 保持 gitignored。
- Thor 是最终边缘部署目标；Orange Pi / RK3588 不是当前主线。

## 下一步

1. 确认 Qwen2.5-VL 的实际服务 endpoint、model ID、认证方式和运行设备。
2. 在 gitignored 本地配置中启用 Review provider，使用固定验收图片做真实 VLM smoke test。
3. 人工核对 confirm / reject / correct / VLM-only finding，以及 JSON 与 Preview 一致性。
4. 测量 detector、Review、Fusion 和完整请求耗时，验证 10 秒超时与降级策略。
5. 根据真实响应稳定性调整 prompt、parser 容错边界和比赛动作策略。
