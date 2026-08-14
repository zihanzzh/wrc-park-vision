# Hardware Notes

本文件记录当前设备分工和多模型部署方向。

## Mac

角色：

- 项目主开发机。
- 使用 Codex 编写共享 Runtime Pipeline、部署代码和配置。
- 管理 GitHub、Obsidian/wiki 和项目文档。
- 进行轻量调试和接口设计。

边界：

- 不作为正式训练主力。
- 不保存 3090 的 `datasets_final/` 副本。
- 不作为最终机器人部署平台。
- 已清理早期数据目录、测试 runs 和 `yolo11n.pt`。

## 3090 Linux 工作站

角色：

- 保存 `datasets_final/prohibited_items/` 和 `datasets_final/garbage/`。
- 基于 `yolo11m.pt` 分别训练 prohibited_items 和 garbage 两个 YOLO11m。
- 单卡顺序训练：先 prohibited_items，成功后再 garbage，不并发占用同一 RTX 3090。
- 保存各自 runs、训练指标和 `weights/best.pt`。
- 进行模型评估、难例分析和数据回流。
- 必要时运行 Qwen / VLM。

权重说明：

- `yolo11m.pt` 是当前训练起点，不是最终自定义权重。
- `yolo26n.pt` 是旧测试或备用预训练权重，当前不属于主线但暂不删除。
- 计划输出 `prohibited_items_yolo11m_best.pt` 和 `garbage_yolo11m_best.pt`。
- 权重和 runs 不提交 GitHub。

`unified_detection` 不得训练；其物理目录是否已删除尚待在 3090 确认。

## NVIDIA Jetson AGX Thor Developer Kit

角色：

- 最终边缘部署目标。
- 部署共享 Runtime Pipeline。
- 加载 prohibited_items、garbage 和后续 behavior module 的多个 TensorRT engine。
- 进行串行/并行策略、显存、功耗和延迟 benchmark。
- 部署并验证 Qwen2.5-VL-32B 的单次 Multi-Image Review、行为协议与冲突裁决。
- 最终连接机器人图片输入与结果输出接口。

当前状态：

- Thor 是独立 Developer Kit。
- 已使用 `/models/Qwen2.5-VL-32B-Instruct-AWQ` 与 served model alias `qwen-vl` 完成禁带品、垃圾、四类行为和完整 Runtime/Competition/Preview 链路真实验收；local 配置的 `WRC_VLM_MODEL_ID` 必须保持该 alias。
- Runtime 已 integration-ready；机器人 transport 与现场产品接线仍由集成方完成。
- 已验收环境的 JetPack、CUDA、TensorRT、Docker 和 vLLM 具体版本/启动参数需要补入 release manifest；ROS2 不是当前 Runtime 的硬依赖。

部署原则：

- 3090 负责训练产生 PyTorch best.pt。
- TensorRT engine 应在 Thor 实际环境中导出、构建或至少实机验证。
- accuracy-first 验证要求 VLM 完成、结果非 degraded，并人工核对物体与四类行为结果。
- timing 继续 profiling，但不再使用固定 10 秒阈值决定模型或触发策略。

## 最终交付包

国内最终收到的应是经过 Thor 实机验证的部署包，而不只是普通 `.pt`。至少包含：

- 模型或 TensorRT engine。
- 各模型 class names。
- model -> `task_group` mapping。
- Runtime code。
- configuration。
- run command。
- sample request / response。
- environment notes。
- 已验证的 Thor 环境和 benchmark 说明。

## Orange Pi / RK3588

- 不是当前主线最终部署目标。
- 仅作为历史备选或测试路线。
- 当前不围绕 RKNN / Orange Pi 设计主线多模型 Runtime。

## 当前硬件路线

```text
Mac 开发共享 Runtime 与文档
  -> 3090 顺序训练两个 YOLO11m
  -> Thor 运行 YOLO / YOLO-World + Qwen2.5-VL-32B accuracy-first 完整验收（已完成）
  -> 产品 transport 联调和 release bundle 交付
```
