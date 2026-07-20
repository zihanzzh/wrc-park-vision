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
- 验证完整链路能否在 10 秒内返回或降级。
- 最终连接机器人图片输入与结果输出接口。

当前状态：

- Thor 是独立 Developer Kit。
- 尚未完成首次初始化和机器人接入。
- 实际 JetPack、CUDA、TensorRT、ROS2 和 Docker 环境待确认。

部署原则：

- 3090 负责训练产生 PyTorch best.pt。
- TensorRT engine 应在 Thor 实际环境中导出、构建或至少实机验证。
- 不声称多个模型一定满足 10 秒，必须 benchmark。
- YOLO 预计不是主要延迟瓶颈，VLM 更可能影响总时延，但最终以 profiling 为准。
- benchmark 后再决定模型串行/并行、模型尺寸、精度模式和 VLM 触发策略。

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
  -> Thor 构建/验证多个 TensorRT engine
  -> Thor benchmark 与 10 秒预算验证
  -> 机器人联调和部署包交付
```
