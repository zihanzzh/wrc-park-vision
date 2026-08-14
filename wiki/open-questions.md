# Open Questions

本文只保留不阻塞当前 integration-ready 交付、但后续产品化需要确认的问题。

## Product integration

- 机器人最终采用 ROS2、HTTP、gRPC 还是其他 transport？该层应包装 `CompetitionResponse`，不能复制 Runtime 逻辑。
- 产品侧如何定义 `partial_success` / `degraded` 的告警等级、重试和人工复核策略？
- `suggestedAction`、语音文案和跨帧 session 状态由哪个机器人策略模块负责？
- Preview 失败但结构化推理成功时，产品侧日志和告警策略是什么？

## Deployment provenance

- 最终两个 detector 文件名、版本、训练 run 和 SHA256 如何登记到部署 manifest？
- 已验收 Thor 的 JetPack、CUDA、TensorRT、Docker、vLLM 版本和完整启动参数是什么？
- 模型/配置/代码/tag 如何形成一个可追溯 release bundle？
- Thor detector 后续保持 sequential，还是在不影响准确率与稳定性的前提下启用 parallel？

## Future versions

- Tracking 和正式 `track_id` 由视觉 Runtime 还是机器人系统负责？
- 是否需要多帧 behavior、pose、区域规则或专用 TensorRT backend？
- Competition/Product schema 若收到新的官方字段要求，如何版本化 adapter？
- 两个 detector 全部失败但 VLM 仍返回 finding 时，未来是否调整当前 `failed` 语义？

上述问题不改变当前已验收的单帧 accuracy-first Pipeline。
