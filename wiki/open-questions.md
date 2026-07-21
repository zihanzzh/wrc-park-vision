# Open Questions

本文件只记录当前仍未确认、不能自行假设的问题。已确认事项见 [[decisions]]。

## 数据状态

- `unified_detection` 在 3090 上是否已经物理删除？若仍存在，是否需要在赛后保留用于调查？
- `datasets_final/prohibited_items/` 中 `roller_skates` 和 `barbecue_grill` 的实际图片数与 bbox 数是否为 0？
- prohibited_items 各类别的最终 train / val / test 分布是什么？
- 两个正式数据集是否需要在训练前再做一次 manifest / image-label / preview 快速验收？
- `garbage/previews/` 是否需要补生成，还是当前人工检查记录已足够？

## 训练与评估

- 3090 上两个任务各自使用的 `batch`、`workers`、`imgsz` 和 `device` 参数是什么？
- 两个训练任务如何自动串联，并在第一项失败时安全停止第二项？
- 最多 200 epochs、`patience=50` 是否适合两个数据集，是否需要分别调整？
- 正式验收指标是什么：mAP、per-class recall、误报率和漏报率各自要求多少？
- `roller_skates` / `barbecue_grill` 若为 0 样本，训练配置是保留空类、临时移除还是先补数据？
- `yolo26n.pt` 后续保留、归档还是删除？

## 多模型 Runtime

- 两个已训练权重的最终文件名、训练 run、class names、版本和 SHA256 是什么？
- prohibited_items 实际权重是完整 8 类版本，还是文档记录的历史 6 类版本？真实权重必须通过 `expected_class_names` 启动校验后才能进入 smoke test。
- 真实权重在 Mac `models/` 中由谁交付和更新？
- prohibited_items 和 garbage detector 在 Thor 上继续 sequential 还是改为并行？
- 多模型是否需要预热并常驻显存？
- behavior module 是否每张图片都运行，还是由 Pipeline 内部条件触发？
- 不同模型在真实照片中产生 `cross_model_overlap` 的频率是多少？是否需要增加业务规则？
- 不同模型 confidence 是否需要校准后再比较？
- 当前 `schema_version: 1.0` 如何封装到最终 ROS2 / HTTP / gRPC 协议，是否需要兼容字段？
- Tracking 何时实现，以及正式 `track_id` 由视觉 Runtime 还是机器人系统负责？当前字段只预留，值为 `null`。
- `suggestedAction` / 语音文案由视觉还是机器人策略层生成？
- 真实 smoke test 使用哪些固定验收图片和期望结果？
- Preview 失败但推理成功时，机器人侧是否仍按推理成功处理？

## 10 秒预算与 VLM

- 10 秒从图片发送、Thor 接收还是 Runtime 开始处理时计时？
- 多个 YOLO engine、结果融合和 VLM 各分配多少预算？
- Qwen / VLM 具体运行在 3090、其他高算力设备还是 Thor？
- 比赛现场是否允许额外设备和联网？
- 已确认单图 Review 使用完整原图；未来 behavior 是否还需要连续多帧序列？
- Qwen2.5-VL 使用哪个准确模型版本、OpenAI-compatible endpoint、认证方式和部署设备？
- 真实服务是否稳定遵循当前严格 JSON contract，是否需要有限重试或结构化输出约束？
- VLM 超时后返回哪些已确认结果，机器人采取什么动作？
- 两个 detector 全部失败但 VLM 返回 finding 时，顶层状态应为 `failure` 还是 `partial_success`？
- 最终是否需要通过模型尺寸或触发策略降低链路时间？

## Thor 部署

- Thor Developer Kit 的实际 JetPack / SDK、CUDA、TensorRT、ROS2 和 Docker 版本是什么？
- Thor 首次初始化由谁负责，完成时间是什么？
- 两个 YOLO11m 使用 FP16、INT8 还是其他 TensorRT 精度？
- INT8 如需校准集，使用哪些数据且如何保留版本？
- 多 engine 的显存、功耗、加载时间和单图延迟是多少？
- 部署包的目录规范、启动方式和版本策略是什么？
- 国内机器人侧最终采用 ROS2、HTTP、gRPC 还是其他接口？

## 不文明行为

- 五类行为的比赛判定边界和验收样例是什么？
- 是否需要独立 YOLO、segmentation、pose、tracking、区域规则或 VLM 组合？
- 需要哪些 bbox、mask、pose、区域或视频片段标注？
- behavior module 的数据负责人、实现负责人和时间表是什么？
- 在比赛剩余时间内，behavior module 的最小可交付范围是什么？

## Repo 与交付

- `scripts/` 中哪些早期 dataset tools 仍需长期保留？
- 最终部署包是否单独建目录，以及哪些配置可以提交 GitHub？
- 训练 run、best.pt、TensorRT engine 和部署包如何建立版本对应关系？
