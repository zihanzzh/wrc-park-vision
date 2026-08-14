# Data Plan

本文件记录正式数据入口、已完成训练的数据治理背景和历史计划。类别见 [[class-list]]，当前交付状态见 [[current-status]]。

## 当前数据策略

当前不再合并禁带品和垃圾 labels。两个经过人工检查的数据集曾作为独立训练入口：

- `datasets_final/prohibited_items/data.yaml`
- `datasets_final/garbage/data.yaml`

数据正确性和可交付性优先于构建单一训练集。当前交付 Runtime 使用 YOLO-World 负责 prohibited/behavior auxiliary proposals，已训练 YOLO11m 负责 garbage；类别仍不通过重写原始 class id 合并。

## Unified Detection 状态

`unified_detection` 曾用于统一 14 类训练。人工检查 train / val / test previews 后发现大量 bbox 显示错误，尤其是 `spray_can`。

当前结论：

- 不得用于训练。
- 不再是正式数据入口。
- 当前不投入主要比赛准备时间修复。
- 若目录仍存在，只能标记为 `deprecated` / `investigation`。
- 其在 3090 上是否已经物理删除尚待确认；本次文档同步不删除任何数据。

## Prohibited Items Dataset

正式入口：`datasets_final/prohibited_items/data.yaml`。

完整比赛类别映射见 [[class-list]]。Runtime 正式 8 类已经通过当前 YOLO-World 配置和 Thor 验收；历史训练数据的精确图片数、bbox 数和 split 分布仍以 3090 的 manifest/README 为准，不影响当前 Runtime 类别 contract。

训练产物属于独立 prohibited_items detector，不与垃圾 class id 混合。

## Garbage Dataset

正式入口：`datasets_final/garbage/data.yaml`。

数据经过 Qwen 预标注、Roboflow 人工检查和修正，共 499 张图片。Roboflow `valid` 已统一为 `val`。

最终 class id：

- `0: crumpled_paper_ball`
- `1: disposable_food_container`
- `2: empty_cigarette_box`
- `3: plastic_drink_bottle`
- `4: plastic_food_wrapper`
- `5: rigid_takeout_bag`

现有 label 第一列与最终 Roboflow `data.yaml` 对应，不重新映射。

## 历史训练计划

以下内容保留用于理解 3090 训练资产来源，不是当前待执行工作。正式 Runtime 已进入 integration-ready，本轮不重新训练。

共同基础权重：`yolo11m.pt`。

3090 单卡顺序执行：

1. prohibited_items YOLO11m。
2. 第一项成功完成后开始 garbage YOLO11m。

当前计划参数：

- 最多 200 epochs。
- `patience=50`。
- 启用 early stopping。
- `batch`、`workers`、`imgsz` 和 `device` 在训练前根据 3090 环境确认。

不要让两个训练任务同时并发占用同一张 RTX 3090。

建议独立输出目录：

- `runs/detect/wrc_prohibited_yolo11m/`
- `runs/detect/wrc_garbage_yolo11m/`

建议交付权重名：

- `prohibited_items_yolo11m_best.pt`
- `garbage_yolo11m_best.pt`

训练最佳权重通常位于各自运行目录的 `weights/best.pt`。重命名时应保留原始 run、训练配置和指标，保证可追溯。

## 权重文件说明

- `yolo11m.pt`：当前两个训练任务的预训练起点，不是最终自定义权重。
- `yolo26n.pt`：旧测试或备用预训练权重，不属于当前主线，暂不删除。
- `prohibited_items_yolo11m_best.pt`：历史独立禁带品训练产物候选；当前正式 proposal layer 使用 YOLO-World。
- `garbage_yolo11m_best.pt`：垃圾自定义权重的历史建议名；交付使用的真实文件名和 SHA256 应写入 release manifest。

训练权重、runs 和 TensorRT engine 不提交 GitHub。在确认训练产物和用途后，再单独整理旧权重。

## 历史训练前验收规则

两个训练任务分别检查：

- `data.yaml` 与 label class id 一致。
- train / val / test 图片和 label 一一对应。
- preview 抽查无系统性 bbox 错位。
- 类别样本数明确，0 样本类别有记录。
- bbox 无明显漏标、错标和越界。
- 输出 project / name 独立，避免 runs 混淆。
- 训练命令、环境和权重版本可追溯。

不得因为 `unified_detection` 已经存在就跳过其质量问题，也不得把其 preview 异常归因于可忽略的显示问题后直接训练。

## 训练后评估与回流

分别评估：

- 各模型 mAP、precision、recall 和 per-class 表现。
- 禁带品误报/漏报，尤其是相似普通物品。
- 垃圾六类之间的混淆。
- 真实园区、露营车/手推车、遮挡、远距离和复杂背景。
- Runtime 中跨模型重复或冲突检测。
- Thor 上每个 engine 和完整 Pipeline 的实际延迟。

低置信、误报、漏报、冲突和 VLM 超时样本回到 3090 进行补标和迭代。

## 不文明行为数据

尚未形成最终数据集。后续数据形式取决于方案，可能包括 bbox、mask、pose、区域、多帧片段和关系标注。它不与当前两个物品 detector 数据集强行合并。

## 数据与 Git 边界

正式数据、Roboflow 导出、训练 runs、`.pt`、`.onnx`、`.engine`、Qwen 模型和大型缓存不提交 GitHub。GitHub 只保留说明、配置、代码、Runtime、部署脚本和 wiki。

Mac 已清理早期 `data/`、`datasets_raw/`、`datasets_stage/`、`datasets_clean/`、测试 `runs/` 和 `yolo11n.pt`，不创建或复制 `datasets_final/`。

## 历史迭代

- Mac 曾构建 3 类 `prohibited_items_3cls` 用于流程验证，已清理，不是当前入口。
- 后续曾计划将禁带品和垃圾合并为 14 类 `unified_detection`。
- `unified_detection` 因 preview bbox 系统性异常被暂停和废弃。
- 当前回到两个独立、已人工检查数据集分别训练，并在 Runtime 层融合。
