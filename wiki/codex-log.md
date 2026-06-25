# Codex Log

本文件记录 Codex 对项目做过的 meaningful change。

## 2026-06-22 Setup Pass

本次 setup pass 完成：

- 创建项目级 `AGENTS.md`，记录 Codex 行为规则和当前阶段限制。
- 创建 `PROJECT_CONTEXT.md`，记录项目背景、目标、阶段、技术方向、硬件角色和职责边界。
- 创建 `README.md`，作为仓库入口说明。
- 创建 `.gitignore`，忽略 Python 缓存、虚拟环境、大型数据目录、模型权重、导出文件和实验输出。
- 创建 `wiki/content-map.md`，作为 Obsidian/wiki 导航入口。
- 创建 `wiki/current-status.md`，记录 repo 初始化阶段状态。
- 创建 `wiki/architecture.md`，记录小模型实时识别 + 大模型低置信度兜底的初步架构。
- 创建 `wiki/decisions.md`，记录当前已确认项目决策。
- 创建 `wiki/open-questions.md`，记录待确认问题。
- 创建 `wiki/class-list.md`，记录初始类别清单。
- 创建 `wiki/data-plan.md`，记录第一阶段数据计划。
- 创建 `wiki/hardware-notes.md`，记录硬件角色和待确认事项。

本次没有实现：

- YOLO 训练代码。
- YOLO 推理代码。
- API 服务。
- Orange Pi / RK3588 部署代码。
- NVIDIA Thor 部署代码。
- 机器人接口代码。

本次没有执行：

- 安装依赖。
- 创建虚拟环境。
- 启动服务。
- git commit。

## 2026-06-22 AGENTS.md 长期规则修正

本次修正完成：

- 修改 `AGENTS.md`，删除将 setup pass 临时限制永久化的表述。
- 保留 Codex 长期工作规则：开始前读取项目上下文、不要擅自实现未要求功能、不要假设接口 / 硬件 / 数据 / 类别已确认、不要自动 commit、meaningful change 后更新 wiki。
- 明确安装依赖、创建虚拟环境、训练模型、启动服务、部署验证等事项不是永久禁止；只有在当前用户请求明确要求或批准后才可以执行。
- 更新 `wiki/current-status.md`，记录 setup pass 已完成，下一步计划是 YOLO baseline 准备阶段，但尚未开始安装依赖或写代码。

本次没有执行：

- 编写 YOLO 代码。
- 安装依赖。
- 创建虚拟环境。
- 启动服务。
- git commit。

## 2026-06-22 比赛规则与架构文档拆分

本次更新完成：

- 创建 `wiki/competition-rules.md`，专门记录“园区管理岗”比赛规则背景、三类任务、计分风险和对视觉系统的影响。
- 更新 `wiki/architecture.md` 为 v0.2 架构草案，区分 Runtime Architecture 和 Development Roadmap。
- 在 `wiki/architecture.md` 中补充图片输入层、任务调度层、小模型检测层、结果稳定层、VLM / 大模型复核层、结果融合层、结果输出层、日志和数据回流层。
- 在 `wiki/architecture.md` 中分别记录禁带品检查、垃圾识别、不文明行为的初步视觉流程。
- 更新 `wiki/content-map.md`，加入 `competition-rules.md` 的用途说明。
- 更新 `wiki/current-status.md`，记录文档 setup 已完成、本地 YOLO11n 预训练模型单图预测已跑通、architecture 已升级为 v0.2 草案。

本次没有执行：

- 编写训练代码。
- 编写推理代码。
- 安装依赖。
- 创建虚拟环境。
- 接入 Orange Pi / RK3588。
- 接入 NVIDIA Thor。
- 编写 API。
- git commit。

## 2026-06-22 v0.3 大小模型协同架构同步

本次更新完成：

- 更新 `wiki/architecture.md` 为 v0.3 架构草案，同步项目负责人提供的“园区管理岗视觉识别整体架构图（大小模型协同方案）”。
- 在 `wiki/architecture.md` 中写入 Thor 边缘小模型、高算力大模型 / VLM 兜底、10 秒超时机制、三场景分流、多帧 tracking / 去重、置信度分级、结果融合、输出接口候选和日志回流。
- 更新 `wiki/hardware-notes.md`，明确 Mac、高算力机器 / 服务器 / 高性能笔记本、NVIDIA Thor、Orange Pi / RK3588 的角色边界。
- 更新 `wiki/decisions.md`，记录 YOLO11m 主力候选、YOLO11s 辅助候选、YOLO11n 环境验证 baseline、Thor 主线部署、Orange Pi 非主线、10 秒超时机制和先确认数据标注计划。
- 更新 `wiki/current-status.md`，记录 architecture 已升级为 v0.3，下一步不是马上训练 YOLO，而是确认数据采集与标注计划。
- 更新 `wiki/open-questions.md`，补充 Thor 环境、VLM 现场使用、接口协议、tracking、segmentation、数据采集和比赛时间约束等待确认事项。

本次没有执行：

- 编写训练代码。
- 编写推理代码。
- 编写 API。
- 安装依赖。
- 创建虚拟环境。
- 运行训练。
- 接入 Orange Pi / RK3588。
- 接入 NVIDIA Thor。
- git commit。

## 2026-06-22 第一版数据采集与标注计划

本次更新完成：

- 更新 `wiki/data-plan.md` 为第一版正式数据采集与标注计划，覆盖目标、第一批优先类别、数据规模、拍摄规范、负样本、标注规则、标注工具、Qwen3-VL-32B 角色、Git 规则和训练前验收标准。
- 更新 `wiki/class-list.md`，区分全量比赛类别、第一批优先训练类别、暂缓类别、辅助类别和负样本示例。
- 更新 `wiki/decisions.md`，记录第一批优先 3 个禁带品 + 3 个垃圾类别、不文明行为暂缓、Qwen3-VL-32B 只做辅助、第一阶段先用 bounding box。
- 更新 `wiki/open-questions.md`，补充道具准备、Roboflow 云平台、机器人视角相机、露营车 / 模拟场地、负样本拍摄、标注负责人、Qwen3-VL-32B 运行环境和本地标注工具待确认问题。
- 更新 `wiki/current-status.md`，记录当前进入数据采集与标注计划阶段，下一步是准备第一批道具和拍摄数据，而不是马上训练 YOLO。
- 更新 `wiki/content-map.md`，同步 `data-plan.md` 和 `class-list.md` 的用途说明。

本次没有执行：

- 编写训练代码。
- 编写推理代码。
- 编写 API。
- 安装依赖。
- 创建虚拟环境。
- 运行训练。
- 接入 Orange Pi / RK3588。
- 接入 NVIDIA Thor。
- 添加真实图片、视频、数据集、标注文件或模型权重。
- git commit。

## 2026-06-25 Roboflow spray_can 单类数据清洗

本次数据清理完成：

- 检查原始数据集 `datasets_raw/roboflow_spray_can_by_kim/`，确认存在 `data.yaml`、`train/images`、`train/labels`、`valid/images`、`valid/labels`。
- 读取原始 `data.yaml`，确认 `names` 为 `['1', 'LED', 'spray can', 'toilet cleaner']`。
- 匹配到原始 `spray can` class id 为 `2`。
- 创建脚本 `scripts/dataset_tools/filter_yolo_single_class.py`，使用 Python 标准库过滤 YOLO 标签。
- 生成 clean 单类数据集 `datasets_clean/spray_can_yolo11_single_class/`。
- 更新 `.gitignore`，忽略 `datasets_raw/` 和 `datasets_clean/`，避免真实数据进入 Git。
- 更新 `wiki/current-status.md`、`wiki/data-plan.md`、`wiki/decisions.md`。

清洗统计：

- 原始 train 图片数：282。
- 原始 valid 图片数：1。
- clean train 图片数：81。
- clean val 图片数：0。
- clean train bbox 数：90。
- clean val bbox 数：0。
- 被删除的非 `spray_can` / 无目标图片数量：202。
- 原始 `spray can` class id：2。
- clean class id：0。

注意：

- clean val 图片数为 0，尚不满足训练前验收标准。
- 后续训练前需要补充验证集，或从 clean train 中重新划分 train / val。

本次没有执行：

- 编写训练代码。
- 编写推理代码。
- 运行 YOLO。
- 安装依赖。
- 创建虚拟环境。
- 接入 Orange Pi / RK3588。
- 接入 NVIDIA Thor。
- git commit。

## 2026-06-25 spray_can train/val 重划分与 bbox preview

本次数据集整理完成：

- 创建 `scripts/dataset_tools/split_yolo_train_val.py`，用于检查 clean YOLO 数据集一致性并按固定 seed 重划分 train / val。
- 创建 `scripts/dataset_tools/preview_yolo_boxes.py`，用于从 YOLO label 绘制 bbox 预览图。
- 检查 `datasets_clean/spray_can_yolo11_single_class/` 中的 image / label 对应关系。
- 确认所有 label 行格式为 `class_id center_x center_y width height`。
- 确认所有 label class id 都是 `0`。
- 使用 seed=`42` 按约 80 / 20 从 train 重新划分 val。
- 生成 preview 图片到 `datasets_clean/spray_can_yolo11_single_class/previews/`。
- 更新 `wiki/current-status.md` 和 `wiki/data-plan.md`。

检查结果：

- 重划分前 train 图片数：79。
- 重划分前 train label 数：79。
- 重划分前 train bbox 数：88。
- 重划分前 val 图片数：0。
- 重划分前 val label 数：0。
- 重划分前 val bbox 数：0。
- 未发现 image / label 不匹配。
- 未发现非 `0` class id。

重划分后统计：

- train 图片数：63。
- train label 数：63。
- train bbox 数：71。
- val 图片数：16。
- val label 数：16。
- val bbox 数：17。

preview 结果：

- train preview 图片数：10。
- val preview 图片数：10。
- preview 输出目录：`datasets_clean/spray_can_yolo11_single_class/previews/`。

数据质量说明：

- 当前 `spray_can` 数据主要来自 Roboflow，白底 / 商品图较多。
- 该数据可用于 pipeline baseline，但不足以代表比赛真实场景。
- 后续需要补充公园、露营车、手推车、安检区视角下的 `spray_can` 图片和相似负样本。

本次没有执行：

- 训练 YOLO。
- 运行 YOLO 推理。
- 安装依赖。
- 创建虚拟环境。
- 接入 Orange Pi / RK3588。
- 接入 NVIDIA Thor。
- git commit。
