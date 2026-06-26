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

## 2026-06-25 spray_can 多来源 canonical clean 数据集合并

本次数据集整理完成：

- 创建 `scripts/dataset_tools/merge_spray_can_sources.py`，用于将多个 `spray_can` / `aerosol` / `spray` 相关来源过滤并合并为单类 YOLO 数据集。
- 更新 `scripts/dataset_tools/preview_yolo_boxes.py`，默认面向 canonical `datasets_clean/spray_can/` 生成 bbox preview。
- 生成 canonical clean 数据集：`datasets_clean/spray_can/`。
- 生成 canonical `data.yaml`，类别为 `0: spray_can`。
- 使用 seed=`42` 按约 80 / 20 重新划分 train / val。
- 生成 preview 图片到 `datasets_clean/spray_can/previews/`。
- 更新 `.gitignore`，补充忽略顶层 `data/`。
- 更新 `wiki/current-status.md`、`wiki/data-plan.md`、`wiki/decisions.md` 和 `wiki/open-questions.md`。

检测到的数据源：

- `datasets_clean/spray_can_yolo11_single_class/`
- `datasets_raw/roboflow_aerosol_trash_detection/`
- `datasets_raw/roboflow_taco_aerosol/`
- `datasets_raw/roboflow_gcplo_spray_products/`

按来源统计：

- Kim clean 来源：原始图片 79；匹配 `0: spray_can`；保留图片 79；保留 bbox 88；跳过图片 0。
- Roboflow aerosol trash detection：原始图片 2783；匹配 `0: Aerosol`；保留图片 0；保留 bbox 0；跳过图片 2783。原因：label 行不是 5 字段 YOLO bbox，多数为 polygon / segmentation 格式。
- Roboflow TACO aerosol：原始图片 1499；匹配 `0: Aerosol`；保留图片 0；保留 bbox 0；跳过图片 1499。原因：label 行不是 5 字段 YOLO bbox，多数为 polygon / segmentation 格式。
- Roboflow GCPLO spray products：原始图片 8494；匹配 10 个 `AER SPRAY` / `Hit Aerosol` 商品类别；保留图片 1553；保留 bbox 10939；跳过图片 6941。

GCPLO 匹配到的原始类别：

- `8: AER SPRAY COOL SURF BLUE 240ml M149P24`
- `9: AER SPRAY FRESHLUSH GRN 240ml M149P24 MT`
- `10: AER SPRAY MORNINGMISTY 240ml M149P24`
- `11: AER SPRAY MUSKAFTER SMOKE 240ml M149P24`
- `12: AER SPRAY PETALCRSH PNK 240ml M149P24 MT`
- `13: AER SPRAY VIOLET VALLEY P24 M149 270ml`
- `83: Hit Aerosol FIK 700ml`
- `84: Hit Aerosol LmFlowrFIK200ml-MT`
- `85: Hit Aerosol LmFlowrFIK425ml-MT`
- `86: Hit Aerosol LmFlowrFIK625ml-MT`

最终 canonical 数据集统计：

- train 图片数：1306。
- val 图片数：326。
- train bbox 数：8856。
- val bbox 数：2171。
- 总图片数：1632。
- 总 bbox 数：11027。
- 所有 label class id 均为 `0`。
- image / label 文件一一对应。
- 未发现非 5 字段 label 行。

preview 结果：

- train preview 图片数：20。
- val preview 图片数：20。
- preview 输出目录：`datasets_clean/spray_can/previews/`。

注意事项：

- TACO / trash 两个来源有 `Aerosol` 类，但本轮未进入 clean 数据集。后续如果要利用这些真实垃圾场景，需要先决定是否将 polygon / segmentation label 转换为 bbox。
- GCPLO 来源以商品图 / 产品图为主，可能存在与比赛真实场景差异大、重复实例多、bbox 密集等风险，需要人工 preview 抽查。
- 当前仍未开始 YOLO11m / YOLO11s 自定义训练。

本次没有执行：

- 训练 YOLO。
- 运行 YOLO 推理。
- 安装依赖。
- 创建虚拟环境。
- 接入 Orange Pi / RK3588。
- 接入 NVIDIA Thor。
- git commit。

## 2026-06-25 Trash / TACO Aerosol polygon-to-bbox 合并

本次数据集增强完成：

- 修改 `scripts/dataset_tools/merge_spray_can_sources.py`，支持将 Trash / TACO 数据源中的 `Aerosol` polygon / segmentation label 转换为 YOLO bbox。
- 修改 `scripts/dataset_tools/preview_yolo_boxes.py`，preview 抽样尽量覆盖不同 source prefix。
- 保留当前 canonical `datasets_clean/spray_can/` 中已有 Kim / GCPLO 数据，不重新处理 GCPLO raw 数据。
- 将 Trash / TACO 转换后的 `Aerosol` 样本追加到 canonical `datasets_clean/spray_can/`。
- 使用 seed=`42` 对合并后的全集重新进行 80 / 20 train / val 划分。
- 重新生成 preview 到 `datasets_clean/spray_can/previews/`。
- 更新 `wiki/current-status.md`、`wiki/data-plan.md`、`wiki/decisions.md` 和 `wiki/open-questions.md`。

数据源结构检查：

- `datasets_raw/roboflow_aerosol_trash_detection/`：识别到 `train/images`、`train/labels`、`valid/images`、`valid/labels`、`test/images`、`test/labels` 和 `data.yaml`。
- `datasets_raw/roboflow_taco_aerosol/`：识别到 `train/images`、`train/labels`、`valid/images`、`valid/labels`、`test/images`、`test/labels` 和 `data.yaml`。

匹配类别：

- Trash / aerosol trash detection：原始 `names` 中 `0: Aerosol`，本次只匹配 class id `0`。
- TACO aerosol：原始 `names` 中 `0: Aerosol`，本次只匹配 class id `0`。
- 本次不把普通 `spray`、`bottle`、`can`、`container` 等模糊类别映射为 `spray_can`。

已有 canonical 数据统计：

- Kim：保留图片 79，保留 bbox 88。
- GCPLO：保留图片 1553，保留 bbox 10939。

Trash / TACO 转换统计：

- Trash / aerosol trash detection：
  - 原始图片数：2783。
  - `Aerosol` class id：0。
  - 转换后保留图片数：10。
  - 转换后保留 bbox 数：10。
  - 跳过图片数：2773。
  - 跳过异常 label 行数：0。
- TACO aerosol：
  - 原始图片数：1499。
  - `Aerosol` class id：0。
  - 转换后保留图片数：10。
  - 转换后保留 bbox 数：10。
  - 跳过图片数：1489。
  - 跳过异常 label 行数：0。

最终 canonical 数据集统计：

- train 图片数：1322。
- val 图片数：330。
- train bbox 数：8743。
- val bbox 数：2304。
- 总图片数：1652。
- 总 bbox 数：11047。
- 所有 label class id 均为 `0`。
- image / label 文件一一对应。
- 未发现非 5 字段 label 行。

最终来源分布：

- train：GCPLO 1240，Kim 66，Trash 8，TACO 8。
- val：GCPLO 313，Kim 13，Trash 2，TACO 2。

preview 结果：

- train preview 图片数：30，来源分布为 GCPLO 8、Kim 8、TACO 7、Trash 7。
- val preview 图片数：30，来源分布为 GCPLO 13、Kim 13、TACO 2、Trash 2。
- preview 输出目录：`datasets_clean/spray_can/previews/`。

注意事项：

- Trash / TACO 虽然更接近真实 / 垃圾场景，但明确标注为 `Aerosol` 的样本很少。
- polygon 转 bbox 可能比原始 mask 更松，需要人工检查 preview 后再决定是否长期保留。
- 当前仍未开始 YOLO11m / YOLO11s 自定义训练。

本次没有执行：

- 训练 YOLO。
- 运行 YOLO 推理。
- 安装依赖。
- 创建虚拟环境。
- 接入 Orange Pi / RK3588。
- 接入 NVIDIA Thor。
- git commit。

## 2026-06-25 spray_can Trash 错误样本删除与 GCPLO 轻量检查

本次小范围数据质量修复完成：

- 从 canonical `datasets_clean/spray_can/` 中删除 2 个用户确认错误的 Trash 样本。
- 仅修改 clean 数据集，不修改 `datasets_raw/`。
- 重新生成普通 bbox preview 到 `datasets_clean/spray_can/previews/`。
- 轻量检查 GCPLO `data.yaml` 中与 `spray` / `aerosol` / `aer` 相关的类别。

删除的 clean 文件：

- `datasets_clean/spray_can/images/train/trash__000102_JPG.rf.LlvL1jE6DUDLC3ItgdVU.JPG`
- `datasets_clean/spray_can/labels/train/trash__000102_JPG.rf.LlvL1jE6DUDLC3ItgdVU.txt`
- `datasets_clean/spray_can/images/train/trash__IMG_5050_JPG.rf.Ja68t5YBDrEhef732ydE.JPG`
- `datasets_clean/spray_can/labels/train/trash__IMG_5050_JPG.rf.Ja68t5YBDrEhef732ydE.txt`

匹配情况：

- `trash_000102_jpg`：clean 文件名实际为 `trash__000102_JPG...`，image 和 label 均找到并删除。
- `trash_IMG_5050_jpg`：clean 文件名实际为 `trash__IMG_5050_JPG...`，image 和 label 均找到并删除。
- clean 中仍存在 TACO 来源的同 basename 样本，本次按用户要求只删除 Trash 来源，没有删除 TACO。

删除后检查结果：

- train 图片数：1320。
- train label 数：1320。
- train bbox 数：8741。
- val 图片数：330。
- val label 数：330。
- val bbox 数：2304。
- image / label 一一对应。
- 所有 label 行均为 5 字段。
- 所有 class id 均为 `0`。

preview 结果：

- train preview 图片数：20。
- val preview 图片数：20。
- preview 输出目录：`datasets_clean/spray_can/previews/`。

GCPLO 轻量检查：

- `datasets_raw/roboflow_gcplo_spray_products/data.yaml` 中包含 `spray` / `aerosol` / `aer` 关键词的类别共 32 个。
- 历史合并时实际纳入的是包含明确 `spray` 或 `aerosol` 的 10 个类别：`AER SPRAY ...` 6 类和 `Hit Aerosol ...` 4 类。
- 当前 `merge_spray_can_sources.py` 不再重建 GCPLO，而是保留 canonical 数据集中已有的 GCPLO clean 样本。
- 初步判断：GCPLO preview 中“图中还有 spray can 没有 bbox”更可能来自原始 GCPLO 只标注部分 SKU / 部分目标；也可能存在少量历史匹配规则未纳入的相关 SKU，例如名字中写作 `SPRCAN` 而不是 `SPRAY` 的类别。
- 不建议本次自动扩大 GCPLO class 匹配范围，因为许多 `AER` 类并不一定是喷雾罐，自动纳入会增加错标风险。
- 建议继续保留 GCPLO，但训练前需要继续人工抽查 preview，并把未标注目标作为 known issue 记录。

本次没有执行：

- 训练 YOLO。
- 运行 YOLO 推理。
- 安装依赖。
- 创建虚拟环境。
- 接入 Orange Pi / RK3588。
- 接入 NVIDIA Thor。
- git commit。

## 2026-06-25 spray_can 移除全部 Trash 来源样本

本次小范围数据质量修复完成：

- 因 Trash 来源多次人工检查发现明显错框，已从 canonical `datasets_clean/spray_can/` 暂时移除全部 `trash__` 样本。
- Kim / GCPLO / TACO 来源保留。
- 未修改 `datasets_raw/roboflow_aerosol_trash_detection/`。
- 未重新合并 Trash 数据。
- 重新生成普通 bbox preview 到 `datasets_clean/spray_can/previews/`。

删除前检查：

- Trash image 数：8。
- Trash label 数：8。
- Trash image / label 一一对应，未发现删除前不匹配。
- train 中 Trash image / label：6 对。
- val 中 Trash image / label：2 对。

删除结果：

- 删除 Trash image 数：8。
- 删除 Trash label 数：8。
- 删除后未发现任何 `trash__` image、label 或 preview。

删除后 canonical 数据集检查：

- train 图片数：1314。
- train label 数：1314。
- train bbox 数：8735。
- val 图片数：328。
- val label 数：328。
- val bbox 数：2302。
- 总图片数：1642。
- 总 bbox 数：11037。
- image / label 一一对应。
- 所有 label 行均为 5 字段 YOLO bbox 格式。
- 所有 class id 均为 `0`。
- bbox 坐标均在 0～1 范围内。
- `data.yaml` 保持 `0: spray_can`。

剩余来源分布：

- train：GCPLO 1240，Kim 66，TACO 8。
- val：GCPLO 313，Kim 13，TACO 2。

preview 结果：

- train preview 图片数：20，来源分布为 GCPLO 7、Kim 7、TACO 6。
- val preview 图片数：20，来源分布为 GCPLO 9、Kim 9、TACO 2。
- preview 输出目录：`datasets_clean/spray_can/previews/`。
- preview 中不再出现 `trash__` 来源。

注意事项：

- Trash 来源暂时不进入 `spray_can` canonical clean dataset，除非后续重新人工修正或重新标注。
- 当前仍未开始 YOLO11m / YOLO11s 自定义训练。

本次没有执行：

- 训练 YOLO。
- 运行 YOLO 推理。
- 安装依赖。
- 创建虚拟环境。
- 接入 Orange Pi / RK3588。
- 接入 NVIDIA Thor。
- git commit。

## 2026-06-25 spray_can 数据目录与工具脚本整理

本次小范围项目整理完成：

- 确认 canonical `datasets_clean/spray_can/` 存在，且 `datasets_clean/spray_can/data.yaml` 存在。
- 删除旧 clean 中间目录 `datasets_clean/spray_can_yolo11_single_class/`。
- 删除早期单源过滤脚本 `scripts/dataset_tools/filter_yolo_single_class.py`。
- 删除早期旧 train / val 划分脚本 `scripts/dataset_tools/split_yolo_train_val.py`。
- 保留当前仍有用的脚本：
  - `scripts/dataset_tools/merge_spray_can_sources.py`
  - `scripts/dataset_tools/preview_yolo_boxes.py`
- 轻量更新 `merge_spray_can_sources.py`，避免后续运行时重新纳入已移除的 Trash 来源。
- 更新 `wiki/current-status.md`、`wiki/data-plan.md` 和 `wiki/decisions.md`。

删除原因：

- `datasets_clean/spray_can_yolo11_single_class/` 是 Kim 单源清洗阶段的中间目录，Kim 数据已经并入 canonical `datasets_clean/spray_can/`。
- `filter_yolo_single_class.py` 只服务早期 Kim 单源过滤流程，输出路径指向已删除的中间目录。
- `split_yolo_train_val.py` 只服务早期中间目录的 train / val 重划分，当前 canonical 数据集已由合并流程维护。

整理后 canonical 数据集检查：

- `data.yaml` 保持：
  - `path: datasets_clean/spray_can`
  - `train: images/train`
  - `val: images/val`
  - `0: spray_can`
- train 图片数：1314。
- train label 数：1314。
- train bbox 数：8735。
- val 图片数：328。
- val label 数：328。
- val bbox 数：2302。
- image / label 一一对应。
- 所有 label 行均为 5 字段 YOLO bbox 格式。
- 所有 class id 均为 `0`。
- bbox 坐标均在 0～1 范围内。
- preview 保持存在，未重新生成。

当前 `spray_can` 保留来源：

- Kim。
- GCPLO。
- TACO。

当前状态：

- Trash 来源已移除。
- 当前仍未开始 YOLO11m / YOLO11s 自定义训练。
- 下一步可以进行 `spray_can` 第一次 YOLO11m baseline 训练，或继续整理下一个禁带品类别。

本次没有执行：

- 训练 YOLO。
- 运行 YOLO 推理。
- 安装依赖。
- 创建虚拟环境。
- 接入 Orange Pi / RK3588。
- 接入 NVIDIA Thor。
- git commit。

## 2026-06-25 dataset_tools 通用化重构

本次小范围项目整理完成：

- 创建 `scripts/dataset_tools/yolo_common.py`，集中放置 YOLO dataset 的通用读取、写入、匹配、polygon-to-bbox、校验和统计函数。
- 创建 `scripts/dataset_tools/extract_class_dataset.py`，用于从 raw YOLO dataset 中提取指定目标类别，输出单类 staging dataset，并将目标类别重映射为 `0`。
- 创建 `scripts/dataset_tools/merge_class_datasets.py`，用于合并多个单类 staging / clean 来源，生成 canonical clean dataset。
- 创建 `scripts/dataset_tools/split_train_val.py`，用于对 clean dataset 重新划分 `train` / `val`。
- 更新 `scripts/dataset_tools/preview_yolo_boxes.py`，使 preview 工具从 `data.yaml` 读取类别名，可用于任意单类或多类 YOLO dataset，不再硬编码 `spray_can`。
- 创建 `scripts/dataset_tools/README.md`，说明工具职责、推荐流程、staging / canonical clean 目录约定，以及 `skateboard` 等后续类别的使用示例。
- 删除 `scripts/dataset_tools/merge_spray_can_sources.py`，因为其功能已由通用 extract / merge / split / preview 工具链覆盖。

文档同步：

- 更新 `wiki/current-status.md`，记录 `scripts/dataset_tools/` 已重构为模块化通用 YOLO dataset tools。
- 更新 `wiki/data-plan.md`，记录新工具链和 `datasets_stage/<class_name>/<source>/` 到 `datasets_clean/<class_name>/` 的推荐流程。
- 更新 `wiki/decisions.md`，新增决策：后续类别复用通用 dataset tools，不再为每个类别长期维护专用合并脚本。

验证：

- 已执行 `python -m py_compile` 做语法检查，`yolo_common.py`、`extract_class_dataset.py`、`merge_class_datasets.py`、`split_train_val.py` 和 `preview_yolo_boxes.py` 均通过。
- 本次不运行 extract / merge / split / preview 脚本，不重建 `datasets_clean/spray_can/`。

本次没有执行：

- 训练 YOLO。
- 运行 YOLO 推理。
- 安装依赖。
- 创建虚拟环境。
- 修改 `datasets_raw/`。
- 重建或修改 canonical `datasets_clean/spray_can/`。
- 接入 Orange Pi / RK3588。
- 接入 NVIDIA Thor。
- git commit。

## 2026-06-25 skateboard 多来源 canonical clean 数据集清理

本次小范围数据集清理完成：

- 检测到 raw skateboard 来源：
  - `datasets_raw/skate_cva2/`
  - `datasets_raw/skate_practicas/`
  - `datasets_raw/skate_unity/`
- 使用通用 `scripts/dataset_tools/extract_class_dataset.py` 提取各来源中的 `skateboard` 类。
- 使用通用 `scripts/dataset_tools/merge_class_datasets.py` 合并 staging 数据。
- 使用通用 `scripts/dataset_tools/split_train_val.py` 按 seed=42 和 val ratio=0.2 重新划分 train / val。
- 使用通用 `scripts/dataset_tools/preview_yolo_boxes.py` 生成 bbox preview。
- 未新增 skateboard 专用脚本。

原始 class names 与匹配结果：

- `skate_cva2`：原始 names 为 `0: Skateboard`，匹配 class id `0`。
- `skate_practicas`：原始 names 为 `0: Skateboard`，匹配 class id `0`。
- `skate_unity`：原始 names 为 `0: skateboard`，匹配 class id `0`。

staging 提取统计：

- `skate_cva2`：
  - 原始图片数：50。
  - 保留图片数：48。
  - 保留 bbox 数：159。
  - 跳过图片数：2。
  - 异常 label 行数：0。
- `skate_practicas`：
  - 原始图片数：200。
  - 保留图片数：200。
  - 保留 bbox 数：200。
  - 跳过图片数：0。
  - 异常 label 行数：0。
- `skate_unity`：
  - 原始图片数：2001。
  - 保留图片数：1967。
  - 保留 bbox 数：1967。
  - 跳过图片数：34。
  - 异常 label 行数：0。

canonical clean 输出：

- 输出目录：`datasets_clean/skateboard/`。
- staging 目录：
  - `datasets_stage/skateboard/cva2/`
  - `datasets_stage/skateboard/practicas/`
  - `datasets_stage/skateboard/unity/`
- `data.yaml` 保持：
  - `path: datasets_clean/skateboard`
  - `train: images/train`
  - `val: images/val`
  - `0: skateboard`

最终 train / val 统计：

- train 图片数：1772。
- train label 数：1772。
- train bbox 数：1869。
- val 图片数：443。
- val label 数：443。
- val bbox 数：457。
- 总图片数：2215。
- 总 bbox 数：2326。
- image / label 一一对应。
- 所有 label 行均为 5 字段 YOLO bbox 格式。
- 所有 class id 均为 `0`。
- bbox 坐标均在 0～1 范围内。

来源分布：

- train：`cva2` 38，`practicas` 162，`unity` 1572。
- val：`cva2` 10，`practicas` 38，`unity` 395。

preview 结果：

- preview 输出目录：`datasets_clean/skateboard/previews/`。
- train preview 图片数：30，来源分布为 `cva2` 10、`practicas` 10、`unity` 10。
- val preview 图片数：30，来源分布为 `cva2` 10、`practicas` 10、`unity` 10。
- preview 只用于人工检查，不参与训练。

注意事项：

- 需要人工检查 preview，确认 bbox 是否真正框住滑板，是否存在把人、轮子、背景或其他运动器材误框为滑板。
- 需要补充园区 / 露营车 / 手推车 / 安检区视角的真实或模拟场景数据。
- `git status --short` 当前显示 `datasets_stage/` 为 untracked；本轮未修改 `.gitignore`，因为当前请求未允许修改该文件。后续应考虑将 `datasets_stage/` 加入忽略规则，避免 staging 数据被误提交。
- 当前仍未开始 YOLO11m / YOLO11s 自定义训练。

本次没有执行：

- 训练 YOLO。
- 运行 YOLO 推理。
- 安装依赖。
- 创建虚拟环境。
- 修改 `datasets_raw/`。
- 修改 `datasets_clean/spray_can/`。
- 接入 Orange Pi / RK3588。
- 接入 NVIDIA Thor。
- git commit。

## 2026-06-25 portable_gas_stove 多来源 canonical clean 数据集清理

本次小范围数据集清理完成：

- 检测到 raw portable gas stove 来源：
  - `datasets_raw/stove_butane/`
  - `datasets_raw/stove_mix/`
- 使用通用 `scripts/dataset_tools/extract_class_dataset.py` 提取炉具 / 炉头本体相关类别。
- 使用通用 `scripts/dataset_tools/merge_class_datasets.py` 合并 staging 数据。
- 使用通用 `scripts/dataset_tools/split_train_val.py` 按 seed=42 和 val ratio=0.2 重新划分 train / val。
- 使用通用 `scripts/dataset_tools/preview_yolo_boxes.py` 生成 bbox preview。
- 未新增 portable_gas_stove 专用脚本。

原始 class names 与匹配结果：

- `stove_butane` 原始 names：
  - `0: Ball Valve`
  - `1: Butane Can`
  - `2: Butane Stove`
  - `3: Cabinet Heater`
  - `4: Camping Burner`
  - `5: Camping Heater`
  - `6: Cast Iron Gas Burners`
  - `7: Fire_Butane Can`
  - `8: Fire_Butane Stove`
  - `9: Fire_Gas Stove`
  - `10: Flexible Hose`
  - `11: Fusecock`
  - `12: Gas Boilers`
  - `13: Gas Meter`
  - `14: Gas Stove`
  - `15: Gas Torch`
  - `16: Gas Tube`
  - `17: Gas Water Heaters`
  - `18: HP Cylinders`
  - `19: Industrial Gas Boilers`
  - `20: LPG Cylinders`
  - `21: Open Gas Water Heaters`
  - `22: Pressure Regulators`
  - `23: Storage Tanks`
  - `24: Vaporizers`
  - `25: Vent Pipe`
- `stove_butane` 匹配 class id：`2`、`4`、`6`、`8`、`9`、`14`。
- `stove_mix` 原始 names：
  - `0: Butane Can`
  - `1: Butane Stove`
  - `2: Cabinet Heater`
  - `3: Cast Iron Gas Burners`
  - `4: Flexible Hose`
  - `5: Gas Boilers`
  - `6: Gas Meter`
  - `7: Gas Water Heaters`
  - `8: HP Cylinders`
  - `9: Industrial Gas Boilers`
  - `10: LPG Cylinders`
  - `11: Open Gas Water Heaters`
  - `12: Storage Tanks`
  - `13: Vaporizers`
- `stove_mix` 匹配 class id：`1`、`3`。

明确跳过的非目标类别：

- `Butane Can`、`Fire_Butane Can`、`HP Cylinders`、`LPG Cylinders`。
- `Gas Torch`、`Cabinet Heater`、`Camping Heater`、`Gas Water Heaters`、`Open Gas Water Heaters`。
- `Ball Valve`、`Fusecock`、`Flexible Hose`、`Gas Tube`、`Gas Meter`、`Pressure Regulators`。
- `Gas Boilers`、`Industrial Gas Boilers`、`Storage Tanks`、`Vaporizers`、`Vent Pipe`。

staging 提取统计：

- `stove_butane`：
  - 原始图片数：1035。
  - 保留图片数：1020。
  - 保留 bbox 数：1418。
  - 跳过图片数：15。
  - 异常 label 行数：0。
- `stove_mix`：
  - 原始图片数：963。
  - 保留图片数：917。
  - 保留 bbox 数：1082。
  - 跳过图片数：46。
  - 异常 label 行数：0。

canonical clean 输出：

- 输出目录：`datasets_clean/portable_gas_stove/`。
- staging 目录：
  - `datasets_stage/portable_gas_stove/butane/`
  - `datasets_stage/portable_gas_stove/mix/`
- `data.yaml` 保持：
  - `path: datasets_clean/portable_gas_stove`
  - `train: images/train`
  - `val: images/val`
  - `0: portable_gas_stove`

最终 train / val 统计：

- train 图片数：1550。
- train label 数：1550。
- train bbox 数：2012。
- val 图片数：387。
- val label 数：387。
- val bbox 数：488。
- 总图片数：1937。
- 总 bbox 数：2500。
- image / label 一一对应。
- 所有 label 行均为 5 字段 YOLO bbox 格式。
- 所有 class id 均为 `0`。
- bbox 坐标均在 0～1 范围内。

来源分布：

- train：`butane` 826，`mix` 724。
- val：`butane` 194，`mix` 193。

preview 结果：

- preview 输出目录：`datasets_clean/portable_gas_stove/previews/`。
- train preview 图片数：30，来源分布为 `butane` 15、`mix` 15。
- val preview 图片数：30，来源分布为 `butane` 15、`mix` 15。
- preview 只用于人工检查，不参与训练。

注意事项：

- 需要人工检查 preview，确认 bbox 是否真正框住炉具本体，是否误把气罐、燃料罐、喷枪、阀门、加热器、管线或气瓶框成 `portable_gas_stove`。
- `Cast Iron Gas Burners`、`Fire_Butane Stove`、`Fire_Gas Stove` 是否长期保留，需要根据人工 preview 和项目类别定义确认。
- `git status --short` 当前显示 `datasets_stage/` 为 untracked；本轮未修改 `.gitignore`，因为当前请求未允许修改该文件。后续应考虑将 `datasets_stage/` 加入忽略规则，避免 staging 数据被误提交。
- 当前仍未开始 YOLO11m / YOLO11s 自定义训练。

本次没有执行：

- 训练 YOLO。
- 运行 YOLO 推理。
- 安装依赖。
- 创建虚拟环境。
- 修改 `datasets_raw/`。
- 修改 `datasets_clean/spray_can/`。
- 修改 `datasets_clean/skateboard/`。
- 接入 Orange Pi / RK3588。
- 接入 NVIDIA Thor。
- git commit。

## 2026-06-25 prohibited_items_3cls 多类别 baseline 数据集合并

本次小范围多类别 YOLO dataset 合并完成：

- 输入单类 clean dataset：
  - `datasets_clean/spray_can/`
  - `datasets_clean/skateboard/`
  - `datasets_clean/portable_gas_stove/`
- 输出多类别 clean dataset：
  - `datasets_clean/prohibited_items_3cls/`
- 创建通用脚本：
  - `scripts/dataset_tools/merge_multiclass_dataset.py`
- 更新 `scripts/dataset_tools/README.md`，增加多类别合并工具说明和示例。

输入 dataset 检查：

- `spray_can`：
  - `data.yaml` names：`0: spray_can`。
  - image / label 一一对应。
  - 所有 label 行均为 5 字段 YOLO bbox 格式。
  - 所有 class id 均为 `0`。
  - bbox 坐标均在 0～1 范围内。
- `skateboard`：
  - `data.yaml` names：`0: skateboard`。
  - image / label 一一对应。
  - 所有 label 行均为 5 字段 YOLO bbox 格式。
  - 所有 class id 均为 `0`。
  - bbox 坐标均在 0～1 范围内。
- `portable_gas_stove`：
  - `data.yaml` names：`0: portable_gas_stove`。
  - image / label 一一对应。
  - 所有 label 行均为 5 字段 YOLO bbox 格式。
  - 所有 class id 均为 `0`。
  - bbox 坐标均在 0～1 范围内。

多类别 class id 映射：

- `0: spray_can`
- `1: skateboard`
- `2: portable_gas_stove`

按类别统计：

- `spray_can`：
  - train 图片数：1314。
  - train bbox 数：8735。
  - val 图片数：328。
  - val bbox 数：2302。
- `skateboard`：
  - train 图片数：1772。
  - train bbox 数：1869。
  - val 图片数：443。
  - val bbox 数：457。
- `portable_gas_stove`：
  - train 图片数：1550。
  - train bbox 数：2012。
  - val 图片数：387。
  - val bbox 数：488。

整体统计：

- train 图片数：4636。
- train label 数：4636。
- train bbox 数：12616。
- val 图片数：1158。
- val label 数：1158。
- val bbox 数：3247。
- 总图片数：5794。
- 总 bbox 数：15863。
- class id 只包含 `0`、`1`、`2`。
- image / label 一一对应。
- 所有 label 行均为 5 字段 YOLO bbox 格式。
- bbox 坐标均在 0～1 范围内。

preview 结果：

- preview 输出目录：`datasets_clean/prohibited_items_3cls/previews/`。
- train preview 图片数：30，类别分布为 `spray_can` 10、`skateboard` 10、`portable_gas_stove` 10。
- val preview 图片数：30，类别分布为 `spray_can` 10、`skateboard` 10、`portable_gas_stove` 10。
- preview 只用于人工检查，不参与训练。

注意事项：

- 需要人工检查 preview，确认三类 label 显示正确、bbox 仍然贴合目标，并确认 `portable_gas_stove` 中没有混入气罐、加热器、喷枪、气瓶或固定工业炉头。
- 该数据集是第一次 3 类 YOLO11m baseline 的数据准备版本，仍缺少真实园区、露营车、手推车和安检区视角数据。
- `git status --short` 当前显示 `datasets_stage/` 为 untracked；本轮未修改 `.gitignore`，因为当前请求未允许修改该文件。后续应考虑将 `datasets_stage/` 加入忽略规则，避免 staging 数据被误提交。
- 当前仍未开始 YOLO11m / YOLO11s 自定义训练。

本次没有执行：

- 训练 YOLO。
- 运行 YOLO 推理。
- 安装依赖。
- 创建虚拟环境。
- 修改 `datasets_raw/`。
- 修改三个单类 clean dataset：`spray_can`、`skateboard`、`portable_gas_stove`。
- 接入 Orange Pi / RK3588。
- 接入 NVIDIA Thor。
- git commit。
