# Decisions

本文件记录已经确认的项目决策和架构选择。未确认内容应写入 `open-questions.md`。

## 已确认决策

### D001：项目默认使用中文文档

- 日期：2026-06-22
- 决策：项目沟通、项目文档和 Obsidian/wiki 默认使用中文。
- 说明：文件名、文件夹名、代码变量、配置字段、API 字段、模型名可以使用英文。

### D002：当前 setup pass 只建立规则和文档结构

- 日期：2026-06-22
- 决策：当前阶段只创建项目记忆、Codex 规则和 wiki 文档结构。
- 说明：不实现 YOLO、训练、推理、API、Orange Pi、Thor 或硬件部署代码。

### D003：采用 YOLO11n / YOLO11s 作为初步 baseline 方向

- 日期：2026-06-22
- 决策：视觉实时小模型 baseline 初步考虑 YOLO11n / YOLO11s。
- 说明：这是早期技术方向。后续已由 D006 更新：YOLO11n 保留为环境验证和最小 baseline，YOLO11m 成为当前主力小模型候选，YOLO11s 作为辅助 / 轻量对比候选。

### D004：采用“两级视觉识别”作为初步架构方向

- 日期：2026-06-22
- 决策：初步架构为小模型实时识别，加大模型 / VLM 对低置信度、遮挡和歧义样本做兜底复核。
- 说明：该方向已扩展为 D007 的大小模型协同 v0.3 架构草案。

### D005：大型数据、模型权重和实验输出不进入 Git

- 日期：2026-06-22
- 决策：大型数据集、图片、视频、标注导出包、模型权重和实验输出不提交到 Git。
- 说明：使用 `.gitignore` 忽略相关目录和文件格式。

### D006：小模型候选策略更新

- 日期：2026-06-22
- 决策：YOLO11n 用于 Mac 本地环境验证和最小 baseline；YOLO11m 作为当前主力小模型候选；YOLO11s 作为辅助 / 轻量对比候选。
- 说明：YOLO11m 不是最终模型，需要通过数据、精度、速度和 Thor 部署性能评估确认。

### D007：采用大小模型协同 v0.3 架构方向

- 日期：2026-06-22
- 决策：系统方向为边缘小模型快速识别 + 高算力大模型 / VLM 兜底分析。
- 说明：大模型 / VLM 只处理低置信度、遮挡、类别歧义、复杂行为判断和漏检补充，不处理全部视频帧。

### D008：大模型复核必须设置 10 秒超时机制

- 日期：2026-06-22
- 决策：大模型 / VLM 请求必须有 10 秒内返回或超时中断 / 回退机制。
- 说明：超时机制用于避免比赛流程被大模型阻塞。

### D009：Thor 作为机器人本体侧主线边缘计算平台

- 日期：2026-06-22
- 决策：NVIDIA Thor 作为机器人本体侧主要边缘计算平台，负责现场小模型实时推理和边缘视觉服务。
- 说明：具体 Thor 型号、系统、CUDA / TensorRT / ROS2 / Docker 环境仍待确认。

### D010：Orange Pi / RK3588 不作为当前主线最终部署平台

- 日期：2026-06-22
- 决策：Orange Pi / RK3588 当前记录为边缘测试板或备用测试设备，不作为主线最终部署平台。
- 说明：当前不让 RKNN / Orange Pi 路线干扰 Thor 主线。

### D011：当前下一步先确认数据采集与标注计划

- 日期：2026-06-22
- 决策：下一步不是马上训练 YOLO，而是先确认数据采集规范、标注工具和第一批数据计划。
- 说明：没有数据和标注，无法进行真正的自定义训练。

### D012：第一批训练数据优先覆盖 3 个禁带品 + 3 个垃圾类别

- 日期：2026-06-22
- 决策：第一批优先类别为 `spray_can`、`portable_gas_stove`、`skateboard`、`plastic_bottle`、`paper_ball`、`food_container`。
- 说明：这些类别覆盖禁带品检查和垃圾识别两个主要任务，道具相对容易准备，适合快速跑通自定义 YOLO 训练闭环。

### D013：不文明行为不进入第一批 YOLO object detection 训练重点

- 日期：2026-06-22
- 决策：踩踏草坪、吸烟、占用消防通道、站立 / 躺在长椅上暂不作为第一批 YOLO object detection 训练重点。
- 说明：不文明行为不是简单 object detection，需要结合关系、区域、姿态 / 动作线索、tracking、规则判断或 VLM 复核单独设计。

### D014：Qwen3-VL-32B 可辅助预标注和复核，但人工确认是最终标签来源

- 日期：2026-06-22
- 决策：Qwen3-VL-32B 可用于预标注、类别复核、漏标检查和难例分析，但不能把自动标注直接当作最终训练标签。
- 说明：训练 YOLO 的 bbox / label 必须经过人工检查和修正，避免错误 VLM 标注增加禁带品误报风险。

### D015：第一阶段先使用 bounding box，不强制 segmentation

- 日期：2026-06-22
- 决策：第一阶段使用 bounding box 标注和 object detection 训练，不强制做 segmentation。
- 说明：如果后续垃圾抓取需要更精确区域，再考虑 YOLO11-seg 或 FastSAM 的 mask 标注 / 分割流程。

### D016：Roboflow spray can by Kim 作为 spray_can 第一版基础数据源

- 日期：2026-06-25
- 决策：使用 Roboflow 下载的 “spray can Computer Vision Dataset by Kim” 作为 `spray_can` 第一版外部基础数据源。
- 说明：该数据源位于 `datasets_raw/roboflow_spray_can_by_kim/`，原始数据保持只读，不直接修改。

### D017：多类别 Roboflow 数据集必须先过滤为项目 class 再进入训练集

- 日期：2026-06-25
- 决策：多类别 Roboflow 数据集不能直接进入训练集，必须先过滤和重映射为项目类别。
- 说明：本次将原始 `spray can` class id `2` 过滤并重映射为项目类别 `spray_can` class id `0`；非目标类别 LED、toilet cleaner 等不进入 clean 数据集。

### D018：datasets_clean/spray_can/ 作为 spray_can canonical clean dataset

- 日期：2026-06-25
- 决策：`datasets_clean/spray_can/` 作为 `spray_can` 类别的 canonical clean 数据集目录。
- 说明：后续 `spray_can` 的公开来源、自采数据和人工修正数据，应优先合并到该目录对应的数据构建流程中。

### D019：同一物品类别的多个公开来源合并到类别级 clean 目录

- 日期：2026-06-25
- 决策：同一物品类别的多个公开来源不再长期拆成多个 clean 目录，而是合并到该物品自己的 canonical clean 目录。
- 说明：可以保留 raw 来源和清洗脚本，clean 输出采用简单稳定命名，方便训练配置和 Obsidian 记录。

### D020：aerosol / spray / spray can 相关来源类别统一映射为 spray_can

- 日期：2026-06-25
- 决策：来源数据中的 `aerosol`、`spray`、`spray can` 以及明确喷雾商品类，进入本项目时统一映射为 `0: spray_can`。
- 说明：不把明显非喷雾罐类别强行映射为 `spray_can`；对于 polygon / segmentation 来源，需先确认是否转换为 bbox 再进入 object detection 训练集。

### D021：segmentation / polygon 数据可转换为 bbox 后作为 detection 数据候选

- 日期：2026-06-25
- 决策：明确属于目标类别的 segmentation / polygon 标注，可以通过外接 bbox 转换为 YOLO detection 训练数据候选。
- 说明：转换数据必须经过 preview 人工检查，确认 bbox 是否过松、类别是否正确、是否适合当前比赛场景后，再决定是否长期保留在训练集。

### D022：spray_can 只保留 canonical clean dataset 和当前数据工具

- 日期：2026-06-25
- 决策：`spray_can` 当前只保留 `datasets_clean/spray_can/` 作为 canonical clean dataset，删除早期 `datasets_clean/spray_can_yolo11_single_class/` 中间目录。
- 说明：早期单源过滤、旧 train / val 划分脚本和 `merge_spray_can_sources.py` 已由模块化通用 YOLO dataset tools 替代。

### D023：dataset_tools 采用通用模块化工具链

- 日期：2026-06-25
- 决策：`scripts/dataset_tools/` 不再为每个类别长期维护专用合并脚本，改为保留通用工具：`yolo_common.py`、`extract_class_dataset.py`、`merge_class_datasets.py`、`split_train_val.py` 和 `preview_yolo_boxes.py`。
- 说明：后续 `skateboard`、`portable_gas_stove`、`plastic_bottle` 等类别应复用同一套 extract / merge / split / preview 流程，避免把单个类别的临时逻辑固化为长期入口。
- 约束：这些工具只负责数据清洗、格式转换、合并、划分、校验和 preview，不负责 YOLO 训练、推理、依赖安装或硬件部署。

### D024：多类别 YOLO 训练集必须显式统一 class id 映射

- 日期：2026-06-25
- 决策：多类别 YOLO 训练集不能直接拼接多个单类 dataset 的 label，必须显式重映射 class id 并生成新的 `data.yaml`。
- 当前 3 类禁带品 baseline class 顺序：
  - `0: spray_can`
  - `1: skateboard`
  - `2: portable_gas_stove`
- 说明：三个单类 clean dataset 内部 class id 都是 `0`，直接拼接会导致所有类别被误认为同一类；因此使用 `merge_multiclass_dataset.py` 生成 `datasets_clean/prohibited_items_3cls/`。
