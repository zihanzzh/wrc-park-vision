# Current Status

## 当前阶段

setup pass 已完成。项目已经建立项目记忆、协作规则和 Obsidian/wiki 文档结构。

本地 YOLO11n 预训练模型单图预测已跑通，用于验证本地 baseline 环境。

architecture 已升级为 v0.3 草案，同步项目负责人提供的“园区管理岗视觉识别整体架构图（大小模型协同方案）”。

当前下一步不是马上训练 YOLO，而是先确认数据采集规范、标注工具和第一批数据计划。没有数据和标注，无法进行真正的自定义训练。

当前已进入数据采集与标注计划阶段。下一步是准备第一批道具和拍摄数据，而不是马上训练 YOLO。

## 已完成

- 创建项目级 `AGENTS.md`，记录 Codex 工作规则。
- 已将 `AGENTS.md` 从 setup-pass 临时限制调整为长期全局工作规则。
- 创建 `PROJECT_CONTEXT.md`，记录项目背景和当前边界。
- 创建 `README.md`，作为仓库入口说明。
- 创建基础 `.gitignore`，避免提交大型数据、模型权重和实验输出。
- 创建 `wiki/` 文档结构。
- 创建 `wiki/competition-rules.md`，将比赛规则和任务背景从架构文档中拆出。
- 更新 `wiki/architecture.md` 为 v0.3 大小模型协同架构草案。
- 已跑通本地 YOLO11n 预训练模型单图预测。
- 同步新版 v0.3 大小模型协同架构：Thor 边缘小模型、高算力大模型兜底、10 秒超时机制、三场景分流。
- 更新硬件分工：Mac 为开发机，高算力机器用于训练 / VLM，Thor 为机器人本体侧主线边缘计算平台，Orange Pi / RK3588 为测试或备用路线。
- 更新模型路线：YOLO11m 为当前主力小模型候选，YOLO11s 为辅助 / 轻量对比候选，YOLO11n 为已跑通的环境验证 baseline。
- 制定第一版数据采集与标注计划，第一批优先 3 个禁带品 + 3 个垃圾类别。
- 明确 Qwen3-VL-32B 可作为预标注、复核和难例分析工具，但训练标签必须以人工确认为准。
- 已下载 Roboflow spray can by Kim 原始数据集到 `datasets_raw/roboflow_spray_can_by_kim/`。
- 已生成单类 `spray_can` clean YOLO 数据集到 `datasets_clean/spray_can_yolo11_single_class/`。
- 已将 `spray_can` clean 数据集按 seed=42 重划分为 train / val。
- 已生成 `spray_can` bbox 可视化预览图到 `datasets_clean/spray_can_yolo11_single_class/previews/`，用于人工检查标注质量。

## 尚未实现

- 尚未形成项目内正式 YOLO11m / YOLO11s baseline 脚本和流程。
- 尚未训练 YOLO11m / YOLO11s 自定义模型。
- 尚未实现训练流程。
- 尚未形成项目内正式推理脚本。
- 尚未实现 FastAPI 或其他 API 服务。
- 尚未确认机器人接口。
- 尚未确认 Thor 具体型号、系统、CUDA / TensorRT / ROS2 / Docker 环境。
- 尚未接入 NVIDIA Thor。
- 尚未接入 Orange Pi / RK3588。
- 尚未确认训练数据来源、标注规范和类别冻结版本。
- 尚未准备第一批 6 类道具和负样本拍摄计划。
- 尚未确认标注工具。
- 尚未采集真实比赛场景图片。
- 除 `spray_can` clean 数据集外，其他第一批类别尚未导出 YOLO 格式数据集。
- `spray_can` clean 数据集主要来自 Roboflow 白底 / 商品图，尚不足以代表比赛真实场景。
- 尚未确认高算力机器 / 服务器 / 高性能笔记本的具体环境。
- 尚未确认 VLM 是否允许现场使用、是否允许联网。

## 当前注意事项

- 当前所有技术方向都属于初步方案，不代表已经完成工程确认。
- 不要在未确认接口和硬件环境前实现部署或服务代码。
- 当前仍未接入 Orange Pi / RK3588、NVIDIA Thor 或机器人接口。
- 不要把 YOLO11m 写死为最终模型，它只是当前主力候选。
- 不要把 VLM 写成实时处理所有帧，它只用于兜底复核。
- 不要把 Orange Pi / RK3588 写成主线最终部署平台。
- 不要把 Qwen3-VL-32B 自动标注直接当作训练标签，必须人工复查。
- 未完成数据采集和标注验收前，不要启动 YOLO11m / YOLO11s 自定义训练。
- 当前仍未开始训练；不要因为已生成 clean 数据集就直接启动自定义训练。
- `spray_can` preview 图需要人工抽查 bbox 质量，尤其确认是否仍有非喷雾罐图片混入。
- 后续 meaningful change 后需要继续更新本文件和 `codex-log.md`。
