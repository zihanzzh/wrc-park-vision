# wrc-park-vision

WRC 园区管理岗视觉识别项目，同时作为 Obsidian vault 维护项目记忆、设计记录和协作日志。

## 项目目标

本项目面向中国机器人比赛中的“园区管理岗”场景，视觉识别部分初步支持：

- 禁带品检查。
- 垃圾识别。
- 不文明行为识别。

## 当前状态

当前仓库处于初始化阶段，只包含项目规则和 wiki 文档结构。

尚未实现：

- YOLO 训练代码。
- YOLO 推理代码。
- FastAPI 或其他 API 服务。
- Orange Pi / RK3588 部署代码。
- NVIDIA Thor 部署代码。
- 机器人接口代码。

## 文档入口

- `PROJECT_CONTEXT.md`：项目背景、目标、阶段和职责边界。
- `AGENTS.md`：Codex / agent 工作规则。
- `wiki/content-map.md`：Obsidian/wiki 内容地图。
- `wiki/current-status.md`：当前项目状态。
- `wiki/architecture.md`：初步视觉架构设计。
- `wiki/open-questions.md`：待确认问题。

## 工作约定

- 默认使用中文沟通和记录项目文档。
- 不在仓库中提交大型数据集、图片、视频、模型权重或实验输出。
- 架构选择记录到 `wiki/decisions.md`。
- 有意义的变更后更新 `wiki/current-status.md` 和 `wiki/codex-log.md`。
