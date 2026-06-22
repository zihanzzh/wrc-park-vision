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
