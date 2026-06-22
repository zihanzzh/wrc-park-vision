# Current Status

## 当前阶段

setup pass 已完成。项目已经建立项目记忆、协作规则和 Obsidian/wiki 文档结构。

下一步计划是进入 YOLO baseline 准备阶段，但尚未开始安装依赖、创建虚拟环境或编写训练 / 推理 / API 代码。

## 已完成

- 创建项目级 `AGENTS.md`，记录 Codex 工作规则。
- 已将 `AGENTS.md` 从 setup-pass 临时限制调整为长期全局工作规则。
- 创建 `PROJECT_CONTEXT.md`，记录项目背景和当前边界。
- 创建 `README.md`，作为仓库入口说明。
- 创建基础 `.gitignore`，避免提交大型数据、模型权重和实验输出。
- 创建 `wiki/` 文档结构。

## 尚未实现

- 尚未实现 YOLO11n / YOLO11s baseline。
- 尚未实现训练流程。
- 尚未实现推理流程。
- 尚未实现 FastAPI 或其他 API 服务。
- 尚未确认机器人接口。
- 尚未确认 Orange Pi 5 Max / RK3588 环境。
- 尚未确认 NVIDIA Thor 型号和环境。
- 尚未确认训练数据来源、标注规范和类别冻结版本。

## 当前注意事项

- 当前所有技术方向都属于初步方案，不代表已经完成工程确认。
- 不要在未确认接口和硬件环境前实现部署或服务代码。
- 后续 meaningful change 后需要继续更新本文件和 `codex-log.md`。
