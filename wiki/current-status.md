# Current Status

## 当前阶段

禁带品与垃圾两个独立 detector 已完成训练，并已在 macOS 和 NVIDIA Thor 上跑通现有 Runtime 检测链路。YOLO-World backend、单图 Behavior Pipeline 和 Qwen2.5-VL Review 已接入共享 Runtime；Thor 上的 7B 完整链路已经跑通，当前重点转向细粒度类别准确率与三类任务系统测试。

本轮已完成代码实现与 mock 自动测试，没有运行真实 Qwen2.5-VL、训练模型、修改数据集、安装依赖或 push。

## Runtime Dual Pass 最终阶段

- Qwen provider 启用时默认固定执行两次互补请求：Pass 1 检查完整原图并处理 YOLO review、明显漏检与行为；Pass 2 在一次请求中接收全部重叠 crops，独立扫描小目标。
- crop 策略由 Pipeline 与 YAML 决定：近方图默认 2x2，宽图默认 3 个横向 crop，高图默认 3 个纵向 crop，重叠率默认 20%。
- Full Image finding 与 Crop finding 都必须提供 normalized bbox；Pipeline 映射为完整原图 geometry，VLM 不修改 corrected YOLO bbox。
- Fusion 跨 YOLO、Pass 1、Pass 2 做同类 IoU 去重并保留来源追踪；不同类别高 IoU 结果继续双方保留并标记冲突。
- Preview 直接绘制最终 observations，并标注 detector、VLM corrected、VLM full image、VLM crop、uncertain 或 review failed。
- 两个 Pass 的 timeout、`max_tokens`、crop 参数、finding IoU 阈值和降级策略均配置化；旧配置仍可加载。
- 双 Pass 真实 Thor 延迟和准确率尚未测试，10 秒目标仍需实测。

## Detection Module 分工 Phase 2

- 单帧 Runtime 保持现有多模块 Pipeline，不重写主流程：同一图片依次进入 YOLO-World object module 和独立 garbage YOLO11m module，输出统一合并为 observations / Detection Summary。
- YOLO-World 现在只负责正式 8 类禁带品和 `person`、`bench`、`grass`、`cigarette`、`vehicle` 五类行为辅助对象；配置与 backend 均明确拒绝 `task_group: garbage`。
- 六类垃圾固定由现有 Ultralytics backend 加载 `garbage_best.pt`。权重元数据已核对为 `crumpled_paper_ball`、`disposable_food_container`、`empty_cigarette_box`、`plastic_drink_bottle`、`plastic_food_wrapper`、`rigid_takeout_bag`，顺序为 ID 0 至 5。
- example 配置表达正式 8 类禁带品；gitignored 的 `runtime.yolo-world.local.yaml` 暂时只启用已验证的 6 类禁带品，并明确保留 `roller_skates`、`barbecue_grill` 的正式定义待后续补齐。
- Phase 2 当时未修改 Dual Pass、Crop Scan 和 Preview；这些剩余能力现已在最终阶段完成。

## Runtime Review/Fusion Phase 1

- VLM Response Parser 已改为逐项容错：`yolo_reviews`、`new_findings`、`behavior_reviews` 分别解析，非法条目跳过并记录结构化 `ReviewIssue`，合法兄弟条目继续保留。
- 只有响应完全无法解析为 JSON object 时 Review 才整体失败；缺失、重复或非法 observation/candidate/class 会成为 item-level issue。
- `rejected` 携带可确定映射的合法纠正类别时规范化为 `corrected`；不从 reasoning 或模糊文本猜测类别。
- 没有确认行为时 `behavior_reviews: []` 是合法结果；无 candidate 的条目只有明确 `confirmed` 才可接受。
- Final Fusion 现在对最终 observations 应用语义：`confirmed` 保留，`corrected` 复用原 YOLO bbox 和 confidence 并更新类别，`rejected` 移除，`uncertain` 默认保留并标记。
- FusionDecision 同时记录 YOLO confidence 与 VLM confidence；Review 整体失败或单条缺失审核时默认 `keep_flagged`，不会静默当作 confirmed。
- Phase 1 当时没有修改 Pipeline、Preview、Crop 或双 Pass；这些剩余能力现已在最终阶段完成。

## Qwen2.5-VL-7B 视觉类别指南

- Thor 上 `Qwen2.5-VL-7B-Instruct-AWQ` 实测：YOLO-World 约 0.87 秒、VLM Review 约 2.61 秒、总时间约 3.50 秒，`status=success`，JSON 可正常解析且每张图片仍只有一次 VLM 请求。
- 当前主要问题由响应格式和延迟转为细粒度类别准确率：测试中的 `kick_scooter` 曾被 7B 误判为 `skateboard`。
- YOLO-World 类别配置新增可选 `visual_description` 与 `distinguishing_rules`，Prompt 会为当前启用类别动态生成紧凑视觉指南。
- 8 类禁带品、6 类垃圾和 5 类行为辅助对象均已补充可见结构定义；重点明确滑板/儿童滑板车、卡式炉/烧烤炉、纸团/塑料瓶/塑料包装等相似类别的差异。
- `empty_cigarette_box` 和 `rigid_takeout_bag` 的 YOLO-World prompts 已收窄，避免把普通烟盒或宽泛外卖袋直接当作目标。
- 旧配置不提供视觉定义时仍可加载并构建 Prompt；非 YOLO-World detector 不受影响。
- 尚未完成禁带品、垃圾和不文明行为三类任务的系统真实图片测试，视觉指南对 Thor 7B 准确率的提升仍需复测确认。

## Qwen2.5-VL-7B Prompt 优化

- Thor 上的 `Qwen2.5-VL-7B-Instruct-AWQ` 已能被 Runtime 成功调用，但旧 Prompt 中的 `"允许的 task_group"` 被 7B 原样复制，导致严格 Parser 拒绝响应。
- Prompt 已改为紧凑规则、明确枚举和基于本次 observation/candidate 动态生成的合法 JSON 模板，不再把说明文字放入 JSON 字段值。
- `reasoning` 默认建议为 `null`，必要时只允许极短句，以减少输出 token。
- Parser 仍严格拒绝非法业务类别，只新增标识符首尾空格清理。
- Provider 在解析失败时会把最长 512 字符的 VLM 原始响应摘录附加到错误信息，便于 Thor 调试，不改变成功路径。
- 单次全图 VLM 请求架构保持不变；Thor 7B 已跑通合法 JSON，下一步复测视觉指南对准确率的影响。

## YOLO-World 隔离实验

- 2026-07-21 在分支 `experiment/yolo-world-smoke` 完成单图 YOLO-World v2 smoke test。
- 使用官方 `yolov8s-worldv2.pt` 和 `set_classes()` 对儿童滑板车测试图运行三组英文 prompts。
- 单独使用 `kick scooter` 未检出；加入同义词后以 `children's scooter` 检出，置信度约 0.30，bbox 覆盖目标合理。
- 完整 8 类续测在杂乱露营车图片中正确检出 `spray can`（约 0.54），但漏检仅局部可见的儿童滑板车和重度遮挡 speaker；同图单类别对照也未检出后两者。
- 该结果只说明开放词汇模型值得继续做更多隔离样本评估，不代表已决定接入正式 Runtime。
- 实验没有修改 Runtime、正式配置、类别映射、训练代码或数据集。

## Runtime 当前链路

```text
完整图片
  -> YOLO-World prohibited_items / behavior object detection
  -> Ultralytics YOLO11m garbage detection
  -> Detection Summary + behavior candidates
  -> 可选 Qwen2.5-VL Pass 1：全图 object/behavior review + 明显漏检
  -> Pipeline 生成重叠 crops
  -> 可选 Qwen2.5-VL Pass 2：一次请求扫描全部 crops
  -> finding bbox 映射与跨来源去重
  -> Final Fusion
  -> PipelineResponse
  -> result.json / preview.jpg
```

已实现：

- 配置驱动的通用 task modules，当前由 YOLO-World 提供 `prohibited_items` / behavior 基础对象，由独立 Ultralytics YOLO11m 提供 `garbage`。
- Ultralytics backend 启动时加载一次模型，并严格校验 `expected_class_names`。
- 单模块故障隔离、稳定 observation ID、跨 task group 冲突保留与标记。
- Detection Summary，向 VLM 提供 YOLO 语义上下文，但不限制 VLM 的全图观察范围。
- Qwen2.5-VL provider interface、OpenAI-compatible HTTP provider、Prompt Builder 和逐项容错 Response Parser。
- VLM 可以确认、拒绝或纠正 YOLO 类别，也可以报告 YOLO 完全漏检的目标。
- corrected 不由 VLM 重新定位；VLM 新 finding 必须提供 normalized bbox，最终 geometry 由 Pipeline 生成。
- Final Fusion 根据 VLM verdict 形成最终 observations，同时在 Detection Summary、Review 和 FusionDecision 中保留可审计的原始检测与双置信度信息。
- Review 或 Fusion 失败时保留 detector 结果，并返回阶段错误和 `partial_success`。
- JSON 与 Preview 使用同一个最终 `PipelineResponse`；有 geometry 的 VLM finding 作为正式 observation 绘框。
- Qwen provider 默认关闭，detector-only 配置继续保持可用。
- Behavior Pipeline 根据配置化关系生成候选：`person + grass`、`person + cigarette`、`vehicle`、`person + bench`。
- candidate 不会直接成为行为；只有 VLM `confirmed` 才生成 `kind: behavior` observation。
- 即使没有行为基础对象，同一次全图 VLM 请求也允许发现四类明显行为。
- 最终行为固定为 `trampling_grass`、`smoking`、`blocking_fire_lane`、`standing_or_lying_on_bench`。

当前明确未完成：

- 三类任务的真实 Qwen2.5-VL 系统效果验证。
- 请求级 10 秒 deadline；当前只有两个 Pass 各自的 HTTP timeout，没有跨阶段共享 deadline。
- 多帧 behavior、tracking、pose 和区域关系增强。
- TensorRT backend、正式 Thor engine 部署与 benchmark。
- API、ROS2、stream、tracking 和并行执行。

## 验证状态

- 自动测试：104 项 Runtime `unittest` 通过。
- Python `compileall`：通过。
- `git diff --check`：通过。
- 本轮双 Pass 请求测试使用 mock HTTP，确认 Pass 1 发送完整图片、Pass 2 在一次请求中发送全部 crops；本轮没有访问真实服务。
- detector 实际运行：macOS 与 NVIDIA Thor 已跑通。
- VLM 实际运行：Thor 7B 已完成单次全图 Review 并返回合法 JSON；当前需要复测配置驱动视觉指南的分类准确率。
- 10 秒完整链路：尚未实测。

## 数据与设备边界

- 正式数据继续只保存在 3090 的 `datasets_final/prohibited_items/` 和 `datasets_final/garbage/`。
- Mac 不保存正式数据集，不修改 labels。
- 本地权重和 `configs/runtime.local.yaml` 保持 gitignored。
- Thor 是最终边缘部署目标；Orange Pi / RK3588 不是当前主线。

## 下一步

1. 在 Thor 上用固定验收图片复测 `skateboard` / `kick_scooter` 等相似类别，记录视觉指南启用前后的准确率与耗时。
2. 对 8 类禁带品、6 类垃圾和四类不文明行为执行系统真实图片测试。
3. 人工核对 confirm / reject / correct、Full Image finding、Crop finding，以及 JSON 与 Preview 一致性。
4. 分别测量 detector、Full Image Review、crop 生成、Crop Scan Review、Fusion、Preview 和完整请求耗时，验证 10 秒目标。
5. 根据真实响应稳定性微调配置中的视觉定义，不放宽 Parser 的业务类别边界。
