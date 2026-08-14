# Architecture

本文只描述当前 integration-ready 架构。被替代方案见 [[decisions]] 与 [[codex-log]]。

## 总体架构

```text
image path / PIL camera frame
  -> 1. Input validation
  -> 2. Detection modules
       - YOLOv8s-WorldV2: prohibited + behavior auxiliary proposals
       - YOLO11m: garbage
  -> 3. Behavior candidate generation
  -> 4. Accuracy-first review policy
  -> 5. Important crops
  -> 6. Multi-image Qwen2.5-VL-32B request
  -> 7. Bounded JSON parser
  -> 8. Optional one-shot truncation fallback
  -> 9. Deterministic Fusion
  -> PipelineResponse
  -> 10. Competition adapter
  -> 11. Preview / artifacts
```

设计分层：

- Detector = proposal / localization layer。
- Qwen2.5-VL-32B = semantic verification + behavior reasoning layer。
- Fusion = deterministic final decision layer。

正式策略是 accuracy-first。600 秒 Runtime 总预算和 300 秒 VLM timeout 只用于异常保护，不是 10 秒性能目标。

## 1. Input validation

产品入口是 `RuntimePipeline.process()`，支持：

- `str` / `Path` 图片路径。
- `PIL.Image.Image` 内存帧。
- 已验证的内部 `ValidatedImage`。

Runtime 解码为 RGB、校验尺寸并创建 request ID。`RequestContext` 可携带 `camera_id`、timestamp、`session_id`、`frame_index`。机器人不需要提供 task type。

## 2. Detection modules

所有模块由 YAML 注册并共享一个 Pipeline：

- YOLOv8s-WorldV2 / YOLO-World：8 类 prohibited items 和 5 类 behavior auxiliary objects。
- 独立 Ultralytics YOLO11m：6 类 garbage；启动时严格校验权重 metadata 的类别数量、名称和顺序。

backend 输出统一转换为 `Observation`。每个 detector 保留自己的 class ID 空间，以 `task_group` 区分。模块只在 Runtime 初始化时加载一次；单模块异常被隔离。

## 3. Behavior candidates

YOLO-World 的 `person`、`bench`、`grass`、`cigarette`、`vehicle` 只提供 object evidence。配置化规则生成四类 candidate：

- `trampling_grass`
- `smoking`
- `blocking_fire_lane`
- `standing_or_lying_on_bench`

Candidate 不是最终行为。只有 VLM confirmed behavior 才进入最终 observations。即使没有 candidate，同一张原图仍执行四类行为主动扫描。

## 4. Accuracy-first review policy

`review_all_task_groups: [garbage, prohibited_items]` 强制全部两组 detector proposals 进入 32B review，不因 detector confidence、尺寸或 conflict 跳过。

Behavior auxiliary objects 继续按低置信、小目标、跨模型 overlap 和 candidate evidence 选择。Detection Summary 保存原始 proposal、geometry、confidence、冲突、review 原因和 behavior candidates，供审计与 Prompt 使用。

## 5. Important crops

Crop Generator 只围绕重要候选扩展上下文：

- 合并重叠区域。
- 按优先级最多保留 5 个。
- 面积达到原图 90% 时跳过，避免重复发送近全图。
- Crop 只辅助视觉判断；所有 finding bbox 始终使用原图 normalized 坐标。

## 6. Multi-image VLM request

正式 VLM 是 `Qwen2.5-VL-32B-Instruct-AWQ`，由 Thor Docker/vLLM 暴露 OpenAI-compatible endpoint，served alias 为 `qwen-vl`。

正常每帧一个统一请求，包含原图、必要 crops、required detector reviews、behavior candidates、类别目录和视觉判据。VLM 不替换 detector localization：confirmed/corrected 复用 YOLO bbox；只有真正漏检的 `new_findings` 提供新 bbox。

## 7. Bounded JSON output protocol

输出顺序为：

```json
{
  "yolo_reviews": [],
  "behavior_reviews": [],
  "new_findings": []
}
```

- 每个 required observation 恰好一条 verdict。
- confirmed/rejected/uncertain object review 只允许 `id`、`verdict`。
- corrected 只额外允许 `task_group`、`class_name`。
- 每个 behavior candidate 恰好一条 verdict；只有 confirmed 可带 confidence、原图 bbox 和一句极短证据。
- 无 candidate 的主动扫描只输出 confirmed behavior。
- `new_findings` 最多 8 条，并拒绝与 detector 或同批 finding 重复的高 IoU 框。
- Parser 不依赖 JSON key 顺序；单条非法项记录 `ReviewIssue`，合法兄弟项继续处理。

## 8. Truncation fallback

只有 primary 返回 `finish_reason=length` 或 JSON 明显因截断未闭合时，才执行一次 compact fallback：

- 复用同一原图和已编码 crops。
- 覆盖全部 required object reviews、behavior candidates 和主动行为扫描。
- 完全关闭 `new_findings`。
- 使用 1800 tokens；禁止递归 retry。

Fallback 成功时 Review 保持 completed，object/behavior decisions 正常进入 Fusion，但记录 `primary_response_truncated`、`missed_object_scan_skipped`、passes 和 metrics；Competition response 标记 `degraded=true`。Fallback 再失败才进入 review failure policy。

截断或 parse failure 的完整 raw response 仅写入请求目录的 `vlm_raw_response.txt`，不进入 Runtime JSON 或 Competition Response。

## 9. Deterministic Fusion

Fusion 按明确规则处理 review：

- confirmed：保留 detector observation 和原 bbox。
- corrected：更新 task/class，保留 detector bbox 和来源审计。
- rejected：从最终 observations 移除。
- uncertain：按配置默认保留并标记 pending。
- review failed / missing：按 `review_failure_policy` 默认保留并标记。
- VLM finding：映射为原图 geometry 后加入。
- 同 task/class 高 IoU：去重并保留来源 trace。
- 不同类别高 IoU：保留双方并标记 conflict，不做猜测性覆盖。

## 10. Competition adapter

内部 `PipelineResponse` 保存完整诊断与审计数据。独立 `build_competition_response()` 只读取 Fusion 后 observations，生成 product-facing V1：

- `frame`
- `status`
- `objects`
- `behaviors`
- `processing_time_ms`
- `degraded`

Adapter 不重新推理、融合或修改 geometry。产品 transport 层如需 ROS2/HTTP/gRPC，只包装该结果，不能复制 Pipeline 逻辑。

## 11. Preview 与 artifacts

CLI 可生成：

- `result.json`：完整 Runtime contract。
- `competition_result.json`：产品 contract。
- `preview.jpg`：直接绘制最终 observations。
- `vlm_raw_response.txt`：仅异常诊断时存在。

Preview 不重新推理或计算 bbox。内存 PIL frame 若需要 Preview，应由上层提供持久化图片路径。

## 12. Failure isolation

- 输入无效：`failed`，不运行 detector。
- 单 detector 失败：其他模块继续；至少一个模块成功时通常为 `partial_success`。
- Review/Fusion 失败：保留安全可用的 detector proposals，并附明确 error/review status。
- Primary 截断且 fallback 成功：结构化结果可用，但 missed-object scan 未完成，`degraded=true`。
- 所有 detector 都失败：`failed`，产品不得当作有效识别结果。
- Preview/output 辅助产物失败：不删除已完成的结构化推理结果。

## Output ownership

Runtime 负责视觉识别、review、Fusion 与结构化结果。机器人/产品层负责 transport、业务动作、语音文案、告警策略和跨帧 session 管理。Tracking、多帧行为和区域规则属于未来版本，不是当前交付 blocker。

## History

以下路线不属于当前架构，仅在 [[decisions]] / [[codex-log]] 保留历史：统一 14 类 `unified_detection`、V1/V2 双 Pass、Qwen2.5-VL-7B、固定 10 秒验收目标、Orange Pi 主线方案。
