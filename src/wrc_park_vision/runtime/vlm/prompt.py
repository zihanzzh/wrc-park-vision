"""Competition-focused prompt construction for full-image review."""

from __future__ import annotations

import json

from ..schemas import DetectionSummary


def build_review_prompt(summary: DetectionSummary, class_catalog: dict[str, list[str]]) -> str:
    summary_json = json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2)
    catalog_json = json.dumps(class_catalog, ensure_ascii=False, indent=2)
    return f"""你是 WRC 园区管理岗视觉系统的语义复核模块。

请独立检查随请求提供的完整原始图片。Detection Summary 只是 YOLO 检测上下文，不能限制你的观察范围。
你需要：
1. 对每一条 YOLO detection 判断 confirmed、rejected、corrected 或 uncertain。
2. 发现图片中被 YOLO 完全漏掉、但属于允许类别的目标，并写入 new_findings。
3. 只负责语义判断，不负责定位。不要输出 bbox、坐标、mask、polygon 或关键点。
4. corrected 只能修正到允许的 task_group 和 class_name；不要创造目录外类别。
5. new_findings 可以没有坐标，这是预期行为。
6. 必须对 Detection Summary 中每个 observation_id 恰好返回一条 yolo_reviews；不要遗漏或重复。
7. behavior_candidates 只是基础对象组合产生的候选，不代表行为已经发生。必须结合完整图片确认或否定。
8. 对每个 behavior candidate 恰好返回一条 behavior_reviews；即使没有 candidate，也要独立扫描 behavior_classes 中的四类行为。
9. 只有画面证据足够时才能确认行为。正常坐在长椅上不属于 standing_or_lying_on_bench。
10. 没有 candidate 的全图行为发现将 candidate_id 设为 null；不要为了填充结果而虚构行为。

允许的物体任务和类别：
{catalog_json}

Detection Summary（包含行为类别和候选）：
{summary_json}

只输出一个 JSON object，不要使用 Markdown。格式必须严格为：
{{
  "yolo_reviews": [
    {{
      "observation_id": "obs-0001",
      "verdict": "confirmed|rejected|corrected|uncertain",
      "corrected_task_group": null,
      "corrected_class_name": null,
      "confidence": 0.0,
      "reasoning": "简短理由"
    }}
  ],
  "new_findings": [
    {{
      "task_group": "允许的 task_group",
      "class_name": "允许的 class_name",
      "confidence": 0.0,
      "reasoning": "简短理由"
    }}
  ],
  "behavior_reviews": [
    {{
      "candidate_id": "behavior-candidate-0001 或 null",
      "class_name": "四类行为之一",
      "verdict": "confirmed|rejected|uncertain",
      "confidence": 0.0,
      "evidence_observation_ids": ["obs-0001"],
      "reasoning": "结合完整图片的简短判断依据"
    }}
  ]
}}
"""
