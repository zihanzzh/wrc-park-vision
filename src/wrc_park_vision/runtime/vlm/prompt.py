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

允许的任务和类别：
{catalog_json}

Detection Summary：
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
  ]
}}
"""
