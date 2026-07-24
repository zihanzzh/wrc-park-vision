"""Competition-focused prompt construction for full-image review."""

from __future__ import annotations

import json

from ..config import BEHAVIOR_CLASS_NAMES
from ..schemas import DetectionSummary


TASK_GROUPS = ("prohibited_items", "garbage", "uncivilized_behavior")


def _build_output_template(summary: DetectionSummary) -> dict[str, object]:
    """Build a parser-valid skeleton using IDs and classes from this request."""
    return {
        "yolo_reviews": [
            {
                "observation_id": item.observation_id,
                "verdict": "uncertain",
                "corrected_task_group": None,
                "corrected_class_name": None,
                "confidence": None,
                "reasoning": None,
            }
            for item in summary.detections
        ],
        "new_findings": [],
        "behavior_reviews": [],
    }


def _build_object_guide(
    class_catalog: dict[str, list[str]],
    visual_class_guide: dict[str, dict[str, dict[str, object]]],
) -> dict[str, dict[str, dict[str, object]]]:
    return {
        task_group: {
            class_name: visual_class_guide.get(task_group, {}).get(class_name, {})
            for class_name in class_names
        }
        for task_group, class_names in class_catalog.items()
    }


def build_review_prompt(
    summary: DetectionSummary,
    class_catalog: dict[str, list[str]],
    visual_class_guide: dict[str, dict[str, dict[str, object]]] | None = None,
) -> str:
    compact = {"ensure_ascii": False, "separators": (",", ":")}
    summary_json = json.dumps(summary.model_dump(mode="json"), **compact)
    object_guide = _build_object_guide(class_catalog, visual_class_guide or {})
    object_guide_json = json.dumps(object_guide, **compact)
    template_json = json.dumps(_build_output_template(summary), **compact)
    task_groups = ", ".join(TASK_GROUPS)
    behavior_classes = ", ".join(BEHAVIOR_CLASS_NAMES)
    return f"""任务
检查完整图片。Detection Summary 仅提供检测上下文。一次完成：审核 YOLO、发现漏检对象、审核行为候选、扫描明显不文明行为。

合法值
- task_group 只能是：{task_groups}
- behavior class_name 只能是：{behavior_classes}

视觉类别指南
以下 JSON 的键是本次启用的 object class_name；visual 是可见结构，distinguish 是关键排除规则：
{object_guide_json}

规则
1. 每个 detection 的 observation_id 在 yolo_reviews 中恰好出现一次。verdict 只能是 confirmed、rejected、corrected、uncertain。
2. 仅 corrected 可设置 corrected_task_group 和 corrected_class_name；其他 verdict 必须为 null。
3. 漏检对象写入 new_findings。每项只含 task_group、class_name、confidence、reasoning。
4. behavior candidate 只是上下文。只有确认发生的行为才写入 behavior_reviews；没有行为时必须返回空数组。
5. 无候选时仍扫描四类行为。只有明确发现行为时才可增加 candidate_id=null、verdict=confirmed 的条目。
6. 正常坐在长椅上不是 standing_or_lying_on_bench。证据不足返回 uncertain，不要虚构目标或行为。
7. 不输出 bbox、坐标、mask、polygon、关键点或其他字段。
8. reasoning 优先为 null；必要时只写一个极短句，不展示分析过程。
9. 必须依据可见结构分类，不能只看总体相似度、颜色或背景。关键结构不可见且多个类别都可能时返回 uncertain。
10. 不得用相似但错误的目录类别代替真实对象。new_findings 的 task_group 和 class_name 必须来自视觉类别指南。

输出
只输出一个 JSON object。不要 Markdown、代码围栏或前后解释。保留三个顶层数组。
以下模板使用本次真实 ID 和类别，属于合法最小响应；根据图片修改判定，可添加合法漏检对象或无候选行为：
{template_json}

当前输入
{summary_json}
"""
