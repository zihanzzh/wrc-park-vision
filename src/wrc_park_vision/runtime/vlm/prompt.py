"""Competition-focused prompt construction for full-image review."""

from __future__ import annotations

import json

from ..config import BEHAVIOR_CLASS_NAMES
from ..crops import ImageCrop
from ..schemas import DetectionSummary, ReviewSummary


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


def build_full_image_prompt(
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
3. 漏检对象写入 new_findings。每项必须包含 task_group、class_name、confidence、bbox_normalized_xyxy、review_pass="full_image"、geometry_source="vlm_full_image"，reasoning 可为 null。
4. behavior candidate 只是上下文。只有确认发生的行为才写入 behavior_reviews；没有行为时必须返回空数组。
5. 无候选时仍扫描四类行为。只有明确发现行为时才可增加 candidate_id=null、verdict=confirmed 的条目。
6. 正常坐在长椅上不是 standing_or_lying_on_bench。证据不足返回 uncertain，不要虚构目标或行为。
7. 新 finding 的 bbox_normalized_xyxy 是相对完整原图的 [x1,y1,x2,y2]，各值在 0 到 1；不要为 yolo_reviews 修改或返回 bbox。
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


def build_review_prompt(
    summary: DetectionSummary,
    class_catalog: dict[str, list[str]],
    visual_class_guide: dict[str, dict[str, dict[str, object]]] | None = None,
) -> str:
    """Backward-compatible name for the full-image prompt."""
    return build_full_image_prompt(
        summary,
        class_catalog,
        visual_class_guide=visual_class_guide,
    )


def build_crop_scan_prompt(
    summary: DetectionSummary,
    class_catalog: dict[str, list[str]],
    crops: list[ImageCrop],
    full_image_review: ReviewSummary,
    visual_class_guide: dict[str, dict[str, dict[str, object]]] | None = None,
) -> str:
    compact = {"ensure_ascii": False, "separators": (",", ":")}
    object_guide = _build_object_guide(class_catalog, visual_class_guide or {})
    crop_catalog = [
        {
            "crop_id": crop.crop_id,
            "original_image_bbox_normalized_xyxy": crop.bbox_normalized_xyxy,
            "width": crop.width,
            "height": crop.height,
        }
        for crop in crops
    ]
    prior_results = {
        "yolo_detections": [
            {
                "observation_id": item.observation_id,
                "task_group": item.task_group,
                "class_name": item.class_name,
                "bbox_normalized_xyxy": item.bbox_normalized_xyxy,
            }
            for item in summary.detections
        ],
        "full_image_findings": [
            {
                "task_group": item.task_group,
                "class_name": item.class_name,
                "bbox_normalized_xyxy": item.bbox_normalized_xyxy,
            }
            for item in full_image_review.findings
        ],
    }
    template = {
        "yolo_reviews": [],
        "new_findings": [
            {
                "task_group": "prohibited_items",
                "class_name": "speaker",
                "confidence": 0.8,
                "reasoning": None,
                "bbox_normalized_xyxy": [0.1, 0.1, 0.8, 0.8],
                "crop_id": crops[0].crop_id if crops else "crop-r1-c1",
                "review_pass": "crop_scan",
                "geometry_source": "vlm_crop",
            }
        ],
        "behavior_reviews": [],
    }
    return f"""任务
这是独立的重叠分块漏检扫描。逐个检查本请求中的全部 crop，专门寻找 YOLO 和完整原图审核遗漏的小目标或易混淆对象。

视觉类别指南
{json.dumps(object_guide, **compact)}

规则
1. 不要假设 YOLO 或 Pass 1 已经完整；同时不要重复 prior_results 中已存在的目标。
2. 只报告视觉类别指南中的 object class。不得增加行为判断。
3. 每个 finding 必须包含对应 crop_id，以及相对该 crop 的 bbox_normalized_xyxy=[x1,y1,x2,y2]。
4. bbox 各值必须在 0 到 1，且 x1<x2、y1<y2。不确定时不要强行输出。
5. review_pass 必须是 crop_scan；geometry_source 必须是 vlm_crop。
6. yolo_reviews 和 behavior_reviews 必须返回空数组。reasoning 优先为 null。

输出
只输出一个 JSON object，不要 Markdown、代码围栏或解释。三个顶层数组与 Pass 1 完全相同：
{json.dumps(template, **compact)}
没有新目标时，new_findings 返回 []。

Crop 目录（图片按此顺序随请求发送）
{json.dumps(crop_catalog, **compact)}

已有结果，仅用于避免重复
{json.dumps(prior_results, **compact)}
"""
