"""Build the compact, unified Runtime V3 multi-image review prompt."""

from __future__ import annotations

import json

from ..config import BEHAVIOR_CLASS_NAMES
from ..review.multi_image_request import MultiImageReviewRequest
from ..schemas import DetectionSummary


TASK_GROUPS = ("prohibited_items", "garbage", "uncivilized_behavior")


def _build_output_template(
    summary: DetectionSummary,
    required_review_observation_ids: tuple[str, ...],
) -> dict[str, object]:
    required_ids = set(required_review_observation_ids)
    return {
        "yolo_reviews": [
            {
                "id": item.observation_id,
                "verdict": "uncertain",
            }
            for item in summary.detections
            if item.observation_id in required_ids
        ],
        "new_findings": [],
        "behavior_reviews": [],
    }


def _build_object_guide(
    class_catalog: dict[str, list[str]],
    visual_class_guide: dict[str, dict[str, dict[str, object]]],
    relevant_class_names: dict[str, set[str]],
) -> dict[str, dict[str, dict[str, object]]]:
    return {
        task_group: {
            class_name: visual_class_guide.get(task_group, {}).get(class_name, {})
            for class_name in class_names
            if class_name in relevant_class_names.get(task_group, set())
        }
        for task_group, class_names in class_catalog.items()
        if relevant_class_names.get(task_group)
    }


def _compact_detection_summary(summary: DetectionSummary) -> dict[str, object]:
    return {
        "detections": [
            {
                "id": item.observation_id,
                "task_group": item.task_group,
                "class_name": item.class_name,
                "confidence": item.confidence,
                "bbox": item.bbox_normalized_xyxy,
                "conflicts": item.conflict_observation_ids,
                "review_reasons": item.review_reasons,
            }
            for item in summary.detections
        ],
        "behavior_classes": [
            {
                "class_name": item.class_name,
                "required_objects": item.required_object_classes,
            }
            for item in summary.behavior_classes
        ],
        "behavior_candidates": [
            {
                "id": item.id,
                "class_name": item.class_name,
                "evidence_ids": item.evidence_observation_ids,
            }
            for item in summary.behavior_candidates
        ],
    }


def _relevant_visual_classes(
    summary: DetectionSummary,
    class_catalog: dict[str, list[str]],
    required_review_observation_ids: tuple[str, ...],
) -> dict[str, set[str]]:
    required_ids = set(required_review_observation_ids)
    relevant: dict[str, set[str]] = {}
    required_groups: set[str] = set()
    for item in summary.detections:
        relevant.setdefault(item.task_group, set()).add(item.class_name)
        if item.observation_id in required_ids:
            required_groups.add(item.task_group)
    for task_group in required_groups:
        relevant.setdefault(task_group, set()).update(
            class_catalog.get(task_group, [])
        )
    return relevant


def _build_prompt(
    summary: DetectionSummary,
    class_catalog: dict[str, list[str]],
    *,
    visual_class_guide: dict[str, dict[str, dict[str, object]]],
    crop_catalog: list[dict[str, object]],
    candidate_catalog: list[dict[str, object]],
    required_review_observation_ids: tuple[str, ...],
) -> str:
    compact = {"ensure_ascii": False, "separators": (",", ":")}
    object_guide = _build_object_guide(
        class_catalog,
        visual_class_guide,
        _relevant_visual_classes(
            summary,
            class_catalog,
            required_review_observation_ids,
        ),
    )
    task_groups = ", ".join(TASK_GROUPS)
    behavior_classes = ", ".join(BEHAVIOR_CLASS_NAMES)
    return f"""任务
同时检查原始完整图片和少量重点裁剪图，一次完成：审核检测、发现漏检对象、确认不文明行为。

图片
- image_id=original_image 是坐标唯一基准。
- 其余图片是原图重点区域，只用于看清细节；crop 元数据给出它在原图的位置。

合法值
- task_group：{task_groups}
- behavior class_name：{behavior_classes}
- object class catalog：{json.dumps(class_catalog, **compact)}

视觉类别指南
{json.dumps(object_guide, **compact)}

规则
1. required_review_observation_ids 中每个 id 必须在 yolo_reviews 中恰好出现一次。未列入该数组的 detection 不需要输出 decision，系统默认保留。
2. yolo_reviews 使用短字段 id 和 verdict。仅 corrected 增加 task_group 和 class_name；其他 verdict 不输出这两个字段。
3. new_findings 只能使用 object class catalog 中的类别。每项必须包含 task_group、class_name、confidence、bbox_normalized_xyxy、review_pass="multi_image"、geometry_source="vlm_multi_image"。crop_id 可为帮助判断该目标的 crop ID，否则为 null。
4. 所有 new_findings 的 bbox_normalized_xyxy 都必须是相对 original_image 的 [x1,y1,x2,y2]，绝不能使用 crop 内坐标。
5. behavior candidate 只是上下文；只有确认发生的行为才写入 behavior_reviews。无候选时仍扫描四类行为，没有行为必须返回空数组。
6. uncertain 只用于图像确实无法判断，不能作为批量默认答案；证据足够时必须作出 confirmed、corrected 或 rejected。
7. cross-model conflict 可能是同一物体的类别冲突，必须按可见结构分别判断。明确为透明塑料饮料瓶时 confirmed plastic_drink_bottle、rejected spray_can；明确为香烟盒时 confirmed empty_cigarette_box、rejected spray_can。
8. 普通塑料饮料瓶、香烟盒和纸团都不是 spray_can。证据不足时仍返回 uncertain，不得猜测。
9. 正常坐在长椅上不是 standing_or_lying_on_bench。没有行为必须返回空数组。
10. reasoning 默认省略；只有必要时才用一个极短句。不要展示分析过程。

输出
只输出一个 JSON object，不要 Markdown、代码围栏或解释。保留三个顶层数组：
{json.dumps(_build_output_template(summary, required_review_observation_ids), **compact)}

Detection Summary
{json.dumps(_compact_detection_summary(summary), **compact)}

required_review_observation_ids
{json.dumps(required_review_observation_ids, **compact)}

重点审核候选
{json.dumps(candidate_catalog, **compact)}

Crop 元数据
{json.dumps(crop_catalog, **compact)}
"""


def build_multi_image_prompt(
    request: MultiImageReviewRequest,
    class_catalog: dict[str, list[str]],
    visual_class_guide: dict[str, dict[str, dict[str, object]]] | None = None,
    *,
    required_review_observation_ids: tuple[str, ...] | None = None,
) -> str:
    crop_catalog = [
        {
            "crop_id": crop.crop_id,
            "original_image_bbox_normalized_xyxy": crop.bbox_normalized_xyxy,
            "reasons": list(crop.reasons),
            "observation_ids": list(crop.observation_ids),
            "behavior_candidate_ids": list(crop.behavior_candidate_ids),
        }
        for crop in request.crops
    ]
    candidate_catalog = [
        {
            "candidate_id": candidate.candidate_id,
            "original_image_bbox_normalized_xyxy": candidate.bbox_normalized_xyxy,
            "reasons": list(candidate.reasons),
            "observation_ids": list(candidate.observation_ids),
            "behavior_candidate_ids": list(candidate.behavior_candidate_ids),
        }
        for candidate in request.candidates
    ]
    return _build_prompt(
        request.summary,
        class_catalog,
        visual_class_guide=visual_class_guide or {},
        crop_catalog=crop_catalog,
        candidate_catalog=candidate_catalog,
        required_review_observation_ids=(
            request.required_review_observation_ids
            if required_review_observation_ids is None
            else required_review_observation_ids
        ),
    )


def build_full_image_prompt(
    summary: DetectionSummary,
    class_catalog: dict[str, list[str]],
    visual_class_guide: dict[str, dict[str, dict[str, object]]] | None = None,
) -> str:
    """Compatibility helper for callers that do not yet provide crops."""
    return _build_prompt(
        summary,
        class_catalog,
        visual_class_guide=visual_class_guide or {},
        crop_catalog=[],
        candidate_catalog=[],
        required_review_observation_ids=tuple(
            item.observation_id for item in summary.detections
        ),
    )


def build_review_prompt(
    summary: DetectionSummary,
    class_catalog: dict[str, list[str]],
    visual_class_guide: dict[str, dict[str, dict[str, object]]] | None = None,
) -> str:
    return build_full_image_prompt(
        summary,
        class_catalog,
        visual_class_guide=visual_class_guide,
    )
