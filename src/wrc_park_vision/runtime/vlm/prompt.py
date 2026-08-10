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
                "verdict": "confirmed",
            }
            for item in summary.detections
            if item.observation_id in required_ids
        ],
        "new_findings": [],
        "behavior_reviews": [
            {
                "candidate_id": item.id,
                "class_name": item.class_name,
                "verdict": "uncertain",
                "confidence": 0.5,
                "reasoning": "Visible evidence is insufficient.",
            }
            for item in summary.behavior_candidates
        ],
    }


def _build_object_guide(
    class_catalog: dict[str, list[str]],
    visual_class_guide: dict[str, dict[str, dict[str, object]]],
    relevant_class_names: dict[str, set[str]],
) -> dict[str, dict[str, dict[str, object]]]:
    guide: dict[str, dict[str, dict[str, object]]] = {}
    for task_group, class_names in class_catalog.items():
        relevant = relevant_class_names.get(task_group, set())
        entries = {
            class_name: visual_class_guide.get(task_group, {}).get(class_name, {})
            for class_name in class_names
            if class_name in relevant
        }
        if entries:
            guide[task_group] = entries
    return guide


def _rounded_bbox(values: tuple[float, float, float, float]) -> list[float]:
    return [round(value, 5) for value in values]


def _compact_review_context(
    summary: DetectionSummary,
    required_review_observation_ids: tuple[str, ...],
    candidate_reasons: dict[str, set[str]],
) -> dict[str, object]:
    required_ids = set(required_review_observation_ids)
    detections: list[dict[str, object]] = []
    for item in summary.detections:
        if item.observation_id not in required_ids:
            continue
        detection: dict[str, object] = {
            "id": item.observation_id,
            "task": item.task_group,
            "class": item.class_name,
            "conf": round(item.confidence, 4),
            "box": _rounded_bbox(item.bbox_normalized_xyxy),
        }
        if item.conflict_observation_ids:
            detection["conflicts"] = item.conflict_observation_ids
        reasons = sorted(
            {
                *item.review_reasons,
                *candidate_reasons.get(item.observation_id, set()),
            }
        )
        if reasons:
            detection["why"] = reasons
        detections.append(detection)

    context: dict[str, object] = {"detections": detections}
    if summary.behavior_classes:
        context["behavior_classes"] = [
            {
                "class": item.class_name,
                "objects": item.required_object_classes,
                "rules": item.decision_rules,
            }
            for item in summary.behavior_classes
        ]
    if summary.behavior_candidates:
        context["behavior_candidates"] = [
            {
                "id": item.id,
                "class": item.class_name,
                "evidence": item.evidence_observation_ids,
            }
            for item in summary.behavior_candidates
        ]
    return context


def _relevant_visual_classes(
    summary: DetectionSummary,
    class_catalog: dict[str, list[str]],
    required_review_observation_ids: tuple[str, ...],
) -> dict[str, set[str]]:
    required_ids = set(required_review_observation_ids)
    relevant: dict[str, set[str]] = {}
    detection_by_id = {
        item.observation_id: item for item in summary.detections
    }
    required_task_groups: set[str] = set()
    for item in summary.detections:
        if item.observation_id not in required_ids:
            continue
        required_task_groups.add(item.task_group)
        relevant.setdefault(item.task_group, set()).add(item.class_name)
        for conflict_id in item.conflict_observation_ids:
            conflict = detection_by_id.get(conflict_id)
            if conflict is not None:
                relevant.setdefault(conflict.task_group, set()).add(
                    conflict.class_name
                )
    for task_group in required_task_groups:
        relevant.setdefault(task_group, set()).update(
            class_catalog.get(task_group, [])
        )
    for behavior_class in summary.behavior_classes:
        for class_name in behavior_class.required_object_classes:
            for task_group, class_names in class_catalog.items():
                if class_name in class_names:
                    relevant.setdefault(task_group, set()).add(class_name)
    return relevant


def _build_prompt(
    summary: DetectionSummary,
    class_catalog: dict[str, list[str]],
    *,
    visual_class_guide: dict[str, dict[str, dict[str, object]]],
    crop_catalog: list[dict[str, object]],
    candidate_reasons: dict[str, set[str]],
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
    output_template = _build_output_template(
        summary,
        required_review_observation_ids,
    )
    review_context = _compact_review_context(
        summary,
        required_review_observation_ids,
        candidate_reasons,
    )
    confirmed_behavior_example = {
        "candidate_id": "behavior-candidate-0001",
        "class_name": "trampling_grass",
        "verdict": "confirmed",
        "confidence": 0.92,
        "bbox_normalized_xyxy": [0.15, 0.08, 0.84, 0.95],
        "reasoning": (
            "The person's feet and body support points are visibly inside "
            "the grass area."
        ),
    }
    return f"""You are the final visual reasoning judge. Detector class labels are proposals, not ground truth. Independently verify the visible object in every required detection; reject or correct detector results when visible evidence conflicts.
一次完成检测审核、漏检扫描和主动行为判断。original_image 是坐标唯一基准；crop 只用于看细节。
合法 task：{json.dumps(TASK_GROUPS, **compact)}
合法 object：{json.dumps(class_catalog, **compact)}
合法 behavior：{json.dumps(BEHAVIOR_CLASS_NAMES, **compact)}
视觉指南：{json.dumps(object_guide, **compact)}
审核输入：{json.dumps(review_context, **compact)}
crop 映射：{json.dumps(crop_catalog, **compact)}

规则：
1. 审核输入中每个 detection id 在 yolo_reviews 恰好出现一次，不得遗漏或重复。verdict 只能是 confirmed/rejected/corrected/uncertain；仅 corrected 增加 task_group、class_name。
2. 对每个 required object 独立检查可见内容：detector confidence 不是语义正确的证据，不得因高置信度默认 confirmed。只有可见物体结构匹配 detector class 才 confirmed；误报必须 rejected；只有允许目录中另一类别的结构明确时才 corrected；uncertain 只用于确实看不清。对 confirmed/corrected/rejected/uncertain 都不得为已有 detector detection 输出 bbox；Runtime 继续使用原 YOLO bbox。
3. spray_can 只有在明确看见 aerosol/pressurized can 结构、圆柱罐体和喷嘴/按压喷头时才能 confirmed。普通塑料瓶不是 spray_can；塑料饮料瓶、香烟盒、纸盒、食品容器、普通包装或垃圾必须 rejected；属于允许目录中的其他类别时 corrected。
4. new_findings 只报允许类别中的明确漏检，字段仅 task_group、class_name、confidence、bbox_normalized_xyxy，可选 crop_id。bbox 必须是 original_image 的 [x1,y1,x2,y2]。
5. behavior candidate 不是结论。存在 candidate 时，每个 candidate_id 必须在 behavior_reviews 中恰好出现一次，不能跳过或返回空数组；verdict 只能是 confirmed/rejected/uncertain。confirmed 必须含 candidate_id（主动全图发现时省略）、class_name、verdict、confidence、原图 bbox_normalized_xyxy 和极短 reasoning；bbox 覆盖主要违规人物、车辆或违规关系区域。rejected/uncertain 可省略 bbox。
6. 无论有无 candidate，都主动检查四类行为：trampling_grass、smoking、blocking_fire_lane、standing_or_lying_on_bench。trampling_grass 仅当脚部、身体支撑点或实际行走位置在草坪上；仅在草坪旁、道路边或草坪背景前必须拒绝。smoking 需要香烟、烟雾、明确手到嘴动作或其他强视觉证据；仅有人、香烟盒或模糊手部动作必须拒绝。blocking_fire_lane 需要车辆实际停放/占用消防或紧急通道、明确禁停区域，或明显阻塞应急通行；普通道路车辆必须拒绝。standing_or_lying_on_bench 仅确认站在或躺在长椅上；正常坐姿必须拒绝。
7. 只输出一个 JSON object；顶层字段必须且只能是 yolo_reviews、new_findings、behavior_reviews，三个值都必须是数组。禁止 behaviors、findings、Markdown、前后解释、null、额外顶层字段和长篇 reasoning。
confirmed behavior 完整格式示例：{json.dumps(confirmed_behavior_example, **compact)}
最小输出模板：{json.dumps(output_template, **compact)}
"""


def build_multi_image_prompt(
    request: MultiImageReviewRequest,
    class_catalog: dict[str, list[str]],
    visual_class_guide: dict[str, dict[str, dict[str, object]]] | None = None,
    *,
    required_review_observation_ids: tuple[str, ...] | None = None,
) -> str:
    crop_catalog: list[dict[str, object]] = []
    for crop in request.crops:
        item: dict[str, object] = {
            "id": crop.crop_id,
            "box": _rounded_bbox(crop.bbox_normalized_xyxy),
        }
        if crop.observation_ids:
            item["observations"] = list(crop.observation_ids)
        if crop.behavior_candidate_ids:
            item["behaviors"] = list(crop.behavior_candidate_ids)
        crop_catalog.append(item)
    candidate_reasons: dict[str, set[str]] = {}
    for candidate in request.candidates:
        for observation_id in candidate.observation_ids:
            candidate_reasons.setdefault(observation_id, set()).update(
                candidate.reasons
            )
    return _build_prompt(
        request.summary,
        class_catalog,
        visual_class_guide=visual_class_guide or {},
        crop_catalog=crop_catalog,
        candidate_reasons=candidate_reasons,
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
        candidate_reasons={},
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
