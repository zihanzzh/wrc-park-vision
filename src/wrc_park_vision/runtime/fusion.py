"""Stable observation ordering and conservative cross-model conflict marking."""

from __future__ import annotations

from .schemas import BBoxGeometry, Conflict, FusionDecision, FusionSummary, Observation, ReviewSummary


def bbox_iou(first: BBoxGeometry, second: BBoxGeometry) -> float:
    ax1, ay1, ax2, ay2 = first.bbox_xyxy
    bx1, by1, bx2, by2 = second.bbox_xyxy
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    if intersection <= 0.0:
        return 0.0
    first_area = (ax2 - ax1) * (ay2 - ay1)
    second_area = (bx2 - bx1) * (by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _geometry_sort_key(observation: Observation) -> tuple[object, ...]:
    if observation.geometry is None:
        return ("",)
    if isinstance(observation.geometry, BBoxGeometry):
        return tuple(round(value, 6) for value in observation.geometry.bbox_xyxy)
    return (observation.geometry.model_dump_json(),)


def _observation_sort_key(observation: Observation) -> tuple[object, ...]:
    return (
        observation.task_group,
        observation.source.module_id,
        observation.kind,
        observation.class_name,
        observation.class_id if observation.class_id is not None else -1,
        -round(observation.confidence, 8),
        *_geometry_sort_key(observation),
    )


def prepare_observations(observations: list[Observation]) -> list[Observation]:
    """Deep-copy observations, sort deterministically, and assign stable IDs."""
    prepared = [observation.model_copy(deep=True) for observation in observations]
    prepared.sort(key=_observation_sort_key)
    for index, observation in enumerate(prepared, 1):
        observation.id = f"obs-{index:04d}"
        observation.conflicts = []
    return prepared


def merge_and_mark_conflicts(observations: list[Observation], iou_threshold: float) -> list[Observation]:
    merged = prepare_observations(observations)

    for index, first in enumerate(merged):
        if not isinstance(first.geometry, BBoxGeometry):
            continue
        for second in merged[index + 1 :]:
            if first.task_group == second.task_group:
                continue
            if not isinstance(second.geometry, BBoxGeometry):
                continue
            if bbox_iou(first.geometry, second.geometry) < iou_threshold:
                continue
            first.conflicts.append(Conflict(observation_id=second.id))
            second.conflicts.append(Conflict(observation_id=first.id))

    for observation in merged:
        observation.conflicts.sort(key=lambda conflict: conflict.observation_id)
    return merged


def fuse_review_results(
    observations: list[Observation],
    review: ReviewSummary,
    behavior_observations: list[Observation] | None = None,
) -> tuple[list[Observation], FusionSummary]:
    """Preserve detector results and append explicit VLM-derived semantic results."""
    preserved = [observation.model_copy(deep=True) for observation in observations]
    review_by_id = {decision.observation_id: decision for decision in review.decisions}
    decisions: list[FusionDecision] = []

    for observation in preserved:
        review_decision = review_by_id.get(observation.id)
        action = "keep_yolo"
        final_task_group = observation.task_group
        final_class_name = observation.class_name
        reasoning = None
        if review_decision is not None:
            reasoning = review_decision.reasoning
            if review_decision.verdict == "rejected":
                action = "reject_yolo"
            elif review_decision.verdict == "corrected":
                action = "correct_yolo"
                final_task_group = review_decision.corrected_task_group
                final_class_name = review_decision.corrected_class_name

        decisions.append(
            FusionDecision(
                id=f"fusion-{len(decisions) + 1:04d}",
                action=action,
                observation_id=observation.id,
                final_task_group=final_task_group,
                final_class_name=final_class_name,
                geometry_source="yolo",
                reasoning=reasoning,
            )
        )

    for finding in review.findings:
        decisions.append(
            FusionDecision(
                id=f"fusion-{len(decisions) + 1:04d}",
                action="add_vlm_finding",
                finding_id=finding.id,
                final_task_group=finding.task_group,
                final_class_name=finding.class_name,
                geometry_source="none",
                reasoning=finding.reasoning,
            )
        )

    for behavior in behavior_observations or []:
        preserved_behavior = behavior.model_copy(deep=True)
        preserved.append(preserved_behavior)
        decisions.append(
            FusionDecision(
                id=f"fusion-{len(decisions) + 1:04d}",
                action="add_behavior",
                observation_id=preserved_behavior.id,
                behavior_review_id=preserved_behavior.metadata.get("behavior_review_id"),
                final_task_group=preserved_behavior.task_group,
                final_class_name=preserved_behavior.class_name,
                geometry_source="none" if preserved_behavior.geometry is None else "yolo",
                evidence_observation_ids=list(preserved_behavior.evidence_observation_ids),
                reasoning=preserved_behavior.reasoning,
            )
        )

    return preserved, FusionSummary(status="completed", decisions=decisions)
