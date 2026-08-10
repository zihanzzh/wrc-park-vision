"""Select observations and behavior relations that merit focused VLM crops."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import CandidateSelectionSettings
from ..schemas import BBoxGeometry, BehaviorCandidate, Float4, Observation


REASON_PRIORITY = {
    "small_object": 1,
    "low_confidence": 2,
    "behavior_candidate": 3,
    "cross_model_conflict": 4,
    "task_group_required": 5,
}


@dataclass(frozen=True)
class ReviewCandidate:
    """A full-image region selected for additional visual detail."""

    candidate_id: str
    bbox_normalized_xyxy: Float4
    reasons: tuple[str, ...]
    observation_ids: tuple[str, ...]
    behavior_candidate_ids: tuple[str, ...] = ()
    priority: int = 0


def normalized_bbox_area(geometry: BBoxGeometry) -> float:
    x1, y1, x2, y2 = geometry.bbox_normalized_xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _union_normalized_bboxes(geometries: list[BBoxGeometry]) -> Float4:
    boxes = [item.bbox_normalized_xyxy for item in geometries]
    return (
        min(item[0] for item in boxes),
        min(item[1] for item in boxes),
        max(item[2] for item in boxes),
        max(item[3] for item in boxes),
    )


def select_review_candidates(
    observations: list[Observation],
    behavior_candidates: list[BehaviorCandidate],
    settings: CandidateSelectionSettings,
    low_confidence_threshold: float,
) -> list[ReviewCandidate]:
    """Return deterministic candidate regions without inventing fixed-grid crops."""
    selected: list[ReviewCandidate] = []
    observation_by_id = {item.id: item for item in observations}

    for observation in observations:
        if not isinstance(observation.geometry, BBoxGeometry):
            continue
        reasons: list[str] = []
        if observation.task_group in settings.review_all_task_groups:
            reasons.append("task_group_required")
        if (
            settings.include_low_confidence
            and observation.confidence < low_confidence_threshold
        ):
            reasons.append("low_confidence")
        if settings.include_cross_model_conflicts and observation.conflicts:
            reasons.append("cross_model_conflict")
        if (
            settings.include_small_objects
            and normalized_bbox_area(observation.geometry)
            <= settings.small_object_area_ratio
        ):
            reasons.append("small_object")
        if not reasons:
            continue
        selected.append(
            ReviewCandidate(
                candidate_id=f"review-candidate-{len(selected) + 1:04d}",
                bbox_normalized_xyxy=observation.geometry.bbox_normalized_xyxy,
                reasons=tuple(reasons),
                observation_ids=(observation.id,),
                priority=max(REASON_PRIORITY[item] for item in reasons),
            )
        )

    if settings.include_behavior_candidates:
        for behavior in behavior_candidates:
            evidence = [
                observation_by_id[item]
                for item in behavior.evidence_observation_ids
                if item in observation_by_id
                and isinstance(observation_by_id[item].geometry, BBoxGeometry)
            ]
            if not evidence:
                continue
            selected.append(
                ReviewCandidate(
                    candidate_id=f"review-candidate-{len(selected) + 1:04d}",
                    bbox_normalized_xyxy=_union_normalized_bboxes(
                        [item.geometry for item in evidence]  # type: ignore[list-item]
                    ),
                    reasons=("behavior_candidate",),
                    observation_ids=tuple(item.id for item in evidence),
                    behavior_candidate_ids=(behavior.id,),
                    priority=REASON_PRIORITY["behavior_candidate"],
                )
            )

    return selected
