"""Build deterministic semantic context from normalized YOLO observations."""

from __future__ import annotations

from collections import Counter

from .config import ReviewSettings
from .schemas import (
    BBoxGeometry,
    BehaviorCandidate,
    BehaviorClassSummary,
    DetectionSummary,
    DetectionSummaryItem,
    Observation,
)


def observation_review_reasons(observation: Observation, settings: ReviewSettings) -> list[str]:
    reasons: list[str] = []
    if observation.confidence < settings.low_confidence_threshold:
        reasons.append("low_confidence")
    if settings.review_cross_task_overlap and observation.conflicts:
        reasons.append("cross_model_overlap")
    return reasons


def build_detection_summary(
    observations: list[Observation],
    settings: ReviewSettings,
    behavior_classes: list[BehaviorClassSummary] | None = None,
    behavior_candidates: list[BehaviorCandidate] | None = None,
) -> DetectionSummary:
    items: list[DetectionSummaryItem] = []
    for observation in observations:
        if not isinstance(observation.geometry, BBoxGeometry):
            continue
        items.append(
            DetectionSummaryItem(
                observation_id=observation.id,
                task_group=observation.task_group,
                class_id=observation.class_id,
                class_name=observation.class_name,
                confidence=observation.confidence,
                bbox_xyxy=observation.geometry.bbox_xyxy,
                bbox_normalized_xyxy=observation.geometry.bbox_normalized_xyxy,
                conflict_observation_ids=[conflict.observation_id for conflict in observation.conflicts],
                review_reasons=observation_review_reasons(observation, settings),
            )
        )
    counts = Counter(item.task_group for item in items)
    return DetectionSummary(
        total_detections=len(items),
        counts_by_task_group=dict(sorted(counts.items())),
        detections=items,
        behavior_classes=behavior_classes or [],
        behavior_candidates=behavior_candidates or [],
    )
