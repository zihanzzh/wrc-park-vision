"""Stable observation ordering and conservative cross-model conflict marking."""

from __future__ import annotations

from .schemas import BBoxGeometry, Conflict, Observation


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
