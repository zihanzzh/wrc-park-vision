"""Generate a small, prioritized set of full-image-referenced review crops."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from ..config import ImportantCropSettings
from ..schemas import Float4, ValidatedImage
from .candidate_selector import ReviewCandidate


Int4 = tuple[int, int, int, int]


@dataclass(frozen=True)
class ReviewCrop:
    crop_id: str
    bbox_xyxy: Int4
    bbox_normalized_xyxy: Float4
    width: int
    height: int
    image: Image.Image
    reasons: tuple[str, ...]
    observation_ids: tuple[str, ...]
    behavior_candidate_ids: tuple[str, ...]
    priority: int


@dataclass
class _CropRegion:
    bbox: Float4
    reasons: set[str]
    observation_ids: set[str]
    behavior_candidate_ids: set[str]
    priority: int


def _fit_interval(center: float, size: float) -> tuple[float, float]:
    size = min(max(size, 0.0), 1.0)
    start = center - size / 2.0
    end = center + size / 2.0
    if start < 0.0:
        end -= start
        start = 0.0
    if end > 1.0:
        start -= end - 1.0
        end = 1.0
    return max(0.0, start), min(1.0, end)


def _expanded_bbox(candidate: ReviewCandidate, settings: ImportantCropSettings) -> Float4:
    x1, y1, x2, y2 = candidate.bbox_normalized_xyxy
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = max((x2 - x1) * settings.context_scale, settings.min_crop_size_ratio)
    height = max((y2 - y1) * settings.context_scale, settings.min_crop_size_ratio)
    left, right = _fit_interval(center_x, width)
    top, bottom = _fit_interval(center_y, height)
    return left, top, right, bottom


def _bbox_iou(first: Float4, second: Float4) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    if intersection <= 0.0:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _merge_regions(first: _CropRegion, second: _CropRegion) -> _CropRegion:
    return _CropRegion(
        bbox=(
            min(first.bbox[0], second.bbox[0]),
            min(first.bbox[1], second.bbox[1]),
            max(first.bbox[2], second.bbox[2]),
            max(first.bbox[3], second.bbox[3]),
        ),
        reasons=first.reasons | second.reasons,
        observation_ids=first.observation_ids | second.observation_ids,
        behavior_candidate_ids=(
            first.behavior_candidate_ids | second.behavior_candidate_ids
        ),
        priority=max(first.priority, second.priority),
    )


def _merge_overlapping_regions(
    regions: list[_CropRegion],
    threshold: float,
) -> list[_CropRegion]:
    merged = list(regions)
    changed = True
    while changed:
        changed = False
        for first_index, first in enumerate(merged):
            match_index = next(
                (
                    second_index
                    for second_index in range(first_index + 1, len(merged))
                    if _bbox_iou(first.bbox, merged[second_index].bbox) >= threshold
                ),
                None,
            )
            if match_index is None:
                continue
            merged[first_index] = _merge_regions(first, merged[match_index])
            del merged[match_index]
            changed = True
            break
    return merged


def generate_important_crops(
    image: ValidatedImage,
    candidates: list[ReviewCandidate],
    settings: ImportantCropSettings,
) -> list[ReviewCrop]:
    """Expand candidate regions, merge overlap, prioritize, and cap crop count."""
    if not settings.enabled or not candidates:
        return []
    regions = [
        _CropRegion(
            bbox=_expanded_bbox(candidate, settings),
            reasons=set(candidate.reasons),
            observation_ids=set(candidate.observation_ids),
            behavior_candidate_ids=set(candidate.behavior_candidate_ids),
            priority=candidate.priority,
        )
        for candidate in candidates
    ]
    merged = _merge_overlapping_regions(regions, settings.merge_iou_threshold)
    merged.sort(
        key=lambda item: (
            -item.priority,
            -(item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]),
            item.bbox,
        )
    )

    crops: list[ReviewCrop] = []
    for region in merged[: settings.max_crops]:
        x1 = max(0, min(image.width - 1, int(region.bbox[0] * image.width)))
        y1 = max(0, min(image.height - 1, int(region.bbox[1] * image.height)))
        x2 = max(x1 + 1, min(image.width, int(round(region.bbox[2] * image.width))))
        y2 = max(y1 + 1, min(image.height, int(round(region.bbox[3] * image.height))))
        normalized = (
            x1 / image.width,
            y1 / image.height,
            x2 / image.width,
            y2 / image.height,
        )
        crops.append(
            ReviewCrop(
                crop_id=f"important-crop-{len(crops) + 1:02d}",
                bbox_xyxy=(x1, y1, x2, y2),
                bbox_normalized_xyxy=normalized,
                width=x2 - x1,
                height=y2 - y1,
                image=image.image.crop((x1, y1, x2, y2)),
                reasons=tuple(sorted(region.reasons)),
                observation_ids=tuple(sorted(region.observation_ids)),
                behavior_candidate_ids=tuple(
                    sorted(region.behavior_candidate_ids)
                ),
                priority=region.priority,
            )
        )
    return crops
