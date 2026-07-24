"""Annotated preview rendering from finalized PipelineResponse objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import PreviewSettings
from .schemas import BBoxGeometry, PipelineResponse


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _observation_display_source(observation: object) -> str:
    metadata = getattr(observation, "metadata", {})
    review = getattr(observation, "review", None)
    if metadata.get("review_source") == "vlm_corrected":
        return "corrected"
    geometry_source = metadata.get("geometry_source")
    if geometry_source == "vlm_full_image":
        return "vlm-full"
    if geometry_source == "vlm_crop":
        return "vlm-crop"
    if review is not None:
        if "review_item_missing_or_failed" in review.reasons:
            return "review-failed"
        if "vlm_uncertain" in review.reasons:
            return "uncertain"
        if review.status == "confirmed":
            return "confirmed"
    return "det"


@dataclass(frozen=True)
class _LabelLayout:
    lines: tuple[str, ...]
    box_xyxy: tuple[int, int, int, int]
    text_origins: tuple[tuple[int, int], ...]


def _text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox(
        (0, 0),
        text,
        font=font,
        anchor="lt",
    )
    return max(1, right - left), max(1, bottom - top)


def _wrap_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    if not text:
        return []
    lines: list[str] = []
    current = ""
    for character in text:
        candidate = current + character
        if current and _text_size(draw, candidate, font)[0] > max_width:
            lines.append(current)
            current = character
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _layout_label(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    class_name: str,
    confidence: float,
    status: str,
    bbox_xyxy: tuple[float, float, float, float],
    image_size: tuple[int, int],
    occupied_boxes: list[tuple[int, int, int, int]] | None = None,
) -> _LabelLayout:
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        return _LabelLayout((), (0, 0, 0, 0), ())

    margin = 1 if image_width > 2 and image_height > 2 else 0
    padding = 2 if image_width > 8 and image_height > 8 else 0
    max_text_width = max(1, image_width - 2 * (margin + padding))
    lines = [
        *_wrap_to_width(draw, class_name, font, max_text_width),
        *_wrap_to_width(
            draw,
            f"{confidence:.2f} | {status}",
            font,
            max_text_width,
        ),
    ]
    if not lines:
        return _LabelLayout((), (0, 0, 0, 0), ())

    measured = [_text_size(draw, line, font) for line in lines]
    line_gap = 1 if image_height > 8 else 0
    content_width = min(max(width for width, _ in measured), max_text_width)
    content_height = sum(height for _, height in measured) + line_gap * (len(lines) - 1)
    box_width = min(image_width, content_width + 2 * padding)
    box_height = min(image_height, content_height + 2 * padding)

    x1, y1, x2, y2 = bbox_xyxy
    preferred_x = int(round(x1))
    max_left = max(0, image_width - box_width - margin)
    max_top = max(0, image_height - box_height - margin)

    def clamp_left(value: int) -> int:
        return min(max(value, margin), max_left)

    def clamp_top(value: int) -> int:
        return min(max(value, margin), max_top)

    occupied = occupied_boxes or []
    left_candidates = list(
        dict.fromkeys(
            (
                clamp_left(preferred_x),
                clamp_left(int(round(x2)) - box_width),
                clamp_left(margin),
                clamp_left(max_left),
                *(
                    candidate
                    for other_left, _, other_right, _ in occupied
                    for candidate in (
                        clamp_left(other_left - box_width - 1),
                        clamp_left(other_right + 1),
                    )
                ),
            )
        )
    )
    top_candidates = list(
        dict.fromkeys(
            (
                clamp_top(int(round(y1)) - box_height - 1),
                clamp_top(int(round(y1))),
                clamp_top(int(round(y2)) + 1),
                clamp_top(int(round(y2)) - box_height),
                clamp_top(margin),
                clamp_top(max_top),
                *(
                    candidate
                    for _, other_top, _, other_bottom in occupied
                    for candidate in (
                        clamp_top(other_top - box_height - 1),
                        clamp_top(other_bottom + 1),
                    )
                ),
            )
        )
    )

    def overlap_area(box: tuple[int, int, int, int]) -> int:
        left, top, right, bottom = box
        return sum(
            max(0, min(right, other_right) - max(left, other_left))
            * max(0, min(bottom, other_bottom) - max(top, other_top))
            for other_left, other_top, other_right, other_bottom in occupied
        )

    candidates = [
        (
            overlap_area((left, top, left + box_width, top + box_height)),
            order,
            left,
            top,
        )
        for order, (top, left) in enumerate(
            (top, left)
            for top in top_candidates
            for left in left_candidates
        )
    ]
    _, _, left, top = min(candidates)

    right = min(image_width, left + box_width)
    bottom = min(image_height, top + box_height)
    origins: list[tuple[int, int]] = []
    text_y = top + padding
    for _, height in measured:
        if text_y + height > bottom:
            break
        origins.append((left + padding, text_y))
        text_y += height + line_gap
    visible_lines = tuple(lines[: len(origins)])
    return _LabelLayout(
        lines=visible_lines,
        box_xyxy=(left, top, right, bottom),
        text_origins=tuple(origins),
    )


def render_preview(
    image_path: Path,
    response: PipelineResponse,
    output_path: Path,
    settings: PreviewSettings,
) -> None:
    """Render only coordinates already present in the final response."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image.load()
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(image, "RGBA")

    occupied_label_boxes: list[tuple[int, int, int, int]] = []
    for observation in response.observations:
        if not isinstance(observation.geometry, BBoxGeometry):
            continue
        color_text = settings.task_group_colors.get(
            observation.task_group,
            settings.task_group_colors.get("unknown", "#d97706"),
        )
        color = _hex_to_rgb(color_text)
        source_label = _observation_display_source(observation)
        if source_label in {"uncertain", "review-failed"}:
            color = (107, 114, 128)
        x1, y1, x2, y2 = observation.geometry.bbox_xyxy
        outline_box = (
            min(max(x1, 0.0), max(0.0, image.width - 1.0)),
            min(max(y1, 0.0), max(0.0, image.height - 1.0)),
            min(max(x2, 0.0), max(0.0, image.width - 1.0)),
            min(max(y2, 0.0), max(0.0, image.height - 1.0)),
        )
        draw.rectangle(outline_box, outline=(*color, 255), width=3)
        layout = _layout_label(
            draw,
            font,
            observation.class_name,
            observation.confidence,
            source_label,
            observation.geometry.bbox_xyxy,
            image.size,
            occupied_boxes=occupied_label_boxes,
        )
        if layout.lines:
            left, top, right, bottom = layout.box_xyxy
            draw.rectangle(
                (left, top, max(left, right - 1), max(top, bottom - 1)),
                fill=(*color, 210),
            )
            for line, origin in zip(layout.lines, layout.text_origins):
                draw.text(
                    origin,
                    line,
                    fill=(255, 255, 255, 255),
                    font=font,
                    anchor="lt",
                )
            occupied_label_boxes.append(layout.box_xyxy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=92)
