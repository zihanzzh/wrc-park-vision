"""Annotated preview rendering from finalized PipelineResponse objects."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import PreviewSettings
from .schemas import BBoxGeometry, PipelineResponse


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


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
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for observation in response.observations:
        if not isinstance(observation.geometry, BBoxGeometry):
            continue
        color_text = settings.task_group_colors.get(
            observation.task_group,
            settings.task_group_colors.get("unknown", "#d97706"),
        )
        color = _hex_to_rgb(color_text)
        x1, y1, x2, y2 = observation.geometry.bbox_xyxy
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f"{observation.task_group} | {observation.class_name} | {observation.confidence:.2f}"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_left, text_top, text_right, text_bottom = text_bbox
        if text_top < 0:
            offset = -text_top
            text_top += offset
            text_bottom += offset
        draw.rectangle((text_left, text_top, text_right, text_bottom), fill=color)
        draw.text((text_left, text_top), label, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=92)

