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
    font = ImageFont.load_default()
    fusion_by_observation = {
        decision.observation_id: decision
        for decision in response.fusion.decisions
        if decision.observation_id is not None
    }
    review_by_observation = {decision.observation_id: decision for decision in response.review.decisions}
    footer_lines: list[str] = []
    for finding in response.review.findings:
        confidence = f" ({finding.confidence:.2f})" if finding.confidence is not None else ""
        footer_lines.append(f"VLM-only | {finding.task_group} | {finding.class_name}{confidence} | no bbox")
    for decision in response.fusion.decisions:
        if decision.action == "correct_yolo":
            footer_lines.append(
                f"Corrected | {decision.observation_id} -> {decision.final_task_group}/{decision.final_class_name}"
            )
        elif decision.action == "reject_yolo":
            footer_lines.append(f"Rejected | {decision.observation_id}")

    if footer_lines:
        line_height = font.getbbox("Ag")[3] + 5
        footer_height = 10 + line_height * len(footer_lines)
        canvas = Image.new("RGB", (image.width, image.height + footer_height), "white")
        canvas.paste(image, (0, 0))
        image = canvas

    draw = ImageDraw.Draw(image)

    for observation in response.observations:
        if not isinstance(observation.geometry, BBoxGeometry):
            continue
        color_text = settings.task_group_colors.get(
            observation.task_group,
            settings.task_group_colors.get("unknown", "#d97706"),
        )
        color = _hex_to_rgb(color_text)
        fusion_decision = fusion_by_observation.get(observation.id)
        suffix = ""
        if fusion_decision is not None:
            if fusion_decision.action == "reject_yolo":
                suffix = " | VLM rejected"
                color = (107, 114, 128)
            elif fusion_decision.action == "correct_yolo":
                suffix = f" | VLM -> {fusion_decision.final_class_name}"
            else:
                review_decision = review_by_observation.get(observation.id)
                if review_decision is not None:
                    suffix = f" | VLM {review_decision.verdict}"
        x1, y1, x2, y2 = observation.geometry.bbox_xyxy
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f"{observation.task_group} | {observation.class_name} | {observation.confidence:.2f}{suffix}"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_left, text_top, text_right, text_bottom = text_bbox
        if text_top < 0:
            offset = -text_top
            text_top += offset
            text_bottom += offset
        draw.rectangle((text_left, text_top, text_right, text_bottom), fill=color)
        draw.text((text_left, text_top), label, fill=(255, 255, 255), font=font)

    if footer_lines:
        line_height = font.getbbox("Ag")[3] + 5
        footer_top = response.input.height + 5
        for index, line in enumerate(footer_lines):
            draw.text((5, footer_top + index * line_height), line, fill=(17, 24, 39), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=92)
