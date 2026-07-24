"""Deterministic overlapping crops and bbox mapping for single-frame VLM review."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image

from .config import CropScanSettings
from .schemas import BBoxGeometry, Float4, ValidatedImage


Int4 = tuple[int, int, int, int]


@dataclass(frozen=True)
class ImageCrop:
    crop_id: str
    bbox_xyxy: Int4
    bbox_normalized_xyxy: Float4
    width: int
    height: int
    image: Image.Image


def _axis_windows(length: int, count: int, overlap: float) -> list[tuple[int, int]]:
    if length <= 0:
        raise ValueError("crop axis length must be positive")
    if count <= 1 or length == 1:
        return [(0, length)]
    window = min(
        length,
        max(1, math.ceil(length / (1.0 + (count - 1) * (1.0 - overlap)))),
    )
    travel = length - window
    starts = [round(index * travel / (count - 1)) for index in range(count)]
    windows: list[tuple[int, int]] = []
    for start in starts:
        item = (max(0, start), min(length, start + window))
        if item[1] > item[0] and item not in windows:
            windows.append(item)
    return windows


def generate_crops(image: ValidatedImage, settings: CropScanSettings) -> list[ImageCrop]:
    """Generate square-grid, wide-strip, or tall-strip crops with full coverage."""
    ratio = image.width / image.height
    if ratio > settings.aspect_ratio_threshold:
        rows, columns = 1, settings.wide_columns
    elif ratio < 1.0 / settings.aspect_ratio_threshold:
        rows, columns = settings.tall_rows, 1
    else:
        rows, columns = settings.square_rows, settings.square_columns

    x_windows = _axis_windows(image.width, columns, settings.overlap)
    y_windows = _axis_windows(image.height, rows, settings.overlap)
    crops: list[ImageCrop] = []
    for row, (y1, y2) in enumerate(y_windows):
        for column, (x1, x2) in enumerate(x_windows):
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue
            crops.append(
                ImageCrop(
                    crop_id=f"crop-r{row + 1}-c{column + 1}",
                    bbox_xyxy=(x1, y1, x2, y2),
                    bbox_normalized_xyxy=(
                        x1 / image.width,
                        y1 / image.height,
                        x2 / image.width,
                        y2 / image.height,
                    ),
                    width=width,
                    height=height,
                    image=image.image.crop((x1, y1, x2, y2)),
                )
            )
    if not crops:
        raise ValueError("crop strategy produced no non-empty crops")
    return crops


def map_crop_bbox_to_image(
    crop: ImageCrop,
    bbox_normalized_xyxy: Float4,
    image_width: int,
    image_height: int,
) -> BBoxGeometry:
    """Map a crop-relative normalized bbox into canonical full-image geometry."""
    x1, y1, x2, y2 = (min(max(float(value), 0.0), 1.0) for value in bbox_normalized_xyxy)
    crop_x1, crop_y1, _, _ = crop.bbox_xyxy
    return BBoxGeometry.from_xyxy(
        (
            crop_x1 + x1 * crop.width,
            crop_y1 + y1 * crop.height,
            crop_x1 + x2 * crop.width,
            crop_y1 + y2 * crop.height,
        ),
        image_width,
        image_height,
    )


def map_full_image_bbox(
    bbox_normalized_xyxy: Float4,
    image_width: int,
    image_height: int,
) -> BBoxGeometry:
    """Convert full-image normalized coordinates to canonical geometry."""
    x1, y1, x2, y2 = (min(max(float(value), 0.0), 1.0) for value in bbox_normalized_xyxy)
    return BBoxGeometry.from_xyxy(
        (x1 * image_width, y1 * image_height, x2 * image_width, y2 * image_height),
        image_width,
        image_height,
    )
