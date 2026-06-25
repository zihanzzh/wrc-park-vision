#!/usr/bin/env python3
"""Render YOLO bounding-box previews for manual dataset inspection."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


DEFAULT_DATASET_DIR = Path("datasets_clean/spray_can_yolo11_single_class")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("PIL/Pillow is not available; cannot generate preview images.") from exc
    return Image, ImageDraw, ImageFont


def image_files(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def read_boxes(label_path: Path) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number} expected 5 fields, got {len(parts)}")
        if parts[0] != "0":
            raise ValueError(f"{label_path}:{line_number} expected class id 0, got {parts[0]}")
        boxes.append(tuple(float(value) for value in parts[1:]))
    return boxes


def draw_preview(image_path: Path, label_path: Path, output_path: Path) -> None:
    Image, ImageDraw, ImageFont = load_pillow()
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font = ImageFont.load_default()

    for center_x, center_y, box_width, box_height in read_boxes(label_path):
        left = (center_x - box_width / 2) * width
        top = (center_y - box_height / 2) * height
        right = (center_x + box_width / 2) * width
        bottom = (center_y + box_height / 2) * height
        draw.rectangle([left, top, right, bottom], outline=(255, 0, 0), width=3)
        label = "spray_can"
        text_box = draw.textbbox((left, top), label, font=font)
        draw.rectangle(text_box, fill=(255, 0, 0))
        draw.text((left, top), label, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def reset_preview_dir(preview_dir: Path) -> None:
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    (preview_dir / "train").mkdir(parents=True, exist_ok=True)
    (preview_dir / "val").mkdir(parents=True, exist_ok=True)


def render_split(dataset_dir: Path, split: str, max_images: int, seed: int) -> int:
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split
    preview_dir = dataset_dir / "previews" / split

    images = image_files(images_dir)
    rng = random.Random(seed)
    rng.shuffle(images)
    selected = sorted(images[:max_images], key=lambda path: path.name)

    count = 0
    for image_path in selected:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for preview image: {image_path}")
        output_path = preview_dir / image_path.name
        draw_preview(image_path, label_path, output_path)
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate bbox preview images for a YOLO dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_pillow()
    reset_preview_dir(args.dataset_dir / "previews")
    train_count = render_split(args.dataset_dir, "train", args.max_images, args.seed)
    val_count = render_split(args.dataset_dir, "val", args.max_images, args.seed)
    print("Preview images generated.")
    print(f"preview_dir: {args.dataset_dir / 'previews'}")
    print(f"train_previews: {train_count}")
    print(f"val_previews: {val_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
