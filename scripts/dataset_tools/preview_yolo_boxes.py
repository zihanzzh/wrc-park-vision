#!/usr/bin/env python3
"""Render YOLO bounding-box previews for manual dataset inspection."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from yolo_common import read_class_names


DEFAULT_DATASET_DIR = Path("datasets_clean/spray_can")
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


def read_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number} expected 5 fields, got {len(parts)}")
        class_id = int(float(parts[0]))
        boxes.append((class_id, *(float(value) for value in parts[1:])))
    return boxes


def source_prefix(file_name: str) -> str:
    if "__" not in file_name:
        return "unknown"
    return file_name.split("__", 1)[0]


def select_preview_images(images: list[Path], max_images: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    groups: dict[str, list[Path]] = {}
    for image_path in images:
        groups.setdefault(source_prefix(image_path.name), []).append(image_path)

    for group in groups.values():
        rng.shuffle(group)

    selected: list[Path] = []
    prefixes = sorted(groups)
    while len(selected) < max_images and prefixes:
        remaining_prefixes: list[str] = []
        for prefix in prefixes:
            group = groups[prefix]
            if group and len(selected) < max_images:
                selected.append(group.pop())
            if group:
                remaining_prefixes.append(prefix)
        prefixes = remaining_prefixes
    return sorted(selected, key=lambda path: path.name)


def draw_preview(image_path: Path, label_path: Path, output_path: Path, class_names: list[str]) -> None:
    Image, ImageDraw, ImageFont = load_pillow()
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font = ImageFont.load_default()

    for class_id, center_x, center_y, box_width, box_height in read_boxes(label_path):
        left = (center_x - box_width / 2) * width
        top = (center_y - box_height / 2) * height
        right = (center_x + box_width / 2) * width
        bottom = (center_y + box_height / 2) * height
        draw.rectangle([left, top, right, bottom], outline=(255, 0, 0), width=3)
        label = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
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


def render_split(dataset_dir: Path, split: str, max_images: int, seed: int, class_names: list[str]) -> int:
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split
    preview_dir = dataset_dir / "previews" / split

    images = image_files(images_dir)
    selected = select_preview_images(images, max_images, seed)

    count = 0
    for image_path in selected:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for preview image: {image_path}")
        output_path = preview_dir / image_path.name
        draw_preview(image_path, label_path, output_path, class_names)
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate bbox preview images for a YOLO dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_pillow()
    class_names = read_class_names(args.dataset_dir / "data.yaml")
    reset_preview_dir(args.dataset_dir / "previews")
    train_count = render_split(args.dataset_dir, "train", args.max_images, args.seed, class_names)
    val_count = render_split(args.dataset_dir, "val", args.max_images, args.seed, class_names)
    print("Preview images generated.")
    print(f"preview_dir: {args.dataset_dir / 'previews'}")
    print(f"train_previews: {train_count}")
    print(f"val_previews: {val_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
