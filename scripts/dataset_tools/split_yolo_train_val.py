#!/usr/bin/env python3
"""Validate and split a clean single-class YOLO dataset into train/val.

The script is intentionally dependency-free and only manipulates the clean
dataset. It never touches datasets_raw/.
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATASET_DIR = Path("datasets_clean/spray_can_yolo11_single_class")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SplitSummary:
    images: int
    labels: int
    bboxes: int


def image_stem_map(images_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in result:
                raise ValueError(f"Duplicate image stem found: {path.stem}")
            result[path.stem] = path
    return result


def label_stem_map(labels_dir: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in sorted(labels_dir.iterdir())
        if path.is_file() and path.suffix == ".txt"
    }


def count_and_validate_label(label_path: Path) -> int:
    bbox_count = 0
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number} expected 5 fields, got {len(parts)}")
        if parts[0] != "0":
            raise ValueError(f"{label_path}:{line_number} expected class id 0, got {parts[0]}")
        for value in parts[1:]:
            try:
                float(value)
            except ValueError as exc:
                raise ValueError(f"{label_path}:{line_number} invalid bbox value: {value}") from exc
        bbox_count += 1
    if bbox_count == 0:
        raise ValueError(f"{label_path} has no bbox rows")
    return bbox_count


def validate_split(dataset_dir: Path, split: str) -> SplitSummary:
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing split directories for {split}")

    images = image_stem_map(images_dir)
    labels = label_stem_map(labels_dir)

    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))
    if missing_labels or missing_images:
        detail = []
        if missing_labels:
            detail.append(f"{split} images without labels: {missing_labels}")
        if missing_images:
            detail.append(f"{split} labels without images: {missing_images}")
        raise ValueError("\n".join(detail))

    bboxes = sum(count_and_validate_label(label_path) for label_path in labels.values())
    return SplitSummary(images=len(images), labels=len(labels), bboxes=bboxes)


def move_existing_val_back_to_train(dataset_dir: Path) -> None:
    """Make the split operation repeatable by restoring any existing val files."""
    for kind in ("images", "labels"):
        val_dir = dataset_dir / kind / "val"
        train_dir = dataset_dir / kind / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(val_dir.iterdir()):
            if not path.is_file():
                continue
            destination = train_dir / path.name
            if destination.exists():
                raise FileExistsError(f"Cannot move {path}; {destination} already exists")
            shutil.move(str(path), str(destination))


def clear_val_dirs(dataset_dir: Path) -> None:
    for kind in ("images", "labels"):
        val_dir = dataset_dir / kind / "val"
        if val_dir.exists():
            shutil.rmtree(val_dir)
        val_dir.mkdir(parents=True, exist_ok=True)


def split_train_val(dataset_dir: Path, val_ratio: float, seed: int) -> None:
    move_existing_val_back_to_train(dataset_dir)
    clear_val_dirs(dataset_dir)

    train_summary = validate_split(dataset_dir, "train")
    if train_summary.images < 2:
        raise ValueError("Need at least 2 image-label pairs to create a val split")

    train_images = list(image_stem_map(dataset_dir / "images" / "train").values())
    rng = random.Random(seed)
    rng.shuffle(train_images)
    val_count = max(1, round(len(train_images) * val_ratio))
    val_images = sorted(train_images[:val_count], key=lambda path: path.name)

    for image_path in val_images:
        label_path = dataset_dir / "labels" / "train" / f"{image_path.stem}.txt"
        shutil.move(str(image_path), str(dataset_dir / "images" / "val" / image_path.name))
        shutil.move(str(label_path), str(dataset_dir / "labels" / "val" / label_path.name))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and split clean YOLO data into train/val.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (0 < args.val_ratio < 1):
        raise ValueError("--val-ratio must be between 0 and 1")

    before_train = validate_split(args.dataset_dir, "train")
    before_val = validate_split(args.dataset_dir, "val")
    print("Before split:")
    print(f"train_images: {before_train.images}")
    print(f"train_labels: {before_train.labels}")
    print(f"train_bboxes: {before_train.bboxes}")
    print(f"val_images: {before_val.images}")
    print(f"val_labels: {before_val.labels}")
    print(f"val_bboxes: {before_val.bboxes}")

    split_train_val(args.dataset_dir, args.val_ratio, args.seed)

    after_train = validate_split(args.dataset_dir, "train")
    after_val = validate_split(args.dataset_dir, "val")
    print("After split:")
    print(f"seed: {args.seed}")
    print(f"val_ratio: {args.val_ratio}")
    print(f"train_images: {after_train.images}")
    print(f"train_labels: {after_train.labels}")
    print(f"train_bboxes: {after_train.bboxes}")
    print(f"val_images: {after_val.images}")
    print(f"val_labels: {after_val.labels}")
    print(f"val_bboxes: {after_val.bboxes}")
    print("all_label_class_ids_are_0: true")
    print("image_label_pairs_match: true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
