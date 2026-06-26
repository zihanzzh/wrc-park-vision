#!/usr/bin/env python3
"""Shared helpers for lightweight YOLO dataset preparation tools."""

from __future__ import annotations

import ast
import shutil
from dataclasses import dataclass, field
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RAW_SPLITS = ("train", "valid", "val", "test")
YOLO_SPLITS = ("train", "val", "test")


@dataclass
class DatasetStats:
    images: int = 0
    labels: int = 0
    bboxes: int = 0
    bad_lines: int = 0
    images_without_labels: list[str] = field(default_factory=list)
    labels_without_images: list[str] = field(default_factory=list)


def normalize_class_name(name: str) -> str:
    """Normalize class names so spaces, underscores, hyphens, and case do not matter."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def read_class_names(data_yaml: Path) -> list[str]:
    """Read YOLO class names from simple Roboflow / Ultralytics data.yaml files."""
    lines = data_yaml.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("names:"):
            continue

        value = stripped.split(":", 1)[1].strip()
        if value:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return [str(parsed[key]) for key in sorted(parsed)]
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            raise ValueError(f"Unsupported names format in {data_yaml}: {value}")

        names: dict[int, str] = {}
        for child in lines[index + 1 :]:
            if not child.startswith((" ", "\t")):
                break
            child = child.strip()
            if not child or ":" not in child:
                continue
            key_text, name_text = child.split(":", 1)
            names[int(key_text.strip())] = name_text.strip().strip("'\"")
        return [names[key] for key in sorted(names)]

    raise ValueError(f"No names field found in {data_yaml}")


def match_class_ids(names: list[str], match_names: list[str]) -> set[int]:
    normalized_targets = {normalize_class_name(name) for name in match_names}
    return {
        class_id
        for class_id, class_name in enumerate(names)
        if normalize_class_name(class_name) in normalized_targets
    }


def image_files(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        return []
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def label_files(labels_dir: Path) -> list[Path]:
    if not labels_dir.is_dir():
        return []
    return sorted(path for path in labels_dir.iterdir() if path.is_file() and path.suffix == ".txt")


def clean_split_name(split: str) -> str:
    return "val" if split == "valid" else split


def find_split_dirs(dataset_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return available YOLO split directories as (split, images_dir, labels_dir)."""
    found: list[tuple[str, Path, Path]] = []

    for split in RAW_SPLITS:
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        if images_dir.is_dir() and labels_dir.is_dir():
            found.append((clean_split_name(split), images_dir, labels_dir))
    if found:
        return found

    for split in YOLO_SPLITS:
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        if images_dir.is_dir() and labels_dir.is_dir():
            found.append((split, images_dir, labels_dir))
    return found


def image_stem_map(images_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in image_files(images_dir):
        if path.stem in result:
            raise ValueError(f"Duplicate image stem found: {path.stem}")
        result[path.stem] = path
    return result


def label_stem_map(labels_dir: Path) -> dict[str, Path]:
    return {path.stem: path for path in label_files(labels_dir)}


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def format_number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def bbox_from_edges(left: float, top: float, right: float, bottom: float) -> list[float] | None:
    left = clamp01(left)
    top = clamp01(top)
    right = clamp01(right)
    bottom = clamp01(bottom)
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return None
    return [left + width / 2, top + height / 2, width, height]


def parse_yolo_target_line(parts: list[str], target_ids: set[int]) -> tuple[list[float] | None, bool]:
    """Parse a target YOLO bbox or segmentation row into bbox coordinates.

    Returns (bbox, malformed). bbox is [center_x, center_y, width, height].
    """
    if len(parts) < 2:
        return None, True
    try:
        class_id = int(float(parts[0]))
    except ValueError:
        return None, True
    if class_id not in target_ids:
        return None, False

    try:
        values = [float(value) for value in parts[1:]]
    except ValueError:
        return None, True

    if len(parts) == 5:
        center_x, center_y, width, height = values
        converted = bbox_from_edges(
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        )
        return converted, converted is None

    if len(values) < 6 or len(values) % 2 != 0:
        return None, True

    xs = values[0::2]
    ys = values[1::2]
    converted = bbox_from_edges(min(xs), min(ys), max(xs), max(ys))
    return converted, converted is None


def yolo_label_line(class_id: int, bbox: list[float]) -> str:
    return " ".join([str(class_id), *(format_number(value) for value in bbox)])


def extract_target_labels(label_path: Path, target_ids: set[int]) -> tuple[list[str], int]:
    if not label_path.exists():
        return [], 0

    output: list[str] = []
    skipped_bad_lines = 0
    for line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        bbox, malformed = parse_yolo_target_line(stripped.split(), target_ids)
        if malformed:
            skipped_bad_lines += 1
            continue
        if bbox:
            output.append(yolo_label_line(0, bbox))
    return output, skipped_bad_lines


def read_yolo_bbox_labels(label_path: Path) -> tuple[list[tuple[int, list[float]]], int]:
    boxes: list[tuple[int, list[float]]] = []
    bad_lines = 0
    for line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            bad_lines += 1
            continue
        try:
            class_id = int(float(parts[0]))
            bbox = [float(value) for value in parts[1:]]
        except ValueError:
            bad_lines += 1
            continue
        if any(value < 0 or value > 1 for value in bbox):
            bad_lines += 1
            continue
        boxes.append((class_id, bbox))
    return boxes, bad_lines


def write_label_file(label_path: Path, lines: list[str]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_data_yaml(dataset_dir: Path, target_name: str) -> None:
    (dataset_dir / "data.yaml").write_text(
        f"path: {dataset_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: {target_name}\n",
        encoding="utf-8",
    )


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_yolo_dirs(dataset_dir: Path, splits: tuple[str, ...] = ("train", "val")) -> None:
    for split in splits:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def unique_name(prefix: str, original_name: str, used_names: set[str]) -> str:
    candidate = f"{prefix}__{original_name}"
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate
    index = 2
    while True:
        candidate = f"{prefix}__{index}__{original_name}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        index += 1


def validate_split(dataset_dir: Path, split: str, require_class_id: int | None = None) -> DatasetStats:
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split
    images = image_stem_map(images_dir)
    labels = label_stem_map(labels_dir)

    stats = DatasetStats(images=len(images), labels=len(labels))
    stats.images_without_labels = sorted(set(images) - set(labels))
    stats.labels_without_images = sorted(set(labels) - set(images))

    for label_path in labels.values():
        boxes, bad_lines = read_yolo_bbox_labels(label_path)
        stats.bad_lines += bad_lines
        for class_id, _ in boxes:
            if require_class_id is not None and class_id != require_class_id:
                stats.bad_lines += 1
            else:
                stats.bboxes += 1
    return stats


def validate_dataset(dataset_dir: Path, splits: tuple[str, ...] = ("train", "val"), require_class_id: int | None = 0) -> dict[str, DatasetStats]:
    return {
        split: validate_split(dataset_dir, split, require_class_id=require_class_id)
        for split in splits
    }


def assert_dataset_valid(dataset_dir: Path, splits: tuple[str, ...] = ("train", "val"), require_class_id: int | None = 0) -> dict[str, DatasetStats]:
    stats_by_split = validate_dataset(dataset_dir, splits=splits, require_class_id=require_class_id)
    problems: list[str] = []
    for split, stats in stats_by_split.items():
        if stats.images_without_labels:
            problems.append(f"{split} images without labels: {stats.images_without_labels}")
        if stats.labels_without_images:
            problems.append(f"{split} labels without images: {stats.labels_without_images}")
        if stats.bad_lines:
            problems.append(f"{split} bad label lines: {stats.bad_lines}")
    if problems:
        raise ValueError("\n".join(problems))
    return stats_by_split
