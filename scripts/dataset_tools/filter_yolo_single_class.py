#!/usr/bin/env python3
"""Filter a multi-class YOLO dataset into one project class.

This script is intentionally small and dependency-free. It copies only images
that have at least one target-class bbox, remaps that class id to 0, and writes
a clean YOLO dataset without modifying the raw Roboflow export.
"""

from __future__ import annotations

import argparse
import ast
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_RAW_DIR = Path("datasets_raw/roboflow_spray_can_by_kim")
DEFAULT_OUT_DIR = Path("datasets_clean/spray_can_yolo11_single_class")
TARGET_CLASS = "spray_can"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SplitStats:
    raw_images: int = 0
    clean_images: int = 0
    clean_bboxes: int = 0
    skipped_images: int = 0


def normalize_name(name: str) -> str:
    """Normalize class names so 'Spray Can', 'spray_can', and 'SprayCan' match."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def read_names(data_yaml: Path) -> list[str]:
    """Read a simple Roboflow-style names field using only the standard library."""
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
            key = int(key_text.strip())
            names[key] = name_text.strip().strip("'\"")
        return [names[key] for key in sorted(names)]

    raise ValueError(f"No names field found in {data_yaml}")


def find_target_class_id(names: list[str], target: str) -> int | None:
    target_normalized = normalize_name(target)
    for class_id, name in enumerate(names):
        if normalize_name(name) == target_normalized:
            return class_id
    return None


def validate_input(raw_dir: Path) -> None:
    required = [
        raw_dir / "data.yaml",
        raw_dir / "train" / "images",
        raw_dir / "train" / "labels",
        raw_dir / "valid" / "images",
        raw_dir / "valid" / "labels",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required dataset paths:\n{formatted}")


def safe_reset_output_dir(out_dir: Path) -> None:
    resolved = out_dir.resolve()
    allowed_parent = (Path.cwd() / "datasets_clean").resolve()
    if allowed_parent not in resolved.parents:
        raise ValueError(f"Refusing to delete output outside datasets_clean/: {out_dir}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def image_files(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def filter_label_file(label_file: Path, target_class_id: int) -> list[str]:
    if not label_file.exists():
        return []

    output_lines: list[str] = []
    for line_number, line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"Expected 5 YOLO fields in {label_file}:{line_number}, got {len(parts)}")

        try:
            class_id = int(float(parts[0]))
        except ValueError as exc:
            raise ValueError(f"Invalid class id in {label_file}:{line_number}: {parts[0]}") from exc

        if class_id == target_class_id:
            output_lines.append(" ".join(["0", *parts[1:]]))

    return output_lines


def process_split(raw_dir: Path, out_dir: Path, raw_split: str, clean_split: str, target_class_id: int) -> SplitStats:
    stats = SplitStats()
    images_dir = raw_dir / raw_split / "images"
    labels_dir = raw_dir / raw_split / "labels"
    out_images_dir = out_dir / "images" / clean_split
    out_labels_dir = out_dir / "labels" / clean_split

    images = image_files(images_dir)
    stats.raw_images = len(images)

    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        filtered_lines = filter_label_file(label_path, target_class_id)
        if not filtered_lines:
            stats.skipped_images += 1
            continue

        shutil.copy2(image_path, out_images_dir / image_path.name)
        (out_labels_dir / f"{image_path.stem}.txt").write_text(
            "\n".join(filtered_lines) + "\n",
            encoding="utf-8",
        )
        stats.clean_images += 1
        stats.clean_bboxes += len(filtered_lines)

    return stats


def write_data_yaml(out_dir: Path) -> None:
    (out_dir / "data.yaml").write_text(
        "path: datasets_clean/spray_can_yolo11_single_class\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: spray_can\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter Roboflow spray can YOLO dataset to one class.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    validate_input(args.raw_dir)
    names = read_names(args.raw_dir / "data.yaml")
    target_class_id = find_target_class_id(names, TARGET_CLASS)
    if target_class_id is None:
        print("ERROR: Could not find spray can class in data.yaml names.")
        print(f"names: {names}")
        return 1

    safe_reset_output_dir(args.out_dir)
    train_stats = process_split(args.raw_dir, args.out_dir, "train", "train", target_class_id)
    val_stats = process_split(args.raw_dir, args.out_dir, "valid", "val", target_class_id)
    write_data_yaml(args.out_dir)

    total_skipped = train_stats.skipped_images + val_stats.skipped_images
    print("Filtered YOLO dataset created.")
    print(f"raw_dir: {args.raw_dir}")
    print(f"clean_dir: {args.out_dir}")
    print(f"source_names: {names}")
    print(f"source_spray_can_class_id: {target_class_id}")
    print("clean_class_id: 0")
    print(f"raw_train_images: {train_stats.raw_images}")
    print(f"raw_valid_images: {val_stats.raw_images}")
    print(f"clean_train_images: {train_stats.clean_images}")
    print(f"clean_val_images: {val_stats.clean_images}")
    print(f"clean_train_bboxes: {train_stats.clean_bboxes}")
    print(f"clean_val_bboxes: {val_stats.clean_bboxes}")
    print(f"removed_non_spray_or_empty_images: {total_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
