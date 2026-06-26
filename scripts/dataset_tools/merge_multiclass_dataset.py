#!/usr/bin/env python3
"""Merge single-class YOLO datasets into one multi-class YOLO dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from yolo_common import (
    assert_dataset_valid,
    ensure_yolo_dirs,
    image_files,
    read_class_names,
    read_yolo_bbox_labels,
    reset_dir,
    unique_name,
    write_label_file,
)


def write_multiclass_data_yaml(dataset_dir: Path, class_names: list[str]) -> None:
    lines = [
        f"path: {dataset_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for class_id, class_name in enumerate(class_names):
        lines.append(f"  {class_id}: {class_name}")
    (dataset_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_multiclass_dataset(dataset_dir: Path, allowed_class_ids: set[int]) -> dict[str, dict[str, object]]:
    stats: dict[str, dict[str, object]] = {}
    problems: list[str] = []

    for split in ("train", "val"):
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        images = {path.stem: path for path in image_files(images_dir)}
        labels = {path.stem: path for path in sorted(labels_dir.glob("*.txt"))}

        images_without_labels = sorted(set(images) - set(labels))
        labels_without_images = sorted(set(labels) - set(images))
        if images_without_labels:
            problems.append(f"{split} images without labels: {images_without_labels}")
        if labels_without_images:
            problems.append(f"{split} labels without images: {labels_without_images}")

        bbox_count = 0
        bad_lines = 0
        class_counts: dict[int, int] = {}
        for label_path in labels.values():
            boxes, line_errors = read_yolo_bbox_labels(label_path)
            bad_lines += line_errors
            for class_id, _ in boxes:
                if class_id not in allowed_class_ids:
                    bad_lines += 1
                    continue
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                bbox_count += 1
        if bad_lines:
            problems.append(f"{split} bad label lines: {bad_lines}")

        stats[split] = {
            "images": len(images),
            "labels": len(labels),
            "bboxes": bbox_count,
            "bad_lines": bad_lines,
            "images_without_labels": images_without_labels,
            "labels_without_images": labels_without_images,
            "class_counts": {str(key): class_counts.get(key, 0) for key in sorted(allowed_class_ids)},
        }

    if problems:
        raise ValueError("\n".join(problems))
    return stats


def remap_label_lines(label_path: Path, target_class_id: int) -> tuple[list[str], int, int]:
    boxes, bad_lines = read_yolo_bbox_labels(label_path)
    output: list[str] = []
    skipped_nonzero = 0

    for source_class_id, bbox in boxes:
        if source_class_id != 0:
            skipped_nonzero += 1
            continue
        output.append(" ".join([str(target_class_id), *(f"{value:.6f}".rstrip("0").rstrip(".") for value in bbox)]))
    return output, bad_lines, skipped_nonzero


def parse_source(values: list[str]) -> tuple[int, str, Path]:
    if len(values) != 3:
        raise ValueError("--source expects: CLASS_ID PREFIX DATASET_DIR")
    return int(values[0]), values[1], Path(values[2])


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge single-class YOLO datasets into a multi-class dataset.")
    parser.add_argument("--output", type=Path, required=True, help="Output multi-class dataset directory.")
    parser.add_argument("--class-names", nargs="+", required=True, help="Class names ordered by final class id.")
    parser.add_argument(
        "--source",
        nargs=3,
        action="append",
        required=True,
        metavar=("CLASS_ID", "PREFIX", "DATASET_DIR"),
        help="Single-class source dataset and final class id. Repeat for each source.",
    )
    args = parser.parse_args()

    allowed_class_ids = set(range(len(args.class_names)))
    sources = [parse_source(values) for values in args.source]
    for class_id, _, source_dir in sources:
        if class_id not in allowed_class_ids:
            raise ValueError(f"Source class id {class_id} is not in allowed ids {sorted(allowed_class_ids)}")
        assert_dataset_valid(source_dir, splits=("train", "val"), require_class_id=0)

    reset_dir(args.output)
    ensure_yolo_dirs(args.output, splits=("train", "val"))
    used_names: set[str] = set()

    stats: dict[str, object] = {
        "output": str(args.output),
        "class_names": {str(index): name for index, name in enumerate(args.class_names)},
        "sources": [],
    }

    for class_id, prefix, source_dir in sources:
        source_names = read_class_names(source_dir / "data.yaml")
        source_stats = {
            "source": str(source_dir),
            "prefix": prefix,
            "target_class_id": class_id,
            "source_names": source_names,
            "splits": {},
            "bad_label_lines": 0,
            "skipped_nonzero_class_lines": 0,
        }

        for split in ("train", "val"):
            images_dir = source_dir / "images" / split
            labels_dir = source_dir / "labels" / split
            split_stats = {"images": 0, "bboxes": 0}

            for image_path in image_files(images_dir):
                label_path = labels_dir / f"{image_path.stem}.txt"
                label_lines, bad_lines, skipped_nonzero = remap_label_lines(label_path, class_id)
                source_stats["bad_label_lines"] += bad_lines
                source_stats["skipped_nonzero_class_lines"] += skipped_nonzero
                if bad_lines or skipped_nonzero:
                    continue
                if not label_lines:
                    continue

                output_name = unique_name(prefix, image_path.name, used_names)
                output_image = args.output / "images" / split / output_name
                output_label = args.output / "labels" / split / f"{Path(output_name).stem}.txt"
                output_image.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, output_image)
                write_label_file(output_label, label_lines)
                split_stats["images"] += 1
                split_stats["bboxes"] += len(label_lines)

            source_stats["splits"][split] = split_stats

        if source_stats["bad_label_lines"] or source_stats["skipped_nonzero_class_lines"]:
            raise ValueError(f"Invalid source labels while processing {source_dir}: {source_stats}")
        stats["sources"].append(source_stats)

    write_multiclass_data_yaml(args.output, args.class_names)
    stats["final"] = validate_multiclass_dataset(args.output, allowed_class_ids)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
