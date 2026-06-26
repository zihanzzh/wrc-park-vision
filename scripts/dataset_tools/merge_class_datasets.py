#!/usr/bin/env python3
"""Merge staged single-class YOLO datasets into one canonical clean dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from yolo_common import (
    assert_dataset_valid,
    ensure_yolo_dirs,
    find_split_dirs,
    image_files,
    read_yolo_bbox_labels,
    reset_dir,
    unique_name,
    write_data_yaml,
    write_label_file,
)


def label_lines_for_single_class(label_path: Path) -> tuple[list[str], int]:
    boxes, bad_lines = read_yolo_bbox_labels(label_path)
    output: list[str] = []
    for class_id, bbox in boxes:
        if class_id != 0:
            bad_lines += 1
            continue
        output.append(" ".join(["0", *(f"{value:.6f}".rstrip("0").rstrip(".") for value in bbox)]))
    return output, bad_lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge staged single-class YOLO datasets.")
    parser.add_argument("--target-name", required=True, help="Project class name for data.yaml.")
    parser.add_argument("--sources", type=Path, nargs="+", required=True, help="Staging dataset directories.")
    parser.add_argument("--output", type=Path, required=True, help="Canonical clean dataset directory.")
    args = parser.parse_args()

    reset_dir(args.output)
    ensure_yolo_dirs(args.output, splits=("train", "val"))
    used_names: set[str] = set()

    stats = {
        "target_name": args.target_name,
        "output": str(args.output),
        "sources": [],
        "kept_images": 0,
        "kept_bboxes": 0,
        "skipped_images": 0,
        "bad_label_lines": 0,
    }

    for source in args.sources:
        source_stats = {
            "source": str(source),
            "raw_images": 0,
            "kept_images": 0,
            "kept_bboxes": 0,
            "skipped_images": 0,
            "bad_label_lines": 0,
        }
        split_dirs = find_split_dirs(source)
        if not split_dirs:
            raise FileNotFoundError(f"No supported YOLO split directories found under {source}")

        for _, images_dir, labels_dir in split_dirs:
            for image_path in image_files(images_dir):
                source_stats["raw_images"] += 1
                label_path = labels_dir / f"{image_path.stem}.txt"
                label_lines, bad_lines = label_lines_for_single_class(label_path)
                source_stats["bad_label_lines"] += bad_lines
                stats["bad_label_lines"] += bad_lines
                if not label_lines:
                    source_stats["skipped_images"] += 1
                    stats["skipped_images"] += 1
                    continue

                source_prefix = source.name or "source"
                output_name = unique_name(source_prefix, image_path.name, used_names)
                output_image = args.output / "images" / "train" / output_name
                output_label = args.output / "labels" / "train" / f"{Path(output_name).stem}.txt"
                output_image.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, output_image)
                write_label_file(output_label, label_lines)
                source_stats["kept_images"] += 1
                source_stats["kept_bboxes"] += len(label_lines)
                stats["kept_images"] += 1
                stats["kept_bboxes"] += len(label_lines)
        stats["sources"].append(source_stats)

    write_data_yaml(args.output, args.target_name)
    assert_dataset_valid(args.output, splits=("train", "val"), require_class_id=0)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
