#!/usr/bin/env python3
"""Extract one target class from a raw YOLO dataset into a staged single-class dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from yolo_common import (
    ensure_yolo_dirs,
    extract_target_labels,
    find_split_dirs,
    image_files,
    match_class_ids,
    read_class_names,
    reset_dir,
    unique_name,
    write_data_yaml,
    write_label_file,
)


def parse_match_classes(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract one class from a YOLO dataset into a staged single-class dataset.")
    parser.add_argument("--source", type=Path, required=True, help="Raw YOLO dataset directory.")
    parser.add_argument("--target-name", required=True, help="Project class name to write as class id 0.")
    parser.add_argument("--match-classes", required=True, help="Comma-separated source class names to match.")
    parser.add_argument("--source-prefix", required=True, help="Prefix added to output filenames.")
    parser.add_argument("--output", type=Path, required=True, help="Output staging dataset directory.")
    args = parser.parse_args()

    data_yaml = args.source / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml: {data_yaml}")

    names = read_class_names(data_yaml)
    target_ids = match_class_ids(names, parse_match_classes(args.match_classes))
    if not target_ids:
        raise ValueError(f"No matching class found. source_names={names}")

    split_dirs = find_split_dirs(args.source)
    if not split_dirs:
        raise FileNotFoundError(f"No supported YOLO split directories found under {args.source}")

    reset_dir(args.output)
    ensure_yolo_dirs(args.output, splits=("train", "val", "test"))
    used_names: set[str] = set()

    stats = {
        "source": str(args.source),
        "output": str(args.output),
        "target_name": args.target_name,
        "source_names": names,
        "matched_classes": [{"id": class_id, "name": names[class_id]} for class_id in sorted(target_ids)],
        "raw_images": 0,
        "kept_images": 0,
        "kept_bboxes": 0,
        "skipped_images": 0,
        "skipped_bad_label_lines": 0,
    }

    for split, images_dir, labels_dir in split_dirs:
        clean_split = "val" if split == "valid" else split
        if clean_split not in {"train", "val", "test"}:
            continue
        for image_path in image_files(images_dir):
            stats["raw_images"] += 1
            label_path = labels_dir / f"{image_path.stem}.txt"
            label_lines, skipped_bad_lines = extract_target_labels(label_path, target_ids)
            stats["skipped_bad_label_lines"] += skipped_bad_lines
            if not label_lines:
                stats["skipped_images"] += 1
                continue

            output_name = unique_name(args.source_prefix, image_path.name, used_names)
            output_image = args.output / "images" / clean_split / output_name
            output_label = args.output / "labels" / clean_split / f"{Path(output_name).stem}.txt"
            output_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, output_image)
            write_label_file(output_label, label_lines)
            stats["kept_images"] += 1
            stats["kept_bboxes"] += len(label_lines)

    write_data_yaml(args.output, args.target_name)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
