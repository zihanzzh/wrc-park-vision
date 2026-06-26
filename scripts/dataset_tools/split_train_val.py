#!/usr/bin/env python3
"""Re-split a YOLO dataset into train/val with a fixed random seed."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from yolo_common import assert_dataset_valid, image_files


def move_val_back_to_train(dataset_dir: Path) -> None:
    for kind in ("images", "labels"):
        train_dir = dataset_dir / kind / "train"
        val_dir = dataset_dir / kind / "val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(val_dir.iterdir()):
            if not path.is_file():
                continue
            destination = train_dir / path.name
            if destination.exists():
                raise FileExistsError(f"Cannot move {path}; {destination} already exists")
            shutil.move(str(path), str(destination))


def clear_val(dataset_dir: Path) -> None:
    for kind in ("images", "labels"):
        val_dir = dataset_dir / kind / "val"
        if val_dir.exists():
            shutil.rmtree(val_dir)
        val_dir.mkdir(parents=True, exist_ok=True)


def split_dataset(dataset_dir: Path, val_ratio: float, seed: int) -> dict[str, object]:
    if not (0 < val_ratio < 1):
        raise ValueError("--val-ratio must be between 0 and 1")

    move_val_back_to_train(dataset_dir)
    clear_val(dataset_dir)
    assert_dataset_valid(dataset_dir, splits=("train",), require_class_id=0)

    train_images = image_files(dataset_dir / "images" / "train")
    if len(train_images) < 2:
        raise ValueError("Need at least 2 image-label pairs to split train/val")

    rng = random.Random(seed)
    shuffled = list(train_images)
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio))
    val_images = sorted(shuffled[:val_count], key=lambda path: path.name)

    for image_path in val_images:
        label_path = dataset_dir / "labels" / "train" / f"{image_path.stem}.txt"
        shutil.move(str(image_path), str(dataset_dir / "images" / "val" / image_path.name))
        shutil.move(str(label_path), str(dataset_dir / "labels" / "val" / label_path.name))

    stats = assert_dataset_valid(dataset_dir, splits=("train", "val"), require_class_id=0)
    return {
        "dataset": str(dataset_dir),
        "seed": seed,
        "val_ratio": val_ratio,
        "train": stats["train"].__dict__,
        "val": stats["val"].__dict__,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-split a YOLO dataset into train/val.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(json.dumps(split_dataset(args.dataset, args.val_ratio, args.seed), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
