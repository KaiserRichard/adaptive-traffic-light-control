"""
split_yolo_dataset.py

Purpose:
- Convert a Roboflow-style YOLO export with only train/images and train/labels
  into a standard YOLO dataset layout:

    images/train
    images/val
    images/test
    labels/train
    labels/val
    labels/test

This script copies files. It does not delete or modify the original dataset.

Example:

python -m yolo_research.scripts.split_yolo_dataset \
  --source yolo_research/datasets/project_after_merging.yolo26 \
  --output yolo_research/datasets/traffic_vehicle_v1 \
  --train-ratio 0.7 \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  --seed 42
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a train-only Roboflow YOLO dataset into train/val/test."
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Source Roboflow YOLO dataset folder. Example: yolo_research/datasets/project_after_merging.yolo26",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output standard YOLO dataset folder. Example: yolo_research/datasets/traffic_vehicle_v1",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training split ratio.",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )

    return parser.parse_args()


def find_input_dirs(source_dir: Path) -> tuple[Path, Path]:
    """
    Expected Roboflow input:

        source/
          train/
            images/
            labels/
          data.yaml
    """

    image_dir = source_dir / "train" / "images"
    label_dir = source_dir / "train" / "labels"

    if not image_dir.exists():
        raise FileNotFoundError(f"Input image directory not found: {image_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"Input label directory not found: {label_dir}")

    return image_dir, label_dir


def list_image_files(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def copy_pair(image_path: Path, label_path: Path, output_dir: Path, split: str) -> None:
    image_out_dir = output_dir / "images" / split
    label_out_dir = output_dir / "labels" / split

    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, image_out_dir / image_path.name)

    if label_path.exists():
        shutil.copy2(label_path, label_out_dir / label_path.name)
    else:
        # Create empty label file if image has no objects.
        # For traffic dataset this should be rare, but YOLO accepts empty .txt labels.
        (label_out_dir / f"{image_path.stem}.txt").write_text("", encoding="utf-8")


def write_data_yaml(output_dir: Path) -> None:
    data = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "car",
            1: "motorbike",
            2: "truck",
            3: "bus",
        },
    }

    with (output_dir / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    args = parse_args()

    source_dir = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio

    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {ratio_sum}"
        )

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                f"Use --overwrite if you want to recreate it."
            )

    image_dir, label_dir = find_input_dirs(source_dir)
    image_files = list_image_files(image_dir)

    if not image_files:
        raise RuntimeError(f"No image files found in {image_dir}")

    random.seed(args.seed)
    random.shuffle(image_files)

    total = len(image_files)
    train_count = int(total * args.train_ratio)
    val_count = int(total * args.val_ratio)

    train_files = image_files[:train_count]
    val_files = image_files[train_count: train_count + val_count]
    test_files = image_files[train_count + val_count:]

    split_map = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    print("=" * 80)
    print("ATLC YOLO DATASET SPLIT")
    print("=" * 80)
    print(f"source: {source_dir}")
    print(f"output: {output_dir}")
    print(f"total images: {total}")
    print(f"train: {len(train_files)}")
    print(f"val: {len(val_files)}")
    print(f"test: {len(test_files)}")
    print("=" * 80)

    for split, files in split_map.items():
        for image_path in files:
            label_path = label_dir / f"{image_path.stem}.txt"
            copy_pair(
                image_path=image_path,
                label_path=label_path,
                output_dir=output_dir,
                split=split,
            )

    write_data_yaml(output_dir)

    print("Done.")
    print(f"Dataset written to: {output_dir}")
    print(f"data.yaml written to: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()