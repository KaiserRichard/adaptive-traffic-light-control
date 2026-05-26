"""
Copy/export the best YOLO weights into the ATLC runtime model folder.

This script intentionally copies .pt locally; .pt files should stay ignored by Git.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export custom YOLO weights into pc_app/models/local.")
    parser.add_argument("--weights", default="yolo_research/outputs/runs/atlc_yolov8n_custom/weights/best.pt", help="Source weights path.")
    parser.add_argument("--out", default="pc_app/models/local/atlc_yolov8n_custom.pt", help="Destination runtime model path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.weights)
    dst = Path(args.out)
    if not src.exists():
        raise FileNotFoundError(f"Weights not found: {src}")
    if dst.exists() and not args.overwrite:
        raise FileExistsError(f"Destination already exists: {dst}. Use --overwrite to replace it.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print("=" * 80)
    print("ATLC YOLO MODEL EXPORT")
    print("=" * 80)
    print(f"Copied from: {src}")
    print(f"Copied to:   {dst}")
    print("Suggested .env:")
    print("DETECTOR_BACKEND=yolo")
    print(f"YOLO_MODEL_PATH=./{dst.as_posix()}")
    print("YOLO_IMGSZ=640")
    print("CONFIDENCE_THRESHOLD=0.25")


if __name__ == "__main__":
    main()
