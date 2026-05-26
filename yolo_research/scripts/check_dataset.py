"""
Command-line dataset validation.

Run from repo root:
    python -m yolo_research.scripts.check_dataset --data yolo_research/configs/data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from yolo_research.src.yolo_utils.dataset_check import check_dataset, print_dataset_summary
from yolo_research.src.yolo_utils.yolo_io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an ATLC YOLO-format dataset.")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml file.")
    parser.add_argument("--out", default="yolo_research/outputs/logs/dataset_check_summary.json", help="Where to save JSON summary.")
    parser.add_argument("--no-save", action="store_true", help="Do not save JSON summary.")
    parser.add_argument("--strict", action="store_true", help="Exit with error code if dataset issues are found.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = check_dataset(args.data)
    print_dataset_summary(summary)
    if not args.no_save:
        save_json(summary, args.out)
        print(f"Saved dataset summary to: {Path(args.out)}")
    has_errors = (
        summary["totals"]["missing_label_count"] > 0
        or summary["totals"]["label_without_image_count"] > 0
        or summary["totals"]["invalid_line_count"] > 0
    )
    if args.strict and has_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
