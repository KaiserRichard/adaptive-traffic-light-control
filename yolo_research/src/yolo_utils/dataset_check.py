"""
dataset_check.py

Validate a YOLO object detection dataset before training.

Current ATLC custom dataset classes:
    0 car
    1 motorbike
    2 truck
    3 bus
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from yolo_research.src.yolo_utils.yolo_io import (
    infer_label_dir_from_image_dir,
    label_path_for_image,
    list_image_files,
    list_label_files,
    load_yaml,
    normalize_class_names,
    resolve_split_image_dir,
)

MAX_EXAMPLES_PER_ERROR_TYPE = 20


def _add_limited_issue(
    issues: list[dict[str, Any]],
    issue_type: str,
    path: Path,
    message: str,
    line_number: int | None = None,
) -> None:
    """Store only a limited number of examples per issue type."""
    existing_count = sum(1 for issue in issues if issue["type"] == issue_type)
    if existing_count >= MAX_EXAMPLES_PER_ERROR_TYPE:
        return
    item: dict[str, Any] = {"type": issue_type, "path": str(path), "message": message}
    if line_number is not None:
        item["line_number"] = line_number
    issues.append(item)


def _parse_label_line(
    line: str,
    label_path: Path,
    line_number: int,
    valid_class_ids: set[int],
    issues: list[dict[str, Any]],
) -> tuple[int | None, float | None, float | None, float | None, float | None]:
    """Parse one YOLO label line: class_id x_center y_center width height."""
    parts = line.strip().split()
    if len(parts) != 5:
        _add_limited_issue(
            issues,
            "invalid_label_format",
            label_path,
            f"Expected 5 values, got {len(parts)}: {line.strip()}",
            line_number,
        )
        return None, None, None, None, None

    try:
        class_id = int(parts[0])
    except ValueError:
        _add_limited_issue(issues, "invalid_class_id_format", label_path, f"class_id must be integer: {parts[0]}", line_number)
        return None, None, None, None, None

    try:
        x_center, y_center, width, height = map(float, parts[1:])
    except ValueError:
        _add_limited_issue(issues, "invalid_bbox_number", label_path, f"bbox values must be numbers: {line.strip()}", line_number)
        return None, None, None, None, None

    if class_id not in valid_class_ids:
        _add_limited_issue(issues, "invalid_class_id", label_path, f"class_id={class_id} is not valid. Valid IDs: {sorted(valid_class_ids)}", line_number)
        return None, None, None, None, None

    bbox_values = [x_center, y_center, width, height]
    if any(value < 0.0 or value > 1.0 for value in bbox_values):
        _add_limited_issue(issues, "bbox_out_of_range", label_path, f"bbox values must be normalized to [0,1]: {bbox_values}", line_number)
        return None, None, None, None, None

    if width <= 0.0 or height <= 0.0:
        _add_limited_issue(issues, "invalid_bbox_size", label_path, f"bbox width/height must be positive: width={width}, height={height}", line_number)
        return None, None, None, None, None

    return class_id, x_center, y_center, width, height


def check_split(split: str, image_dir: Path, label_dir: Path, class_names: dict[int, str]) -> dict[str, Any]:
    """Validate one split: train, val, or test."""
    valid_class_ids = set(class_names.keys())
    issues: list[dict[str, Any]] = []

    image_files = list_image_files(image_dir)
    label_files = list_label_files(label_dir)

    image_stems = {path.stem for path in image_files}
    label_stems = {path.stem for path in label_files}

    missing_label_stems = sorted(image_stems - label_stems)
    label_without_image_stems = sorted(label_stems - image_stems)

    for stem in missing_label_stems[:MAX_EXAMPLES_PER_ERROR_TYPE]:
        _add_limited_issue(issues, "image_missing_label", image_dir / f"{stem}.*", f"Image stem '{stem}' has no matching label file.")

    for stem in label_without_image_stems[:MAX_EXAMPLES_PER_ERROR_TYPE]:
        _add_limited_issue(issues, "label_without_image", label_dir / f"{stem}.txt", f"Label stem '{stem}' has no matching image file.")

    class_counter: Counter[int] = Counter()
    bbox_widths: list[float] = []
    bbox_heights: list[float] = []
    bbox_areas: list[float] = []

    empty_label_files = 0
    valid_object_count = 0
    invalid_line_count = 0

    for image_path in image_files:
        label_path = label_path_for_image(image_path, label_dir)
        if not label_path.exists():
            continue

        raw_text = label_path.read_text(encoding="utf-8").strip()
        if not raw_text:
            empty_label_files += 1
            _add_limited_issue(issues, "empty_label_file", label_path, "Label file exists but contains no objects.")
            continue

        for line_number, line in enumerate(raw_text.splitlines(), start=1):
            if not line.strip():
                continue
            class_id, _, _, width, height = _parse_label_line(line, label_path, line_number, valid_class_ids, issues)
            if class_id is None:
                invalid_line_count += 1
                continue
            class_counter[class_id] += 1
            valid_object_count += 1
            bbox_widths.append(float(width))
            bbox_heights.append(float(height))
            bbox_areas.append(float(width) * float(height))

    class_counts_by_name = {class_names[class_id]: int(class_counter.get(class_id, 0)) for class_id in sorted(class_names.keys())}

    return {
        "split": split,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "image_count": len(image_files),
        "label_count": len(label_files),
        "missing_label_count": len(missing_label_stems),
        "label_without_image_count": len(label_without_image_stems),
        "empty_label_file_count": empty_label_files,
        "valid_object_count": valid_object_count,
        "invalid_line_count": invalid_line_count,
        "class_counts": class_counts_by_name,
        "bbox_stats": {
            "count": len(bbox_areas),
            "min_area": min(bbox_areas) if bbox_areas else None,
            "max_area": max(bbox_areas) if bbox_areas else None,
            "avg_area": sum(bbox_areas) / len(bbox_areas) if bbox_areas else None,
            "avg_width": sum(bbox_widths) / len(bbox_widths) if bbox_widths else None,
            "avg_height": sum(bbox_heights) / len(bbox_heights) if bbox_heights else None,
        },
        "issues": issues,
    }


def check_dataset(data_yaml_path: str | Path) -> dict[str, Any]:
    """Validate a YOLO dataset using data.yaml and return summary."""
    data_yaml_path = Path(data_yaml_path)
    data_yaml = load_yaml(data_yaml_path)
    class_names = normalize_class_names(data_yaml.get("names"))

    split_summaries = {}
    for split in ["train", "val", "test"]:
        if split not in data_yaml:
            if split in {"train", "val"}:
                raise ValueError(f"data.yaml is missing required split: {split}")
            continue
        image_dir = resolve_split_image_dir(data_yaml_path, data_yaml, split)
        label_dir = infer_label_dir_from_image_dir(image_dir, split)
        split_summaries[split] = check_split(split, image_dir, label_dir, class_names)

    total_class_counts: Counter[str] = Counter()
    for split_summary in split_summaries.values():
        for class_name, count in split_summary["class_counts"].items():
            total_class_counts[class_name] += int(count)

    return {
        "data_yaml": str(data_yaml_path),
        "class_names": class_names,
        "splits": split_summaries,
        "totals": {
            "image_count": sum(item["image_count"] for item in split_summaries.values()),
            "label_count": sum(item["label_count"] for item in split_summaries.values()),
            "valid_object_count": sum(item["valid_object_count"] for item in split_summaries.values()),
            "invalid_line_count": sum(item["invalid_line_count"] for item in split_summaries.values()),
            "missing_label_count": sum(item["missing_label_count"] for item in split_summaries.values()),
            "label_without_image_count": sum(item["label_without_image_count"] for item in split_summaries.values()),
            "class_counts": dict(total_class_counts),
        },
    }


def print_dataset_summary(summary: dict[str, Any]) -> None:
    """Print clean terminal summary."""
    print("=" * 80)
    print("YOLO DATASET CHECK SUMMARY")
    print("=" * 80)
    print(f"data.yaml: {summary['data_yaml']}")

    print("\nClasses:")
    for class_id, class_name in summary["class_names"].items():
        print(f"  {class_id}: {class_name}")

    print("\nSplit Summary:")
    for split, split_summary in summary["splits"].items():
        print("-" * 80)
        print(f"Split: {split}")
        print(f"Images: {split_summary['image_count']}")
        print(f"Labels: {split_summary['label_count']}")
        print(f"Valid objects: {split_summary['valid_object_count']}")
        print(f"Missing labels: {split_summary['missing_label_count']}")
        print(f"Labels without images: {split_summary['label_without_image_count']}")
        print(f"Empty label files: {split_summary['empty_label_file_count']}")
        print(f"Invalid label lines: {split_summary['invalid_line_count']}")
        print("Class counts:")
        for class_name, count in split_summary["class_counts"].items():
            print(f"  {class_name}: {count}")
        bbox_stats = split_summary["bbox_stats"]
        if bbox_stats["count"] > 0:
            print("BBox stats:")
            print(f"  avg area: {bbox_stats['avg_area']:.6f}")
            print(f"  avg width: {bbox_stats['avg_width']:.6f}")
            print(f"  avg height: {bbox_stats['avg_height']:.6f}")
        if split_summary["issues"]:
            print("Warnings / issues:")
            for issue in split_summary["issues"][:10]:
                line_info = f":line {issue['line_number']}" if "line_number" in issue else ""
                print(f"  [WARN] {issue['type']} {issue['path']}{line_info} - {issue['message']}")
            if len(split_summary["issues"]) > 10:
                print(f"  ... and {len(split_summary['issues']) - 10} more issues")

    print("-" * 80)
    print("Totals:")
    for key, value in summary["totals"].items():
        if key != "class_counts":
            print(f"{key}: {value}")
    print("Total class counts:")
    for class_name, count in summary["totals"]["class_counts"].items():
        print(f"  {class_name}: {count}")
    print("=" * 80)

    has_errors = (
        summary["totals"]["missing_label_count"] > 0
        or summary["totals"]["label_without_image_count"] > 0
        or summary["totals"]["invalid_line_count"] > 0
    )
    print("[WARN] Dataset has issues. Fix them before serious training." if has_errors else "[OK] Dataset structure looks valid.")
