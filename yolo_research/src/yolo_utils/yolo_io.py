"""
yolo_io.py

Shared helper functions for YOLO research scripts.

This module keeps YAML loading, path resolution, image/label listing,
and JSON saving in one place so training, validation, evaluation, and
plotting scripts do not duplicate path logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML and return a dictionary."""
    yaml_path = Path(path).expanduser()
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"YAML file is empty: {yaml_path}")
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping/dictionary: {yaml_path}")
    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as YAML."""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as formatted JSON."""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as Path."""
    output_dir = Path(path).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def normalize_class_names(names: Any) -> dict[int, str]:
    """
    Normalize YOLO data.yaml names into {0: "car", 1: "motorbike", ...}.

    Ultralytics accepts either:
    - list: ["car", "motorbike", ...]
    - dict: {0: "car", 1: "motorbike", ...}
    """
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    raise ValueError("data.yaml field 'names' must be a list or dictionary.")


def get_dataset_root(data_yaml_path: str | Path, data_yaml: dict[str, Any]) -> Path:
    """Resolve dataset root from data.yaml `path` field."""
    data_yaml_path = Path(data_yaml_path).expanduser()
    raw_root = data_yaml.get("path")
    if raw_root is None:
        raise ValueError("data.yaml must contain a 'path' field.")
    root = Path(str(raw_root)).expanduser()
    if not root.is_absolute():
        root = (data_yaml_path.parent / root).resolve()
    return root


def resolve_split_image_dir(data_yaml_path: str | Path, data_yaml: dict[str, Any], split: str) -> Path:
    """Resolve the images directory for a split such as train, val, or test."""
    if split not in data_yaml:
        raise ValueError(f"data.yaml missing split field: {split}")
    dataset_root = get_dataset_root(data_yaml_path, data_yaml)
    split_value = Path(str(data_yaml[split])).expanduser()
    if split_value.is_absolute():
        return split_value
    return dataset_root / split_value


def infer_label_dir_from_image_dir(image_dir: str | Path, split: str) -> Path:
    """
    Infer YOLO label directory from image directory.

    Preferred layout:
        images/train -> labels/train
    """
    image_dir = Path(image_dir).expanduser()
    parts = list(image_dir.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts)
    return image_dir.parent.parent / "labels" / split


def list_image_files(image_dir: str | Path) -> list[Path]:
    """Return sorted image files in one directory."""
    image_dir = Path(image_dir).expanduser()
    if not image_dir.exists():
        return []
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def list_label_files(label_dir: str | Path) -> list[Path]:
    """Return sorted YOLO label .txt files in one directory."""
    label_dir = Path(label_dir).expanduser()
    if not label_dir.exists():
        return []
    return sorted(path for path in label_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt")


def label_path_for_image(image_path: str | Path, label_dir: str | Path) -> Path:
    """Return expected label path for an image."""
    image_path = Path(image_path)
    label_dir = Path(label_dir)
    return label_dir / f"{image_path.stem}.txt"
