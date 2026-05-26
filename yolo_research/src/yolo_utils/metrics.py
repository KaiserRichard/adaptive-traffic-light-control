"""
metrics.py

Save YOLO validation metrics into stable JSON files for report writing and
model comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yolo_research.src.yolo_utils.yolo_io import save_json


def _safe_getattr(obj: Any, attr_name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, attr_name)
    except Exception:
        return default


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def extract_yolo_val_metrics(val_results: Any) -> dict[str, Any]:
    """
    Extract common object-detection validation metrics from Ultralytics results.

    Defensive extraction is used because the exact result object can change
    slightly across Ultralytics versions.
    """
    box = _safe_getattr(val_results, "box", None)
    metrics: dict[str, Any] = {
        "precision": None,
        "recall": None,
        "map50": None,
        "map50_95": None,
        "per_class_map50_95": None,
    }

    if box is not None:
        metrics["precision"] = _safe_float(_safe_getattr(box, "mp", None))
        metrics["recall"] = _safe_float(_safe_getattr(box, "mr", None))
        metrics["map50"] = _safe_float(_safe_getattr(box, "map50", None))
        metrics["map50_95"] = _safe_float(_safe_getattr(box, "map", None))
        maps = _safe_getattr(box, "maps", None)
        if maps is not None:
            try:
                metrics["per_class_map50_95"] = [float(x) for x in maps]
            except Exception:
                metrics["per_class_map50_95"] = None

    results_dict = _safe_getattr(val_results, "results_dict", None)
    if isinstance(results_dict, dict):
        metrics["raw_results_dict"] = {
            str(key): _safe_float(value) if isinstance(value, (int, float)) else str(value)
            for key, value in results_dict.items()
        }

    return metrics


def save_yolo_val_metrics(val_results: Any, output_path: str | Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract validation metrics and save as JSON."""
    metrics = extract_yolo_val_metrics(val_results)
    if extra is not None:
        metrics["extra"] = extra
    save_json(metrics, output_path)
    return metrics
