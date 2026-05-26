"""
plotting.py

Generate report-ready PNG evidence for ATLC Phase 11.

Evidence groups:
A. Dataset/model evidence
B. Embedded system/control evidence

Rule:
Every figure should answer an engineering question. Missing input files should
produce warnings, not crashes.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from yolo_research.src.yolo_utils.yolo_io import (
    infer_label_dir_from_image_dir,
    list_image_files,
    list_label_files,
    load_yaml,
    normalize_class_names,
    resolve_split_image_dir,
)

FIGURE_DPI = 200


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def save_figure(path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {output_path}")


def _read_yolo_labels(label_dir: Path, valid_class_ids: set[int]) -> list[dict[str, Any]]:
    rows = []
    for label_path in list_label_files(label_dir):
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        for line in text.splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
            except ValueError:
                continue
            if class_id not in valid_class_ids:
                continue
            if not all(0.0 <= v <= 1.0 for v in [x_center, y_center, width, height]):
                continue
            if width <= 0 or height <= 0:
                continue
            rows.append({
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "area": width * height,
                "label_file": str(label_path),
            })
    return rows


def _load_all_label_rows(data_yaml_path: str | Path) -> tuple[pd.DataFrame, dict[int, str]]:
    data_yaml_path = Path(data_yaml_path)
    data_yaml = load_yaml(data_yaml_path)
    class_names = normalize_class_names(data_yaml["names"])
    valid_class_ids = set(class_names.keys())
    all_rows = []
    for split in ["train", "val", "test"]:
        if split not in data_yaml:
            continue
        image_dir = resolve_split_image_dir(data_yaml_path, data_yaml, split)
        label_dir = infer_label_dir_from_image_dir(image_dir, split)
        rows = _read_yolo_labels(label_dir, valid_class_ids)
        for row in rows:
            row["split"] = split
        all_rows.extend(rows)
    return pd.DataFrame(all_rows), class_names


def plot_images_per_split(data_yaml_path: str | Path, outdir: str | Path) -> Path | None:
    data_yaml_path = Path(data_yaml_path)
    outdir = Path(outdir)
    data_yaml = load_yaml(data_yaml_path)
    split_counts = {}
    for split in ["train", "val", "test"]:
        if split not in data_yaml:
            continue
        image_dir = resolve_split_image_dir(data_yaml_path, data_yaml, split)
        split_counts[split] = len(list_image_files(image_dir))
    if not split_counts:
        warn("No split image directories found. Skipping images_per_split.png.")
        return None
    plt.figure(figsize=(7, 5))
    plt.bar(split_counts.keys(), split_counts.values())
    plt.title("Images per Dataset Split")
    plt.xlabel("Split")
    plt.ylabel("Number of Images")
    for idx, value in enumerate(split_counts.values()):
        plt.text(idx, value, str(value), ha="center", va="bottom")
    output_path = outdir / "01_images_per_split.png"
    save_figure(output_path)
    return output_path


def plot_class_distribution(data_yaml_path: str | Path, outdir: str | Path) -> Path | None:
    outdir = Path(outdir)
    df, class_names = _load_all_label_rows(data_yaml_path)
    if df.empty:
        warn("No valid labels found. Skipping class_distribution.png.")
        return None
    counts = df["class_id"].value_counts().to_dict()
    labels = [class_names[class_id] for class_id in sorted(class_names)]
    values = [counts.get(class_id, 0) for class_id in sorted(class_names)]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title("Object Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Object Count")
    for idx, value in enumerate(values):
        plt.text(idx, value, str(value), ha="center", va="bottom")
    output_path = outdir / "02_class_distribution.png"
    save_figure(output_path)
    return output_path


def plot_bbox_size_distribution(data_yaml_path: str | Path, outdir: str | Path) -> Path | None:
    outdir = Path(outdir)
    df, class_names = _load_all_label_rows(data_yaml_path)
    if df.empty:
        warn("No valid labels found. Skipping bbox_size_distribution.png.")
        return None
    plt.figure(figsize=(8, 5))
    for class_id in sorted(class_names):
        subset = df[df["class_id"] == class_id]
        if not subset.empty:
            plt.hist(subset["area"], bins=40, alpha=0.5, label=class_names[class_id])
    plt.title("Bounding Box Area Distribution")
    plt.xlabel("Normalized BBox Area")
    plt.ylabel("Object Count")
    plt.legend()
    output_path = outdir / "03_bbox_size_distribution.png"
    save_figure(output_path)
    return output_path


def plot_sample_labels_grid(data_yaml_path: str | Path, outdir: str | Path, split: str = "train", max_images: int = 9) -> Path | None:
    data_yaml_path = Path(data_yaml_path)
    outdir = Path(outdir)
    data_yaml = load_yaml(data_yaml_path)
    class_names = normalize_class_names(data_yaml["names"])
    if split not in data_yaml:
        warn(f"Split '{split}' not found. Skipping sample_labels_grid.png.")
        return None
    image_dir = resolve_split_image_dir(data_yaml_path, data_yaml, split)
    label_dir = infer_label_dir_from_image_dir(image_dir, split)
    image_files = list_image_files(image_dir)
    if not image_files:
        warn(f"No images found in {image_dir}. Skipping sample_labels_grid.png.")
        return None
    selected_images = image_files[:max_images]
    cols = 3
    rows = int(np.ceil(len(selected_images) / cols))
    plt.figure(figsize=(cols * 5, rows * 4))
    for idx, image_path in enumerate(selected_images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            text = label_path.read_text(encoding="utf-8").strip()
            for line in text.splitlines() if text else []:
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    class_id = int(parts[0])
                    x_center, y_center, bw, bh = map(float, parts[1:])
                except ValueError:
                    continue
                x1 = int((x_center - bw / 2) * w)
                y1 = int((y_center - bh / 2) * h)
                x2 = int((x_center + bw / 2) * w)
                y2 = int((y_center + bh / 2) * h)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, class_names.get(class_id, str(class_id)), (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        plt.subplot(rows, cols, idx)
        plt.imshow(image)
        plt.title(image_path.name)
        plt.axis("off")
    plt.suptitle(f"Sample YOLO Labels Grid ({split})")
    plt.tight_layout()
    output_path = outdir / "04_sample_labels_grid.png"
    save_figure(output_path)
    return output_path


def _load_results_csv(run_dir: str | Path) -> pd.DataFrame | None:
    run_dir = Path(run_dir)
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        warn(f"results.csv not found at {results_csv}. Skipping training curve plots.")
        return None
    df = pd.read_csv(results_csv)
    df.columns = [col.strip() for col in df.columns]
    return df


def plot_train_loss_curve(run_dir: str | Path, outdir: str | Path) -> Path | None:
    outdir = Path(outdir)
    df = _load_results_csv(run_dir)
    if df is None or df.empty:
        return None
    epoch_col = "epoch" if "epoch" in df.columns else None
    loss_cols = [col for col in df.columns if "train/" in col and "loss" in col]
    if not loss_cols:
        warn("No train loss columns found in results.csv. Skipping train_loss_curve.png.")
        return None
    x = df[epoch_col] if epoch_col else range(len(df))
    plt.figure(figsize=(9, 5))
    for col in loss_cols:
        plt.plot(x, df[col], label=col)
    plt.title("YOLO Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path = outdir / "05_train_loss_curve.png"
    save_figure(output_path)
    return output_path


def plot_metrics_summary_bar(run_dir: str | Path, outdir: str | Path) -> Path | None:
    outdir = Path(outdir)
    df = _load_results_csv(run_dir)
    if df is None or df.empty:
        return None
    metric_candidates = {
        "Precision": ["metrics/precision(B)", "metrics/precision"],
        "Recall": ["metrics/recall(B)", "metrics/recall"],
        "mAP50": ["metrics/mAP50(B)", "metrics/mAP50"],
        "mAP50-95": ["metrics/mAP50-95(B)", "metrics/mAP50-95"],
    }
    last_row = df.iloc[-1]
    labels, values = [], []
    for label, candidates in metric_candidates.items():
        found_col = next((col for col in candidates if col in df.columns), None)
        if found_col is not None:
            labels.append(label)
            values.append(float(last_row[found_col]))
    if not labels:
        warn("No metric columns found in results.csv. Skipping metrics_summary_bar.png.")
        return None
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title("Final YOLO Validation Metrics")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    for idx, value in enumerate(values):
        plt.text(idx, value, f"{value:.3f}", ha="center", va="bottom")
    output_path = outdir / "06_metrics_summary_bar.png"
    save_figure(output_path)
    return output_path


def copy_ultralytics_figures(run_dir: str | Path, outdir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "confusion_matrix.png": "07_confusion_matrix.png",
        "confusion_matrix_normalized.png": "08_confusion_matrix_normalized.png",
        "PR_curve.png": "09_precision_recall_curve.png",
        "F1_curve.png": "10_f1_curve.png",
        "results.png": "11_ultralytics_results_overview.png",
    }
    copied = []
    for src_name, dst_name in mapping.items():
        src = run_dir / src_name
        if not src.exists():
            warn(f"{src_name} not found in {run_dir}. Skipping.")
            continue
        dst = outdir / dst_name
        shutil.copy2(src, dst)
        print(f"[OK] copied {src} -> {dst}")
        copied.append(dst)
    return copied


def plot_prediction_samples_grid(predictions_dir: str | Path, outdir: str | Path, max_images: int = 9) -> Path | None:
    predictions_dir = Path(predictions_dir)
    outdir = Path(outdir)
    if not predictions_dir.exists():
        warn(f"Predictions directory not found: {predictions_dir}. Skipping prediction_samples_grid.png.")
        return None
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        image_files.extend(predictions_dir.rglob(ext))
    image_files = sorted(image_files)
    if not image_files:
        warn(f"No prediction images found in {predictions_dir}. Skipping prediction_samples_grid.png.")
        return None
    selected_images = image_files[:max_images]
    cols = 3
    rows = int(np.ceil(len(selected_images) / cols))
    plt.figure(figsize=(cols * 5, rows * 4))
    for idx, image_path in enumerate(selected_images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(rows, cols, idx)
        plt.imshow(image)
        plt.title(image_path.name)
        plt.axis("off")
    plt.suptitle("Prediction Samples Grid")
    plt.tight_layout()
    output_path = outdir / "12_prediction_samples_grid.png"
    save_figure(output_path)
    return output_path


def _read_optional_csv(path: str | Path, description: str) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        warn(f"{description} not found: {path}. Skipping related plots.")
        return None
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    if df.empty:
        warn(f"{description} is empty: {path}. Skipping related plots.")
        return None
    return df


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def plot_density_over_time(pipeline_log_csv: str | Path, outdir: str | Path) -> Path | None:
    df = _read_optional_csv(pipeline_log_csv, "pipeline log")
    outdir = Path(outdir)
    if df is None:
        return None
    x_col = _find_col(df, ["frame", "frame_index", "time_s", "timestamp"])
    a_col = _find_col(df, ["smoothed_pce_a", "density_a", "direction_A_density"])
    b_col = _find_col(df, ["smoothed_pce_b", "density_b", "direction_B_density"])
    if x_col is None or a_col is None or b_col is None:
        warn("Missing columns for density_over_time.png.")
        return None
    plt.figure(figsize=(10, 5))
    plt.plot(df[x_col], df[a_col], label="Direction A")
    plt.plot(df[x_col], df[b_col], label="Direction B")
    plt.title("Traffic Density Over Time")
    plt.xlabel(x_col)
    plt.ylabel("Smoothed PCE Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path = outdir / "13_density_over_time.png"
    save_figure(output_path)
    return output_path


def plot_green_time_over_time(pipeline_log_csv: str | Path, outdir: str | Path) -> Path | None:
    df = _read_optional_csv(pipeline_log_csv, "pipeline log")
    outdir = Path(outdir)
    if df is None:
        return None
    x_col = _find_col(df, ["frame", "frame_index", "time_s", "timestamp"])
    a_col = _find_col(df, ["green_a", "active_green_a"])
    b_col = _find_col(df, ["green_b", "active_green_b"])
    if x_col is None or a_col is None or b_col is None:
        warn("Missing columns for green_time_over_time.png.")
        return None
    plt.figure(figsize=(10, 5))
    plt.plot(df[x_col], df[a_col], label="Green A")
    plt.plot(df[x_col], df[b_col], label="Green B")
    plt.title("Adaptive Green Time Over Time")
    plt.xlabel(x_col)
    plt.ylabel("Green Time (seconds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path = outdir / "14_green_time_over_time.png"
    save_figure(output_path)
    return output_path


def plot_signal_state_timeline(pipeline_log_csv: str | Path, outdir: str | Path) -> Path | None:
    df = _read_optional_csv(pipeline_log_csv, "pipeline log")
    outdir = Path(outdir)
    if df is None:
        return None
    x_col = _find_col(df, ["frame", "frame_index", "time_s", "timestamp"])
    state_col = _find_col(df, ["runtime_state", "state", "signal_state"])
    if x_col is None or state_col is None:
        warn("Missing columns for signal_state_timeline.png.")
        return None
    states = list(dict.fromkeys(df[state_col].astype(str).tolist()))
    state_to_id = {state: idx for idx, state in enumerate(states)}
    y = df[state_col].astype(str).map(state_to_id)
    plt.figure(figsize=(11, 5))
    plt.step(df[x_col], y, where="post")
    plt.yticks(list(state_to_id.values()), list(state_to_id.keys()))
    plt.title("Signal Runtime State Timeline")
    plt.xlabel(x_col)
    plt.ylabel("Traffic Signal State")
    plt.grid(True, alpha=0.3)
    output_path = outdir / "15_signal_state_timeline.png"
    save_figure(output_path)
    return output_path


def plot_pipeline_runtime_breakdown(profile_log_csv: str | Path, outdir: str | Path) -> Path | None:
    df = _read_optional_csv(profile_log_csv, "profile log")
    outdir = Path(outdir)
    if df is None:
        return None
    timing_cols = [col for col in ["read_ms", "detect_ms", "logic_ms", "uart_ms", "draw_ms", "writer_ms", "display_ms"] if col in df.columns]
    if not timing_cols:
        warn("No timing columns found for pipeline_runtime_breakdown.png.")
        return None
    avg_values = df[timing_cols].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(avg_values.index, avg_values.values)
    plt.title("Average Pipeline Runtime Breakdown")
    plt.xlabel("Pipeline Stage")
    plt.ylabel("Average Time (ms)")
    plt.xticks(rotation=30, ha="right")
    for idx, value in enumerate(avg_values.values):
        plt.text(idx, value, f"{value:.1f}", ha="center", va="bottom")
    output_path = outdir / "16_pipeline_runtime_breakdown.png"
    save_figure(output_path)
    return output_path


def plot_uart_ack_latency(uart_log_csv: str | Path, outdir: str | Path) -> Path | None:
    df = _read_optional_csv(uart_log_csv, "UART log")
    outdir = Path(outdir)
    if df is None:
        return None
    x_col = _find_col(df, ["plan_id", "message_id", "frame"])
    latency_col = _find_col(df, ["latency_ms", "ack_latency_ms"])
    if x_col is None or latency_col is None:
        warn("Missing columns for uart_ack_latency.png.")
        return None
    plt.figure(figsize=(10, 5))
    plt.plot(df[x_col], df[latency_col], marker="o")
    plt.title("UART ACK Latency")
    plt.xlabel(x_col)
    plt.ylabel("Latency (ms)")
    plt.grid(True, alpha=0.3)
    output_path = outdir / "17_uart_ack_latency.png"
    save_figure(output_path)
    return output_path


def plot_uart_success_rate(uart_log_csv: str | Path, outdir: str | Path) -> Path | None:
    df = _read_optional_csv(uart_log_csv, "UART log")
    outdir = Path(outdir)
    if df is None:
        return None
    status_col = _find_col(df, ["success", "status", "uart"])
    if status_col is None:
        warn("Missing status column for uart_success_rate.png.")
        return None
    counts = df[status_col].astype(str).value_counts()
    plt.figure(figsize=(7, 5))
    plt.bar(counts.index, counts.values)
    plt.title("UART Communication Result Counts")
    plt.xlabel("Result")
    plt.ylabel("Count")
    for idx, value in enumerate(counts.values):
        plt.text(idx, value, str(value), ha="center", va="bottom")
    output_path = outdir / "18_uart_success_rate.png"
    save_figure(output_path)
    return output_path


def write_plot_metadata_table(outdir: str | Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [
        ["01_images_per_split.png", "data.yaml + image folders", "Split", "Image count", "Is the dataset split reasonable?", "Train should contain most images; val/test should be large enough for evaluation."],
        ["02_class_distribution.png", "YOLO label files", "Class name", "Object count", "Is the dataset balanced?", "If motorbike count is much lower than car count, motorbike recall may suffer."],
        ["03_bbox_size_distribution.png", "YOLO label files", "Normalized bbox area", "Object count", "Are vehicles too small?", "Many tiny boxes suggest higher image size or closer camera view may be needed."],
        ["04_sample_labels_grid.png", "Images + YOLO labels", "N/A", "N/A", "Are labels visually correct?", "Bad boxes or wrong classes should be fixed before training."],
        ["05_train_loss_curve.png", "results.csv", "Epoch", "Loss", "Did training converge?", "Loss should generally decrease and stabilize."],
        ["06_metrics_summary_bar.png", "results.csv", "Metric", "Score", "What is final model quality?", "Low recall may indicate missed vehicles."],
        ["07_confusion_matrix.png", "Ultralytics validation output", "Predicted class", "True class", "Which classes are confused?", "Motorbike confusion with car/truck suggests more local data or better labels are needed."],
        ["09_precision_recall_curve.png", "Ultralytics validation output", "Recall", "Precision", "What confidence trade-off exists?", "This helps choose CONFIDENCE_THRESHOLD for runtime inference."],
        ["12_prediction_samples_grid.png", "Prediction images", "N/A", "N/A", "Does the model look correct qualitatively?", "Check missed motorbikes, false positives, and crowded-scene behavior."],
        ["13_density_over_time.png", "pipeline_log.csv", "Frame/time", "Smoothed PCE density", "Does detection become a stable control signal?", "Density should reflect traffic load and should not jump randomly."],
        ["14_green_time_over_time.png", "pipeline_log.csv", "Frame/time", "Green seconds", "Does scheduler adapt timing?", "Higher-density direction should receive longer green time."],
        ["15_signal_state_timeline.png", "pipeline_log.csv", "Frame/time", "Signal state", "Does runtime FSM execute stably?", "State order should follow stable traffic-light phases."],
        ["16_pipeline_runtime_breakdown.png", "profile_log.csv", "Pipeline stage", "Average ms", "Where is the runtime bottleneck?", "YOLO/display/writer should dominate; control logic should stay lightweight."],
        ["17_uart_ack_latency.png", "uart_log.csv", "Plan ID/frame", "Latency ms", "Is MCU communication stable?", "Low and stable latency means UART communication is reliable."],
        ["18_uart_success_rate.png", "uart_log.csv", "Result", "Count", "How reliable is UART communication?", "Most messages should be successful ACKs; timeouts/errors should be rare."],
    ]
    output_path = outdir / "plot_metadata.csv"
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PNG file", "Source data", "X-axis", "Y-axis", "Engineering question", "Interpretation template"])
        writer.writerows(rows)
    print(f"[OK] saved {output_path}")
    return output_path
