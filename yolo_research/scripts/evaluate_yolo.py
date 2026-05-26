"""
Evaluate a trained YOLO model and save structured metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from yolo_research.src.yolo_utils.metrics import save_yolo_val_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model for ATLC.")
    parser.add_argument("--weights", required=True, help="Path to trained weights, for example best.pt.")
    parser.add_argument("--data", default="yolo_research/configs/data.yaml", help="Path to YOLO data.yaml.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--device", default="0", help="Device: 0 for GPU, cpu for CPU.")
    parser.add_argument("--project", default="yolo_research/outputs/evaluation", help="Output project directory for validation results.")
    parser.add_argument("--name", default="atlc_yolov8n_custom_val", help="Validation run name.")
    parser.add_argument("--conf", type=float, default=None, help="Optional confidence threshold for validation.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS during validation.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow writing into existing validation output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    data_path = Path(args.data)
    project_dir = Path(args.project)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")
    project_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ATLC YOLO EVALUATION")
    print("=" * 80)
    print(f"weights: {weights_path}")
    print(f"data: {data_path}")
    print(f"imgsz: {args.imgsz}")
    print(f"batch: {args.batch}")
    print(f"device: {args.device}")
    print(f"project: {project_dir}")
    print(f"name: {args.name}")
    print("=" * 80)

    model = YOLO(str(weights_path))
    val_kwargs = {
        "data": str(data_path),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": str(project_dir),
        "name": args.name,
        "iou": args.iou,
        "plots": True,
        "save_json": True,
        "exist_ok": args.exist_ok,
    }
    if args.conf is not None:
        val_kwargs["conf"] = args.conf
    val_results = model.val(**val_kwargs)

    output_dir = project_dir / args.name
    metrics_path = output_dir / "metrics_summary.json"
    metrics = save_yolo_val_metrics(
        val_results=val_results,
        output_path=metrics_path,
        extra={
            "weights": str(weights_path),
            "data": str(data_path),
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "conf": args.conf,
            "iou": args.iou,
        },
    )
    print("Saved metrics summary to:", metrics_path)
    print("Extracted metrics:")
    print(metrics)


if __name__ == "__main__":
    main()
