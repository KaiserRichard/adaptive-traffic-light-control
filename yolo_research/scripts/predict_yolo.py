"""
Run trained YOLO prediction on sample images/videos and save visual outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO prediction and save report-ready examples.")
    parser.add_argument("--weights", required=True, help="Path to trained weights, for example best.pt.")
    parser.add_argument("--source", required=True, help="Image, folder, video, or camera source.")
    parser.add_argument("--imgsz", type=int, default=640, help="Prediction image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS.")
    parser.add_argument("--device", default="0", help="Device: 0 for GPU, cpu for CPU.")
    parser.add_argument("--project", default="yolo_research/outputs/predictions", help="Prediction output root.")
    parser.add_argument("--name", default="atlc_yolov8n_custom_predictions", help="Prediction run name.")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per image.")
    parser.add_argument("--save-txt", action="store_true", help="Save prediction labels as txt.")
    parser.add_argument("--save-conf", action="store_true", help="Save confidence values in txt labels.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow writing into existing output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    Path(args.project).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ATLC YOLO PREDICTION")
    print("=" * 80)
    print(f"weights: {weights_path}")
    print(f"source: {args.source}")
    print(f"imgsz: {args.imgsz}")
    print(f"conf: {args.conf}")
    print(f"device: {args.device}")
    print(f"project: {args.project}")
    print(f"name: {args.name}")
    print("=" * 80)

    model = YOLO(str(weights_path))
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_det=args.max_det,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )
    print("Prediction finished.")
    print("Saved to:", Path(args.project) / args.name)


if __name__ == "__main__":
    main()
