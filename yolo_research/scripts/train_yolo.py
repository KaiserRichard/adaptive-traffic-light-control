"""
Reproducible YOLO training script for ATLC custom dataset.

Example:
    python -m yolo_research.scripts.train_yolo \
      --data yolo_research/configs/data.yaml \
      --model yolo26n.pt \
      --epochs 50 \
      --imgsz 640 \
      --batch 16 \
      --device 0 \
      --project yolo_research/outputs/runs \
      --name atlc_yolo26n_custom \
      --exist-ok
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO model for ATLC custom vehicle dataset.")
    parser.add_argument("--data", default="yolo_research/configs/data.yaml", help="Path to YOLO data.yaml.")
    parser.add_argument("--model", default="yolo26n.pt", help="Base model path/name.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", default="16", help="Batch size. Use integer like 16 or 'auto'.")
    parser.add_argument("--device", default="0", help="Training device. Use 0 for first GPU or cpu for CPU.")
    parser.add_argument("--project", default="yolo_research/outputs/runs", help="Output project directory.")
    parser.add_argument("--name", default="atlc_yolo26n_custom", help="Experiment/run name.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow writing into existing run directory.")
    return parser.parse_args()


def parse_batch(batch_value: str) -> int | str:
    if batch_value.strip().lower() == "auto":
        return "auto"
    return int(batch_value)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    project_dir = Path(args.project)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")
    project_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ATLC YOLO TRAINING")
    print("=" * 80)
    print(f"data: {data_path}")
    print(f"model: {args.model}")
    print(f"epochs: {args.epochs}")
    print(f"imgsz: {args.imgsz}")
    print(f"batch: {args.batch}")
    print(f"device: {args.device}")
    print(f"project: {project_dir}")
    print(f"name: {args.name}")
    print("=" * 80)

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=parse_batch(args.batch),
        device=args.device,
        project=str(project_dir),
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        exist_ok=args.exist_ok,
    )

    run_dir = project_dir / args.name
    best_pt = run_dir / "weights" / "best.pt"
    print("=" * 80)
    print("TRAINING FINISHED")
    print(f"Run directory: {run_dir}")
    print(f"Best weights expected at: {best_pt}")
    print("=" * 80)


if __name__ == "__main__":
    main()
