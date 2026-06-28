"""
Export the trained ATLC YOLO model from Ultralytics .pt format to ONNX.

Phase 16.1 deliberately exports only FP32/static ONNX by default. INT8,
TensorRT, TFLite, and full image/video inference are later deployment phases.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ATLC YOLO weights to ONNX.")
    parser.add_argument(
        "--weights",
        default="yolo_research/outputs/runs/atlc_yolo26n_custom/weights/best.pt",
        help="Input Ultralytics .pt weights path.",
    )
    parser.add_argument(
        "--output",
        default="deployment/onnx/atlc_yolo26n_custom.onnx",
        help="Output ONNX file path.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size.")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version.")
    parser.add_argument("--dynamic", action="store_true", help="Export dynamic input shapes.")
    parser.add_argument("--simplify", action="store_true", help="Simplify the exported ONNX graph.")
    parser.add_argument("--half", action="store_true", help="Export FP16 ONNX when supported.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    output_path = Path(args.output)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if output_path.suffix.lower() != ".onnx":
        raise ValueError(f"Output path must end with .onnx: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ATLC PHASE 16.1 YOLO ONNX EXPORT")
    print("=" * 80)
    print(f"weights:  {weights_path}")
    print(f"output:   {output_path}")
    print(f"imgsz:    {args.imgsz}")
    print(f"opset:    {args.opset}")
    print(f"dynamic:  {args.dynamic}")
    print(f"simplify: {args.simplify}")
    print(f"half:     {args.half}")
    print("=" * 80)

    model = YOLO(str(weights_path))
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
        half=args.half,
    )

    exported_path = Path(exported)
    if not exported_path.exists():
        raise FileNotFoundError(f"Ultralytics reported an ONNX export path that does not exist: {exported_path}")

    if exported_path.resolve() != output_path.resolve():
        shutil.copy2(exported_path, output_path)

    if not output_path.exists():
        raise FileNotFoundError(f"Expected ONNX output was not created: {output_path}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print("=" * 80)
    print("EXPORT FINISHED")
    print(f"Ultralytics export path: {exported_path}")
    print(f"Phase 16.1 ONNX path:   {output_path}")
    print(f"ONNX size:              {size_mb:.2f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()
