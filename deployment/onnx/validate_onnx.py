"""
Validate that an exported ATLC ONNX model is structurally valid and loadable.

This script does not run image or video inference. Phase 16.1 only verifies
that the exported ONNX graph can be checked and opened by ONNX Runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an ATLC YOLO ONNX model.")
    parser.add_argument(
        "--model",
        default="deployment/onnx/atlc_yolo26n_custom.onnx",
        help="Path to the ONNX model.",
    )
    return parser.parse_args()


def format_shape(shape: list[int | str | None]) -> str:
    return "[" + ", ".join("?" if dim is None else str(dim) for dim in shape) + "]"


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    size_mb = model_path.stat().st_size / (1024 * 1024)

    print("=" * 80)
    print("ATLC PHASE 16.1 ONNX VALIDATION")
    print("=" * 80)
    print(f"model: {model_path}")
    print(f"size:  {size_mb:.2f} MB")
    print("=" * 80)

    onnx_model = onnx.load(str(model_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX checker: PASS")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    print("ONNX Runtime: PASS")

    print("=" * 80)
    print("INPUTS")
    for input_meta in session.get_inputs():
        print(f"- name: {input_meta.name}")
        print(f"  shape: {format_shape(input_meta.shape)}")
        print(f"  type: {input_meta.type}")

    print("=" * 80)
    print("OUTPUTS")
    for output_meta in session.get_outputs():
        print(f"- name: {output_meta.name}")
        print(f"  shape: {format_shape(output_meta.shape)}")
        print(f"  type: {output_meta.type}")
    print("=" * 80)


if __name__ == "__main__":
    main()
