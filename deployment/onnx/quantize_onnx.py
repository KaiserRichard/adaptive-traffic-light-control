"""
File:
    quantize_onnx.py

Phase:
    Phase 16.5 - ONNX Quantization Experiment

Purpose:
    - Create a quantized ONNX model from the FP32 ONNX model.
    - Report model size before and after quantization.

Responsibilities:
    - Load the FP32 ONNX model path.
    - Generate a quantized ONNX model.
    - Print size comparison.
    - Keep quantization as an experiment.

This file should NOT:
    - Run TensorRT.
    - Run TFLite.
    - Perform ROI counting.
    - Send UART messages.
    - Modify firmware.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def parse_args() -> argparse.Namespace:
    """
    Parse command-line options for the ONNX quantization experiment.

    Why:

        Quantized model files are deployment artifacts. Explicit input, output,
        and mode arguments make the experiment reproducible without modifying
        the original FP32 ONNX model.
    """

    parser = argparse.ArgumentParser(description="Quantize the ATLC ONNX model.")
    parser.add_argument(
        "--input",
        default="deployment/onnx/atlc_yolo26n_custom.onnx",
        help="Input FP32 ONNX model path.",
    )
    parser.add_argument(
        "--output",
        default="deployment/onnx/atlc_yolo26n_custom_dynamic_int8.onnx",
        help="Output quantized ONNX model path.",
    )
    parser.add_argument(
        "--mode",
        default="dynamic",
        choices=["dynamic"],
        help="Quantization mode. Phase 16.5 implements dynamic quantization first.",
    )
    return parser.parse_args()


def model_size_mb(path: Path) -> float:
    """
    Return a model file size in MiB.

    Why:

        Model size is one of the first edge-deployment tradeoffs to inspect.
        Smaller files are easier to store and transfer, but size reduction alone
        does not prove that a model is faster or accurate enough.
    """

    return path.stat().st_size / (1024 * 1024)


def size_reduction_percent(fp32_size_mb: float, quantized_size_mb: float) -> float:
    """
    Compute size reduction as a percentage of the FP32 model size.

    Why:

        A percentage is easier to compare across future model variants than raw
        megabytes alone.
    """

    if fp32_size_mb <= 0:
        raise ValueError("FP32 model size must be positive.")
    return (1.0 - quantized_size_mb / fp32_size_mb) * 100.0


def quantize_model(input_path: Path, output_path: Path, mode: str) -> None:
    """
    Quantize an ONNX model using ONNX Runtime quantization tools.

    Why:

        Dynamic quantization is a low-friction first experiment because it does
        not require a calibration dataset. This script uses QUInt8 weights
        because the local CPUExecutionProvider can load that dynamic model for
        this YOLO export, while QInt8 ConvInteger output may not be supported
        by every ONNX Runtime CPU build.
    """

    if mode != "dynamic":
        raise ValueError(f"Unsupported quantization mode for Phase 16.5: {mode}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input ONNX model not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must not overwrite the FP32 input model.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QUInt8,
    )

    if not output_path.exists():
        raise FileNotFoundError(f"Quantized ONNX output was not created: {output_path}")


def main() -> None:
    """
    Run the Phase 16.5 quantization experiment.

    Why:

        This script only creates a candidate quantized model and reports size.
        Runtime and detection quality are checked separately by the comparison
        script before deciding whether the quantized model is useful.
    """

    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    print("=" * 80)
    print("ATLC PHASE 16.5 ONNX QUANTIZATION")
    print("=" * 80)
    print(f"Input model path:     {input_path}")
    print(f"Output model path:    {output_path}")
    print(f"Quantization mode:    {args.mode}")
    print("Weight type:          QUInt8")
    print("=" * 80)

    try:
        quantize_model(input_path, output_path, args.mode)
    except Exception as exc:
        print("Quantization failed.")
        print(f"Error: {type(exc).__name__}: {exc}")
        raise

    fp32_size = model_size_mb(input_path)
    quantized_size = model_size_mb(output_path)
    reduction = size_reduction_percent(fp32_size, quantized_size)

    print("Quantization finished.")
    print(f"FP32 model size:      {fp32_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction:       {reduction:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
