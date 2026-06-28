"""
File:
    compare_fp32_quantized_onnx_image.py

Phase:
    Phase 16.5 - ONNX Quantization Experiment

Purpose:
    - Compare FP32 ONNX inference with quantized ONNX inference on the same image.
    - Report model size, detection counts, and basic runtime.
    - Save side-by-side annotated comparison output.

Responsibilities:
    - Load FP32 and quantized ONNX Runtime sessions.
    - Reuse corrected letterbox preprocessing.
    - Run both models on the same input image.
    - Measure basic runtime for repeated runs.
    - Save visual comparison artifacts.

This file should NOT:
    - Run full benchmarking.
    - Use TensorRT or TFLite.
    - Perform ROI counting.
    - Send UART messages.
    - Modify firmware.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ONNX_HELPER_DIR = Path(__file__).resolve().parents[1] / "onnx"
sys.path.insert(0, str(ONNX_HELPER_DIR))

from infer_onnx_image import (  # noqa: E402
    create_session,
    draw_detections,
    filter_detections,
    letterbox_image,
    load_image,
    parse_providers,
    run_inference,
    validate_model_io,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line options for the FP32 vs quantized ONNX comparison.

    Why:

        This script compares two deployment artifacts on one image. Explicit
        paths and run settings make the experiment repeatable without becoming
        a full benchmark framework.
    """

    parser = argparse.ArgumentParser(description="Compare FP32 and quantized ONNX image inference for ATLC YOLO.")
    parser.add_argument(
        "--fp32-model",
        default="deployment/onnx/atlc_yolo26n_custom.onnx",
        help="Path to the FP32 ONNX model.",
    )
    parser.add_argument(
        "--quantized-model",
        default="deployment/onnx/atlc_yolo26n_custom_dynamic_int8.onnx",
        help="Path to the quantized ONNX model.",
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--output", required=True, help="Path for the side-by-side comparison image.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for both models.")
    parser.add_argument(
        "--providers",
        default="CPUExecutionProvider",
        help="Comma-separated ONNX Runtime providers, for example CPUExecutionProvider.",
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs per model.")
    return parser.parse_args()


def validate_inputs(fp32_model_path: Path, quantized_model_path: Path, image_path: Path, runs: int) -> None:
    """
    Validate required files and run count before creating ONNX sessions.

    Why:

        Quantization experiments can fail because the generated model is absent
        or unusable. Clear path validation separates missing-file problems from
        ONNX Runtime operator support problems.
    """

    if not fp32_model_path.exists():
        raise FileNotFoundError(f"FP32 ONNX model not found: {fp32_model_path}")
    if not quantized_model_path.exists():
        raise FileNotFoundError(f"Quantized ONNX model not found: {quantized_model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if runs <= 0:
        raise ValueError("--runs must be a positive integer.")


def model_size_mb(path: Path) -> float:
    """
    Return a model file size in MiB.

    Why:

        Quantization is often considered to reduce storage footprint. The size
        comparison is useful even if runtime or visual quality is not better.
    """

    return path.stat().st_size / (1024 * 1024)


def size_reduction_percent(fp32_size_mb: float, quantized_size_mb: float) -> float:
    """
    Compute size reduction as a percentage of the FP32 model size.

    Why:

        The percentage makes it easier to judge whether the quantized artifact
        has a meaningful storage advantage.
    """

    return (1.0 - quantized_size_mb / fp32_size_mb) * 100.0


def output_paths(side_by_side_path: Path) -> tuple[Path, Path, Path]:
    """
    Derive FP32, quantized, and side-by-side image paths.

    Why:

        One `--output` argument keeps the CLI simple while still producing the
        individual visual artifacts needed for inspection.
    """

    suffix = side_by_side_path.suffix or ".jpg"
    base = side_by_side_path.with_suffix("")
    return (
        base.with_name(f"{base.name}_fp32").with_suffix(suffix),
        base.with_name(f"{base.name}_quantized").with_suffix(suffix),
        side_by_side_path.with_suffix(suffix),
    )


def run_onnx_model(session, input_name: str, image_bgr: np.ndarray, imgsz: int, conf: float, runs: int) -> tuple[list[dict], float]:
    """
    Run repeated ONNX inference and return detections plus average time.

    Why:

        Both FP32 and quantized models must use the corrected Phase 16.4
        letterbox preprocessing path. This prevents quantization results from
        being confused with preprocessing mismatch.
    """

    detections = []
    total_time = 0.0

    for run_index in range(runs):
        started_at = time.perf_counter()
        input_tensor, ratio, pad_x, pad_y = letterbox_image(image_bgr, imgsz)
        output = run_inference(session, input_name, input_tensor)
        detections_for_run = filter_detections(output, conf, image_bgr.shape[:2], ratio, pad_x, pad_y)
        total_time += time.perf_counter() - started_at

        if run_index == runs - 1:
            detections = detections_for_run

    return detections, total_time / runs


def add_title(image_bgr: np.ndarray, title: str) -> np.ndarray:
    """
    Add a title band above one annotated image.

    Why:

        The side-by-side output should be inspectable outside the terminal, so
        each half is labeled directly in the image.
    """

    title_height = 42
    titled = cv2.copyMakeBorder(image_bgr, title_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(titled, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return titled


def make_side_by_side(fp32_image: np.ndarray, quantized_image: np.ndarray) -> np.ndarray:
    """
    Combine FP32 and quantized annotated outputs into one comparison image.

    Why:

        Runtime and detection counts are not enough. Visual inspection is
        required to decide whether quantization preserves acceptable detection
        quality.
    """

    fp32_titled = add_title(fp32_image, "FP32 ONNX")
    quantized_titled = add_title(quantized_image, "Quantized ONNX")
    return np.hstack((fp32_titled, quantized_titled))


def save_image(path: Path, image_bgr: np.ndarray) -> None:
    """
    Save an image artifact and fail if OpenCV cannot write it.

    Why:

        A quantization experiment needs inspectable visual artifacts. Checking
        the write result avoids silently reporting a missing comparison image.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"OpenCV could not write image: {path}")


def runtime_interpretation(fp32_avg: float, quantized_avg: float) -> str:
    """
    Summarize the basic runtime comparison without overclaiming.

    Why:

        Phase 16.5 is an experiment, not a benchmark report. The interpretation
        should describe this run only.
    """

    tolerance = 0.05
    if abs(fp32_avg - quantized_avg) / fp32_avg <= tolerance:
        return "Quantized model runtime is similar to FP32 in this basic comparison."
    if quantized_avg < fp32_avg:
        return f"Quantized model is faster than FP32 in this basic comparison ({fp32_avg / quantized_avg:.2f}x by average time)."
    return f"Quantized model is slower than FP32 in this basic comparison ({quantized_avg / fp32_avg:.2f}x by average time)."


def main() -> None:
    """
    Run the complete Phase 16.5 FP32 vs quantized ONNX comparison.

    Why:

        This script checks whether the quantized model is smaller, runtime
        competitive, and visually reasonable on the same input image. It does
        not decide final deployment by itself.
    """

    args = parse_args()
    fp32_model_path = Path(args.fp32_model)
    quantized_model_path = Path(args.quantized_model)
    image_path = Path(args.image)
    side_by_side_path = Path(args.output)
    providers = parse_providers(args.providers)

    validate_inputs(fp32_model_path, quantized_model_path, image_path, args.runs)
    fp32_image_path, quantized_image_path, side_by_side_path = output_paths(side_by_side_path)

    image_bgr = load_image(image_path)
    input_tensor, _, _, _ = letterbox_image(image_bgr, args.imgsz)

    fp32_session = create_session(fp32_model_path, providers)
    quantized_session = create_session(quantized_model_path, providers)
    fp32_input_name, _ = validate_model_io(fp32_session, input_tensor)
    quantized_input_name, _ = validate_model_io(quantized_session, input_tensor)

    fp32_detections, fp32_avg = run_onnx_model(fp32_session, fp32_input_name, image_bgr, args.imgsz, args.conf, args.runs)
    quantized_detections, quantized_avg = run_onnx_model(
        quantized_session,
        quantized_input_name,
        image_bgr,
        args.imgsz,
        args.conf,
        args.runs,
    )

    fp32_image = draw_detections(image_bgr, fp32_detections)
    quantized_image = draw_detections(image_bgr, quantized_detections)
    side_by_side = make_side_by_side(fp32_image, quantized_image)

    save_image(fp32_image_path, fp32_image)
    save_image(quantized_image_path, quantized_image)
    save_image(side_by_side_path, side_by_side)

    fp32_size = model_size_mb(fp32_model_path)
    quantized_size = model_size_mb(quantized_model_path)
    reduction = size_reduction_percent(fp32_size, quantized_size)

    print("=" * 80)
    print("ATLC PHASE 16.5 FP32 VS QUANTIZED ONNX COMPARISON")
    print("=" * 80)
    print(f"Image path:                  {image_path}")
    print(f"FP32 model:                  {fp32_model_path}")
    print(f"Quantized model:             {quantized_model_path}")
    print(f"Provider:                    {', '.join(fp32_session.get_providers())}")
    print(f"Runs:                        {args.runs}")
    print("")
    print(f"FP32 model size:             {fp32_size:.2f} MB")
    print(f"Quantized model size:        {quantized_size:.2f} MB")
    print(f"Size reduction:              {reduction:.2f}%")
    print("")
    print(f"FP32 average time:           {fp32_avg:.4f} s")
    print(f"Quantized average time:      {quantized_avg:.4f} s")
    print("")
    print(f"FP32 detections:             {len(fp32_detections)}")
    print(f"Quantized detections:        {len(quantized_detections)}")
    print("")
    print(f"Output FP32 image:           {fp32_image_path}")
    print(f"Output quantized image:      {quantized_image_path}")
    print(f"Output side-by-side image:   {side_by_side_path}")
    print("")
    print("Interpretation:")
    print(f"    Quantized model is {'smaller' if quantized_size < fp32_size else 'not smaller'} than FP32.")
    print(f"    {runtime_interpretation(fp32_avg, quantized_avg)}")
    print(f"    Detection counts are {'same' if len(fp32_detections) == len(quantized_detections) else 'different'}.")
    print("    Visual inspection is required.")
    print("=" * 80)


if __name__ == "__main__":
    main()
