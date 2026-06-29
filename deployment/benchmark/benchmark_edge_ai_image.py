"""
File:
    benchmark_edge_ai_image.py

Phase:
    Phase 16.6 - Edge AI Benchmark Report

Purpose:
    - Run a small repeatable image benchmark for available ATLC deployment backends.
    - Compare model size, average latency, basic FPS, and detection count.

Responsibilities:
    - Load the FP32 ONNX model.
    - Load the quantized ONNX model if available.
    - Run repeated inference on one or more test images.
    - Write a small benchmark summary.

This file should NOT:
    - Perform full mAP evaluation.
    - Convert models to TFLite.
    - Run TensorRT.
    - Perform ROI counting.
    - Send UART messages.
    - Modify firmware.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import onnxruntime as ort

ONNX_HELPER_DIR = Path(__file__).resolve().parents[1] / "onnx"
sys.path.insert(0, str(ONNX_HELPER_DIR))

from infer_onnx_image import (  # noqa: E402
    create_session,
    filter_detections,
    letterbox_image,
    load_image,
    parse_providers,
    run_inference,
    validate_model_io,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    """
    Parse benchmark command-line options.

    Why:

        Benchmark results are only useful when the model paths, inputs,
        confidence threshold, provider, and run count are explicit and
        repeatable from the terminal.
    """

    parser = argparse.ArgumentParser(description="Run a small ATLC Edge AI image benchmark.")
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
    parser.add_argument("--images", required=True, help="Path to one image or a directory of images.")
    parser.add_argument("--max-images", type=int, default=5, help="Maximum images to use when --images is a directory.")
    parser.add_argument("--imgsz", type=int, default=640, help="Model input image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument(
        "--providers",
        default="CPUExecutionProvider",
        help="Comma-separated ONNX Runtime providers, for example CPUExecutionProvider.",
    )
    parser.add_argument("--runs", type=int, default=10, help="Timed runs per image per backend.")
    parser.add_argument(
        "--output",
        default="/tmp/atlc_phase16_6_benchmark.json",
        help="Path to save the benchmark JSON summary.",
    )
    return parser.parse_args()


def collect_images(images_path: Path, max_images: int) -> list[Path]:
    """
    Resolve one image path or a small sorted image subset from a directory.

    Why:

        Phase 16.6 is a smoke benchmark, not full dataset evaluation. Limiting a
        directory input keeps the run quick while still allowing a small
        representative sanity check.
    """

    if max_images <= 0:
        raise ValueError("--max-images must be a positive integer.")

    if images_path.is_file():
        if images_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {images_path}")
        return [images_path]

    if images_path.is_dir():
        image_paths = sorted(path for path in images_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not image_paths:
            raise FileNotFoundError(f"No supported images found in directory: {images_path}")
        return image_paths[:max_images]

    raise FileNotFoundError(f"Image path not found: {images_path}")


def model_size_mb(path: Path) -> float:
    """
    Return model size in MiB.

    Why:

        Edge deployment decisions often trade storage footprint against latency
        and detection behavior. Size is recorded with the runtime metrics.
    """

    return path.stat().st_size / (1024 * 1024)


def load_backends(fp32_model: Path, quantized_model: Path, providers: list[str], sample_image, imgsz: int) -> list[dict[str, Any]]:
    """
    Create ONNX Runtime sessions for benchmark backends.

    Why:

        Session creation is intentionally outside the timed loop. The benchmark
        measures repeated image inference latency, not one-time model loading.
    """

    if not fp32_model.exists():
        raise FileNotFoundError(f"FP32 ONNX model not found: {fp32_model}")

    input_tensor, _, _, _ = letterbox_image(sample_image, imgsz)
    backends: list[dict[str, Any]] = []

    fp32_session = create_session(fp32_model, providers)
    fp32_input_name, _ = validate_model_io(fp32_session, input_tensor)
    backends.append(
        {
            "name": "FP32 ONNX Runtime",
            "model_path": fp32_model,
            "model_size_mb": model_size_mb(fp32_model),
            "session": fp32_session,
            "input_name": fp32_input_name,
            "note": "Deployment baseline.",
        }
    )

    if quantized_model.exists():
        quantized_session = create_session(quantized_model, providers)
        quantized_input_name, _ = validate_model_io(quantized_session, input_tensor)
        backends.append(
            {
                "name": "Quantized ONNX Runtime",
                "model_path": quantized_model,
                "model_size_mb": model_size_mb(quantized_model),
                "session": quantized_session,
                "input_name": quantized_input_name,
                "note": "Dynamic QUInt8 candidate.",
            }
        )
    else:
        print(f"Warning: quantized model not found, skipping: {quantized_model}")

    return backends


def benchmark_backend(backend: dict[str, Any], images: list[tuple[Path, Any]], imgsz: int, conf: float, runs: int) -> dict[str, Any]:
    """
    Benchmark one backend across all selected images.

    Why:

        Each timed run includes preprocessing, ONNX Runtime inference, and
        confidence filtering. That gives a practical deployment latency number
        for the image inference path while still avoiding a full benchmark
        framework.
    """

    if runs <= 0:
        raise ValueError("--runs must be a positive integer.")

    latencies: list[float] = []
    detections_by_image: dict[str, int] = {}

    for image_path, image_bgr in images:
        detections = []
        for run_index in range(runs):
            started_at = time.perf_counter()
            input_tensor, ratio, pad_x, pad_y = letterbox_image(image_bgr, imgsz)
            output = run_inference(backend["session"], backend["input_name"], input_tensor)
            detections_for_run = filter_detections(output, conf, image_bgr.shape[:2], ratio, pad_x, pad_y)
            latencies.append(time.perf_counter() - started_at)

            if run_index == runs - 1:
                detections = detections_for_run

        detections_by_image[str(image_path)] = len(detections)

    average_latency = sum(latencies) / len(latencies)
    return {
        "backend": backend["name"],
        "model_path": str(backend["model_path"]),
        "model_size_mb": backend["model_size_mb"],
        "images_tested": len(images),
        "runs_per_image": runs,
        "confidence_threshold": conf,
        "average_latency_seconds": average_latency,
        "min_latency_seconds": min(latencies),
        "max_latency_seconds": max(latencies),
        "approx_fps": 1.0 / average_latency if average_latency > 0 else 0.0,
        "detections_by_image": detections_by_image,
        "total_detections": sum(detections_by_image.values()),
        "visual_quality_note": "Detection counts only; visual inspection and mAP are outside Phase 16.6.",
        "backend_note": backend["note"],
    }


def build_summary(results: list[dict[str, Any]]) -> str:
    """
    Build a conservative interpretation from benchmark results.

    Why:

        The report should summarize measured tradeoffs without claiming that a
        short CPU smoke test proves final deployment performance.
    """

    if len(results) < 2:
        return "Only one backend was benchmarked, so no runtime tradeoff conclusion is available."

    fastest = min(results, key=lambda item: item["average_latency_seconds"])
    smallest = min(results, key=lambda item: item["model_size_mb"])
    return (
        f"{fastest['backend']} was fastest in this CPU smoke benchmark. "
        f"{smallest['backend']} was smallest. "
        "Use these results as deployment guidance, not as final mAP or hardware performance proof."
    )


def print_results_table(results: list[dict[str, Any]]) -> None:
    """
    Print a compact benchmark table.

    Why:

        A readable terminal table makes it easy to paste the result into the
        Phase 16.6 report while the JSON file keeps raw values reproducible.
    """

    print("")
    print("ATLC PHASE 16.6 EDGE AI IMAGE BENCHMARK")
    print("=" * 96)
    print(f"{'Backend':<26} {'Size MB':>9} {'Avg s':>10} {'Min s':>10} {'Max s':>10} {'FPS':>10} {'Detections':>12}")
    print("-" * 96)
    for result in results:
        print(
            f"{result['backend']:<26} "
            f"{result['model_size_mb']:>9.2f} "
            f"{result['average_latency_seconds']:>10.4f} "
            f"{result['min_latency_seconds']:>10.4f} "
            f"{result['max_latency_seconds']:>10.4f} "
            f"{result['approx_fps']:>10.2f} "
            f"{result['total_detections']:>12}"
        )
    print("=" * 96)


def write_json_report(output_path: Path, payload: dict[str, Any]) -> None:
    """
    Save benchmark data as JSON.

    Why:

        Raw benchmark outputs should stay outside Git for this phase, but a
        JSON file in `/tmp` makes the report traceable while the command is
        being run locally.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    """
    Run the Phase 16.6 image benchmark.

    Why:

        This benchmark organizes Phase 16 deployment tradeoffs: FP32 ONNX is
        the current deployment baseline, while dynamic quantized ONNX is a
        smaller candidate that must be checked for speed and detection drift.
    """

    args = parse_args()
    providers = parse_providers(args.providers)
    image_paths = collect_images(Path(args.images), args.max_images)
    images = [(path, load_image(path)) for path in image_paths]

    backends = load_backends(Path(args.fp32_model), Path(args.quantized_model), providers, images[0][1], args.imgsz)
    results = [benchmark_backend(backend, images, args.imgsz, args.conf, args.runs) for backend in backends]
    summary = build_summary(results)

    payload = {
        "hardware_note": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "software_note": {
            "python": sys.version.split()[0],
            "opencv": cv2.__version__,
            "onnxruntime": ort.__version__,
            "providers_requested": providers,
        },
        "images": [str(path) for path in image_paths],
        "backends": [
            {
                "name": backend["name"],
                "model_path": str(backend["model_path"]),
                "model_size_mb": backend["model_size_mb"],
                "providers": backend["session"].get_providers(),
            }
            for backend in backends
        ],
        "results": results,
        "summary": summary,
    }

    print_results_table(results)
    print(summary)
    write_json_report(Path(args.output), payload)
    print(f"Benchmark JSON saved: {args.output}")


if __name__ == "__main__":
    main()
