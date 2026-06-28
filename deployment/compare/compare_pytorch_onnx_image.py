"""
File:
    compare_pytorch_onnx_image.py

Phase:
    Phase 16.4 - PyTorch vs ONNX Runtime Comparison

Purpose:
    - Compare Ultralytics PyTorch inference with ONNX Runtime inference on the same image.
    - Report basic detection counts and runtime.
    - Save side-by-side annotated comparison output.

Responsibilities:
    - Load the PyTorch .pt model.
    - Load the ONNX model.
    - Run both on the same input image.
    - Measure basic runtime.
    - Save visual comparison output.

This file should NOT:
    - Run full benchmarking.
    - Perform quantization.
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
from ultralytics import YOLO

ONNX_HELPER_DIR = Path(__file__).resolve().parents[1] / "onnx"
sys.path.insert(0, str(ONNX_HELPER_DIR))

from infer_onnx_image import (  # noqa: E402
    create_session,
    draw_detections,
    filter_detections,
    load_image,
    parse_providers,
    preprocess_image,
    run_inference,
    validate_model_io,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line options for the basic PyTorch vs ONNX comparison.

    Why:

        This phase compares two backends on the same image. Making paths,
        confidence, provider, image size, and run count explicit keeps the
        comparison reproducible without turning it into a full benchmark suite.
    """

    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX Runtime image inference for ATLC YOLO.")
    parser.add_argument(
        "--pt-model",
        default="yolo_research/outputs/runs/atlc_yolo26n_custom/weights/best.pt",
        help="Path to the trained Ultralytics .pt model.",
    )
    parser.add_argument(
        "--onnx-model",
        default="deployment/onnx/atlc_yolo26n_custom.onnx",
        help="Path to the exported ONNX model.",
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the side-by-side comparison image.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for both backends.")
    parser.add_argument(
        "--providers",
        default="CPUExecutionProvider",
        help="Comma-separated ONNX Runtime providers, for example CPUExecutionProvider.",
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs per backend.")
    return parser.parse_args()


def validate_inputs(pt_model_path: Path, onnx_model_path: Path, image_path: Path, runs: int) -> None:
    """
    Validate required files and run count before loading heavy model runtimes.

    Why:

        Failing fast on missing paths prevents spending time loading one backend
        before discovering that the other model or image path is invalid.
    """

    if not pt_model_path.exists():
        raise FileNotFoundError(f"PyTorch model not found: {pt_model_path}")
    if not onnx_model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if runs <= 0:
        raise ValueError("--runs must be a positive integer.")


def output_paths(side_by_side_path: Path) -> tuple[Path, Path, Path]:
    """
    Derive companion output paths from the requested side-by-side path.

    Why:

        The CLI should stay simple with one `--output` argument while still
        saving all three artifacts needed for visual inspection: PyTorch only,
        ONNX only, and side-by-side.
    """

    suffix = side_by_side_path.suffix or ".jpg"
    base = side_by_side_path.with_suffix("")
    return (
        base.with_name(f"{base.name}_pytorch").with_suffix(suffix),
        base.with_name(f"{base.name}_onnx").with_suffix(suffix),
        side_by_side_path.with_suffix(suffix),
    )


def pytorch_result_to_detections(result) -> list[dict]:
    """
    Convert one Ultralytics result into the local detection dictionary format.

    Why:

        ONNX Phase 16.2 drawing expects dictionaries with `box`, `confidence`,
        and `class_id`. Converting PyTorch detections to the same shape allows
        both backends to use the same visual style.
    """

    detections = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy()

    for box, confidence, class_id in zip(xyxy, confidences, class_ids):
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        detections.append(
            {
                "box": (x1, y1, x2, y2),
                "confidence": float(confidence),
                "class_id": int(round(float(class_id))),
            }
        )
    return detections


def run_pytorch_comparison(model: YOLO, image_bgr: np.ndarray, imgsz: int, conf: float, runs: int) -> tuple[list[dict], float]:
    """
    Run repeated Ultralytics PyTorch inference and return detections plus average time.

    Why:

        A few repeated runs reduce one-off startup noise, but this is still only
        a practical comparison. It does not control enough variables to support
        final performance claims.
    """

    detections = []
    total_time = 0.0

    for run_index in range(runs):
        started_at = time.perf_counter()
        results = model.predict(source=image_bgr, imgsz=imgsz, conf=conf, device="cpu", verbose=False)
        total_time += time.perf_counter() - started_at

        if run_index == runs - 1:
            detections = pytorch_result_to_detections(results[0])

    return detections, total_time / runs


def run_onnx_comparison(
    session,
    input_name: str,
    image_bgr: np.ndarray,
    imgsz: int,
    conf: float,
    runs: int,
) -> tuple[list[dict], float]:
    """
    Run repeated ONNX Runtime inference and return detections plus average time.

    Why:

        The ONNX path reuses Phase 16.2 preprocessing and postprocessing so the
        comparison checks the deployment path that already passed image and
        video smoke tests.
    """

    detections = []
    total_time = 0.0

    for run_index in range(runs):
        started_at = time.perf_counter()
        input_tensor = preprocess_image(image_bgr, imgsz)
        output = run_inference(session, input_name, input_tensor)
        detections_for_run = filter_detections(output, conf, image_bgr.shape[:2], imgsz)
        total_time += time.perf_counter() - started_at

        if run_index == runs - 1:
            detections = detections_for_run

    return detections, total_time / runs


def add_title(image_bgr: np.ndarray, title: str) -> np.ndarray:
    """
    Add a small title band above one annotated image.

    Why:

        Side-by-side outputs are easier to inspect when each backend is labeled
        directly in the image, especially when the standalone files are opened
        outside the terminal context.
    """

    title_height = 42
    titled = cv2.copyMakeBorder(image_bgr, title_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(titled, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return titled


def make_side_by_side(pytorch_image: np.ndarray, onnx_image: np.ndarray) -> np.ndarray:
    """
    Combine PyTorch and ONNX annotated outputs into one comparison image.

    Why:

        Detection counts are useful, but visual similarity is the practical
        question in this phase. A side-by-side image makes box placement and
        labels easy to compare before any formal benchmark work.
    """

    pytorch_titled = add_title(pytorch_image, "PyTorch / Ultralytics")
    onnx_titled = add_title(onnx_image, "ONNX Runtime")
    if pytorch_titled.shape[0] != onnx_titled.shape[0]:
        target_height = pytorch_titled.shape[0]
        scale = target_height / onnx_titled.shape[0]
        onnx_titled = cv2.resize(onnx_titled, (int(onnx_titled.shape[1] * scale), target_height))
    return np.hstack((pytorch_titled, onnx_titled))


def save_image(path: Path, image_bgr: np.ndarray) -> None:
    """
    Save one comparison artifact and fail if the file cannot be written.

    Why:

        The comparison is only useful if it leaves inspectable artifacts. OpenCV
        write failures are easy to miss unless the return value is checked.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"OpenCV could not write image: {path}")


def speed_interpretation(pytorch_avg: float, onnx_avg: float) -> str:
    """
    Produce a short practical runtime interpretation.

    Why:

        Phase 16.4 should answer whether ONNX is faster or slower in this basic
        CPU comparison, while avoiding overclaiming benchmark-grade results.
    """

    if onnx_avg < pytorch_avg:
        ratio = pytorch_avg / onnx_avg
        return f"ONNX is faster than PyTorch in this basic CPU comparison ({ratio:.2f}x by average time)."
    if onnx_avg > pytorch_avg:
        ratio = onnx_avg / pytorch_avg
        return f"ONNX is slower than PyTorch in this basic CPU comparison ({ratio:.2f}x by average time)."
    return "ONNX and PyTorch have the same average time in this basic CPU comparison."


def count_interpretation(pytorch_count: int, onnx_count: int) -> str:
    """
    Produce a short detection-count interpretation.

    Why:

        Equal counts do not prove identical detections, and different counts do
        not automatically mean failure. The count is a quick smoke-test signal
        that must be paired with visual inspection.
    """

    if pytorch_count == onnx_count:
        return "Detection counts are the same. Visual inspection is still required."
    return "Detection counts are different. Visual inspection is required."


def main() -> None:
    """
    Run the complete Phase 16.4 image comparison workflow.

    Why:

        This script gives a practical answer about PyTorch vs ONNX behavior on
        the same input image. It intentionally stops before full benchmarking,
        quantization, accelerator runtimes, ROI logic, planning, UART, or
        firmware changes.
    """

    args = parse_args()
    pt_model_path = Path(args.pt_model)
    onnx_model_path = Path(args.onnx_model)
    image_path = Path(args.image)
    side_by_side_path = Path(args.output)
    providers = parse_providers(args.providers)

    validate_inputs(pt_model_path, onnx_model_path, image_path, args.runs)
    pytorch_path, onnx_path, side_by_side_path = output_paths(side_by_side_path)

    image_bgr = load_image(image_path)

    pytorch_model = YOLO(str(pt_model_path))
    onnx_session = create_session(onnx_model_path, providers)
    input_tensor = preprocess_image(image_bgr, args.imgsz)
    input_name, _ = validate_model_io(onnx_session, input_tensor)

    pytorch_detections, pytorch_avg = run_pytorch_comparison(
        pytorch_model,
        image_bgr,
        args.imgsz,
        args.conf,
        args.runs,
    )
    onnx_detections, onnx_avg = run_onnx_comparison(
        onnx_session,
        input_name,
        image_bgr,
        args.imgsz,
        args.conf,
        args.runs,
    )

    pytorch_image = draw_detections(image_bgr, pytorch_detections)
    onnx_image = draw_detections(image_bgr, onnx_detections)
    side_by_side = make_side_by_side(pytorch_image, onnx_image)

    save_image(pytorch_path, pytorch_image)
    save_image(onnx_path, onnx_image)
    save_image(side_by_side_path, side_by_side)

    print("=" * 80)
    print("ATLC PHASE 16.4 PYTORCH VS ONNX COMPARISON")
    print("=" * 80)
    print(f"Image path:                  {image_path}")
    print(f"PyTorch model:               {pt_model_path}")
    print(f"ONNX model:                  {onnx_model_path}")
    print(f"Provider:                    {', '.join(onnx_session.get_providers())}")
    print(f"Runs:                        {args.runs}")
    print("")
    print(f"PyTorch average time:        {pytorch_avg:.4f} s")
    print(f"ONNX average time:           {onnx_avg:.4f} s")
    print(f"PyTorch detections:          {len(pytorch_detections)}")
    print(f"ONNX detections:             {len(onnx_detections)}")
    print("")
    print(f"Output PyTorch image:        {pytorch_path}")
    print(f"Output ONNX image:           {onnx_path}")
    print(f"Output side-by-side image:   {side_by_side_path}")
    print("")
    print(speed_interpretation(pytorch_avg, onnx_avg))
    print(count_interpretation(len(pytorch_detections), len(onnx_detections)))
    print("=" * 80)


if __name__ == "__main__":
    main()
