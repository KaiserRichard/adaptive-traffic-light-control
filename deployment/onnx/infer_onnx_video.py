"""
File:
    infer_onnx_video.py

Phase:
    Phase 16.3 - ONNX Runtime Video Inference

Purpose:
    - Run an input video through the exported ONNX model.
    - Save an annotated output video.

Responsibilities:
    - Open video input.
    - Read frames.
    - Reuse ONNX image inference logic per frame.
    - Draw detections.
    - Write output video.
    - Print basic runtime statistics.

This file should NOT:
    - Run a full benchmark framework.
    - Perform quantization.
    - Use TensorRT or TFLite.
    - Perform ROI counting.
    - Send UART messages.
    - Modify firmware.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from infer_onnx_image import (
    create_session,
    draw_detections,
    filter_detections,
    letterbox_image,
    parse_providers,
    run_inference,
    validate_model_io,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line options for one-video ONNX inference.

    Why:

        Video inference has more ways to create large artifacts than image
        inference, so the input video and output video are explicit arguments.
        The safe defaults remain the model path, image size, confidence
        threshold, CPU provider, and max-frame behavior.
    """

    parser = argparse.ArgumentParser(description="Run ONNX Runtime video inference for ATLC YOLO.")
    parser.add_argument(
        "--model",
        default="deployment/onnx/atlc_yolo26n_custom.onnx",
        help="Path to the exported ONNX model.",
    )
    parser.add_argument("--video", required=True, help="Path to one input video.")
    parser.add_argument("--output", required=True, help="Path where the annotated output video will be saved.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size expected by the ONNX model.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections.")
    parser.add_argument(
        "--providers",
        default="CPUExecutionProvider",
        help="Comma-separated ONNX Runtime providers, for example CPUExecutionProvider.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process. Use 0 to process the whole video.",
    )
    return parser.parse_args()


def open_video(video_path: Path) -> cv2.VideoCapture:
    """
    Open the input video and fail early if OpenCV cannot read it.

    Why:

        VideoCapture may silently fail for missing paths, unsupported codecs, or
        damaged files. Checking the handle before inference prevents creating an
        empty output file that looks like a successful deployment artifact.
    """

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    return capture


def read_video_metadata(capture: cv2.VideoCapture) -> tuple[int, int, float, int]:
    """
    Read basic metadata needed to create an output video with matching shape.

    Why:

        The writer must know frame width, height, and FPS before the first frame
        is written. Frame count is useful for reporting, but some codecs or
        streams report it as zero, so it is treated as informational only.
    """

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid input video resolution: {width}x{height}")
    if fps <= 0:
        fps = 30.0

    return width, height, fps, frame_count


def create_video_writer(output_path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    """
    Create an OpenCV writer for the annotated output video.

    Why:

        Phase 16.3 should produce a normal reviewable video artifact. MP4 output
        uses the `mp4v` codec; other extensions may still work depending on the
        local OpenCV build, but `.mp4` is the recommended path for this phase.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create output video: {output_path}")
    return writer


def process_video(
    capture: cv2.VideoCapture,
    writer: cv2.VideoWriter,
    session,
    input_name: str,
    imgsz: int,
    conf_threshold: float,
    max_frames: int,
) -> tuple[int, float]:
    """
    Run frame-by-frame ONNX inference and write annotated frames.

    Why:

        Video inference is just repeated image inference with careful resource
        handling. Reusing Phase 16.2 preprocessing, model execution,
        confidence filtering, and drawing keeps the model contract identical:
        input [1, 3, 640, 640], output [1, 300, 6].
    """

    if max_frames < 0:
        raise ValueError("--max-frames must be 0 or a positive integer.")

    frames_processed = 0
    started_at = time.perf_counter()

    while True:
        if max_frames > 0 and frames_processed >= max_frames:
            break

        ok, frame_bgr = capture.read()
        if not ok:
            break

        input_tensor, ratio, pad_x, pad_y = letterbox_image(frame_bgr, imgsz)
        output = run_inference(session, input_name, input_tensor)
        detections = filter_detections(output, conf_threshold, frame_bgr.shape[:2], ratio, pad_x, pad_y)
        annotated = draw_detections(frame_bgr, detections)
        writer.write(annotated)
        frames_processed += 1

    elapsed = time.perf_counter() - started_at
    return frames_processed, elapsed


def main() -> None:
    """
    Run the complete Phase 16.3 ONNX video inference pipeline.

    Why:

        This function keeps the deployment flow intentionally narrow:

            video -> frames -> preprocessing -> ONNX Runtime -> detections
            -> annotated frames -> output video

        It prints basic runtime FPS for operator feedback, but it does not
        become a benchmark framework. Controlled benchmarking belongs in a later
        phase with hardware/software versions, inputs, configs, and metrics.
    """

    args = parse_args()
    model_path = Path(args.model)
    video_path = Path(args.video)
    output_path = Path(args.output)
    providers = parse_providers(args.providers)

    capture = open_video(video_path)
    writer = None

    try:
        width, height, input_fps, frame_count = read_video_metadata(capture)
        writer = create_video_writer(output_path, width, height, input_fps)

        session = create_session(model_path, providers)
        first_input_tensor, _, _, _ = letterbox_image(__read_first_frame_for_validation(capture), args.imgsz)
        input_name, _ = validate_model_io(session, first_input_tensor)
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frames_processed, elapsed = process_video(
            capture=capture,
            writer=writer,
            session=session,
            input_name=input_name,
            imgsz=args.imgsz,
            conf_threshold=args.conf,
            max_frames=args.max_frames,
        )
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    approx_processing_fps = frames_processed / elapsed if elapsed > 0 else 0.0

    print("=" * 80)
    print("ATLC PHASE 16.3 ONNX VIDEO INFERENCE")
    print("=" * 80)
    print(f"Model path:              {model_path}")
    print(f"Video path:              {video_path}")
    print(f"Output path:             {output_path}")
    print(f"Provider:                {', '.join(session.get_providers())}")
    print(f"Input video FPS:         {input_fps:.2f}")
    print(f"Input video resolution:  {width}x{height}")
    print(f"Input frame count:       {frame_count if frame_count > 0 else 'unknown'}")
    print(f"Frames processed:        {frames_processed}")
    print(f"Total processing time:   {elapsed:.2f} s")
    print(f"Approx processing FPS:   {approx_processing_fps:.2f}")
    print(f"Output video saved:      {output_path}")
    print("=" * 80)


def __read_first_frame_for_validation(capture: cv2.VideoCapture):
    """
    Read the first frame once to validate model input/output before processing.

    Why:

        The reused Phase 16.2 validator needs a real preprocessed tensor. Reading
        one frame before the loop catches model/session shape problems before
        the script spends time writing a partial output video.
    """

    ok, frame_bgr = capture.read()
    if not ok:
        raise RuntimeError("Input video contains no readable frames.")
    return frame_bgr


if __name__ == "__main__":
    main()
