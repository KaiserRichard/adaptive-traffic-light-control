"""
benchmark_detector.py

Purpose:
- Benchmark detector runtime on a fixed video.
- Save metrics.json.
- Save run_notes.md.
- Automatically name output folder by model, format, and device.

Run:
    python -m experiments.benchmark_detector
"""

import json
import platform
import time
from pathlib import Path

import cv2

from pc_app.config import VIDEO_SOURCE, DETECTOR_BACKEND, YOLO_MODEL_PATH, BENCHMARK_MAX_FRAMES
from pc_app.vision.detector_factory import create_detector


def get_device_name() -> str:
    """
    Return a simple device name for benchmark folder naming.
    """

    machine = platform.machine().lower()
    system = platform.system().lower()

    if "arm" in machine or "aarch64" in machine:
        return "raspi"

    if system in {"darwin", "windows", "linux"}:
        return "pc"

    return "unknown"


def get_model_format(model_path: str) -> str:
    """
    Return model format:
    - pt
    - onnx
    - ncnn
    - roboflow
    """

    backend = DETECTOR_BACKEND.lower().strip()

    if backend == "roboflow":
        return "roboflow"

    lower_path = model_path.lower()
    path = Path(model_path)

    if "ncnn" in lower_path or path.is_dir():
        return "ncnn"

    suffix = path.suffix.lower().replace(".", "")

    if suffix:
        return suffix

    return "unknown"


def get_model_name(model_path: str) -> str:
    """
    Extract model name from model path.

    Example:
    ./pc_app/models/local/yolov8n.pt -> yolov8n
    ./pc_app/models/local/yolo26n.pt -> yolo26n
    """

    backend = DETECTOR_BACKEND.lower().strip()

    if backend == "roboflow":
        return "roboflow"

    path = Path(model_path)

    if path.suffix:
        name = path.stem
    else:
        name = path.name

    name = name.replace("_ncnn_model", "")

    return name.lower()


def get_benchmark_id() -> str:
    """
    Build benchmark ID.

    Format:
        model_format_device

    Example:
        yolov8n_pt_pc
        yolo26n_pt_pc
    """

    model_name = get_model_name(YOLO_MODEL_PATH)
    model_format = get_model_format(YOLO_MODEL_PATH)
    device_name = get_device_name()

    return f"{model_name}_{model_format}_{device_name}"


def get_output_dir() -> Path:
    """
    Build output directory path.
    """

    return Path("outputs") / "benchmarks" / get_benchmark_id()


def write_run_notes(output_dir: Path, metrics: dict) -> None:
    """
    Write human-readable benchmark notes.
    """

    notes_path = output_dir / "run_notes.md"

    lines = [
        f"# Benchmark: {metrics['benchmark_id']}",
        "",
        "## Purpose",
        "",
        "This benchmark records local detector runtime on a fixed traffic video.",
        "",
        "## Detector Configuration",
        "",
        "```text",
        f"DETECTOR_BACKEND={metrics['detector_backend']}",
        f"YOLO_MODEL_PATH={metrics['yolo_model_path']}",
        f"VIDEO_SOURCE={metrics['video_source']}",
        "```",
        "",
        "## Metrics",
        "",
        "```text",
        f"Frame count: {metrics['frame_count']}",
        f"Total detections: {metrics['total_detections']}",
        f"Average detections per frame: {metrics['avg_detections_per_frame']:.4f}",
        f"Average inference time: {metrics['avg_inference_ms']:.4f} ms/frame",
        f"Average FPS: {metrics['avg_fps']:.4f}",
        f"Minimum inference time: {metrics['min_inference_ms']:.4f} ms",
        f"Maximum inference time: {metrics['max_inference_ms']:.4f} ms",
        "```",
        "",
        "## Interpretation",
        "",
        "This benchmark measures detector inference only.",
        "",
        "It does not include ROI splitting, density estimation, signal scheduling, or visualization overhead.",
        "",
    ]

    notes_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """
    Main benchmark function.
    """

    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Benchmark started")
    print("Detector backend:", DETECTOR_BACKEND)
    print("YOLO model path:", YOLO_MODEL_PATH)
    print("Video source:", VIDEO_SOURCE)
    print("Output dir:", output_dir)

    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

    detector = create_detector()

    frame_count = 0
    total_detections = 0
    inference_times = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        start_time = time.perf_counter()
        detections = detector.detect(frame)
        end_time = time.perf_counter()

        inference_time = end_time - start_time

        inference_times.append(inference_time)
        total_detections += len(detections)
        frame_count += 1


        if frame_count % 30 == 0:
            current_fps = 1.0 / inference_time if inference_time > 0 else 0.0
            print(
                {
                    "frame": frame_count,
                    "detections": len(detections),
                    "inference_ms": round(inference_time * 1000, 2),
                    "fps": round(current_fps, 2),
                }
            )

        if frame_count >= BENCHMARK_MAX_FRAMES:
            print(f"Reached BENCHMARK_MAX_FRAMES={BENCHMARK_MAX_FRAMES}. Stopping benchmark.")
            break

    cap.release()

    if frame_count == 0:
        raise RuntimeError("No frames were processed. Check VIDEO_SOURCE.")

    total_inference_time = sum(inference_times)
    avg_inference_time = total_inference_time / frame_count
    avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0

    metrics = {
        "benchmark_id": get_benchmark_id(),
        "benchmark_max_frames": BENCHMARK_MAX_FRAMES,
        "detector_backend": DETECTOR_BACKEND,
        "video_source": VIDEO_SOURCE,
        "yolo_model_path": YOLO_MODEL_PATH,
        "device": get_device_name(),
        "model_name": get_model_name(YOLO_MODEL_PATH),
        "model_format": get_model_format(YOLO_MODEL_PATH),
        "frame_count": frame_count,
        "total_detections": total_detections,
        "avg_detections_per_frame": total_detections / frame_count,
        "avg_inference_ms": avg_inference_time * 1000,
        "avg_fps": avg_fps,
        "min_inference_ms": min(inference_times) * 1000,
        "max_inference_ms": max(inference_times) * 1000,
    }

    metrics_path = output_dir / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    write_run_notes(output_dir, metrics)

    print("Benchmark finished")
    print(json.dumps(metrics, indent=2))
    print("Saved metrics to:", metrics_path)
    print("Saved notes to:", output_dir / "run_notes.md")


if __name__ == "__main__":
    main()
