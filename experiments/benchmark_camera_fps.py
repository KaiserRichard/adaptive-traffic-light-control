"""
benchmark_camera_fps.py

Purpose:
- Measure raw camera read FPS.
- This benchmark does NOT run YOLO.
- This benchmark does NOT run ROI, density, scheduler, UART, or visualization.
- It only measures how fast OpenCV can read frames from the camera/video source.

Why this exists:
- If raw camera FPS is low, the bottleneck is camera capture or camera driver.
- If raw camera FPS is high but full pipeline FPS is low, the bottleneck is likely
  YOLO inference, display, video writing, or later processing stages.

Run:
    python -m experiments.benchmark_camera_fps

Typical .env:
    VIDEO_SOURCE=0
    CAMERA_WIDTH=1280
    CAMERA_HEIGHT=720
    CAMERA_FPS=30
"""

import json
import time
from pathlib import Path

import cv2

from pc_app.config import (
    VIDEO_SOURCE,
    VIDEO_SOURCE_RAW,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
    OUTPUTS_DIR,
)


BENCHMARK_FRAMES = 300
WARMUP_FRAMES = 30


def configure_camera(cap: cv2.VideoCapture) -> None:
    """
    Request camera properties from OpenCV.
    """

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)


def get_camera_properties(cap: cv2.VideoCapture) -> dict:
    """
    Read actual camera properties reported by OpenCV.
    """

    return {
        "actual_width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "actual_height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "actual_fps": cap.get(cv2.CAP_PROP_FPS),
    }


def main() -> None:
    """
    Run raw camera FPS benchmark.
    """

    print("Raw camera FPS benchmark started")
    print("VIDEO_SOURCE_RAW:", VIDEO_SOURCE_RAW)
    print("VIDEO_SOURCE parsed:", VIDEO_SOURCE)

    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera/video source: {VIDEO_SOURCE!r}. "
            "Check VIDEO_SOURCE in .env and verify camera connection."
        )

    configure_camera(cap)

    camera_properties = get_camera_properties(cap)

    print(
        {
            "requested_width": CAMERA_WIDTH,
            "requested_height": CAMERA_HEIGHT,
            "requested_fps": CAMERA_FPS,
            **camera_properties,
        }
    )

    # Warm up camera.
    warmup_success = 0

    for _ in range(WARMUP_FRAMES):
        ret, frame = cap.read()
        if ret and frame is not None:
            warmup_success += 1

    print({"warmup_frames": WARMUP_FRAMES, "warmup_success": warmup_success})

    frame_count = 0
    failed_reads = 0
    read_times = []

    benchmark_start = time.perf_counter()

    while frame_count < BENCHMARK_FRAMES:
        read_start = time.perf_counter()
        ret, frame = cap.read()
        read_end = time.perf_counter()

        if not ret or frame is None:
            failed_reads += 1
            continue

        read_times.append(read_end - read_start)
        frame_count += 1

        if frame_count % 30 == 0:
            elapsed = time.perf_counter() - benchmark_start
            current_fps = frame_count / elapsed if elapsed > 0 else 0.0
            print(
                {
                    "frame": frame_count,
                    "current_avg_fps": round(current_fps, 2),
                    "last_read_ms": round((read_end - read_start) * 1000, 2),
                }
            )

    benchmark_end = time.perf_counter()
    cap.release()

    total_time = benchmark_end - benchmark_start
    avg_fps = frame_count / total_time if total_time > 0 else 0.0

    avg_read_ms = (
        sum(read_times) / len(read_times) * 1000.0
        if read_times
        else 0.0
    )

    metrics = {
        "video_source_raw": VIDEO_SOURCE_RAW,
        "video_source_parsed": str(VIDEO_SOURCE),
        "requested_width": CAMERA_WIDTH,
        "requested_height": CAMERA_HEIGHT,
        "requested_fps": CAMERA_FPS,
        **camera_properties,
        "benchmark_frames": BENCHMARK_FRAMES,
        "warmup_frames": WARMUP_FRAMES,
        "warmup_success": warmup_success,
        "successful_frames": frame_count,
        "failed_reads": failed_reads,
        "total_time_seconds": total_time,
        "avg_camera_read_fps": avg_fps,
        "avg_read_ms": avg_read_ms,
        "min_read_ms": min(read_times) * 1000.0 if read_times else 0.0,
        "max_read_ms": max(read_times) * 1000.0 if read_times else 0.0,
    }

    output_dir = OUTPUTS_DIR / "benchmarks" / "camera_raw_fps"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    notes_path = output_dir / "run_notes.md"

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    notes_path.write_text(
        "\n".join(
            [
                "# Raw Camera FPS Benchmark",
                "",
                "## Purpose",
                "",
                "This benchmark measures raw OpenCV camera read performance.",
                "",
                "It does not run YOLO, ROI splitting, density estimation, scheduling, UART, display, or video writing.",
                "",
                "## Configuration",
                "",
                "```text",
                f"VIDEO_SOURCE={VIDEO_SOURCE_RAW}",
                f"CAMERA_WIDTH={CAMERA_WIDTH}",
                f"CAMERA_HEIGHT={CAMERA_HEIGHT}",
                f"CAMERA_FPS={CAMERA_FPS}",
                "```",
                "",
                "## Results",
                "",
                "```text",
                f"Successful frames: {metrics['successful_frames']}",
                f"Failed reads: {metrics['failed_reads']}",
                f"Average camera read FPS: {metrics['avg_camera_read_fps']:.2f}",
                f"Average read time: {metrics['avg_read_ms']:.2f} ms",
                f"Minimum read time: {metrics['min_read_ms']:.2f} ms",
                f"Maximum read time: {metrics['max_read_ms']:.2f} ms",
                "```",
                "",
                "## Interpretation",
                "",
                "If this FPS is high but the full pipeline FPS is low, the bottleneck is likely YOLO inference, display, video writing, or later processing stages.",
                "",
                "If this FPS is already low, the bottleneck may be the camera, camera driver, resolution, USB bandwidth, or Raspberry Pi camera configuration.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("Raw camera FPS benchmark finished")
    print(json.dumps(metrics, indent=2))
    print("Saved metrics to:", metrics_path)
    print("Saved notes to:", notes_path)


if __name__ == "__main__":
    main()