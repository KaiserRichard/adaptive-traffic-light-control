"""
test_camera_capture.py

Purpose:
- Smoke test camera capture using OpenCV.
- Verify that the camera can be opened.
- Read a small number of frames.
- Save one sample frame for visual inspection.

Why this exists:
- Before debugging YOLO, ROI, scheduler, or UART, we need to prove
  that OpenCV can read from the camera.
- This test isolates camera access from the rest of the pipeline.

Run:
    python -m experiments.test_camera_capture

Typical .env for camera:
    VIDEO_SOURCE=0
    CAMERA_WIDTH=1280
    CAMERA_HEIGHT=720
    CAMERA_FPS=30
"""

from pathlib import Path
import time

import cv2

from pc_app.config import (
    VIDEO_SOURCE,
    VIDEO_SOURCE_RAW,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
    OUTPUTS_DIR,
)


def configure_camera(cap: cv2.VideoCapture) -> None:
    """
    Request camera properties from OpenCV.

    Note:
    The camera driver may ignore or adjust these values.
    Always read back actual values after setting them.
    """

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)


def print_camera_properties(cap: cv2.VideoCapture) -> None:
    """
    Print actual camera properties reported by OpenCV.
    """

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(
        {
            "video_source_raw": VIDEO_SOURCE_RAW,
            "video_source_parsed": VIDEO_SOURCE,
            "requested_width": CAMERA_WIDTH,
            "requested_height": CAMERA_HEIGHT,
            "requested_fps": CAMERA_FPS,
            "actual_width": actual_width,
            "actual_height": actual_height,
            "actual_fps": actual_fps,
        }
    )


def main() -> None:
    """
    Run camera smoke test.
    """

    print("Camera smoke test started")
    print("VIDEO_SOURCE_RAW:", VIDEO_SOURCE_RAW)
    print("VIDEO_SOURCE parsed:", VIDEO_SOURCE)

    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera/video source: {VIDEO_SOURCE!r}. "
            "Check VIDEO_SOURCE in .env and verify camera connection."
        )

    configure_camera(cap)
    print_camera_properties(cap)

    output_dir = OUTPUTS_DIR / "camera_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_frame_path = output_dir / "camera_sample_frame.jpg"

    successful_reads = 0
    first_valid_frame = None

    # Read several frames because some cameras need warm-up time.
    for frame_index in range(30):
        ret, frame = cap.read()

        if not ret or frame is None:
            print({"frame": frame_index, "status": "read_failed"})
            time.sleep(0.05)
            continue

        successful_reads += 1

        if first_valid_frame is None:
            first_valid_frame = frame.copy()

        print(
            {
                "frame": frame_index,
                "status": "ok",
                "shape": frame.shape,
            }
        )

        time.sleep(0.02)

    cap.release()

    if first_valid_frame is None:
        raise RuntimeError("Camera opened but no valid frame was captured.")

    cv2.imwrite(str(sample_frame_path), first_valid_frame)

    print("Camera smoke test finished")
    print("Successful reads:", successful_reads)
    print("Saved sample frame:", sample_frame_path)


if __name__ == "__main__":
    main()