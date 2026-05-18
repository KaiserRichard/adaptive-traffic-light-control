"""
setup_roi.py

Purpose:
    Interactive ROI setup tool.

    Supports:
    - video file
    - camera index
    - image file

Controls:
    - Left mouse click: add a point
    - n: finish current ROI and move to next one
    - s: save ROI JSON and exit
    - q: quit without saving
    - r: reset current ROI points

Workflow:
    1) Draw ROI A with at least 3 points, press 'n'
    2) Draw ROI B with at least 3 points, press 'n'
    3) Press 's' to save configuration

Output:
    Saves ROI JSON at ROI_CONFIG_PATH.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from pc_app.config import VIDEO_SOURCE, ROI_CONFIG_PATH


current_points = []
roi_a = []
roi_b = []
stage = "A"


def resolve_source(source):
    """
    Convert camera index string into int if needed.

    Example:
        "0" -> 0
        "./video.mp4" -> "./video.mp4"
        "./frame.jpg" -> "./frame.jpg"
    """

    if isinstance(source, str) and source.isdigit():
        return int(source)

    return source


def load_base_frame(source):
    """
    Load one frame from image, video, or camera.
    """

    source = resolve_source(source)

    if isinstance(source, str):
        suffix = Path(source).suffix.lower()

        if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            frame = cv2.imread(source)

            if frame is None:
                raise RuntimeError(f"Cannot read image source: {source}")

            return frame

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video/camera source: {source}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to read first frame from video/camera source.")

    return frame


def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback for adding polygon points.
    """

    global current_points

    if event == cv2.EVENT_LBUTTONDOWN:
        print("CLICK:", x, y)
        current_points.append((x, y))


def draw_points_and_polygon(display, points, color, closed=False):
    """
    Draw selected points and polygon lines.
    """

    for pt in points:
        cv2.circle(
            img=display,
            center=pt,
            radius=4,
            color=color,
            thickness=-1,
        )

    if len(points) > 1:
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(
            img=display,
            pts=[pts],
            isClosed=closed,
            color=color,
            thickness=2 if closed else 1,
        )


def draw_completed_roi(display, roi, label, color):
    """
    Draw completed ROI polygon with label.
    """

    if not roi:
        return

    pts = np.array(roi, dtype=np.int32)

    cv2.polylines(
        img=display,
        pts=[pts],
        isClosed=True,
        color=color,
        thickness=2,
    )

    cv2.putText(
        img=display,
        text=label,
        org=roi[0],
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        color=color,
        thickness=2,
    )


def save_roi_config() -> None:
    """
    Save ROI configuration to JSON.
    """

    data = {
        "direction_a_name": "A",
        "direction_b_name": "B",
        "roi_a": roi_a,
        "roi_b": roi_b,
    }

    output_path = Path(ROI_CONFIG_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved ROI config to: {output_path}")


def main():
    """
    Main interactive ROI setup loop.
    """

    global current_points, roi_a, roi_b, stage

    print("ROI setup started")
    print("Source:", VIDEO_SOURCE)
    print("Output:", ROI_CONFIG_PATH)

    base_frame = load_base_frame(VIDEO_SOURCE)

    cv2.namedWindow("ROI Setup")
    cv2.setMouseCallback("ROI Setup", mouse_callback)

    while True:
        display = base_frame.copy()

        draw_completed_roi(display, roi_a, "ROI A", (0, 255, 0))
        draw_completed_roi(display, roi_b, "ROI B", (255, 0, 0))

        draw_points_and_polygon(
            display=display,
            points=current_points,
            color=(0, 255, 255),
            closed=False,
        )

        message = (
            f"Drawing ROI {stage} | click: add point | "
            "n: next | r: reset current | s: save | q: quit"
        )

        cv2.putText(
            img=display,
            text=message,
            org=(20, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(255, 255, 255),
            thickness=2,
        )

        cv2.imshow("ROI Setup", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("n"):
            if stage == "A":
                if len(current_points) >= 3:
                    roi_a = current_points.copy()
                    current_points = []
                    stage = "B"
                    print("ROI A completed")
                else:
                    print("ROI A needs at least 3 points.")

            elif stage == "B":
                if len(current_points) >= 3:
                    roi_b = current_points.copy()
                    current_points = []
                    stage = "DONE"
                    print("ROI B completed. Press 's' to save.")
                else:
                    print("ROI B needs at least 3 points.")

            else:
                print("Both ROIs are already completed. Press 's' to save.")

        elif key == ord("r"):
            current_points = []
            print(f"Reset current ROI points for stage {stage}")

        elif key == ord("s"):
            if roi_a and roi_b:
                save_roi_config()
                break

            print("Both ROI A and ROI B must be defined before saving.")

        elif key == ord("q"):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()