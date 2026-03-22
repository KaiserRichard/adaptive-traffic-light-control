"""
visualize.py

Purpose:
    - Draw bounding boxes
    - Draw text labels
    - Draw ROI polygons
    - Draw bbox center points for ROI debugging
    - Draw count, density, and timing panels

Previous phase additions:
    - Phase 2 -> ROI polygons, direction-aware detections, bbox centers
    - Phase 3 -> density blocks
    - Phase 4 -> timing block
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import cv2
from pc_app.vision.roi import get_bbox_center


def draw_detections(frame, detections: List[Dict[str, Any]], color=(0, 255, 0)):
    """
    Draw bounding boxes and class labels on the frame.

    Each detection is expected to look like:
        {
            "bbox": [x1, y1, x2, y2],
            "conf": 0.88,
            "class_name": "car"
        }
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["conf"]
        cls = det["class_name"]

        cv2.rectangle(
            img=frame,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=color,
            thickness=2
        )

        label = f"{cls} {conf:.2f}"

        cv2.putText(
            img=frame,
            text=label,
            org=(x1, max(20, y1 - 8)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=color,
            thickness=2,
        )

    return frame


def draw_bbox_centers(frame, detections: List[Dict[str, Any]], color=(0, 0, 255)):
    """
    Draw the center point of each bounding box.
    Useful for debugging ROI assignment.
    """
    for det in detections:
        cx, cy = get_bbox_center(det["bbox"])
        cv2.circle(frame, (cx, cy), 4, color, -1)

    return frame


def draw_polygon(frame, polygon: List[Tuple[int, int]], label: str, color):
    """
    Draw a closed polygon ROI and a text label.
    """
    polygon_np = np.array(polygon, dtype=np.int32)
    cv2.polylines(
        img=frame,
        pts=[polygon_np],
        isClosed=True,
        color=color,
        thickness=2
    )

    label_x, label_y = polygon[0]
    cv2.putText(
        img=frame,
        text=f"ROI {label}",
        org=(label_x, max(20, label_y - 10)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        color=color,
        thickness=2
    )

    return frame


def draw_counts_block(
    frame,
    title: str,
    counts: Dict[str, int],
    top_left: Tuple[int, int],
    color=(255, 255, 255),
):
    """
    Draw one block of class counts.
    """
    x, y = top_left

    cv2.putText(
        img=frame,
        text=title,
        org=(x, y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.75,
        color=color,
        thickness=2
    )
    y += 28

    if not counts:
        cv2.putText(
            img=frame,
            text="None",
            org=(x, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.75,
            color=color,
            thickness=2
        )
        return frame

    for cls_name in sorted(counts.keys()):
        cv2.putText(
            img=frame,
            text=f"{cls_name}: {counts[cls_name]}",
            org=(x, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.65,
            color=color,
            thickness=2,
        )
        y += 26

    return frame


# PHASE 3:
# Draw one density summary block.
def draw_density_block(
    frame,
    title: str,
    raw_density: float,
    pce_density: float,
    smoothed_pce_density: float,
    top_left: Tuple[int, int],
    color=(255, 255, 255)
):
    x, y = top_left

    cv2.putText(
        frame,
        title,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
    )
    y += 28

    lines = [
        f"raw density: {raw_density:.2f}",
        f"pce density: {pce_density:.2f}",
        f"smoothed pce: {smoothed_pce_density:.2f}",
    ]

    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
        )
        y += 24

    return frame


# PHASE 4:
# Draw one block showing the computed traffic signal timing plan.
def draw_timing_block(
    frame,
    title: str,
    signal_plan: Dict[str, int | float],
    top_left: Tuple[int, int],
    color=(255, 255, 255)
):
    x, y = top_left

    cv2.putText(
        frame,
        title,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2
    )
    y += 28

    lines = [
        f"green A: {signal_plan['green_a']} s",
        f"green B: {signal_plan['green_b']} s",
        f"yellow: {signal_plan['yellow']} s",
        f"all red: {signal_plan['all_red']} s",
    ]

    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2
        )
        y += 24

    return frame


def draw_status_panel(
    frame,
    fps: float,
    total_detections: int,
    counts_a: Dict[str, int],
    counts_b: Dict[str, int],
    density_a: Dict[str, float],
    density_b: Dict[str, float],
    signal_plan: Dict[str, float | int],   # PHASE 4
):
    """
    Draw overall FPS, total detections, count summaries,
    density summaries, and timing summaries.
    """
    cv2.putText(
        img=frame,
        text=f"FPS: {fps:.2f}",
        org=(20, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.75,
        color=(0, 255, 255),
        thickness=2
    )

    cv2.putText(
        img=frame,
        text=f"Total detections: {total_detections}",
        org=(20, 60),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.75,
        color=(0, 255, 255),
        thickness=2
    )

    frame = draw_counts_block(frame, "Direction A", counts_a, (20, 100), color=(0, 255, 0))
    frame = draw_counts_block(frame, "Direction B", counts_b, (20, 250), color=(255, 0, 0))

    # PHASE 3:
    # Draw density summary blocks for Direction A and Direction B.
    frame = draw_density_block(
        frame,
        "Direction A Density",
        raw_density=density_a["raw_density"],
        pce_density=density_a["pce_density"],
        smoothed_pce_density=density_a["smoothed_pce_density"],
        top_left=(800, 100),
        color=(255, 255, 255),
    )

    frame = draw_density_block(
        frame,
        "Direction B Density",
        raw_density=density_b["raw_density"],
        pce_density=density_b["pce_density"],
        smoothed_pce_density=density_b["smoothed_pce_density"],
        top_left=(800, 220),
        color=(255, 255, 255),
    )

    # PHASE 4:
    # Draw timing block after density blocks.
    frame = draw_timing_block(
        frame=frame,
        title="Signal Plan",
        signal_plan=signal_plan,
        top_left=(800, 360),
        color=(255, 255, 255)
    )

    return frame