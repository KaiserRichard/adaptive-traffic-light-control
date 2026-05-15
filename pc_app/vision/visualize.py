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
    - Phase 8 -> traffic light runtime panel
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

# PHASE 8: Draw a traffic light runtime panel showing current state, countdown, active plan, pending plan, and UART status.
def draw_signal_runtime_panel(
    frame,
    runtime_snapshot,
    uart_status=None,
):
    """
    Draw a virtual traffic light runtime panel on the video.

    Purpose:
    - Show current traffic state.
    - Show countdown.
    - Show active plan.
    - Show pending plan if available.
    - Show UART status.

    This helps compare the software-side runtime state with the physical MCU LEDs.
    """

    import cv2

    h, w = frame.shape[:2]

    panel_w = 430
    panel_h = 260

    x1 = w - panel_w - 20
    y1 = 20
    x2 = w - 20
    y2 = y1 + panel_h

    # Background panel.
    cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 2)

    state = runtime_snapshot.state
    remaining = runtime_snapshot.remaining_seconds
    active_plan = runtime_snapshot.active_plan
    pending_plan = runtime_snapshot.pending_plan

    # Color rules in BGR.
    off_color = (60, 60, 60)
    red_color = (0, 0, 255)
    yellow_color = (0, 255, 255)
    green_color = (0, 255, 0)

    a_red = off_color
    a_yellow = off_color
    a_green = off_color

    b_red = off_color
    b_yellow = off_color
    b_green = off_color

    if state == "A_GREEN":
        a_green = green_color
        b_red = red_color
    elif state == "A_YELLOW":
        a_yellow = yellow_color
        b_red = red_color
    elif state == "ALL_RED_AFTER_A":
        a_red = red_color
        b_red = red_color
    elif state == "B_GREEN":
        a_red = red_color
        b_green = green_color
    elif state == "B_YELLOW":
        a_red = red_color
        b_yellow = yellow_color
    elif state == "ALL_RED_AFTER_B":
        a_red = red_color
        b_red = red_color

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        frame,
        "SIGNAL RUNTIME",
        (x1 + 15, y1 + 30),
        font,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"State: {state}",
        (x1 + 15, y1 + 65),
        font,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Remaining: {remaining}s",
        (x1 + 15, y1 + 95),
        font,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Draw virtual lights.
    light_radius = 13

    a_x = x1 + 75
    b_x = x1 + 175
    lights_y = y1 + 135

    cv2.putText(frame, "A", (a_x - 8, lights_y - 30), font, 0.6, (255, 255, 255), 2)
    cv2.circle(frame, (a_x, lights_y), light_radius, a_red, -1)
    cv2.circle(frame, (a_x, lights_y + 35), light_radius, a_yellow, -1)
    cv2.circle(frame, (a_x, lights_y + 70), light_radius, a_green, -1)

    cv2.putText(frame, "B", (b_x - 8, lights_y - 30), font, 0.6, (255, 255, 255), 2)
    cv2.circle(frame, (b_x, lights_y), light_radius, b_red, -1)
    cv2.circle(frame, (b_x, lights_y + 35), light_radius, b_yellow, -1)
    cv2.circle(frame, (b_x, lights_y + 70), light_radius, b_green, -1)

    # Active plan text.
    text_x = x1 + 230
    text_y = y1 + 125

    cv2.putText(
        frame,
        "Active plan:",
        (text_x, text_y),
        font,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    plan_lines = [
        f"A green: {active_plan['green_a']}s",
        f"B green: {active_plan['green_b']}s",
        f"Yellow: {active_plan['yellow']}s",
        f"All-red: {active_plan['all_red']}s",
    ]

    for i, line in enumerate(plan_lines):
        cv2.putText(
            frame,
            line,
            (text_x, text_y + 25 + i * 22),
            font,
            0.47,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    # Pending plan info.
    if pending_plan is not None:
        pending_text = (
            f"Pending: A{pending_plan['green_a']} "
            f"B{pending_plan['green_b']}"
        )
    else:
        pending_text = "Pending: none"

    cv2.putText(
        frame,
        pending_text,
        (x1 + 15, y2 - 45),
        font,
        0.48,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )

    # UART status.
    if uart_status is None:
        uart_text = "UART: disabled/idle"
    else:
        uart_text = uart_status

    cv2.putText(
        frame,
        uart_text[:48],
        (x1 + 15, y2 - 18),
        font,
        0.45,
        (180, 255, 180),
        1,
        cv2.LINE_AA,
    )

    return frame