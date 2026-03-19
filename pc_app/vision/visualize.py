'''
visualize.py

Purpose: 
    - Draw bouding boxes
    - Draw text labels
    - Count detections by class
    - Draw a small status panel (FPS + counts)
'''

import numpy as np
from typing import List, Dict, Any, Tuple
import cv2
from pc_app.vision.roi import get_bbox_center

def draw_detections(frame, detections: List[Dict[str, Any]], color=(0, 255, 0)):
    '''
    Draw bounding boxes and class labels on the frame.

    Each detection is expected to look like: 
        {
            "bbox": [x1, y1, x2, y2],
            "conf": 0.88,
            "class_name": "car"
        }
    
    We draw: 
    - Green rectangle
    - class label + confidence score above the box
    '''

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["conf"]
        cls = det["class_name"]

        # Draw bounding box
        cv2.rectangle(
            img=frame,
            pt1=(x1,y1),
            pt2=(x2,y2),
            color=color, # Green
            thickness=2
        )

        # Build text label
        label = f"{cls} {conf:.2f}"
        
        # Put label slightly above the box
        cv2.putText(
            img=frame,
            text=label,
            org=(x1, max(20, y1-8)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=color,
            thickness=2,
        )

    return frame

# Draw the center point of each bounding box for debugging ROI assignment
def draw_bbox_centers(frame, detections: List[Dict[str, Any]], color=(0 ,0, 255)):
    for det in detections:
        cx, cy = get_bbox_center(det["bbox"])
        cv2.circle(frame, (cx, cy), 4, color, -1)
    
    return frame

# Draw a closed polygon ROI and a text label.
def draw_polygon(frame, polygon: List[Tuple[int, int]], label: str, color):
    polygon_np = np.array(polygon, dtype=np.int32)
    cv2.polylines(
        img=frame,
        pts=[polygon_np],
        isClosed=True,
        color=color,
        thickness=2
    )

    # Draw the label (ROI A / ROI B)
    label_x, label_y = polygon[0]
    cv2.putText(
        img=frame,
        text= f"ROI {label}",
        org=(label_x, max(20, label_y-10)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        color=color,
        thickness=2
    )

    return frame

# Draw one block of class counts
def draw_counts_block(
        frame,
        title: str,
        counts: Dict[str, int],
        top_left: Tuple[int, int],
        color=(255,255,255),
):
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
    
    # Draw each class count
    for cls_name in sorted(counts.keys()):
        cv2.putText(
        img=frame, 
        text=f"{cls_name}: {counts[cls_name]}",
        org=(x,y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.65,
        color=color,
        thickness=2,
        )
        y += 26
    
    return frame

# Phase 3: Draw one density summary block.
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

# Draw overall FPS, total detections, and per-direction count blocks.
def draw_status_panel(
        frame,
        fps: float,
        total_detections: int,
        counts_a: Dict[str, int],
        counts_b: Dict[str, int],
        density_a: Dict[str, float], # PHASE 3
        density_b: Dict[str, float],
):
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

    # PHASE 2: Draw density block
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

    return frame