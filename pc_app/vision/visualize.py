'''
visualize.py

Purpose: 
    - Draw bouding boxes
    - Draw text labels
    - Count detections by class
    - Draw a small status panel (FPS + counts)
'''

from collections import Counter
from typing import List, Dict, Any
import cv2

def draw_detections(frame, detections: List[Dict[str, Any]]):
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
            color=(0, 255, 0),
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
            color=(0, 255, 0),
            thickness=2,
        )

    return frame

def count_by_class(detections: List[Dict[str, Any]]) -> Dict[str, int]:
    '''
    Count how many detections belong to each class.
    '''

    counts = Counter()

    for det in detections:
        counts[det["class_name"]] += 1

    # Convert Counter object to a normal dictionary
    return dict(counts)

def draw_counts_panel(frame, counts: Dict[str, int], fps: float):
    '''
    Draw FPS and class counts at the top-left corner.
    
    Layout:
        FPS: xx.xx
        bus: 1
        car: 4
        motorbike: 12
        truck: 1

    y starts at 30 to leave some top margin
    '''

    # Draw FPS text
    y= 30 # Location where we put FPS text
    cv2.putText(
        img=frame, 
        text=f"FPS: {fps:.2f}",
        org=(20,y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        color=(0,255,255),
        thickness=2,
    )

    y += 30

    # Draw each class count
    for cls_name in sorted(counts.keys()):
        cv2.putText(
        img=frame, 
        text=f"{cls_name}: {counts[cls_name]}",
        org=(20,y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        color=(0,255,255),
        thickness=2,
        )
        y += 30

    return frame