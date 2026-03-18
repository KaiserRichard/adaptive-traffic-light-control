"""
roi.py: decides where detections belong

Purpose:
    - Load region-of-interest (ROI) polygons from JSON configuration.
    - Assign detetions into Direction A, B, or outside both regions
    - Provide helper functions for bounding-box center and point-in-polygon tests. 
"""
from typing import Tuple, Dict, Any, List
import json
import numpy as np
import cv2

Point = Tuple[int, int]
Polygon = List[Point]

# Load ROI configuration from a JSON file
def load_roi_config(json_path: str) -> Dict[str, Any]:
    '''
    Expected JSON format:
        {
        "direction_a_name": "A",
        "direction_b_name": "B",
        "roi_a": [[x1,y1], [x2, y2], ...],
        "roi_b": [[x1,y1], [x2, y2], ...],
        }
    '''

    with open(file=json_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    return{
        "direction_a_name": data["direction_a_name"],
        "direction_b_name": data["direction_b_name"],
        "roi_a": [tuple(point) for point in data["roi_a"]], 
        "roi_b": [tuple(point) for point in data["roi_b"]]
    }


# Compute the center point of a bounding box
def get_bbox_center(bbox: List[int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy

# Check whether a point lies inside or on the edge of a polygon 
def point_in_polygon(point: Tuple[int, int], polygon: Polygon) -> bool:
    '''
    OpenCV pointPolygonTest returns:
        > 0 : inside
        = 0 : on edge
        < 0 : outside
    '''
    polygon_np = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(contour=polygon_np, pt=point, measureDist=False) # We dont need to measure Distance
    return result >= 0 

# Check whether a detection belongs to a polygon ROI using the center of the bounding box.
def detection_in_roi(detection: Dict[str, Any], polygon: Polygon) -> bool :
    center = get_bbox_center(detection["bbox"])
    return point_in_polygon(center, polygon)

'''
Split detections into:
    - detections_a and detections_b: detections whose centers fall inside ROI A and B
    - detections_outside: detectiosn outside both ROIs
'''
def split_detections_by_direction(
        detections: List[Dict[str, Any]],
        roi_a: Polygon, 
        roi_b: Polygon):
    detections_a = []
    detections_b = []
    detections_outside = []

    for det in detections:
        center = get_bbox_center(det["bbox"])
        in_a = point_in_polygon(center, roi_a)
        in_b = point_in_polygon(center, roi_b)

        if in_a and not in_b:
            detections_a.append(det)
        elif in_b and not in_a:
            detections_b.append(det)
        # If both polygons overlap and a detection center is inside both -> closer ROI centroid
        elif in_a and in_b:
            # Resolve overlap by nearest x-centroid
            ax = np.mean([p[0] for p in roi_a])
            bx = np.mean([p[0] for p in roi_b])

            if abs(center[0] - ax) <= abs(center[0] - bx):
                detections_a.append(det)
            else: 
                detections_b.append(det)

        else: 
            detections_outside.append(det)

    return detections_a, detections_b, detections_outside