'''
test_roi.py

Purpose:
    - Verify ROI loading and direction assignment on the first video frame.
    - Helps debug ROI JSON and split logic before running full pipelin
'''

import cv2
from pc_app.config import VIDEO_SOURCE, ROI_CONFIG_PATH
from pc_app.vision.detector import RoboflowDetector
from pc_app.vision.roi import load_roi_config,  split_detections_by_direction

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")
    
    detector = RoboflowDetector()

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read first frame.")
    
    roi_cfg = load_roi_config(ROI_CONFIG_PATH)
    roi_a = roi_cfg["roi_a"]
    roi_b = roi_cfg["roi_b"]

    detections = detector.detect(frame=frame)
    detections_a, detections_b, detections_outside = split_detections_by_direction(detections, roi_a, roi_b)

    print("Total detections:", len(detections))
    print("Direction A detections:", len(detections_a))
    print("Direction B detections:", len(detections_b))
    print("Outside detections:", len(detections_outside))

    cap.release()

if __name__ == '__main__':
    main()
    
