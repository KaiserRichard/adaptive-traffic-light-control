"""
test_detector.py

Purpose:
- Quick test before running the full application
- Check:
    1. video can open
    2. first frame can be read
    3. detector works on one frame
    4. returned detections look correct
"""

import cv2
from pc_app.config import VIDEO_SOURCE
from pc_app.vision.detector import RoboflowDetector


def main():
    # Open video
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

    # Initialize detector
    detector = RoboflowDetector()

    # Read only the first frame
    ret, frame = cap.read()

    if not ret:
        raise RuntimeError("Failed to read first frame.")

    # Run detection on one frame only
    detections = detector.detect(frame)

    print("First-frame detections:")
    for det in detections[:10]:
        print(det)

    cap.release()


if __name__ == "__main__":
    main()