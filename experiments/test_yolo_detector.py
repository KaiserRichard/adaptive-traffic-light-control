"""
test_yolo_detector.py

Purpose:
- Test local YOLO detector on the first frame of a video.
- Verify model loading, inference, class normalization, and output format.
"""

# Phase 6: YOLO smoke test on first frame before running the full app.
import cv2

from pc_app.config import VIDEO_SOURCE
from pc_app.vision.detector_yolo import LocalYoloDetector


def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to read first frame.")

    detector = LocalYoloDetector()
    detections = detector.detect(frame)

    print("Local YOLO detections:")
    print("Total:", len(detections))

    for det in detections[:10]:
        print(det)


if __name__ == "__main__":
    main()