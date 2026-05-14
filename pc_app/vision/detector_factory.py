"""
detector_factory.py

Purpose:
- Select the detector backend from config.
- Keep main.py clean.
- Support both Roboflow baseline and local YOLO deployment.
"""

from pc_app.config import DETECTOR_BACKEND


def create_detector():
    """
    Create detector backend based on DETECTOR_BACKEND.

    Supported:
    - roboflow
    - yolo
    """

    backend = DETECTOR_BACKEND.strip().lower()

    if backend == "roboflow":
        from pc_app.vision.detector import RoboflowDetector

        return RoboflowDetector()

    if backend == "yolo":
        from pc_app.vision.detector_yolo import LocalYoloDetector

        return LocalYoloDetector()

    raise ValueError(
        f"Unsupported DETECTOR_BACKEND: {DETECTOR_BACKEND}. "
        "Expected 'roboflow' or 'yolo'."
    )