"""
detector_yolo.py

Purpose:
- Provide a local YOLO detector backend using Ultralytics.
- Replace hosted Roboflow inference for Raspberry Pi / edge deployment.
- Keep the same output format as the existing detector pipeline.

Output format:
[
    {
        "bbox": [x1, y1, x2, y2],
        "conf": 0.87,
        "class_name": "car"
    }
]
"""

from typing import List, Dict, Any
from pathlib import Path

from ultralytics import YOLO

from pc_app.config import (
    CONFIDENCE_THRESHOLD,
    CLASS_NORMALIZATION,
    YOLO_MODEL_PATH,
    YOLO_IMGSZ,
    YOLO_VERBOSE,
)
from pc_app.vision.classes import normalize_class_name


class LocalYoloDetector:
    """
    Local YOLO detector wrapper.

    This class hides Ultralytics-specific output details from the rest
    of the project.

    The rest of the pipeline only calls:

        detector.detect(frame)

    This makes it possible to switch between:
    - YOLO26 .pt
    - YOLOv8 .pt
    - ONNX export
    - NCNN export

    without changing ROI, density, scheduler, or visualization logic.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or YOLO_MODEL_PATH

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"YOLO model file not found: {self.model_path}\n"
                "Place the model under pc_app/models/local/ or update YOLO_MODEL_PATH."
            )

        self.model = YOLO(self.model_path)

    def detect(self, frame) -> List[Dict[str, Any]]:
        """
        Run local YOLO inference on one OpenCV frame.

        Input:
            frame: OpenCV BGR image

        Output:
            list of normalized detections
        """

        results = self.model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=YOLO_IMGSZ,
            verbose=YOLO_VERBOSE,
        )

        detections: List[Dict[str, Any]] = []

        for result in results:
            names = result.names
            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                raw_class = names.get(cls_id, str(cls_id))

                normalized = normalize_class_name(raw_class, CLASS_NORMALIZATION)

                if normalized is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf,
                        "class_name": normalized,
                    }
                )

        return detections