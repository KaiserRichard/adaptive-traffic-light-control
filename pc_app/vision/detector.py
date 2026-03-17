'''
detector.py

Purpose:
    - Wrap the detector inside a class
    - Provide ONE public method:
        detect(frame) -> list of detections
'''

from typing import List, Dict, Any
from inference_sdk import InferenceHTTPClient

from pc_app.config import (
    ROBOFLOW_API_KEY,
    ROBOFLOW_MODEL_ID,
    CONFIDENCE_THRESHOLD,
    CLASS_NORMALIZATION,
)

from pc_app.vision.classes import normalize_class_name

'''
Detector wrapper for Roboflow inference
'''
class RoboflowDetector:
    # Initialize the Roboflow client
    def __init__(self) -> None:
        # Immediately validate the API key and model ID 
        if not ROBOFLOW_API_KEY:
            raise ValueError("ROBOFLOW_API_KEY is missing in environment.")

        if not ROBOFLOW_MODEL_ID:
            raise ValueError("ROBOFLOW_MODEL_ID is missing in environment.")
        
        # Roboflow inference client
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_API_KEY,
        )

        self.model_id = ROBOFLOW_MODEL_ID
    
    def detect(self, frame) -> List[Dict[str, Any]]:

        # Send frame to Roboflow model and receive predicted dictionary
        result = self.client.infer(frame,model_id=self.model_id)

        detections: List[Dict[str, Any]] = []

        # If there is no predictions, use empty list instead of crashing
        predictions = result.get("predictions", [])

        for pred in predictions:
            # Read confidence score safely
            conf = float(pred.get("confidence", 0.0))

            # Skip weak detections
            if conf < CONFIDENCE_THRESHOLD:
                continue

            # Raw class name from model
            raw_class = pred.get("class", "")

            # Normalize class
            normalized = normalize_class_name(raw_name=raw_class, mapping=CLASS_NORMALIZATION)

            # Skip unsupported classes
            if normalized is None:
                continue

            # Roboflow box format : center x, center y, width, height
            x_center, y_center = float(pred["x"]), float(pred["y"])
            width, height = float(pred["width"]), float(pred["height"])

            # Convert center-based box -> corner-based box
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "class_name": normalized,
            }) 

        return detections
    
    