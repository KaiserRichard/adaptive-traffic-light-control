''''
config.py

Purpose:
- Keep all configuration variables in one place.
- Avoid hard-coding paths, API keys, class mappings, and thresholds
'''

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment from .env file
load_dotenv(override=True)

# BASE_DIR points to the root folder of the project.
# altc_project/
BASE_DIR = Path(__file__).resolve().parent.parent

# Standard folders used
DATASETS_DIR = BASE_DIR / "datasets"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Video source: 
# Can be a file path or webcam
VIDEO_SOURCE = os.getenv(
    "VIDEO_SOURCE",
    str(DATASETS_DIR / "sample_videos" / "test.mp4")
)

# Roboflow credentials
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "")

# Detection confidence threshold:
# Only keep predictions whose confidence >= 0.4 (means 40%)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4")) 

# Class Normalization
'''
Finish later
'''
CLASS_NORMALIZATION = {
    # MOTORBIKE
    "motorbike": "motorbike",
    "scooter": "motorbike",
    "bicycle": "motorbike",

    # CAR GROUP
    "car": "car",
    "suv": "car",
    "taxi": "car",
    "policecar": "car",
    "minivan": "car",
    "van": "car",
    "pickup": "car",
    "garbagevan": "car",
    "army vehicle": "car",

    # BUS
    "bus": "bus",
    "minibus": "bus",

    # TRUCK
    "truck": "truck",

    # RICKSHAW (VERY IMPORTANT)
    "auto rickshaw": "rickshaw",
    "three wheelers (cng)": "rickshaw",
    "human hauler": "rickshaw",
    "rickshaw": "rickshaw",
}

# Window display
SHOW_WINDOW = True

# Save annotated output video
SAVE_OUTPUT_VIDEO = True
OUTPUT_VIDEO_PATH = str(OUTPUTS_DIR / "phase1_detector_output.mp4")
CONFIGS_DIR = BASE_DIR / "pc_app" / "configs" # New Directory

# PHASE 2: ROI CONFIGURATION FILE
# This JSON stores ROI polygons for Direction A and B
ROI_CONFIG_PATH = str(CONFIGS_DIR / "roi_example.json")

# Direction display names
DIRECTION_A_NAME = "A"
DIRECTION_B_NAME = "B"