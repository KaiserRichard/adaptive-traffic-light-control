"""
config.py

Purpose:
- Keep all configuration variables in one place.
- Avoid hard-coding paths, API keys, model paths, thresholds, and runtime modes.
"""

from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent.parent

DATASETS_DIR = BASE_DIR / "datasets"
OUTPUTS_DIR = BASE_DIR / "outputs"
CONFIGS_DIR = BASE_DIR / "pc_app" / "configs"

VIDEO_SOURCE = os.getenv(
    "VIDEO_SOURCE",
    str(DATASETS_DIR / "sample_videos" / "test.mp4"),
)

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "")

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))

CLASS_NORMALIZATION = {
    "motorbike": "motorbike",
    "motorcycle": "motorbike",
    "scooter": "motorbike",
    "bicycle": "bicycle",

    "car": "car",
    "suv": "car",
    "taxi": "car",
    "policecar": "car",
    "minivan": "car",
    "van": "car",
    "pickup": "car",
    "garbagevan": "car",
    "army vehicle": "car",

    "bus": "bus",
    "minibus": "bus",

    "truck": "truck",

    "auto rickshaw": "rickshaw",
    "three wheelers (cng)": "rickshaw",
    "human hauler": "rickshaw",
    "rickshaw": "rickshaw",
}

SHOW_WINDOW = os.getenv("SHOW_WINDOW", "true").lower() == "true"
SAVE_OUTPUT_VIDEO = os.getenv("SAVE_OUTPUT_VIDEO", "true").lower() == "true"

OUTPUT_VIDEO_PATH = os.getenv(
    "OUTPUT_VIDEO_PATH",
    str(OUTPUTS_DIR / "benchmarks" / "local_yolo_pt_pc" / "annotated_full_pipeline.mp4"),
)

ROI_CONFIG_PATH = os.getenv(
    "ROI_CONFIG_PATH",
    str(CONFIGS_DIR / "roi_example.json"),
)

DIRECTION_A_NAME = "A"
DIRECTION_B_NAME = "B"

ENABLE_RAW_DENSITY = True
ENABLE_PCE_DENSITY = True

DENSITY_SMOOTHING_ALPHA = float(os.getenv("DENSITY_SMOOTHING_ALPHA", "0.30"))

PCE_WEIGHTS = {
    "bicycle": 0.3,
    "motorbike": 0.5,
    "car": 1.0,
    "rickshaw": 1.2,
    "bus": 2.5,
    "truck": 2.5,
}

BASE_GREEN_TIME = int(os.getenv("BASE_GREEN_TIME", "20"))
MIN_GREEN_TIME = int(os.getenv("MIN_GREEN_TIME", "10"))
MAX_GREEN_TIME = int(os.getenv("MAX_GREEN_TIME", "45"))

YELLOW_TIME = int(os.getenv("YELLOW_TIME", "3"))
ALL_RED_TIME = int(os.getenv("ALL_RED_TIME", "1"))

DENSITY_EPSILON = float(os.getenv("DENSITY_EPSILON", "1e-6"))

DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "yolo")

YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    str(BASE_DIR / "pc_app" / "models" / "local" / "yolo26n.pt"),
)

YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
YOLO_VERBOSE = os.getenv("YOLO_VERBOSE", "false").lower() == "true"

BENCHMARK_MAX_FRAMES = int(os.getenv("BENCHMARK_MAX_FRAMES", "300"))

# ---------------------------------------------------------------------
# Phase 8: UART / MCU communication
# ---------------------------------------------------------------------

# Enable or disable sending signal plans to MCU.
# Keep this false by default so users can run the vision pipeline
# without physical hardware.
ENABLE_UART = os.getenv("ENABLE_UART", "false").lower() == "true"

# Serial port for the MCU.
# Examples:
# - Arduino Uno on macOS: /dev/cu.usbmodem141011
# - ESP32 on macOS: /dev/cu.usbserial-XXXX
# - Linux/Raspberry Pi: /dev/ttyUSB0 or /dev/ttyACM0
UART_PORT = os.getenv("UART_PORT", "")

# Serial baud rate. Must match MCU firmware Serial.begin(...).
UART_BAUD = int(os.getenv("UART_BAUD", "115200"))

# Send one UART plan every N processed frames.
# This prevents sending a new plan every video frame.
UART_SEND_INTERVAL_FRAMES = int(os.getenv("UART_SEND_INTERVAL_FRAMES", "30"))

# Timeout in seconds while waiting for ACK from MCU.
UART_ACK_TIMEOUT = float(os.getenv("UART_ACK_TIMEOUT", "2.0"))

# Minimum difference in green time before sending a new plan.
# Example:
# If previous green_a=34 and new green_a=34, do not resend.
# If previous green_a=34 and new green_a=36, resend.
UART_MIN_GREEN_CHANGE = int(os.getenv("UART_MIN_GREEN_CHANGE", "1"))