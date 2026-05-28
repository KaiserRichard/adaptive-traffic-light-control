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

def parse_video_source(value: str):
    """
    Parse VIDEO_SOURCE from environment.

    Why this exists:
    - .env values are always strings.
    - OpenCV expects camera index as an integer, not string "0".
    - Without this helper, VIDEO_SOURCE=0 may be treated as a file path "0".

    Examples:
        "0" -> 0
        "1" -> 1
        "./datasets/sample_videos/test.mov" -> "./datasets/sample_videos/test.mov"
        "/dev/video0" -> "/dev/video0"
    """

    if value is None:
        return 0

    value = value.strip()

    if value.isdigit():
        return int(value)

    return value


BASE_DIR = Path(__file__).resolve().parent.parent

DATASETS_DIR = BASE_DIR / "datasets"
OUTPUTS_DIR = BASE_DIR / "outputs"
CONFIGS_DIR = BASE_DIR / "pc_app" / "configs"

# For Phase 10: This allows VIDEO_SOURCE=0 to work correctly with OpenCV
VIDEO_SOURCE_RAW = os.getenv(
    "VIDEO_SOURCE",
    str(DATASETS_DIR / "sample_videos" / "test.mp4"),
)

VIDEO_SOURCE = parse_video_source(VIDEO_SOURCE_RAW)

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

# ---------------------------------------------------------------------
# Phase 10: Camera input configuration
# ---------------------------------------------------------------------

# Camera width requested from OpenCV.
# The camera may not always accept the exact requested value.
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1280"))

# Camera height requested from OpenCV.
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))

# Camera FPS requested from OpenCV.
# Actual camera FPS should be measured using experiments/benchmark_camera_fps.py.
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "30"))

# Run YOLO once every N frames.
# 1 = detect every frame.
# 2 = detect every 2 frames.
# 3 = detect every 3 frames.
#
# This reduces YOLO inference load while still allowing the camera stream
# to be read continuously.
DETECT_EVERY_N_FRAMES = int(os.getenv("DETECT_EVERY_N_FRAMES", "1"))

if DETECT_EVERY_N_FRAMES < 1:
    raise ValueError("DETECT_EVERY_N_FRAMES must be >= 1.")

# Runtime log directory 
RUNTIME_LOG_DIR = os.getenv(
    "RUNTIME_LOG_DIR",
    "./outputs/runtime_logs"
)