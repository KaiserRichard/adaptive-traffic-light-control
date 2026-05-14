# Adaptive Traffic Light Control

Computer-vision-based adaptive traffic light control system.

The system detects vehicles from a traffic video or camera input, assigns vehicles into road directions using ROI polygons, estimates traffic density, and computes adaptive green-light timing.

The current main direction is:

```text
Local YOLO inference
→ Raspberry Pi edge deployment
→ UART communication
→ ESP32/STM32 traffic light controller

Roboflow is preserved only as an early hosted inference baseline.

1. Features
Local YOLO vehicle detection
Roboflow legacy baseline support
ROI-based direction splitting
Vehicle class counting
Raw and PCE-weighted density estimation
EMA-smoothed traffic density
Adaptive green-time scheduler
Detector-only benchmarking
Full-pipeline profiling
Raspberry Pi deployment preparation
2. Repository Structure
adaptive-traffic-light-control/
├── datasets/
│   ├── sample_videos/
│   ├── benchmark_videos/
│   └── roboflow_export/
│
├── docs/
│   ├── baselines/
│   ├── benchmarks/
│   └── deployment/
│
├── experiments/
│   ├── test_yolo_detector.py
│   └── benchmark_detector.py
│
├── outputs/
│   ├── baselines/
│   └── benchmarks/
│
├── pc_app/
│   ├── configs/
│   ├── control/
│   ├── models/
│   │   └── local/
│   ├── vision/
│   ├── config.py
│   └── main.py
│
├── requirements.txt
├── requirements-roboflow.txt
└── README.md
3. Setup
3.1 Clone the repository
git clone https://github.com/KaiserRichard/adaptive-traffic-light-control.git
cd adaptive-traffic-light-control
3.2 Create virtual environment

macOS / Linux:

python3 -m venv .venv
source .venv/bin/activate

Windows PowerShell:

python -m venv .venv
.venv\Scripts\Activate.ps1
3.3 Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

Verify core packages:

python -c "import cv2, torch, numpy, ultralytics; print('imports ok'); print('torch:', torch.__version__); print('numpy:', numpy.__version__); print('ultralytics:', ultralytics.__version__)"

Expected main environment:

numpy==1.26.4
torch==2.2.2
ultralytics==8.4.22
4. Model Setup

Model files are not committed to GitHub.

Place local YOLO model files under:

pc_app/models/local/

Expected examples:

pc_app/models/local/yolov8n.pt
pc_app/models/local/yolo26n.pt
4.1 Download YOLOv8n
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mv yolov8n.pt pc_app/models/local/yolov8n.pt
4.2 Download YOLO26n
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
mv yolo26n.pt pc_app/models/local/yolo26n.pt
4.3 Check model files
ls -lh pc_app/models/local/

Expected:

README.md
yolov8n.pt
yolo26n.pt

If a model file is missing, the app will raise:

FileNotFoundError: YOLO model file not found
5. Video Setup

Sample videos are not committed if they are large.

Place test videos under:

datasets/sample_videos/

Example:

datasets/sample_videos/test.mov
datasets/sample_videos/test.mp4

Create the folder if needed:

mkdir -p datasets/sample_videos

Then copy your test video into that folder.

6. Environment Configuration

Create .env:

cp .env.example .env

Recommended .env for local YOLO benchmarking:

DETECTOR_BACKEND=yolo

YOLO_MODEL_PATH=./pc_app/models/local/yolo26n.pt
YOLO_IMGSZ=640
YOLO_VERBOSE=false

VIDEO_SOURCE=./datasets/sample_videos/test.mov
CONFIDENCE_THRESHOLD=0.3

SHOW_WINDOW=false
SAVE_OUTPUT_VIDEO=false
OUTPUT_VIDEO_PATH=./outputs/benchmarks/yolo26n_pt_pc/annotated_full_pipeline.mp4

DENSITY_SMOOTHING_ALPHA=0.30
BASE_GREEN_TIME=20
MIN_GREEN_TIME=10
MAX_GREEN_TIME=45
YELLOW_TIME=3
ALL_RED_TIME=1
DENSITY_EPSILON=1e-6

BENCHMARK_MAX_FRAMES=300

To test YOLOv8n instead:

YOLO_MODEL_PATH=./pc_app/models/local/yolov8n.pt

To test YOLO26n:

YOLO_MODEL_PATH=./pc_app/models/local/yolo26n.pt
7. Run the Project
7.1 Smoke test local YOLO

Use this to verify model loading and first-frame inference:

python -m experiments.test_yolo_detector

Expected output:

Local YOLO detections:
Total: ...
{'bbox': [...], 'conf': ..., 'class_name': 'car'}

This confirms:

video source can be opened
model file exists
YOLO inference works
class normalization works
detection output format is compatible with the pipeline
7.2 Run detector-only benchmark
python -m experiments.benchmark_detector

Output is saved under:

outputs/benchmarks/<model_name>_<model_format>_<device>/

Example:

outputs/benchmarks/yolo26n_pt_pc/
├── metrics.json
└── run_notes.md

Check metrics:

cat outputs/benchmarks/yolo26n_pt_pc/metrics.json

The benchmark measures detector-only performance:

frame → YOLO inference → detections

It does not include ROI, density, scheduler, drawing, video writing, or GUI display.

7.3 Run full pipeline
python -m pc_app.main

The full pipeline includes:

frame read
→ YOLO inference
→ ROI split
→ vehicle counting
→ density estimation
→ EMA smoothing
→ adaptive signal timing
→ visualization drawing
→ optional video writing
→ optional GUI display
8. Runtime Modes

Use .env to choose runtime behavior.

Benchmark mode

Best for FPS measurement:

SHOW_WINDOW=false
SAVE_OUTPUT_VIDEO=false
Evidence export mode

Best for generating annotated video:

SHOW_WINDOW=false
SAVE_OUTPUT_VIDEO=true
Live demo mode

Best for visual demonstration:

SHOW_WINDOW=true
SAVE_OUTPUT_VIDEO=false
Debug mode

Only for short runs:

SHOW_WINDOW=true
SAVE_OUTPUT_VIDEO=true

Debug mode has the lowest FPS because both GUI display and video writing are enabled.

9. Benchmark Results

Current local PC result using YOLO26n:

Detector-only FPS: approximately 11–13 FPS
Inference time: approximately 75–90 ms/frame

Full-pipeline results:

Display OFF, video saving OFF: approximately 10–12 FPS
Display OFF, video saving ON:  approximately 7–8.5 FPS
Display ON, video saving ON:   approximately 3–5 FPS

Typical profiling:

detect_ms:   ~75–100 ms
logic_ms:    ~0.1 ms
draw_ms:     ~1.5 ms
writer_ms:   ~30–50 ms
display_ms:  ~80–140 ms

Interpretation:

The adaptive traffic-light logic is lightweight.
The main bottleneck is YOLO inference.
VideoWriter and GUI display add significant overhead.
10. Raspberry Pi Deployment

The Raspberry Pi will act as the edge-AI host.

Target architecture:

Camera / video input
→ Raspberry Pi
→ Local YOLO detector
→ ROI split
→ density estimation
→ adaptive timing
→ UART to ESP32/STM32
10.1 Install system packages

On Raspberry Pi:

sudo apt update
sudo apt install -y python3-venv python3-pip git libgl1 libglib2.0-0
10.2 Clone repo
git clone https://github.com/KaiserRichard/adaptive-traffic-light-control.git
cd adaptive-traffic-light-control
10.3 Create environment
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

If PyTorch installation fails on Raspberry Pi, use a Raspberry Pi-specific requirements file later.

10.4 Copy model file to Raspberry Pi

From your PC/Mac:

scp pc_app/models/local/yolo26n.pt pi@<RASPI_IP>:~/adaptive-traffic-light-control/pc_app/models/local/

Optional YOLOv8n:

scp pc_app/models/local/yolov8n.pt pi@<RASPI_IP>:~/adaptive-traffic-light-control/pc_app/models/local/
10.5 Copy test video to Raspberry Pi
scp datasets/sample_videos/test.mov pi@<RASPI_IP>:~/adaptive-traffic-light-control/datasets/sample_videos/
10.6 Raspberry Pi .env

Recommended:

DETECTOR_BACKEND=yolo

YOLO_MODEL_PATH=./pc_app/models/local/yolo26n.pt
YOLO_IMGSZ=640
YOLO_VERBOSE=false

VIDEO_SOURCE=./datasets/sample_videos/test.mov
CONFIDENCE_THRESHOLD=0.3

SHOW_WINDOW=false
SAVE_OUTPUT_VIDEO=false
BENCHMARK_MAX_FRAMES=300

DENSITY_SMOOTHING_ALPHA=0.30
BASE_GREEN_TIME=20
MIN_GREEN_TIME=10
MAX_GREEN_TIME=45
YELLOW_TIME=3
ALL_RED_TIME=1
DENSITY_EPSILON=1e-6
10.7 Run tests on Raspberry Pi

Smoke test:

python -m experiments.test_yolo_detector

Detector benchmark:

python -m experiments.benchmark_detector

Full pipeline:

python -m pc_app.main

Expected benchmark folder:

outputs/benchmarks/yolo26n_pt_raspi/
├── metrics.json
└── run_notes.md
11. Roboflow Baseline

Roboflow was used as an early hosted inference baseline.

It is preserved for documentation and comparison, but it is not the main deployment path.

Roboflow environment is separated in:

requirements-roboflow.txt

Do not install Roboflow dependencies into the main YOLO environment unless compatibility has been tested.

12. ONNX / NCNN Deployment Strategy

Do not start with ONNX or NCNN immediately.

First benchmark .pt on Raspberry Pi.

Recommended order:

1. YOLO26n.pt on PC
2. YOLO26n.pt on Raspberry Pi
3. Reduce YOLO_IMGSZ if needed
4. Try frame skipping if needed
5. Export ONNX if .pt is too slow
6. Try NCNN if ONNX is still not good enough

Correct mental model:

YOLO26n = model
.pt / ONNX / NCNN = deployment formats

Choose the model first. Then benchmark deployment formats.

13. Project Roadmap Summary
Phase 1 — Basic vehicle detection
Phase 2 — ROI and direction-based counting
Phase 3 — Density estimation
Phase 4 — Adaptive signal timing
Phase 5 — Requirements cleanup + Roboflow baseline preservation
Phase 6 — Local YOLO + Raspberry Pi deployment testbed
Phase 7 — UART communication with ESP32/STM32
Phase 8 — Robustness: fake MCU, watchdog, 74HC595, fail-safe

Current status:

Phase 6 is focused on local YOLO integration, benchmarking, and Raspberry Pi deployment preparation.
14. Common Issues
NumPy conflict

If you see:

A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x

Run:

pip uninstall -y numpy
pip install numpy==1.26.4

Verify:

python -c "import torch, numpy; print(torch.__version__); print(numpy.__version__)"
Missing model file

Check:

ls -lh pc_app/models/local/

Then download/copy the model or update:

YOLO_MODEL_PATH=./pc_app/models/local/yolo26n.pt
Missing video file

Check:

ls -lh datasets/sample_videos/

Then copy the video or update:

VIDEO_SOURCE=./datasets/sample_videos/test.mov
No metrics.json

metrics.json is created only when the benchmark script finishes.

Run:

python -m experiments.benchmark_detector

Then check:

find outputs/benchmarks -name "metrics.json"

Make sure this is set:

BENCHMARK_MAX_FRAMES=300
15. Git Policy

Do not commit:

*.pt
*.onnx
*_ncnn_model/
*.mp4
*.mov
*.avi

Commit:

README.md
docs/
source code
configuration examples
run_notes.md
metrics.json if useful for report evidence
16. Final Project Direction
Roboflow hosted baseline
→ Local YOLO detector
→ Raspberry Pi edge deployment
→ UART communication
→ ESP32/STM32 traffic light controller

Final goal:
A real adaptive traffic light testbed combining computer vision, edge AI, and embedded-system control.

Sau khi thay README:

```bash
git add README.md
git commit -m "docs: update README with setup and model deployment guide"
git push
