# Adaptive Traffic Light Control

Computer-vision-based adaptive traffic light control system.

This project detects vehicles from a traffic video or camera input, assigns vehicles to road directions using ROI polygons, estimates traffic density, and computes adaptive traffic light timing.

Current main direction:

```text
Local YOLO detector
→ Raspberry Pi edge deployment
→ UART communication
→ ESP32/STM32 traffic light controller
```

Roboflow is kept only as a legacy hosted inference baseline.

---

## 1. What This Project Does

The current pipeline is:

```text
Video / Camera Input
→ Vehicle Detection
→ ROI Split
→ Vehicle Counting
→ Density Estimation
→ EMA Smoothing
→ Adaptive Signal Timing
→ Visualization / Benchmark Output
```

Current supported detector backends:

```text
yolo      = local Ultralytics YOLO detector
roboflow  = legacy hosted Roboflow detector
```

The main development path is now:

```text
DETECTOR_BACKEND=yolo
```

---

## 2. Repository Structure

```text
adaptive-traffic-light-control/
├── datasets/
│   ├── sample_videos/
│   ├── benchmark_videos/
│   ├── roboflow_export/
│   └── snapshots/
│
├── docs/
│   ├── baselines/
│   ├── benchmarks/
│   └── deployment/
│
├── experiments/
│   ├── test_yolo_detector.py
│   ├── benchmark_detector.py
│   ├── test_counter.py
│   ├── test_density.py
│   ├── test_detector.py
│   ├── test_roi.py
│   └── test_scheduler.py
│
├── outputs/
│   ├── baselines/
│   └── benchmarks/
│
├── pc_app/
│   ├── configs/
│   │   └── roi_example.json
│   ├── control/
│   │   └── scheduler.py
│   ├── models/
│   │   └── local/
│   │       └── README.md
│   ├── vision/
│   │   ├── detector.py
│   │   ├── detector_yolo.py
│   │   ├── detector_factory.py
│   │   ├── classes.py
│   │   ├── counter.py
│   │   ├── density.py
│   │   ├── roi.py
│   │   └── visualize.py
│   ├── config.py
│   └── main.py
│
├── requirements.txt
├── requirements-roboflow.txt
├── .env.example
└── README.md
```

---

## 3. Quick Setup

### 3.1 Clone the repository

```bash
git clone https://github.com/KaiserRichard/adaptive-traffic-light-control.git
cd adaptive-traffic-light-control
```

### 3.2 Create virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3.3 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.4 Verify installation

```bash
python -c "import cv2, torch, numpy, ultralytics; print('imports ok'); print('torch:', torch.__version__); print('numpy:', numpy.__version__); print('ultralytics:', ultralytics.__version__)"
```

Expected main versions:

```text
torch: 2.2.2
numpy: 1.26.4
ultralytics: 8.4.22
```

---

## 4. Model Setup

Model files are not committed to GitHub.

Put local YOLO models here:

```text
pc_app/models/local/
```

Expected examples:

```text
pc_app/models/local/yolov8n.pt
pc_app/models/local/yolo26n.pt
```

### 4.1 Create model folder

```bash
mkdir -p pc_app/models/local
```

### 4.2 Download YOLOv8n

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mv yolov8n.pt pc_app/models/local/yolov8n.pt
```

### 4.3 Download YOLO26n

```bash
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
mv yolo26n.pt pc_app/models/local/yolo26n.pt
```

### 4.4 Check model files

```bash
ls -lh pc_app/models/local/
```

Expected:

```text
README.md
yolov8n.pt
yolo26n.pt
```

If the model file is missing, the program will raise:

```text
FileNotFoundError: YOLO model file not found
```

---

## 5. Video Setup

Video files are not committed to GitHub if they are large.

Put test videos here:

```text
datasets/sample_videos/
```

Create the folder if needed:

```bash
mkdir -p datasets/sample_videos
```

Expected examples:

```text
datasets/sample_videos/test.mov
datasets/sample_videos/test.mp4
```

If your video has another name, update `VIDEO_SOURCE` in `.env`.

---

## 6. Environment Configuration

Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

Recommended `.env` for local YOLO testing:

```env
# Detector backend
DETECTOR_BACKEND=yolo

# Local YOLO model
YOLO_MODEL_PATH=./pc_app/models/local/yolo26n.pt
YOLO_IMGSZ=640
YOLO_VERBOSE=false

# Video input
VIDEO_SOURCE=./datasets/sample_videos/test.mov

# Detection threshold
CONFIDENCE_THRESHOLD=0.3

# Runtime mode
SHOW_WINDOW=false
SAVE_OUTPUT_VIDEO=false
OUTPUT_VIDEO_PATH=./outputs/benchmarks/yolo26n_pt_pc/annotated_full_pipeline.mp4

# Density and scheduler
DENSITY_SMOOTHING_ALPHA=0.30
BASE_GREEN_TIME=20
MIN_GREEN_TIME=10
MAX_GREEN_TIME=45
YELLOW_TIME=3
ALL_RED_TIME=1
DENSITY_EPSILON=1e-6

# Benchmark
BENCHMARK_MAX_FRAMES=300
```

To test YOLOv8n instead:

```env
YOLO_MODEL_PATH=./pc_app/models/local/yolov8n.pt
```

To test YOLO26n:

```env
YOLO_MODEL_PATH=./pc_app/models/local/yolo26n.pt
```

---

## 7. Runtime Modes

Choose mode by changing `.env`.

### Benchmark mode

Best for measuring FPS:

```env
SHOW_WINDOW=false
SAVE_OUTPUT_VIDEO=false
```

### Evidence export mode

Best for creating annotated output video:

```env
SHOW_WINDOW=false
SAVE_OUTPUT_VIDEO=true
```

### Live demo mode

Best for visual demo:

```env
SHOW_WINDOW=true
SAVE_OUTPUT_VIDEO=false
```

### Debug mode

Use only for short debugging:

```env
SHOW_WINDOW=true
SAVE_OUTPUT_VIDEO=true
```

Debug mode is slowest because it both displays and writes video.

---

## 8. Run Local YOLO Smoke Test

This checks whether the model loads and detects objects on the first frame.

```bash
python -m experiments.test_yolo_detector
```

Expected output:

```text
Local YOLO detections:
Total: ...
{'bbox': [...], 'conf': ..., 'class_name': 'car'}
```

This verifies:

- video file can be opened
- YOLO model file exists
- YOLO inference works
- class normalization works
- detector output format matches the rest of the pipeline

---

## 9. Run Detector Benchmark

This measures detector-only performance.

```bash
python -m experiments.benchmark_detector
```

The benchmark output is saved here:

```text
outputs/benchmarks/<model_name>_<model_format>_<device>/
```

Example for YOLO26n on PC:

```text
outputs/benchmarks/yolo26n_pt_pc/
├── metrics.json
└── run_notes.md
```

Example for YOLOv8n on PC:

```text
outputs/benchmarks/yolov8n_pt_pc/
├── metrics.json
└── run_notes.md
```

Check metrics:

```bash
cat outputs/benchmarks/yolo26n_pt_pc/metrics.json
```

Important:

`metrics.json` is created only after the benchmark finishes.

If the benchmark is interrupted with `Ctrl+C`, `metrics.json` may not be created.

Use this in `.env` to limit runtime:

```env
BENCHMARK_MAX_FRAMES=300
```

---

## 10. Run Full Pipeline

Run the full adaptive traffic light pipeline:

```bash
python -m pc_app.main
```

This includes:

```text
frame reading
→ YOLO inference
→ ROI split
→ vehicle counting
→ density estimation
→ EMA smoothing
→ adaptive signal timing
→ visualization drawing
→ optional video writing
→ optional GUI display
```

The program prints profiling information every 30 frames, for example:

```text
detect_ms
logic_ms
draw_ms
writer_ms
display_ms
total_ms
full_loop_fps_est
```

---

## 11. Current Benchmark Reference

Current local PC result using YOLO26n:

```text
Detector-only FPS: around 11–13 FPS
Inference time: around 75–90 ms/frame
```

Full pipeline reference:

```text
Display OFF, video saving OFF: around 10–12 FPS
Display OFF, video saving ON:  around 7–8.5 FPS
Display ON, video saving ON:   around 3–5 FPS
```

Typical profiling:

```text
detect_ms:   ~75–100 ms
logic_ms:    ~0.1 ms
draw_ms:     ~1.5 ms
writer_ms:   ~30–50 ms
display_ms:  ~80–140 ms
```

Interpretation:

```text
The adaptive traffic-light logic is lightweight.
The main bottleneck is YOLO inference.
VideoWriter and GUI display add significant overhead.
```

---

## 12. Raspberry Pi Deployment

The Raspberry Pi will be used as the edge-AI host.

Target architecture:

```text
Camera / video input
→ Raspberry Pi
→ Local YOLO detector
→ ROI split
→ density estimation
→ adaptive timing
→ UART to ESP32/STM32
```

### 12.1 Install system packages on Raspberry Pi

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git libgl1 libglib2.0-0
```

### 12.2 Clone repository

```bash
git clone https://github.com/KaiserRichard/adaptive-traffic-light-control.git
cd adaptive-traffic-light-control
```

### 12.3 Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch installation fails on Raspberry Pi, use a Raspberry Pi-specific requirements file later.

### 12.4 Copy model file to Raspberry Pi

From your PC/Mac:

```bash
scp pc_app/models/local/yolo26n.pt pi@<RASPI_IP>:~/adaptive-traffic-light-control/pc_app/models/local/
```

Optional YOLOv8n:

```bash
scp pc_app/models/local/yolov8n.pt pi@<RASPI_IP>:~/adaptive-traffic-light-control/pc_app/models/local/
```

### 12.5 Copy test video to Raspberry Pi

From your PC/Mac:

```bash
scp datasets/sample_videos/test.mov pi@<RASPI_IP>:~/adaptive-traffic-light-control/datasets/sample_videos/
```

### 12.6 Create `.env` on Raspberry Pi

Recommended Raspberry Pi `.env`:

```env
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
```

### 12.7 Run tests on Raspberry Pi

Smoke test:

```bash
python -m experiments.test_yolo_detector
```

Detector benchmark:

```bash
python -m experiments.benchmark_detector
```

Full pipeline:

```bash
python -m pc_app.main
```

Expected benchmark folder on Raspberry Pi:

```text
outputs/benchmarks/yolo26n_pt_raspi/
├── metrics.json
└── run_notes.md
```

---

## 13. Roboflow Baseline

Roboflow was used as an early hosted inference baseline.

It is preserved for documentation and comparison only.

Roboflow dependencies are separated in:

```text
requirements-roboflow.txt
```

Do not install `requirements-roboflow.txt` into the same environment as `requirements.txt` unless compatibility has been tested.

Reason:

```text
Roboflow inference SDK may require NumPy 2.x.
The main YOLO/PyTorch environment uses NumPy 1.26.4.
```

---

## 14. ONNX / NCNN Strategy

Do not start with ONNX or NCNN immediately.

First benchmark `.pt` on Raspberry Pi.

Recommended order:

```text
1. YOLO26n.pt on PC
2. YOLO26n.pt on Raspberry Pi
3. Reduce YOLO_IMGSZ if needed
4. Try frame skipping if needed
5. Export ONNX if .pt is too slow
6. Try NCNN if ONNX is still not good enough
```

Correct mental model:

```text
YOLO26n = model
.pt / ONNX / NCNN = deployment formats
```

Choose the model first. Then benchmark deployment formats.

---

## 15. Common Issues

### 15.1 NumPy conflict

If you see:

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

Run:

```bash
pip uninstall -y numpy
pip install numpy==1.26.4
```

Verify:

```bash
python -c "import torch, numpy; print(torch.__version__); print(numpy.__version__)"
```

### 15.2 Missing model file

If you see:

```text
FileNotFoundError: YOLO model file not found
```

Check:

```bash
ls -lh pc_app/models/local/
```

Then download/copy the model or update:

```env
YOLO_MODEL_PATH=./pc_app/models/local/yolo26n.pt
```

### 15.3 Missing video file

If the video cannot be opened, check:

```bash
ls -lh datasets/sample_videos/
```

Then copy the video or update:

```env
VIDEO_SOURCE=./datasets/sample_videos/test.mov
```

### 15.4 No `metrics.json`

`metrics.json` is created only when the benchmark script finishes.

Run:

```bash
python -m experiments.benchmark_detector
```

Then check:

```bash
find outputs/benchmarks -name "metrics.json"
```

Make sure `.env` contains:

```env
BENCHMARK_MAX_FRAMES=300
```

---

## 16. Git Policy

Do not commit:

```text
*.pt
*.onnx
*_ncnn_model/
*.mp4
*.mov
*.avi
```

Commit:

```text
source code
README.md
docs/
.env.example
run_notes.md
metrics.json if useful for report evidence
```

---

## 17. Short Roadmap

```text
Phase 1 — Basic vehicle detection
Phase 2 — ROI and direction-based counting
Phase 3 — Density estimation
Phase 4 — Adaptive signal timing
Phase 5 — Requirements cleanup + Roboflow baseline preservation
Phase 6 — Local YOLO + Raspberry Pi deployment testbed
Phase 7 — UART communication with ESP32/STM32
Phase 8 — Robustness: fake MCU, watchdog, 74HC595, fail-safe
```

---

## 18. Final Direction

```text
Roboflow hosted baseline
→ Local YOLO detector
→ Raspberry Pi edge deployment
→ UART communication
→ ESP32/STM32 traffic light controller
```

Final goal:

```text
A real adaptive traffic light testbed combining computer vision, edge AI, and embedded-system control.
```
