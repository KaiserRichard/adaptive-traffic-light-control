# Raspberry Pi Setup Notes

## Goal

Prepare a Raspberry Pi to run the ATLC Edge AI inference path.

This is setup guidance only. Hardware validation is pending until the commands are run on a real Raspberry Pi.

## Recommended Hardware

Recommended starting point:

- Raspberry Pi 4 or Raspberry Pi 5
- 4 GB RAM or more
- Raspberry Pi OS 64-bit
- active cooling for sustained video inference
- camera module, USB camera, or copied test videos
- reliable 5 V power supply
- microSD card or SSD with enough space for Python dependencies, model artifacts, and test media

Avoid treating Raspberry Pi Zero or older low-memory boards as the first target for YOLO video inference.

## Recommended OS

Use Raspberry Pi OS 64-bit when possible.

Why:

- Python machine-learning wheels are usually easier to install on 64-bit ARM.
- ONNX Runtime and TFLite package availability can depend on OS architecture.
- A 64-bit OS better matches the Raspberry Pi 4/5 deployment target.

Check architecture:

```bash
uname -m
getconf LONG_BIT
python3 --version
```

Expected for 64-bit OS:

```text
aarch64
64
```

## System Packages

Install basic build and Python support:

```bash
sudo apt update
sudo apt install -y \
  git \
  python3 \
  python3-venv \
  python3-pip \
  python3-dev \
  libatlas-base-dev \
  libopenblas-dev \
  libjpeg-dev \
  zlib1g-dev
```

For camera/video work, OpenCV may need extra system libraries:

```bash
sudo apt install -y \
  libgl1 \
  libglib2.0-0 \
  v4l-utils
```

## Clone Repository

```bash
git clone https://github.com/KaiserRichard/adaptive-traffic-light-control.git
cd adaptive-traffic-light-control
```

## Python Environment

Use a virtual environment. Recent Raspberry Pi OS versions protect the system Python environment, so a venv avoids package-management conflicts.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r deployment/raspberry_pi/requirements_pi.txt
```

If `opencv-python` fails or GUI libraries are not needed, try:

```bash
python -m pip uninstall -y opencv-python
python -m pip install opencv-python-headless
```

## ONNX Runtime Notes

Try the standard CPU package first:

```bash
python -m pip install onnxruntime
```

Then verify:

```bash
python - <<'PY'
import onnxruntime as ort
print(ort.__version__)
print(ort.get_available_providers())
PY
```

Expected minimum provider:

```text
CPUExecutionProvider
```

If the wheel is unavailable for the selected Raspberry Pi OS/Python combination, use a 64-bit Raspberry Pi OS image or build/install an ONNX Runtime wheel compatible with that platform.

## Model File Placement

The repository intentionally does not commit generated `.onnx` model artifacts.

Place or generate the model at:

```text
deployment/onnx/atlc_yolo26n_custom.onnx
```

Optional quantized candidate:

```text
deployment/onnx/atlc_yolo26n_custom_dynamic_int8.onnx
```

The FP32 ONNX model is the current recommended baseline.

## Quick Verification

```bash
.venv/bin/python deployment/onnx/validate_onnx.py \
  --model deployment/onnx/atlc_yolo26n_custom.onnx
```

Expected contract:

```text
Input:
    images
    [1, 3, 640, 640]

Output:
    output0
    [1, 300, 6]
```

## Reference Links

- ONNX Runtime install documentation: https://onnxruntime.ai/docs/install/
- ONNX Runtime Raspberry Pi tutorial: https://onnxruntime.ai/docs/tutorials/iot-edge/rasp-pi-cv.html

