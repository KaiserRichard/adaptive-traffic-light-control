# Raspberry Pi Deployment Path

This folder prepares the Adaptive Traffic Light Control Edge AI host for Raspberry Pi deployment.

Status:

```text
Deployment path prepared.
Real Raspberry Pi hardware validation pending.
```

## Raspberry Pi Role

The Raspberry Pi is the AI host, not the deterministic traffic-light controller.

Intended system split:

```text
Camera / Video Input
        |
        v
Raspberry Pi AI Host
        |
        v
YOLO Inference
        |
        v
ROI Counting / Density Estimation
        |
        v
Adaptive PLAN Generation
        |
        v
UART PLAN Message
        |
        v
STM32F103C8T6 / ESP32 Controller
        |
        v
Traffic Light FSM
        |
        v
LED / PCB Output
```

The Raspberry Pi handles computer vision and high-level planning. The MCU handles deterministic light control, safe fallback behavior, and physical outputs.

## Recommended Runtime Path

Use FP32 ONNX Runtime first.

Reason:

- Phase 16.1 validated the ONNX model contract.
- Phase 16.4 fixed the PyTorch/ONNX preprocessing mismatch with letterbox preprocessing.
- Phase 16.6 showed FP32 ONNX Runtime is faster than the dynamic quantized ONNX model on the tested CPU environment.
- Quantized ONNX is much smaller, but it was not faster in local testing.

TFLite remains an investigation path. Do not make it the default until conversion, output decoding, and Raspberry Pi runtime behavior are validated.

## Folder Contents

| File | Purpose |
| --- | --- |
| `setup_raspberry_pi.md` | Raspberry Pi OS, Python, OpenCV, and ONNX Runtime setup notes |
| `run_onnx_on_pi.md` | Image, video, and benchmark commands for Raspberry Pi |
| `tflite_investigation.md` | TFLite benefits, risks, and possible conversion paths |
| `requirements_pi.txt` | Lightweight Python dependency list for the Pi |
| `run_pi_inference.sh` | Small convenience wrapper for ONNX image/video smoke tests |

## Hardware Validation Status

Real Raspberry Pi validation has not been completed in this repository environment.

Pending tests:

- install dependencies on Raspberry Pi OS
- copy or generate the ONNX model artifact on the Pi
- run image inference
- run short video inference
- run the Phase 16.6 benchmark script on-device
- record latency, FPS, CPU load, memory use, thermals, and power if available
- later connect AI host output to the MCU UART integration path

