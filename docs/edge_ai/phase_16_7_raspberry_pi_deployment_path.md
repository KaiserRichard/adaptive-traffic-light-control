# Phase 16.7 Raspberry Pi Deployment Path

## Goal

Create a Raspberry Pi deployment path for the ATLC Edge AI host.

This phase prepares setup documentation, runtime commands, dependency guidance, ONNX Runtime deployment notes, and TFLite investigation notes.

It does not claim final Raspberry Pi performance.

## Scope

Included:

- Raspberry Pi role in the ATLC system
- recommended hardware and OS
- Python environment setup
- OpenCV and ONNX Runtime notes
- model file placement
- image inference command
- video inference command
- benchmark command
- TFLite investigation notes

Not included:

- TensorRT
- Jetson deployment
- STM32 firmware changes
- ESP32 firmware changes
- UART integration
- traffic planner changes
- ROI counting changes
- full hardware demo
- PCB bring-up

## Why Raspberry Pi Is Used

Raspberry Pi is the planned AI host because it can run Linux, Python, OpenCV, camera/video input, and ONNX Runtime while communicating with a microcontroller over UART.

The Raspberry Pi is not the deterministic traffic-light controller. It handles computer vision and planning. The MCU handles real-time signal execution and hardware outputs.

## Raspberry Pi System Role

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

## Recommended Deployment Path

Use FP32 ONNX Runtime first.

Reason:

- the project already has a validated FP32 ONNX pipeline
- Phase 16.4 fixed preprocessing consistency with letterbox preprocessing
- Phase 16.6 selected FP32 ONNX Runtime as the current deployment baseline
- dynamic quantized ONNX is much smaller but was slower on the tested CPU environment

## ONNX Runtime Path

Setup guide:

```text
deployment/raspberry_pi/setup_raspberry_pi.md
```

Run guide:

```text
deployment/raspberry_pi/run_onnx_on_pi.md
```

Requirements:

```text
deployment/raspberry_pi/requirements_pi.txt
```

Convenience wrapper:

```text
deployment/raspberry_pi/run_pi_inference.sh
```

## TFLite Investigation

TFLite is a reasonable future option for Raspberry Pi because it is designed for lightweight edge inference.

However, this phase does not force TFLite conversion because:

- ONNX Runtime is already validated for this model
- TFLite conversion may change output tensor shape
- unsupported operators may appear during conversion
- postprocessing must not be guessed
- real Raspberry Pi speed still needs to be measured

Investigation notes:

```text
deployment/raspberry_pi/tflite_investigation.md
```

## Setup Steps

1. Install Raspberry Pi OS 64-bit.
2. Install Python and system dependencies.
3. Clone the repository.
4. Create a virtual environment.
5. Install `deployment/raspberry_pi/requirements_pi.txt`.
6. Place the FP32 ONNX model at `deployment/onnx/atlc_yolo26n_custom.onnx`.
7. Run ONNX validation.
8. Run image inference.
9. Run short video inference.
10. Run the Phase 16.6 benchmark on-device.

## Commands

Image inference:

```bash
.venv/bin/python deployment/onnx/infer_onnx_image.py \
  --model deployment/onnx/atlc_yolo26n_custom.onnx \
  --image yolo_research/datasets/atlc_2000/images/test/09150440_jpg.rf.DvAoBPo7uxkzXD4hgu8H.jpg \
  --output /tmp/atlc_pi_onnx_test.jpg \
  --imgsz 640 \
  --conf 0.25 \
  --providers CPUExecutionProvider
```

Video inference:

```bash
.venv/bin/python deployment/onnx/infer_onnx_video.py \
  --model deployment/onnx/atlc_yolo26n_custom.onnx \
  --video datasets/sample_videos/test.mov \
  --output /tmp/atlc_pi_onnx_video_test.mp4 \
  --imgsz 640 \
  --conf 0.25 \
  --providers CPUExecutionProvider \
  --max-frames 60
```

Benchmark:

```bash
.venv/bin/python deployment/benchmark/benchmark_edge_ai_image.py \
  --fp32-model deployment/onnx/atlc_yolo26n_custom.onnx \
  --quantized-model deployment/onnx/atlc_yolo26n_custom_dynamic_int8.onnx \
  --images yolo_research/datasets/atlc_2000/images/test \
  --max-images 5 \
  --imgsz 640 \
  --conf 0.25 \
  --providers CPUExecutionProvider \
  --runs 10 \
  --output /tmp/atlc_pi_phase16_6_benchmark.json
```

## Expected Outputs

Image inference should print:

```text
Input tensor shape:  [1, 3, 640, 640]
Output tensor shape: [1, 300, 6]
Number detections:   <count>
Output image path:   /tmp/atlc_pi_onnx_test.jpg
```

Benchmark should print a table with:

- model size
- average latency
- minimum latency
- maximum latency
- approximate FPS
- detection count

## Limitations

- Real Raspberry Pi performance is pending.
- Camera input is not integrated in this phase.
- UART PLAN generation is not integrated in this phase.
- TFLite conversion is not implemented in this phase.
- Quantized ONNX is not the default recommendation yet.
- Results from the Mac CPU benchmark should not be copied as Raspberry Pi performance claims.

## Next Phase

Skip Jetson / TensorRT for now because no Jetson hardware is available.

Recommended next phase:

```text
Phase 16.9 - AI Host PLAN Generation Interface
```

