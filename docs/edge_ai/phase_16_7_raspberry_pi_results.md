# Phase 16.7 Raspberry Pi Results

## Status

```text
Deployment path prepared.
Real Raspberry Pi benchmark pending.
```

Phase 16.7 created documentation and setup files for Raspberry Pi deployment. It did not run on Raspberry Pi hardware in the current environment.

## Result Table

| Item | Status | Notes |
| --- | --- | --- |
| ONNX Runtime path | Prepared | Uses validated FP32 ONNX model |
| Quantized ONNX path | Candidate | Smaller, not faster on Mac CPU |
| TFLite path | Investigated | Conversion/runtime validation pending |
| Raspberry Pi benchmark | Pending | Requires real device test |
| AI-to-MCU UART integration | Planned | Later phase |
| Camera capture on Pi | Pending | Image/video files should be tested first |
| Hardware validation | Pending | Requires Raspberry Pi and camera/video input |

## Deployment Recommendation

Use FP32 ONNX Runtime as the first Raspberry Pi runtime path.

Reason:

- it is already validated in the repository
- it uses the corrected letterbox preprocessing path
- it is the current Phase 16.6 deployment baseline
- quantized ONNX needs on-device validation before adoption
- TFLite needs conversion and output-shape validation before adoption

## Commands To Run On Raspberry Pi

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

Short video smoke test:

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

## Metrics To Record Later

When Raspberry Pi hardware is available, record:

- Raspberry Pi model
- OS version and architecture
- Python version
- ONNX Runtime version
- OpenCV version
- model path and model size
- image/video source
- latency and approximate FPS
- CPU load
- memory use
- temperature
- detection counts
- visual inspection notes
- whether camera input works

## Known Limitations

- No Raspberry Pi benchmark has been run yet.
- No TFLite model has been generated yet.
- No camera capture path has been validated yet.
- No UART integration is included in this phase.
- No claim is made about real Raspberry Pi FPS.

