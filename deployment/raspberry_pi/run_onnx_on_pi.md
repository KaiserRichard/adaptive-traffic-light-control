# Run ONNX Runtime on Raspberry Pi

## Goal

Run the existing ATLC ONNX inference scripts on Raspberry Pi with the same preprocessing and output contract validated in earlier Phase 16 work.

## Runtime Recommendation

Use FP32 ONNX Runtime first.

Reason:

- the FP32 ONNX path is already exported and validated
- Phase 16.4 fixed preprocessing with letterbox behavior
- Phase 16.6 showed FP32 ONNX is the current deployment baseline
- quantized ONNX is smaller but was slower on the tested CPU environment

## Image Inference

```bash
.venv/bin/python deployment/onnx/infer_onnx_image.py \
  --model deployment/onnx/atlc_yolo26n_custom.onnx \
  --image yolo_research/datasets/atlc_2000/images/test/09150440_jpg.rf.DvAoBPo7uxkzXD4hgu8H.jpg \
  --output /tmp/atlc_pi_onnx_test.jpg \
  --imgsz 640 \
  --conf 0.25 \
  --providers CPUExecutionProvider
```

Expected output shape:

```text
Input tensor shape:  [1, 3, 640, 640]
Output tensor shape: [1, 300, 6]
```

The output image is saved to:

```text
/tmp/atlc_pi_onnx_test.jpg
```

## Video Inference

Place a small test video at:

```text
datasets/sample_videos/test.mov
```

Then run:

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

Use `--max-frames 60` for an initial smoke test. Use `--max-frames 0` only after confirming the script runs correctly on the Pi.

## Benchmark Command

Run the Phase 16.6 image benchmark on Raspberry Pi:

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

Record:

- average latency
- approximate FPS
- model size
- detection counts
- CPU load
- memory use
- temperature if available

## Convenience Script

The wrapper script can run image or video smoke tests:

```bash
deployment/raspberry_pi/run_pi_inference.sh image
deployment/raspberry_pi/run_pi_inference.sh video
```

Environment variables can override defaults:

```bash
MODEL=deployment/onnx/atlc_yolo26n_custom.onnx \
IMAGE=yolo_research/datasets/atlc_2000/images/test/09150440_jpg.rf.DvAoBPo7uxkzXD4hgu8H.jpg \
OUTPUT=/tmp/atlc_pi_onnx_test.jpg \
deployment/raspberry_pi/run_pi_inference.sh image
```

## Known Limitations

- Raspberry Pi performance is not validated yet.
- Camera capture is not integrated in this phase.
- UART PLAN generation is not integrated in this phase.
- Quantized ONNX should be benchmarked on-device before adoption.
- TFLite is not implemented in this phase.

