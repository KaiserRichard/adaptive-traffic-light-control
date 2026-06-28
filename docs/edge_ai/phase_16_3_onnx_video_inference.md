# Phase 16.3 ONNX Runtime Video Inference

## Goal

Run an input video frame-by-frame through the exported ATLC YOLO ONNX model and save an annotated output video.

## Scope

This phase only implements ONNX Runtime video inference. It does not implement a full benchmark framework, TensorRT, TFLite, INT8 quantization, ROI counting, traffic density estimation, green-time planning, UART communication, ESP32 firmware, or STM32 firmware.

Basic FPS printing is included only as runtime feedback.

## Video Pipeline

```text
Input video
  -> read frame-by-frame
  -> preprocess each frame
  -> run ONNX Runtime inference
  -> filter detections
  -> draw boxes
  -> write annotated output video
```

The script reuses Phase 16.2 image inference helpers from `deployment/onnx/infer_onnx_image.py` for preprocessing, session creation, model execution, confidence filtering, and drawing.

## Input And Output Paths

Verified ONNX model:

```text
deployment/onnx/atlc_yolo26n_custom.onnx
```

Available sample video:

```text
datasets/sample_videos/test.mov
```

Recommended output folder:

```text
results/onnx_video_predictions/
```

Only `.gitkeep` should be committed from the output folder. Generated videos should remain local artifacts.

## Command To Run

Short smoke test:

```bash
.venv/bin/python deployment/onnx/infer_onnx_video.py \
  --model deployment/onnx/atlc_yolo26n_custom.onnx \
  --video datasets/sample_videos/test.mov \
  --output /tmp/atlc_phase16_3_video_onnx.mp4 \
  --imgsz 640 \
  --conf 0.25 \
  --providers CPUExecutionProvider \
  --max-frames 60
```

Project-local output, if you intentionally want a saved result artifact:

```bash
.venv/bin/python deployment/onnx/infer_onnx_video.py \
  --model deployment/onnx/atlc_yolo26n_custom.onnx \
  --video datasets/sample_videos/test.mov \
  --output results/onnx_video_predictions/test_onnx.mp4 \
  --imgsz 640 \
  --conf 0.25 \
  --providers CPUExecutionProvider \
  --max-frames 60
```

Set `--max-frames 0` to process the whole video.

## Expected Output

The script prints:

```text
ATLC PHASE 16.3 ONNX VIDEO INFERENCE
Model path:              deployment/onnx/atlc_yolo26n_custom.onnx
Video path:              datasets/sample_videos/test.mov
Output path:             /tmp/atlc_phase16_3_video_onnx.mp4
Provider:                CPUExecutionProvider
Input video FPS:         <fps>
Input video resolution:  <width>x<height>
Frames processed:        60
Total processing time:   <seconds> s
Approx processing FPS:   <fps>
Output video saved:      /tmp/atlc_phase16_3_video_onnx.mp4
```

The output video should contain annotated frames with bounding boxes and class/confidence labels.

## Why This Is Not Full Benchmarking Yet

Phase 16.3 prints approximate processing FPS so the operator can see whether the script is running normally. This is not a benchmark because it does not control hardware state, software versions, input duration, warmup, repeated runs, CPU/RAM/GPU usage, dropped frames, or end-to-end latency.

Full benchmarking should be handled in a later phase with a reproducible experiment record.

## Common Mistakes

| Mistake | Symptom | Fix |
| --- | --- | --- |
| Passing an image path to `--video` | OpenCV cannot open video | Use a real video such as `datasets/sample_videos/test.mov` |
| Forgetting `--output` | CLI error | Provide an output `.mp4` path |
| Expecting TensorRT speed | CPU-only runtime speed | Keep Phase 16.3 on `CPUExecutionProvider` |
| Committing generated videos | Large accidental commits | Keep generated `.mp4` outputs local and commit only `.gitkeep` |
| Treating FPS as benchmark data | Weak performance claim | Use a later benchmark phase with controlled metrics |

## Next Phase

Phase 16.4 should compare PyTorch/Ultralytics and ONNX Runtime behavior on image and video inputs. It should compare output shapes, detection plausibility, runtime behavior, and visible annotation quality without adding TensorRT, TFLite, INT8, ROI, UART, or firmware changes.
