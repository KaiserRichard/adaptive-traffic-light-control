# Phase 16.2 ONNX Runtime Image Inference

## Goal

Run one image through the exported ATLC YOLO ONNX model with ONNX Runtime and save an annotated image.

This phase is image-only. It does not implement video inference, benchmarking, TensorRT, TFLite, INT8 quantization, ROI counting, traffic density estimation, green-time planning, UART, ESP32 firmware, or STM32 firmware.

## Pipeline

```text
Image
  -> preprocessing
  -> ONNX Runtime
  -> output tensor
  -> confidence filtering
  -> bounding boxes
  -> annotated image
```

## Preprocessing

The Phase 16.1 validated model input is:

```text
name: images
shape: [1, 3, 640, 640]
type: tensor(float)
```

The inference script prepares the image by:

- loading the image with OpenCV
- resizing to `640 x 640` by default
- converting BGR to RGB
- normalizing pixels from `0..255` to `0..1`
- converting `HWC` layout to `CHW`
- adding the batch dimension

Phase 16.2 uses direct resize for a simple smoke test. Letterbox preprocessing can be evaluated later if comparison against Ultralytics inference shows a meaningful mismatch.

## ONNX Runtime

The default provider is:

```text
CPUExecutionProvider
```

The script accepts `--providers` as a comma-separated provider list, but Phase 16.2 should remain CPU-first unless a later phase explicitly scopes accelerator validation.

## Postprocessing

The Phase 16.1 validated output is:

```text
name: output0
shape: [1, 300, 6]
type: tensor(float)
```

Each row is treated as an already-decoded detection:

```text
x1, y1, x2, y2, confidence, class_id
```

The script does not implement a generic YOLO decoder for `[1,84,8400]` or `[1,8400,84]`. If the observed output shape is not `[1,300,6]`, the script stops and reports the observed shape.

## Commands

Install dependencies if needed:

```bash
python3 -m pip install -r requirements.txt
```

Run one-image ONNX inference:

```bash
python3 deployment/onnx/infer_onnx_image.py \
  --model deployment/onnx/atlc_yolo26n_custom.onnx \
  --image yolo_research/datasets/atlc_2000/images/test/09150440_jpg.rf.DvAoBPo7uxkzXD4hgu8H.jpg \
  --output results/onnx_image_predictions/09150440_onnx.jpg \
  --imgsz 640 \
  --conf 0.25 \
  --providers CPUExecutionProvider
```

Use another image by changing only `--image` and `--output`.

## Expected Output

The command should print:

```text
Model path:          deployment/onnx/atlc_yolo26n_custom.onnx
Image path:          <input image>
Provider:            CPUExecutionProvider
Input tensor shape:  [1, 3, 640, 640]
Output tensor shape: [1, 300, 6]
Number detections:   <filtered detection count>
Output image path:   results/onnx_image_predictions/<name>.jpg
```

The output image should contain green bounding boxes and class/confidence labels for detections above the confidence threshold.

## Common Mistakes

| Mistake | Symptom | Fix |
| --- | --- | --- |
| Using a generic YOLO decoder | Incorrect boxes or shape errors | Use the verified `[1,300,6]` output contract |
| Passing a missing image path | `Image not found` | Use an existing file under `yolo_research/datasets/atlc_2000/images/test/` |
| Running with an unavailable provider | Provider error before inference | Use `--providers CPUExecutionProvider` |
| Forgetting dependency installation | `ModuleNotFoundError` | Run `python3 -m pip install -r requirements.txt` |
| Expecting video output | Only one annotated image is created | Leave video inference for Phase 16.3 |

## Next Phase

Phase 16.3 should implement ONNX Runtime video inference using the same validated model contract, while still avoiding TensorRT, TFLite, INT8 quantization, ROI counting, UART, and firmware changes unless those are explicitly scoped.
