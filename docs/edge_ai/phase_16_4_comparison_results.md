# Phase 16.4 Comparison Results

Record PyTorch vs ONNX Runtime image comparison results here after running `deployment/compare/compare_pytorch_onnx_image.py`.

## Fix Summary

Initial Phase 16.4 result:

```text
PyTorch detections: 7
ONNX detections:    2
```

Visual inspection showed that ONNX missed several obvious motorbikes. The root cause was preprocessing mismatch:

```text
PyTorch / Ultralytics: letterbox preprocessing
ONNX Runtime path:     direct resize preprocessing
```

Fix applied:

```text
deployment/onnx/infer_onnx_image.py now uses letterbox preprocessing and letterbox-aware box restoration.
```

The ONNX model output contract remains unchanged:

```text
[1, 300, 6] = x1, y1, x2, y2, confidence, class_id
```

## Result Table

| Field | Value |
| --- | --- |
| Input image | `yolo_research/datasets/atlc_2000/images/test/09150440_jpg.rf.DvAoBPo7uxkzXD4hgu8H.jpg` |
| PyTorch model | `yolo_research/outputs/runs/atlc_yolo26n_custom/weights/best.pt` |
| ONNX model | `deployment/onnx/atlc_yolo26n_custom.onnx` |
| Provider | `CPUExecutionProvider` |
| Runs | `5` |
| PyTorch average time | `0.1303 s` |
| ONNX average time | `0.0634 s` |
| PyTorch detections | `7` |
| ONNX detections | `6` |
| Result | Improved; accepted for Phase 16.4 with visual review note |
| Notes | Letterbox preprocessing fixed the major mismatch. At `conf=0.25`, ONNX still misses one lower-confidence motorbike that PyTorch keeps. Visual inspection of `/tmp/atlc_phase16_4_comparison_fixed.jpg` shows box alignment is now close. |

## Lower Confidence Check

| Field | Value |
| --- | --- |
| Confidence threshold | `0.15` |
| PyTorch average time | `0.0910 s` |
| ONNX average time | `0.0422 s` |
| PyTorch detections | `7` |
| ONNX detections | `7` |
| Result | PASS |
| Notes | ONNX recovers the missing motorbike at the lower threshold. Visual inspection of `/tmp/atlc_phase16_4_comparison_fixed_conf015.jpg` shows comparable boxes. |

## Small Multi-Image Sanity Check

| Image | Confidence | PyTorch detections | ONNX detections | Note |
| --- | --- | --- | --- | --- |
| `09150440_jpg.rf.DvAoBPo7uxkzXD4hgu8H.jpg` | `0.25` | `7` | `6` | Main smoke test; one low-confidence ONNX miss remains |
| `image--1945-_jpg.rf.XmQpI5JRv1aAxLWykOOH.jpg` | `0.25` | `6` | `6` | Counts matched |
| `BUS_118_jpeg.rf.gzkbLP7HPzf03OtSB8yr.jpeg` | `0.25` | `2` | `1` | Counts differed; visual review still needed |
| `CAME0223_jpg.rf.jtovPaCwHQRN0WhhQ6R6.jpg` | `0.25` | `7` | `7` | Counts matched |

The small check is not a benchmark. It only confirms that letterbox preprocessing makes the ONNX path much closer to PyTorch on representative images.

## Smoke Test Command

```bash
.venv/bin/python deployment/compare/compare_pytorch_onnx_image.py \
  --pt-model yolo_research/outputs/runs/atlc_yolo26n_custom/weights/best.pt \
  --onnx-model deployment/onnx/atlc_yolo26n_custom.onnx \
  --image yolo_research/datasets/atlc_2000/images/test/09150440_jpg.rf.DvAoBPo7uxkzXD4hgu8H.jpg \
  --output /tmp/atlc_phase16_4_comparison_fixed.jpg \
  --imgsz 640 \
  --conf 0.25 \
  --providers CPUExecutionProvider \
  --runs 5
```

## Lower Confidence Command

```bash
.venv/bin/python deployment/compare/compare_pytorch_onnx_image.py \
  --pt-model yolo_research/outputs/runs/atlc_yolo26n_custom/weights/best.pt \
  --onnx-model deployment/onnx/atlc_yolo26n_custom.onnx \
  --image yolo_research/datasets/atlc_2000/images/test/09150440_jpg.rf.DvAoBPo7uxkzXD4hgu8H.jpg \
  --output /tmp/atlc_phase16_4_comparison_fixed_conf015.jpg \
  --imgsz 640 \
  --conf 0.15 \
  --providers CPUExecutionProvider \
  --runs 5
```

## Scope Notes

- This phase is a practical PyTorch vs ONNX image comparison.
- Runtime numbers are basic comparison values, not final benchmark results.
- No TensorRT, TFLite, INT8 quantization, ROI counting, traffic density estimation, green-time planning, UART communication, ESP32 firmware, or STM32 firmware was added.
