# Phase 16.2 Inference Results

Record one-image ONNX Runtime inference results here after running `deployment/onnx/infer_onnx_image.py`.

## Result Table

| Field | Value |
| --- | --- |
| Input image | `yolo_research/datasets/atlc_2000/images/test/09150440_jpg.rf.DvAoBPo7uxkzXD4hgu8H.jpg` |
| Output image | `results/onnx_image_predictions/09150440_onnx.jpg` |
| Inference provider | `CPUExecutionProvider` |
| Input tensor shape | `[1, 3, 640, 640]` |
| Output tensor shape | `[1, 300, 6]` |
| Confidence threshold | `0.25` |
| Detections | `2` in the smoke test run |
| Result | PASS |
| Notes | Smoke test used the validated `[1, 300, 6]` output format and wrote the annotated image successfully. |

## Command Used

```bash
.venv/bin/python deployment/onnx/infer_onnx_image.py \
  --model deployment/onnx/atlc_yolo26n_custom.onnx \
  --image yolo_research/datasets/atlc_2000/images/test/09150440_jpg.rf.DvAoBPo7uxkzXD4hgu8H.jpg \
  --output results/onnx_image_predictions/09150440_onnx.jpg \
  --imgsz 640 \
  --conf 0.25 \
  --providers CPUExecutionProvider
```

The implementation smoke test wrote to `/tmp/atlc_phase16_2_09150440_onnx.jpg` to avoid committing generated prediction images.

## Scope Notes

- This result is only for one-image ONNX Runtime inference.
- No video inference, benchmark framework, TensorRT, TFLite, INT8 quantization, ROI counting, traffic density estimation, green-time planning, UART, ESP32 firmware, or STM32 firmware was added in this phase.
