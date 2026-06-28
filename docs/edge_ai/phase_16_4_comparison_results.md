# Phase 16.4 Comparison Results

Record PyTorch vs ONNX Runtime image comparison results here after running `deployment/compare/compare_pytorch_onnx_image.py`.

## Result Table

| Field | Value |
| --- | --- |
| Input image | `yolo_research/datasets/atlc_2000/images/test/09150440_jpg.rf.DvAoBPo7uxkzXD4hgu8H.jpg` |
| PyTorch model | `yolo_research/outputs/runs/atlc_yolo26n_custom/weights/best.pt` |
| ONNX model | `deployment/onnx/atlc_yolo26n_custom.onnx` |
| Provider | `CPUExecutionProvider` |
| Runs | `5` |
| PyTorch average time | `0.1048 s` |
| ONNX average time | `0.0403 s` |
| PyTorch detections | `7` |
| ONNX detections | `2` |
| Result | PASS with review needed |
| Notes | ONNX was faster in this basic CPU comparison, but detection counts differed. Visual inspection of `/tmp/atlc_phase16_4_comparison.jpg` is required before drawing quality conclusions. |

## Smoke Test Command

```bash
.venv/bin/python deployment/compare/compare_pytorch_onnx_image.py \
  --pt-model yolo_research/outputs/runs/atlc_yolo26n_custom/weights/best.pt \
  --onnx-model deployment/onnx/atlc_yolo26n_custom.onnx \
  --image yolo_research/datasets/atlc_2000/images/test/09150440_jpg.rf.DvAoBPo7uxkzXD4hgu8H.jpg \
  --output /tmp/atlc_phase16_4_comparison.jpg \
  --imgsz 640 \
  --conf 0.25 \
  --providers CPUExecutionProvider \
  --runs 5
```

## Scope Notes

- This phase is a practical PyTorch vs ONNX image comparison.
- Runtime numbers are basic comparison values, not final benchmark results.
- No TensorRT, TFLite, INT8 quantization, ROI counting, traffic density estimation, green-time planning, UART communication, ESP32 firmware, or STM32 firmware was added.
