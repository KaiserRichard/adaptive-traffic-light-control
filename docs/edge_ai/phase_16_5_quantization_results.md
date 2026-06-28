# Phase 16.5 Quantization Results

This file records the Phase 16.5 ONNX quantization smoke-test results.

## Primary Result Table

| Field | Value |
| --- | --- |
| FP32 model path | `deployment/onnx/atlc_yolo26n_custom.onnx` |
| Quantized model path | `deployment/onnx/atlc_yolo26n_custom_dynamic_int8.onnx` |
| Quantization mode | `dynamic` |
| Dynamic weight type | `QUInt8` |
| FP32 model size | `9.31 MB` |
| Quantized model size | `2.72 MB` |
| Size reduction | `70.75%` |
| Input image | `yolo_research/datasets/atlc_2000/images/test/09150440_jpg.rf.DvAoBPo7uxkzXD4hgu8H.jpg` |
| Provider | `CPUExecutionProvider` |
| Runs | `5` |
| Confidence threshold | `0.25` |
| FP32 average time | `0.0461 s` |
| Quantized average time | `0.0620 s` |
| FP32 detections | `6` |
| Quantized detections | `6` |
| Result | Quantized model is much smaller but slower in this basic CPU smoke test. Detection count matches FP32 at `conf=0.25`. |
| Notes | Visual inspection of `/tmp/atlc_phase16_5_quantization_comparison.jpg` showed similar boxes and reasonable detections. |

## Lower-Confidence Check

| Field | Value |
| --- | --- |
| Confidence threshold | `0.15` |
| FP32 average time | `0.0455 s` |
| Quantized average time | `0.0618 s` |
| FP32 detections | `7` |
| Quantized detections | `6` |
| Output image | `/tmp/atlc_phase16_5_quantization_comparison_conf015.jpg` |
| Result | Quantized model missed one lower-confidence FP32 detection. |
| Notes | Quantization is visually reasonable for the main smoke test, but it changes marginal detections. It should not be accepted as final deployment without broader evaluation. |

## Operator Support Note

A trial with signed dynamic `QInt8` weights produced a smaller model but failed to create an ONNX Runtime CPU session with:

```text
NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10) node with name '/model.0/conv/Conv_quant'
```

The committed quantization script therefore uses dynamic `QUInt8` weights for this Phase 16.5 CPU smoke test.

## Quantization Command

```bash
.venv/bin/python deployment/onnx/quantize_onnx.py \
  --input deployment/onnx/atlc_yolo26n_custom.onnx \
  --output deployment/onnx/atlc_yolo26n_custom_dynamic_int8.onnx \
  --mode dynamic
```

## Comparison Command

```bash
.venv/bin/python deployment/compare/compare_fp32_quantized_onnx_image.py \
  --fp32-model deployment/onnx/atlc_yolo26n_custom.onnx \
  --quantized-model deployment/onnx/atlc_yolo26n_custom_dynamic_int8.onnx \
  --image yolo_research/datasets/atlc_2000/images/test/09150440_jpg.rf.DvAoBPo7uxkzXD4hgu8H.jpg \
  --output /tmp/atlc_phase16_5_quantization_comparison.jpg \
  --imgsz 640 \
  --conf 0.25 \
  --providers CPUExecutionProvider \
  --runs 5
```

## Scope Notes

- This is a quantization experiment, not a final deployment decision.
- Generated quantized `.onnx` and comparison `.jpg` files should not be committed.
- No TensorRT, TFLite, Raspberry Pi deployment, Jetson deployment, ROI counting, traffic density estimation, green-time planning, UART communication, ESP32 firmware, STM32 firmware, or full mAP evaluation was added.

## Interpretation

The dynamic `QUInt8` ONNX model is worth keeping as an edge-deployment candidate because it reduces model size by about `70.75%` and preserves the detection count at `conf=0.25` on the Phase 16.4 smoke-test image.

It is not yet a clear runtime win on this CPU environment. The quantized model was about `1.34x` to `1.36x` slower than FP32 in these short five-run checks. Phase 16.6 should measure this more rigorously across images/videos and record hardware/software details before making a deployment decision.
