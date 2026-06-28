# Phase 16.5 ONNX Quantization Experiment

## Goal

Run a controlled ONNX quantization experiment and compare the FP32 ONNX model against a quantized ONNX model.

The goal is to answer:

- Is the quantized model smaller?
- Is the quantized model faster, slower, or similar on CPU?
- Are detections still visually reasonable?
- Are detection counts similar?
- Is the quantized model worth considering for edge deployment?

## Scope

This phase only covers ONNX quantization and one-image FP32-vs-quantized comparison.

It does not implement TensorRT, TFLite, Raspberry Pi deployment, Jetson deployment, ROI counting, traffic density estimation, green-time planning, UART communication, ESP32 firmware, STM32 firmware, or full dataset mAP evaluation.

## Why Quantization Matters

Quantization can reduce model size and sometimes improve CPU inference speed. That can matter for edge deployment where storage, memory, and compute are limited.

Quantization is not automatically beneficial. A quantized model can be slower, unsupported by the runtime, or less accurate. Phase 16.5 treats quantization as an experiment, not as a final deployment decision.

## Dynamic Quantization

Phase 16.5 starts with ONNX Runtime dynamic quantization:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
```

Dynamic quantization does not require a calibration dataset, which makes it a safe first experiment. The generated model must still be checked with ONNX Runtime inference before considering it useful.

The Phase 16.5 script uses dynamic `QUInt8` weights because this local ONNX Runtime `CPUExecutionProvider` can load that output for the YOLO ONNX export. A trial with signed `QInt8` weights produced unsupported `ConvInteger` nodes on this CPU runtime, so the experiment records operator support as part of the deployment result.

## FP32 Vs Quantized ONNX Comparison

The comparison script runs both models on the same image:

```text
same image
  -> FP32 ONNX Runtime inference
  -> quantized ONNX Runtime inference
  -> model size comparison
  -> average runtime comparison
  -> detection count comparison
  -> side-by-side visual inspection
```

The comparison reuses the corrected Phase 16.4 letterbox preprocessing and box restoration logic from `deployment/onnx/infer_onnx_image.py`.

## What Is Measured

The scripts report:

- FP32 model size
- quantized model size
- size reduction percentage
- dynamic quantization weight type
- FP32 average runtime
- quantized average runtime
- FP32 detection count
- quantized detection count
- output image paths for visual inspection

## Why This Is Not Final Deployment Yet

This is a single-image smoke test and model-size experiment. It is not a benchmark report and not a final accuracy evaluation.

Before adopting quantization, later work should measure behavior across more images/videos and record hardware, software versions, runtime settings, and raw results.

## Commands

Quantize:

```bash
.venv/bin/python deployment/onnx/quantize_onnx.py \
  --input deployment/onnx/atlc_yolo26n_custom.onnx \
  --output deployment/onnx/atlc_yolo26n_custom_dynamic_int8.onnx \
  --mode dynamic
```

Compare at `conf=0.25`:

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

Optional lower-threshold check:

```bash
.venv/bin/python deployment/compare/compare_fp32_quantized_onnx_image.py \
  --fp32-model deployment/onnx/atlc_yolo26n_custom.onnx \
  --quantized-model deployment/onnx/atlc_yolo26n_custom_dynamic_int8.onnx \
  --image yolo_research/datasets/atlc_2000/images/test/09150440_jpg.rf.DvAoBPo7uxkzXD4hgu8H.jpg \
  --output /tmp/atlc_phase16_5_quantization_comparison_conf015.jpg \
  --imgsz 640 \
  --conf 0.15 \
  --providers CPUExecutionProvider \
  --runs 5
```

Generated `.onnx` and `.jpg` outputs should not be committed.

## Expected Output

The quantization script should print:

```text
Input model path
Output model path
Quantization mode
Weight type
FP32 model size
Quantized model size
Size reduction
```

The comparison script should print:

```text
FP32 model size
Quantized model size
Size reduction
FP32 average time
Quantized average time
FP32 detections
Quantized detections
Output image paths
Interpretation
```

## Common Mistakes

| Mistake | Symptom | Fix |
| --- | --- | --- |
| Committing quantized `.onnx` output | Large generated artifact in Git | Keep `deployment/onnx/*.onnx` ignored |
| Comparing with direct resize | Detection mismatch unrelated to quantization | Reuse `letterbox_image()` from Phase 16.4 |
| Treating one image as final accuracy | Unsupported deployment claim | Use this only as a smoke test |
| Ignoring unsupported operator errors | Runtime comparison fails later | Document the error and do not fake results |
| Using a quantized operator unsupported by the CPU provider | ONNX Runtime session creation fails | Try a CPU-supported dynamic weight type and record the result |
| Assuming smaller means better | Poor detections or slower runtime | Compare size, speed, and visual quality together |

## Next Phase

Phase 16.6 should create an Edge AI benchmark report with controlled inputs, repeatable commands, hardware/software details, and recorded raw results.
