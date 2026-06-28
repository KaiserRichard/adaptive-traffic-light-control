# Phase 16.4 PyTorch vs ONNX Runtime Comparison

## Goal

Compare Ultralytics PyTorch inference and ONNX Runtime inference on the same image.

This phase answers practical deployment questions:

- Does ONNX Runtime produce reasonable detections compared with PyTorch?
- Is ONNX Runtime faster or slower on CPU in a basic comparison?
- Are bounding boxes visually similar?
- Are detection counts similar?

## Scope

This is not a final benchmark. It does not add TensorRT, TFLite, INT8 quantization, ROI counting, traffic density estimation, green-time planning, UART communication, ESP32 firmware, or STM32 firmware.

## Why Compare PyTorch And ONNX

PyTorch/Ultralytics is the training and reference inference path. ONNX Runtime is the deployment path introduced in Phase 16.1 through Phase 16.3.

Comparing both on the same image checks whether the exported ONNX model behaves plausibly before investing in heavier optimization work.

## Image Comparison Pipeline

```text
same image
  -> PyTorch / Ultralytics inference
  -> ONNX Runtime inference
  -> detection count comparison
  -> annotated PyTorch image
  -> annotated ONNX image
  -> side-by-side visual comparison
```

The ONNX path reuses Phase 16.2 helper functions from `deployment/onnx/infer_onnx_image.py`.

## Runtime Comparison Meaning

The script runs each backend several times with `--runs` and reports average time. This helps identify whether ONNX Runtime is faster or slower on CPU for a quick smoke test.

Do not treat this as a benchmark claim. A real benchmark needs controlled hardware state, warmup policy, repeated trials, CPU/RAM/GPU measurements, input set definition, and raw result tracking.

## Detection-Count Comparison Meaning

Detection counts are a quick sanity signal:

- Same count means the two paths are broadly aligned, but visual inspection is still required.
- Different count means the visual output must be checked carefully before assuming a deployment problem.

Detection count alone does not prove matching boxes, matching classes, or matching confidence values.

## Commands

Smoke test:

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

Project-local output, if intentionally saving artifacts:

```bash
.venv/bin/python deployment/compare/compare_pytorch_onnx_image.py \
  --pt-model yolo_research/outputs/runs/atlc_yolo26n_custom/weights/best.pt \
  --onnx-model deployment/onnx/atlc_yolo26n_custom.onnx \
  --image yolo_research/datasets/atlc_2000/images/test/09150440_jpg.rf.DvAoBPo7uxkzXD4hgu8H.jpg \
  --output results/comparison/09150440_side_by_side.jpg \
  --imgsz 640 \
  --conf 0.25 \
  --providers CPUExecutionProvider \
  --runs 5
```

Generated `.jpg` result files should not be staged or committed unless explicitly requested.

## Expected Output

```text
ATLC PHASE 16.4 PYTORCH VS ONNX COMPARISON

Image path:                  <image>
PyTorch model:               <best.pt>
ONNX model:                  <model.onnx>
Provider:                    CPUExecutionProvider
Runs:                        5

PyTorch average time:        <seconds> s
ONNX average time:           <seconds> s
PyTorch detections:          <count>
ONNX detections:             <count>

Output PyTorch image:        <path>
Output ONNX image:           <path>
Output side-by-side image:   <path>
```

The script also prints whether ONNX is faster or slower in this basic CPU comparison and whether detection counts are the same or different.

## Common Mistakes

| Mistake | Symptom | Fix |
| --- | --- | --- |
| Treating this as a final benchmark | Unsupported performance claim | Use this only as a practical comparison |
| Comparing different input images | Detection differences are meaningless | Use the exact same `--image` path |
| Forgetting the `.pt` model | `PyTorch model not found` | Check `yolo_research/outputs/runs/atlc_yolo26n_custom/weights/best.pt` |
| Forgetting the ONNX model | `ONNX model not found` | Re-run Phase 16.1 export if needed |
| Committing generated comparison images | Large or noisy commits | Keep smoke-test outputs in `/tmp`, or save project-local outputs without staging the generated `.jpg` files |

## Next Phase

Phase 16.5 should be a quantization experiment. It should define the quantization goal, model format, calibration/input data, expected accuracy risk, and measurable results before changing the deployment path.
