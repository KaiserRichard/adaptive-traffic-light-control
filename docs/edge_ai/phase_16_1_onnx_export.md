# Phase 16.1 ONNX Export

## Goal

Export the trained ATLC YOLO model from Ultralytics `.pt` format to ONNX and validate that the ONNX model can be loaded by ONNX Runtime.

This phase is intentionally narrow. It does not add TensorRT, TFLite, INT8 quantization, image/video ONNX inference, ROI counting, UART logic, ESP32 firmware changes, or STM32 firmware changes.

## Why ONNX Export Matters

ONNX is an intermediate model format for deployment runtimes. Exporting the trained YOLO model to ONNX is the first step toward edge deployment because it separates the trained PyTorch/Ultralytics model from later runtime-specific optimization stages.

For ATLC, ONNX export matters because it allows the vehicle detector to be validated before choosing a target edge runtime such as ONNX Runtime, TensorRT, or a later embedded inference path.

## Input Model Path

```text
yolo_research/outputs/runs/atlc_yolo26n_custom/weights/best.pt
```

This path matches the current YOLO training configuration in `yolo_research/configs/train_config.yaml`.

## Output ONNX Path

```text
deployment/onnx/atlc_yolo26n_custom.onnx
```

The generated `.onnx` file should remain a deployment artifact. Do not commit large generated model binaries unless the repository policy is changed intentionally.

## Export Parameters

Default Phase 16.1 export settings:

| Parameter | Default | Notes |
| --- | --- | --- |
| `--weights` | `yolo_research/outputs/runs/atlc_yolo26n_custom/weights/best.pt` | Trained YOLO checkpoint |
| `--output` | `deployment/onnx/atlc_yolo26n_custom.onnx` | Phase 16.1 ONNX output |
| `--imgsz` | `640` | Matches training image size |
| `--opset` | `12` | Conservative ONNX opset for compatibility |
| `--dynamic` | disabled | Static input shape by default |
| `--simplify` | disabled | Optional graph simplification |
| `--half` | disabled | Safe FP32 export by default |

Do not use INT8 in Phase 16.1.

## Commands

Install or update Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Export the ONNX model:

```bash
python3 deployment/onnx/export_onnx.py \
  --weights yolo_research/outputs/runs/atlc_yolo26n_custom/weights/best.pt \
  --output deployment/onnx/atlc_yolo26n_custom.onnx \
  --imgsz 640 \
  --opset 12
```

Validate the ONNX model:

```bash
python3 deployment/onnx/validate_onnx.py \
  --model deployment/onnx/atlc_yolo26n_custom.onnx
```

Optional static simplified export:

```bash
python3 deployment/onnx/export_onnx.py \
  --weights yolo_research/outputs/runs/atlc_yolo26n_custom/weights/best.pt \
  --output deployment/onnx/atlc_yolo26n_custom.onnx \
  --imgsz 640 \
  --opset 12 \
  --simplify
```

## Expected Output

The export command should print the export configuration, the Ultralytics-generated ONNX path, the requested Phase 16.1 output path, and the model size.

The validation command should print:

- ONNX file size
- `ONNX checker: PASS`
- `ONNX Runtime: PASS`
- input names, shapes, and types
- output names, shapes, and types

Typical static YOLO input shape is similar to:

```text
shape: [1, 3, 640, 640]
type: tensor(float)
```

The exact output tensor shape depends on the YOLO version and export settings.

## Common Errors

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `Weights not found` | The `.pt` path is missing or the training run was not copied locally | Check `yolo_research/outputs/runs/.../weights/best.pt` |
| `ModuleNotFoundError: No module named 'ultralytics'` | Python dependencies are not installed in the active environment | Run `python3 -m pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'onnx'` | ONNX validation dependencies are missing | Reinstall dependencies from `requirements.txt` |
| Simplified export fails | ONNX graph simplifier dependency is missing or incompatible | Reinstall dependencies, then retry without `--simplify` if needed |
| ONNX checker failure | Export produced an invalid graph or incompatible opset | Retry with default FP32 static settings and opset 12 |
| ONNX Runtime session failure | Runtime cannot load the exported graph | Check ONNX Runtime version and retry without `--simplify` or `--half` |

## Next Phase

Phase 16.2 should run a minimal ONNX Runtime inference smoke test on one known image and compare output tensor shape and basic detection plausibility against the PyTorch/Ultralytics model. It should still avoid TensorRT, TFLite, INT8 quantization, and firmware changes unless those are explicitly scoped.
