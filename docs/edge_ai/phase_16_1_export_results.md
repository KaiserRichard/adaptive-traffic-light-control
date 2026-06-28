# Phase 16.1 Export Results

Fill this file after running the Phase 16.1 export and validation commands.

## Export Result Table

| Field | Value |
| --- | --- |
| Phase | 16.1 ONNX export and load validation |
| Input `.pt` path | `yolo_research/outputs/runs/atlc_yolo26n_custom/weights/best.pt` |
| Output `.onnx` path | `deployment/onnx/atlc_yolo26n_custom.onnx` |
| Image size | `640` |
| Opset | `12` |
| Dynamic/static | Static by default |
| Simplify enabled | No by default |
| Model size | Run `python3 deployment/onnx/validate_onnx.py --model deployment/onnx/atlc_yolo26n_custom.onnx` |
| ONNX checker result | Pending |
| ONNX Runtime loading result | Pending |
| Commit hash | Pending |

## Input Metadata

| Name | Shape | Type |
| --- | --- | --- |
| Pending | Pending | Pending |

## Output Metadata

| Name | Shape | Type |
| --- | --- | --- |
| Pending | Pending | Pending |

## Commands Used

```bash
python3 deployment/onnx/export_onnx.py \
  --weights yolo_research/outputs/runs/atlc_yolo26n_custom/weights/best.pt \
  --output deployment/onnx/atlc_yolo26n_custom.onnx \
  --imgsz 640 \
  --opset 12

python3 deployment/onnx/validate_onnx.py \
  --model deployment/onnx/atlc_yolo26n_custom.onnx
```

## Notes

- Phase 16.1 only validates ONNX export and ONNX Runtime loading.
- No full image/video inference has been implemented in this phase.
- No TensorRT, TFLite, INT8 quantization, ROI counting, UART logic, ESP32 firmware changes, or STM32 firmware changes were added.
