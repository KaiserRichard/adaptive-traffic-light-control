# Local YOLO Models

This folder stores local YOLO model files used for edge deployment testing.

Model files are not committed to GitHub because they can be large.

Expected examples:

- yolov8n.pt
- yolo26n.pt
- yolo26s.pt
- yolo26x.pt
- yolo26n.onnx
- yolo26n_ncnn_model/

## Main Project Direction

The project uses local YOLO inference for Raspberry Pi deployment.

Roboflow is preserved only as an early hosted inference baseline.

## Model Selection

Recommended order:

1. YOLOv8n as a stable local baseline
2. YOLO26n as the main edge-oriented candidate
3. YOLO26s if better detection is needed and speed is still acceptable
4. YOLO26x only for quality comparison, not Raspberry Pi deployment

## Notes

Do not commit `.pt`, `.onnx`, or NCNN model folders.

Store model files locally or in external storage.