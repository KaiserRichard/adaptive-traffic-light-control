# traffic_jam_01.mp4

## Purpose

Benchmark video for comparing detection performance across different detector backends.

## Scenario

- Traffic jam / congested road scene
- Fixed camera angle
- Used for testing vehicle detection, ROI assignment, density estimation, and FPS

## Planned Comparisons

- Roboflow hosted inference baseline
- Local YOLO PyTorch model
- YOLO ONNX export
- YOLO NCNN export on Raspberry Pi

## Metrics to Compare

- FPS
- detection count stability
- precision
- recall
- F1-score
- qualitative detection quality
- missed vehicles
- false positives

## Notes

This video should remain unchanged so that all models are evaluated on the same input.