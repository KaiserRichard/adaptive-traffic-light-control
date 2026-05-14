# Benchmark: yolo26n_pt_pc

## Purpose

This benchmark records local detector runtime on a fixed traffic video.

## Detector Configuration

```text
DETECTOR_BACKEND=yolo
YOLO_MODEL_PATH=./pc_app/models/local/yolo26n.pt
VIDEO_SOURCE=./datasets/sample_videos/test.mov
```

## Metrics

```text
Frame count: 300
Total detections: 2783
Average detections per frame: 9.2767
Average inference time: 89.9067 ms/frame
Average FPS: 11.1226
Minimum inference time: 69.5356 ms
Maximum inference time: 278.9915 ms
```

## Interpretation

This benchmark measures detector inference only.

It does not include ROI splitting, density estimation, signal scheduling, or visualization overhead.
