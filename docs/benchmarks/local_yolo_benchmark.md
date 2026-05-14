# Local YOLO Benchmark

## Purpose

This document records the benchmark strategy for local YOLO inference.

The goal is to compare:

- Roboflow hosted inference baseline
- YOLOv8n PyTorch local baseline
- YOLO26n PyTorch local candidate
- future YOLO26n ONNX export
- future YOLO26n NCNN export on Raspberry Pi

## Input Video

The benchmark should use the same input video for fair comparison.

Example:

datasets/sample_videos/test.mov

## Detector-only Benchmark

Command:

python -m experiments.benchmark_detector

This measures:

frame → detector.detect(frame) → detections

It does not measure:

ROI split, density estimation, scheduler, drawing, video writing, display window.

## Full-pipeline Benchmark

Command:

python -m pc_app.main

This measures the full application pipeline:

frame read → YOLO inference → ROI split → counting → density estimation → EMA smoothing → adaptive timing scheduler → visualization drawing → video writing → display

## Runtime Modes

### Benchmark mode

SHOW_WINDOW=false
SAVE_OUTPUT_VIDEO=false

Expected behavior:

Highest FPS, closest to detector-only speed.

### Evidence export mode

SHOW_WINDOW=false
SAVE_OUTPUT_VIDEO=true

Expected behavior:

Lower FPS due to VideoWriter overhead. Creates annotated output video.

### Live demo mode

SHOW_WINDOW=true
SAVE_OUTPUT_VIDEO=false

Expected behavior:

Interactive visual demo without video writing overhead.

### Debug mode

SHOW_WINDOW=true
SAVE_OUTPUT_VIDEO=true

Expected behavior:

Lowest FPS. Useful only for short visual debugging.

## Current YOLO26n PC Results

Observed approximate results on the local PC testbed:

Detector-only FPS: around 11–13 FPS

Full pipeline, display OFF, video saving OFF: around 10–12 FPS

Full pipeline, display OFF, video saving ON: around 7–8.5 FPS

Full pipeline, display ON, video saving ON: around 3–5 FPS

## Profiling Result

Typical profiling values:

detect_ms: ~75–100 ms
logic_ms: ~0.1 ms
draw_ms: ~1.5 ms
writer_ms: ~30–50 ms
display_ms: ~80–140 ms

## Interpretation

The adaptive traffic-light logic is computationally lightweight.

The main runtime bottlenecks are:

1. YOLO inference
2. video writing
3. GUI display

ROI assignment, density estimation, EMA smoothing, and adaptive scheduling introduce negligible overhead compared with YOLO inference and output/display operations.

## Report Sentence

Detector-only benchmarking measured raw YOLO inference throughput, while full-pipeline profiling included frame reading, inference, ROI assignment, density estimation, scheduling, visualization, video writing, and display overhead. The results showed that ROI, density, and scheduling logic introduced negligible overhead compared with YOLO inference, video encoding, and GUI display.