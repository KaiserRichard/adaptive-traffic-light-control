# Phase 10 — Camera Input and Real-Time Inference Optimization

## 1. Purpose

Phase 10 moves the project from video-file testing toward real camera testing.

Previous phases mainly used stored video files as input. This was useful for repeatable development and benchmarking, but the final system should operate with a live camera stream.

The goal of Phase 10 is to:

- verify that OpenCV can open the camera
- measure raw camera read FPS
- create camera-specific ROI configuration
- run the existing full pipeline with `VIDEO_SOURCE=0`
- reduce YOLO inference load using frame skipping
- prepare the pipeline for Raspberry Pi camera-based deployment

The full pipeline remains:

```text
Camera input
→ YOLO detector
→ ROI split
→ vehicle counting
→ density estimation
→ adaptive scheduler
→ signal runtime controller
→ optional UART to MCU