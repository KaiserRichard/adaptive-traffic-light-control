# Roboflow Baseline Detector

## Purpose

Roboflow was used in the early stage of the project as a hosted inference baseline.

The goal was to quickly validate:

- video loading
- vehicle detection
- class filtering
- class normalization
- bounding box visualization
- per-frame counting
- FPS measurement

## Why Roboflow Was Useful

Roboflow allowed the project to test vehicle detection before training or deploying a local YOLO model.

This reduced early development risk because the team could verify the application pipeline first.

## Why the Main Pipeline Later Moved to Local YOLO

For Raspberry Pi deployment and embedded-system integration, the project later moved toward local YOLO inference.

Reasons:

- no internet/API dependency
- better control over model format
- easier benchmarking on Raspberry Pi
- better fit for edge AI deployment
- avoids dependency conflicts in the main environment

## Environment Note

Roboflow dependencies are kept separate in:

requirements-roboflow.txt

The main project uses:

requirements.txt

Do not install both environments together unless dependency compatibility has been teste.d†