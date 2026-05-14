# Raspberry Pi YOLO Deployment Plan

## Purpose

This document describes how the project will move from PC-based local YOLO inference to Raspberry Pi edge deployment.

## Deployment Goal

The Raspberry Pi will act as the edge-AI host.

Responsibilities:

- read camera/video input
- run local YOLO inference
- compute ROI-based vehicle counts
- estimate traffic density
- compute adaptive signal timing
- later send timing plans to ESP32/STM32 through UART

## Target Architecture

Camera / video input → Raspberry Pi → Local YOLO detector → ROI split → density estimation → adaptive timing → UART to ESP32/STM32

## Initial Deployment Steps

1. Install Raspberry Pi OS.
2. Enable camera support.
3. Test camera capture.
4. Clone the project repository.
5. Create Python virtual environment.
6. Install `requirements.txt`.
7. Copy local YOLO model into `pc_app/models/local/`.
8. Configure `.env`.
9. Run YOLO smoke test.
10. Run detector benchmark.
11. Run full pipeline.
12. Only then integrate UART with ESP32/STM32.

## Setup Commands

```bash
git clone https://github.com/KaiserRichard/adaptive-traffic-light-control.git
cd adaptive-traffic-light-control

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt