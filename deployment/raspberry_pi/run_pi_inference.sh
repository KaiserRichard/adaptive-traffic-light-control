#!/usr/bin/env bash
#
# Run a Raspberry Pi ONNX Runtime smoke test for ATLC.
#
# Usage:
#   deployment/raspberry_pi/run_pi_inference.sh image
#   deployment/raspberry_pi/run_pi_inference.sh video
#
# Environment variables can override defaults:
#   PYTHON_BIN=.venv/bin/python
#   MODEL=deployment/onnx/atlc_yolo26n_custom.onnx
#   IMAGE=path/to/image.jpg
#   VIDEO=path/to/video.mov
#   OUTPUT=/tmp/output.jpg

set -euo pipefail

MODE="${1:-image}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODEL="${MODEL:-deployment/onnx/atlc_yolo26n_custom.onnx}"
IMAGE="${IMAGE:-yolo_research/datasets/atlc_2000/images/test/09150440_jpg.rf.DvAoBPo7uxkzXD4hgu8H.jpg}"
VIDEO="${VIDEO:-datasets/sample_videos/test.mov}"
IMGSZ="${IMGSZ:-640}"
CONF="${CONF:-0.25}"
PROVIDERS="${PROVIDERS:-CPUExecutionProvider}"
MAX_FRAMES="${MAX_FRAMES:-60}"

case "${MODE}" in
  image)
    OUTPUT="${OUTPUT:-/tmp/atlc_pi_onnx_test.jpg}"
    "${PYTHON_BIN}" deployment/onnx/infer_onnx_image.py \
      --model "${MODEL}" \
      --image "${IMAGE}" \
      --output "${OUTPUT}" \
      --imgsz "${IMGSZ}" \
      --conf "${CONF}" \
      --providers "${PROVIDERS}"
    ;;
  video)
    OUTPUT="${OUTPUT:-/tmp/atlc_pi_onnx_video_test.mp4}"
    "${PYTHON_BIN}" deployment/onnx/infer_onnx_video.py \
      --model "${MODEL}" \
      --video "${VIDEO}" \
      --output "${OUTPUT}" \
      --imgsz "${IMGSZ}" \
      --conf "${CONF}" \
      --providers "${PROVIDERS}" \
      --max-frames "${MAX_FRAMES}"
    ;;
  *)
    echo "Unsupported mode: ${MODE}" >&2
    echo "Use: image or video" >&2
    exit 2
    ;;
esac

