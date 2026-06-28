# Phase 16.3 Video Inference Results

Record ONNX Runtime video inference results here after running `deployment/onnx/infer_onnx_video.py`.

## Result Table

| Field | Value |
| --- | --- |
| Input video | `datasets/sample_videos/test.mov` |
| Output video | `/tmp/atlc_phase16_3_video_onnx.mp4` for smoke test |
| Provider | `CPUExecutionProvider` |
| Input resolution | `1920x1080` |
| Input FPS | `30.00` |
| Frames processed | `60` |
| Total processing time | `4.01 s` |
| Approx processing FPS | `14.96` |
| Confidence threshold | `0.25` |
| Result | PASS |
| Notes | Smoke test wrote the annotated video to `/tmp/atlc_phase16_3_video_onnx.mp4`; generated video was not committed. |

## Smoke Test Command

```bash
.venv/bin/python deployment/onnx/infer_onnx_video.py \
  --model deployment/onnx/atlc_yolo26n_custom.onnx \
  --video datasets/sample_videos/test.mov \
  --output /tmp/atlc_phase16_3_video_onnx.mp4 \
  --imgsz 640 \
  --conf 0.25 \
  --providers CPUExecutionProvider \
  --max-frames 60
```

## Scope Notes

- This phase only validates ONNX Runtime video inference.
- Basic processing FPS is printed for operational feedback only.
- No full benchmark framework, TensorRT, TFLite, INT8 quantization, ROI counting, traffic density estimation, green-time planning, UART communication, ESP32 firmware, or STM32 firmware was added.
