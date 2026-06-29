# Phase 16.6 Benchmark Results

This file records the Phase 16.6 Edge AI image benchmark results.

## One-Image Benchmark

Command:

```bash
.venv/bin/python deployment/benchmark/benchmark_edge_ai_image.py \
  --fp32-model deployment/onnx/atlc_yolo26n_custom.onnx \
  --quantized-model deployment/onnx/atlc_yolo26n_custom_dynamic_int8.onnx \
  --images yolo_research/datasets/atlc_2000/images/test/09150440_jpg.rf.DvAoBPo7uxkzXD4hgu8H.jpg \
  --imgsz 640 \
  --conf 0.25 \
  --providers CPUExecutionProvider \
  --runs 10 \
  --output /tmp/atlc_phase16_6_benchmark.json
```

| Backend | Model Size | Avg Latency | Approx FPS | Detections | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| FP32 ONNX Runtime | `9.31 MB` | `0.0421 s` | `23.74` | `6` | Deployment baseline; fastest in this run |
| Quantized ONNX Runtime | `2.72 MB` | `0.0811 s` | `12.34` | `6` | Smaller candidate; slower in this run |

## Five-Image Benchmark

Command:

```bash
.venv/bin/python deployment/benchmark/benchmark_edge_ai_image.py \
  --fp32-model deployment/onnx/atlc_yolo26n_custom.onnx \
  --quantized-model deployment/onnx/atlc_yolo26n_custom_dynamic_int8.onnx \
  --images yolo_research/datasets/atlc_2000/images/test \
  --max-images 5 \
  --imgsz 640 \
  --conf 0.25 \
  --providers CPUExecutionProvider \
  --runs 10 \
  --output /tmp/atlc_phase16_6_benchmark_5_images.json
```

| Backend | Model Size | Avg Latency | Approx FPS | Total Detections | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| FP32 ONNX Runtime | `9.31 MB` | `0.0391 s` | `25.60` | `32` | Deployment baseline; fastest in this run |
| Quantized ONNX Runtime | `2.72 MB` | `0.0560 s` | `17.85` | `36` | Smaller candidate; detection total differs |

## Five-Image Detection Counts

| Image | FP32 ONNX | Quantized ONNX |
| --- | ---: | ---: |
| `09150440_jpg.rf.DvAoBPo7uxkzXD4hgu8H.jpg` | `6` | `6` |
| `BUS_118_jpeg.rf.gzkbLP7HPzf03OtSB8yr.jpeg` | `1` | `1` |
| `CAME0223_jpg.rf.jtovPaCwHQRN0WhhQ6R6.jpg` | `7` | `7` |
| `CAME0227_jpg.rf.9ISx8Ub3x8GhGXS6SBWK.jpg` | `9` | `10` |
| `frame_112_jpg.rf.BCmsR4bQX7w1C0in04fO.jpg` | `9` | `12` |

## Important Conclusion

Dynamic `QUInt8` quantization reduced model size significantly but did not improve runtime on the tested CPU environment.

In the one-image benchmark, both backends produced `6` detections at `conf=0.25`.

In the five-image benchmark, the quantized model produced more total detections than FP32 (`36` versus `32`). This does not automatically mean better quality; it means quantization changes confidence/box behavior enough that broader validation is required before adopting it.

## Deployment Recommendation

Use FP32 ONNX Runtime as the current deployment baseline.

Keep the quantized ONNX model as a candidate for storage-constrained deployment, but do not make it the default runtime until it is tested on the target Raspberry Pi hardware.

## Test Environment

| Field | Value |
| --- | --- |
| Platform | `macOS-15.7.7-x86_64-i386-64bit` |
| Machine | `x86_64` |
| Processor | `i386` |
| Python | `3.11.0rc2` |
| OpenCV | `4.10.0` |
| ONNX Runtime | `1.20.1` |
| Provider | `CPUExecutionProvider` |

## Raw Outputs

Raw JSON benchmark outputs are generated under `/tmp` and should not be committed:

- `/tmp/atlc_phase16_6_benchmark.json`
- `/tmp/atlc_phase16_6_benchmark_5_images.json`
