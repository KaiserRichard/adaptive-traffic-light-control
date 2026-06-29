# Phase 16.6 Edge AI Benchmark Report

## Goal

Phase 16.6 organizes the Edge AI deployment results from Phase 16.1 through Phase 16.5 into a benchmark-style report.

The report answers:

- which current model/runtime is the best deployment baseline
- which model is fastest in the local CPU smoke benchmark
- which model is smallest
- whether quantization is currently worth adopting
- what should be tested later on Raspberry Pi

## Scope

This phase is limited to Edge AI benchmark reporting for image inference.

It does not implement TFLite conversion, Raspberry Pi deployment, TensorRT, Jetson deployment, ROI counting, traffic planning, UART communication, ESP32 firmware, STM32 firmware, or full dataset mAP evaluation.

## Test Environment

The benchmark is run locally using ONNX Runtime with `CPUExecutionProvider`.

Recorded local environment for this run:

| Field | Value |
| --- | --- |
| Platform | `macOS-15.7.7-x86_64-i386-64bit` |
| Machine | `x86_64` |
| Processor | `i386` |
| Python | `3.11.0rc2` |
| OpenCV | `4.10.0` |
| ONNX Runtime | `1.20.1` |
| Provider | `CPUExecutionProvider` |

## Benchmark Methodology

The benchmark script runs repeated image inference on one image or a small image directory subset.

For each backend, the timed path includes:

```text
OpenCV image already loaded
    -> letterbox preprocessing
    -> ONNX Runtime inference
    -> confidence filtering
    -> letterbox-aware box restoration
```

Session creation and model loading are intentionally outside the timed loop.

This is a practical smoke benchmark. It is not full accuracy evaluation and it is not final hardware benchmarking.

## Models Compared

| Backend | Model Path | Role |
| --- | --- | --- |
| FP32 ONNX Runtime | `deployment/onnx/atlc_yolo26n_custom.onnx` | Current deployment baseline |
| Quantized ONNX Runtime | `deployment/onnx/atlc_yolo26n_custom_dynamic_int8.onnx` | Dynamic QUInt8 size-reduction candidate |

PyTorch / Ultralytics remains the reference path from Phase 16.4, but Phase 16.6 focuses on ONNX Runtime deployment candidates.

## Metrics

The benchmark records:

- backend name
- model path
- model size in MB
- number of images tested
- confidence threshold
- number of timed runs
- average latency per image
- minimum latency
- maximum latency
- approximate FPS
- detection count per image
- basic visual quality note

Approximate FPS is calculated as:

```text
FPS = 1 / average_latency_seconds
```

## Results Table

See [Phase 16.6 benchmark results](phase_16_6_benchmark_results.md).

## Interpretation

Phase 16.5 already showed that dynamic `QUInt8` quantization greatly reduced model size but did not improve local CPU runtime on the main smoke-test image.

Phase 16.6 turns that into a structured benchmark report by measuring FP32 ONNX Runtime and quantized ONNX Runtime with the same corrected letterbox preprocessing path.

Measured outcome:

- FP32 ONNX Runtime was faster in both the one-image and five-image CPU smoke benchmarks.
- Quantized ONNX Runtime was much smaller: `2.72 MB` versus `9.31 MB`.
- Detection counts were the same on the main Phase 16.4 image at `conf=0.25`.
- Detection totals differed on the five-image smoke set, so quantized output should not be treated as equivalent without broader validation.

## Deployment Recommendation

Use FP32 ONNX Runtime as the current deployment baseline.

Reason:

- it uses the corrected letterbox preprocessing path
- it has already been validated against the PyTorch reference
- it is faster than the quantized model in the current CPU smoke benchmark
- it preserves acceptable detection behavior for Phase 16 deployment work

Keep the dynamic quantized model as a candidate for storage-constrained targets, but do not adopt it as the default runtime yet.

## Limitations

- This is not full mAP evaluation.
- This is not a full benchmark framework.
- CPU results from this development machine may not match Raspberry Pi results.
- The benchmark does not measure memory, power, thermals, or end-to-end camera latency.
- The benchmark does not include video throughput or dropped-frame behavior.

## Next Phase

Recommended Phase 16.7:

```text
Phase 16.7 - TFLite / Raspberry Pi Deployment Path
```
