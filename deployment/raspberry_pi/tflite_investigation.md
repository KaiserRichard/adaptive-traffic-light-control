# TFLite Investigation

## Goal

Document whether TensorFlow Lite should be considered for Raspberry Pi deployment.

This phase does not perform TFLite conversion or runtime validation.

## Why TFLite Is Considered

TFLite is designed for edge and embedded inference. On Raspberry Pi, it may provide:

- smaller runtime footprint than full TensorFlow
- CPU-friendly inference path
- compatibility with `tflite_runtime`
- possible future accelerator support, depending on hardware

## Why TFLite Is Not the First Runtime Path

The current project already has a validated ONNX Runtime path:

- Phase 16.1 exported and validated ONNX.
- Phase 16.2 added ONNX image inference.
- Phase 16.3 added ONNX video inference.
- Phase 16.4 fixed letterbox preprocessing and box restoration.
- Phase 16.6 selected FP32 ONNX Runtime as the current deployment baseline.

Moving to TFLite introduces new conversion and postprocessing risk. It should be tested after the ONNX Runtime path is working on Raspberry Pi.

## Possible Conversion Paths

### Path A - Ultralytics Export

```text
Ultralytics YOLO .pt
    -> TFLite export
    -> Raspberry Pi TFLite runtime validation
```

This is likely the cleanest path if the installed Ultralytics version supports direct TFLite export for this model.

Validation required:

- confirm exported input shape
- confirm output tensor format
- verify whether output is still `[1, 300, 6]` or a different YOLO head format
- update postprocessing only after observing the real output

### Path B - ONNX to TensorFlow to TFLite

```text
PyTorch YOLO
    -> ONNX
    -> TensorFlow SavedModel
    -> TFLite
```

This path may work, but it has more conversion steps and more opportunity for operator or output-format changes.

Validation required:

- conversion compatibility
- numerical behavior
- preprocessing consistency
- output shape and decoder requirements
- Raspberry Pi runtime support

## Runtime Package Notes

For Raspberry Pi Python inference, the lightweight runtime package may be:

```bash
python -m pip install tflite-runtime
```

Package support depends on Raspberry Pi OS architecture and Python version. The official LiteRT Python quickstart documents `tflite_runtime` as the small runtime package for executing `.tflite` models.

## Risks

| Risk | Why It Matters |
| --- | --- |
| Output tensor shape changes | Existing ONNX postprocessing assumes `[1, 300, 6]` |
| Unsupported operators | Conversion may fail or require Select TF ops |
| Different preprocessing expectations | Direct resize versus letterbox can cause missed detections |
| New runtime dependency | Raspberry Pi package availability must be checked |
| Accuracy drift | TFLite conversion and quantization may change detection confidence |

## Recommendation

Do not force TFLite in Phase 16.7.

Recommended order:

1. Run FP32 ONNX Runtime on Raspberry Pi.
2. Benchmark FP32 ONNX on real hardware.
3. Benchmark dynamic quantized ONNX on real hardware.
4. Attempt TFLite export only after the ONNX Runtime path is measured.
5. Validate TFLite output shape before implementing a decoder.

## Reference Links

- LiteRT Python quickstart: https://developers.google.com/edge/litert/microcontrollers/python
- TensorFlow Lite Interpreter API: https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter

