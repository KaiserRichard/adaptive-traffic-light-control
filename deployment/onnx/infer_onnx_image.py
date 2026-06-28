"""
File:
    infer_onnx_image.py

Phase:
    Phase 16.2 - ONNX Runtime Image Inference

Purpose:
    - Run one image through the exported ONNX model.
    - Save an annotated output image.

Responsibilities:
    - Load ONNX Runtime session.
    - Preprocess image.
    - Execute inference.
    - Draw detections.

This file should NOT:
    - Run video inference.
    - Benchmark FPS.
    - Perform quantization.
    - Send UART messages.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


CLASS_NAMES = {
    0: "car",
    1: "motorbike",
    2: "truck",
    3: "bus",
}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line options for one-image ONNX inference.

    Why:

        Phase 16.2 should be reproducible from the terminal. Explicit
        arguments make the model path, input image, output image, input size,
        confidence threshold, and ONNX Runtime provider visible in the command
        history instead of hiding them as hard-coded values.
    """

    parser = argparse.ArgumentParser(description="Run one-image ONNX Runtime inference for ATLC YOLO.")
    parser.add_argument(
        "--model",
        default="deployment/onnx/atlc_yolo26n_custom.onnx",
        help="Path to the exported ONNX model.",
    )
    parser.add_argument("--image", required=True, help="Path to one input image.")
    parser.add_argument(
        "--output",
        default="results/onnx_image_predictions/atlc_onnx_prediction.jpg",
        help="Path where the annotated image will be saved.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size expected by the ONNX model.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections.")
    parser.add_argument(
        "--providers",
        default="CPUExecutionProvider",
        help="Comma-separated ONNX Runtime providers, for example CPUExecutionProvider.",
    )
    return parser.parse_args()


def parse_providers(provider_text: str) -> list[str]:
    """
    Convert a comma-separated provider string into an ONNX Runtime provider list.

    Why:

        ONNX Runtime accepts providers as an ordered list. Keeping the CLI as a
        comma-separated string makes the default simple while still allowing a
        later GPU-capable environment to pass a different provider order.
    """

    providers = [provider.strip() for provider in provider_text.split(",") if provider.strip()]
    if not providers:
        raise ValueError("At least one ONNX Runtime provider must be specified.")
    return providers


def load_image(image_path: Path) -> np.ndarray:
    """
    Load an input image with OpenCV and fail clearly if it cannot be read.

    Why:

        OpenCV returns None for missing, unsupported, or corrupt images. Raising
        an explicit error avoids passing invalid data into preprocessing where
        the failure would be less obvious.
    """

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    return image_bgr


def letterbox_geometry(original_width: int, original_height: int, imgsz: int) -> tuple[float, int, int, int, int]:
    """
    Compute Ultralytics-style letterbox resize ratio and padding.

    Why:

        Ultralytics preserves aspect ratio before inference and pads the image
        to the requested square input size. ONNX must use the same geometry so
        returned boxes can be restored to the original image coordinates.
    """

    ratio = min(imgsz / original_height, imgsz / original_width)
    resized_width = int(round(original_width * ratio))
    resized_height = int(round(original_height * ratio))

    pad_width = imgsz - resized_width
    pad_height = imgsz - resized_height
    pad_x = int(round(pad_width / 2 - 0.1))
    pad_y = int(round(pad_height / 2 - 0.1))

    return ratio, pad_x, pad_y, resized_width, resized_height


def letterbox_image(image_bgr: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, int, int]:
    """
    Convert an OpenCV image into a letterboxed ONNX input tensor.

    Why:

        Ultralytics PyTorch inference uses letterbox preprocessing instead of a
        direct resize. Matching that behavior keeps aspect ratio intact, reduces
        box distortion, and makes ONNX output more comparable with PyTorch.

        This function returns the tensor plus the resize ratio and top-left
        padding so decoded boxes can be restored with:

            x = (x - pad_x) / ratio
            y = (y - pad_y) / ratio
    """

    original_height, original_width = image_bgr.shape[:2]
    ratio, pad_x, pad_y, resized_width, resized_height = letterbox_geometry(original_width, original_height, imgsz)

    resized_bgr = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    letterboxed_bgr = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    letterboxed_bgr[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized_bgr

    letterboxed_rgb = cv2.cvtColor(letterboxed_bgr, cv2.COLOR_BGR2RGB)
    normalized = letterboxed_rgb.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0), ratio, pad_x, pad_y


def preprocess_image(image_bgr: np.ndarray, imgsz: int) -> np.ndarray:
    """
    Convert an OpenCV image into an ONNX input tensor.

    Why:

        The validated Phase 16.1 ONNX model expects:

            [1, 3, 640, 640]

        Therefore this function:

            - letterboxes the image to the fixed model input size
            - preserves aspect ratio and adds padding
            - converts OpenCV BGR pixels to RGB pixels
            - normalizes uint8 pixels from 0..255 to float32 0..1
            - converts HWC layout to CHW layout
            - adds the batch dimension

        Letterbox preprocessing is required for comparison with Ultralytics
        PyTorch inference because direct resize distorts boxes on non-square
        images.
    """

    input_tensor, _, _, _ = letterbox_image(image_bgr, imgsz)
    return input_tensor


def create_session(model_path: Path, providers: list[str]) -> ort.InferenceSession:
    """
    Create an ONNX Runtime session for the exported model.

    Why:

        Provider selection affects where inference runs. Phase 16.2 defaults to
        CPUExecutionProvider so the result is portable and does not depend on
        TensorRT, CUDA, TFLite, or embedded accelerators.
    """

    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    available = set(ort.get_available_providers())
    missing = [provider for provider in providers if provider not in available]
    if missing:
        raise ValueError(f"Unavailable ONNX Runtime provider(s): {missing}. Available providers: {sorted(available)}")

    return ort.InferenceSession(str(model_path), providers=providers)


def validate_model_io(session: ort.InferenceSession, input_tensor: np.ndarray) -> tuple[str, str]:
    """
    Validate that the ONNX model matches the Phase 16.1 verified tensor contract.

    Why:

        This project already verified that the exported model input is
        [1, 3, 640, 640] and the output is [1, 300, 6]. A generic YOLO decoder
        for [1, 84, 8400] or [1, 8400, 84] would be wrong for this phase, so the
        script checks the contract before postprocessing detections.
    """

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise RuntimeError(f"Expected one model input, observed {len(inputs)} inputs.")
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one model output, observed {len(outputs)} outputs.")

    input_name = inputs[0].name
    output_name = outputs[0].name
    expected_input_shape = (1, 3, input_tensor.shape[2], input_tensor.shape[3])
    if tuple(input_tensor.shape) != expected_input_shape:
        raise RuntimeError(f"Unexpected input tensor shape: {input_tensor.shape}")

    return input_name, output_name


def run_inference(session: ort.InferenceSession, input_name: str, input_tensor: np.ndarray) -> np.ndarray:
    """
    Execute one ONNX Runtime forward pass and return the first output tensor.

    Why:

        Phase 16.2 needs only a single-image smoke test. The returned output is
        intentionally kept as a raw tensor until shape validation confirms it is
        the verified [1, 300, 6] detection format.
    """

    outputs = session.run(None, {input_name: input_tensor})
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one output tensor, observed {len(outputs)} tensors.")

    output = outputs[0]
    if output.shape != (1, 300, 6):
        raise RuntimeError(
            "Unexpected ONNX output shape. "
            f"Expected [1, 300, 6] for [x1, y1, x2, y2, confidence, class_id], observed {list(output.shape)}."
        )
    return output


def scale_box_to_original(box: np.ndarray, original_width: int, original_height: int, imgsz: int) -> tuple[int, int, int, int]:
    """
    Restore one letterboxed detection box back to the source image.

    Why:

        ONNX boxes are in the padded model input coordinate system. Because
        preprocessing preserves aspect ratio and adds padding, box restoration
        must remove the padding before dividing by the resize ratio.
    """

    ratio, pad_x, pad_y, _, _ = letterbox_geometry(original_width, original_height, imgsz)
    x1 = int(round((float(box[0]) - pad_x) / ratio))
    y1 = int(round((float(box[1]) - pad_y) / ratio))
    x2 = int(round((float(box[2]) - pad_x) / ratio))
    y2 = int(round((float(box[3]) - pad_y) / ratio))

    x1 = max(0, min(original_width - 1, x1))
    y1 = max(0, min(original_height - 1, y1))
    x2 = max(0, min(original_width - 1, x2))
    y2 = max(0, min(original_height - 1, y2))
    return x1, y1, x2, y2


def filter_detections(output: np.ndarray, conf_threshold: float, original_shape: tuple[int, int], imgsz: int) -> list[dict]:
    """
    Convert the verified [1, 300, 6] output tensor into drawable detections.

    Why:

        Phase 16.1 validated that each row is already:

            x1, y1, x2, y2, confidence, class_id

        Because the model already returns decoded detections, this function
        only filters by confidence, scales boxes to the original image, and
        rejects invalid boxes. It does not implement a generic YOLO head decoder.
    """

    original_height, original_width = original_shape
    detections = []

    for detection in output[0]:
        if not np.all(np.isfinite(detection)):
            continue

        confidence = float(detection[4])
        if confidence < conf_threshold:
            continue

        x1, y1, x2, y2 = scale_box_to_original(detection[:4], original_width, original_height, imgsz)
        if x2 <= x1 or y2 <= y1:
            continue

        class_id = int(round(float(detection[5])))
        detections.append(
            {
                "box": (x1, y1, x2, y2),
                "confidence": confidence,
                "class_id": class_id,
            }
        )

    return detections


def format_label(class_id: int, confidence: float) -> str:
    """
    Build the compact label drawn above each detection.

    Why:

        ATLC training uses four vehicle classes. Showing the class name and
        confidence on the output image makes the smoke-test result inspectable
        without adding ROI counting, traffic density estimation, or planning.
    """

    class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
    return f"{class_name} {confidence:.2f}"


def draw_detections(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Draw filtered detections on a copy of the original image.

    Why:

        Keeping drawing separate from inference makes the data flow easier to
        verify: raw model output is filtered first, then visualization is added
        only as a final artifact for human inspection.
    """

    annotated = image_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        confidence = detection["confidence"]
        class_id = detection["class_id"]
        label = format_label(class_id, confidence)

        color = (0, 180, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)

        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_width, text_height = text_size
        label_y1 = max(0, y1 - text_height - baseline - 4)
        label_y2 = max(text_height + baseline + 4, y1)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + text_width + 6, label_y2), color, thickness=-1)
        cv2.putText(
            annotated,
            label,
            (x1 + 3, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return annotated


def save_image(output_path: Path, image_bgr: np.ndarray) -> None:
    """
    Save the annotated image and fail if OpenCV cannot write it.

    Why:

        A successful inference run is not useful for Phase 16.2 unless it leaves
        an inspectable artifact. Checking the write result prevents silent
        failures caused by bad paths or unsupported file extensions.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), image_bgr)
    if not success:
        raise RuntimeError(f"OpenCV could not write output image: {output_path}")


def main() -> None:
    """
    Run the complete Phase 16.2 one-image ONNX inference pipeline.

    Why:

        The main function keeps the phase boundary explicit:

            image -> preprocessing -> ONNX Runtime -> [1, 300, 6] tensor
            -> confidence filtering -> bounding boxes -> annotated image

        It intentionally stops before video inference, benchmarking, ROI
        counting, traffic planning, UART, and firmware integration.
    """

    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)
    output_path = Path(args.output)
    providers = parse_providers(args.providers)

    image_bgr = load_image(image_path)
    input_tensor = preprocess_image(image_bgr, args.imgsz)
    session = create_session(model_path, providers)
    input_name, _ = validate_model_io(session, input_tensor)
    output = run_inference(session, input_name, input_tensor)
    detections = filter_detections(output, args.conf, image_bgr.shape[:2], args.imgsz)
    annotated = draw_detections(image_bgr, detections)
    save_image(output_path, annotated)

    print("=" * 80)
    print("ATLC PHASE 16.2 ONNX IMAGE INFERENCE")
    print("=" * 80)
    print(f"Model path:          {model_path}")
    print(f"Image path:          {image_path}")
    print(f"Provider:            {', '.join(session.get_providers())}")
    print(f"Input tensor shape:  {list(input_tensor.shape)}")
    print(f"Output tensor shape: {list(output.shape)}")
    print(f"Number detections:   {len(detections)}")
    print(f"Output image path:   {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
