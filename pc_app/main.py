"""
main.py

Phase 2 full pipeline:
1. Open video
2. Initialize detector
3. Load ROI configuration (JSON)
4. Detect vehicles
5. Split detections by direction (A / B / outside)
6. Count vehicles per direction
7. Draw ROI + detections + status panel
8. Show on screen
9. Optionally save annotated video

Key difference from Phase 1:
- Phase 1 → total detection + total counts
- Phase 2 → direction-aware (ROI A / ROI B)
"""

import time
from pathlib import Path
import cv2

from pc_app.config import (
    VIDEO_SOURCE,
    SHOW_WINDOW, 
    SAVE_OUTPUT_VIDEO,
    OUTPUT_VIDEO_PATH,
    ROI_CONFIG_PATH
)

from pc_app.vision.detector import RoboflowDetector
from pc_app.vision.roi import load_roi_config, split_detections_by_direction
from pc_app.vision.counter import count_by_class
from pc_app.vision.visualize import (
    draw_detections,
    draw_bbox_centers,
    draw_polygon,
    draw_status_panel,
)


def main():
    """
    Main loop for Phase 2
    """

    # Initialize detector (same as Phase 1)
    detector = RoboflowDetector()

    # Open video source
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")
    
    # Phase 2: Load ROI configuration from JSON
    # This replaces hardcoded ROI and allows reuse across scenes
    roi_cfg = load_roi_config(ROI_CONFIG_PATH)

    roi_a = roi_cfg["roi_a"]
    roi_b = roi_cfg["roi_b"]

    direction_a_name = roi_cfg["direction_a_name"]
    direction_b_name = roi_cfg["direction_b_name"]

    # Video writer (same logic as Phase 1)
    writer = None

    if SAVE_OUTPUT_VIDEO:
        Path(OUTPUT_VIDEO_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Store previous frame timestamp for FPS
    prev_time = time.time()

    while True:
        # Read frame
        ret, frame = cap.read()

        if not ret:
            break

        # 1) Detect objects (same as Phase 1)
        detections = detector.detect(frame=frame)

        # 2) Phase 2: Split detections into directions
        detections_a, detections_b, detections_outside = split_detections_by_direction(
            detections, roi_a, roi_b
        )

        # 3) Phase 2: Count objects per direction
        counts_a = count_by_class(detections_a)
        counts_b = count_by_class(detections_b)

        # 4) Compute FPS (same as Phase 1)
        current_time = time.time()
        dt = current_time - prev_time

        fps = 1.0 / dt if dt > 0 else 0.0
        prev_time = current_time

        # 5) Phase 2: Draw ROI and detections

        # Draw ROI polygons first
        frame = draw_polygon(frame, roi_a, direction_a_name, (0, 255, 0))   # A → green
        frame = draw_polygon(frame, roi_b, direction_b_name, (255, 0, 0))   # B → blue

        # Draw detections by region
        frame = draw_detections(frame, detections_a, color=(0, 255, 0))
        frame = draw_detections(frame, detections_b, color=(255, 0, 0))
        frame = draw_detections(frame, detections_outside, color=(128, 128, 128))

        # Draw bbox centers (useful for debugging ROI assignment)
        frame = draw_bbox_centers(frame, detections_a, color=(0, 255, 0))
        frame = draw_bbox_centers(frame, detections_b, color=(255, 0, 0))
        frame = draw_bbox_centers(frame, detections_outside, color=(128, 128, 128))

        # Draw status panel (per-direction counts)
        frame = draw_status_panel(
            frame=frame,
            fps=fps,
            total_detections=len(detections),
            counts_a=counts_a,
            counts_b=counts_b
        )

        # 6) Create writer after first frame
        if SAVE_OUTPUT_VIDEO and writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                filename=OUTPUT_VIDEO_PATH,
                fourcc=fourcc,
                fps=20,
                frameSize=(w, h)
            )

        # 7) Save frame
        if writer is not None:
            writer.write(frame)

        # 8) Show window
        if SHOW_WINDOW:
            cv2.imshow("ATLC Phase 2 - ROI and Direction Counting", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # 9) Console output
        print({
            "total_detections": len(detections),
            "direction_A_counts": counts_a,
            "direction_B_counts": counts_b,
            "outside_roi": len(detections_outside),
            "fps": round(fps, 2),
        })

    # Cleanup
    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()