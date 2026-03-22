"""
main.py

Phase 4 full pipeline:
1. Open video
2. Initialize detector
3. Load ROI configuration
4. Detect vehicles
5. Split detections by direction
6. Count classes per direction
7. Compute density per direction
8. Smooth density over time (EMA)
9. Convert density into signal timing
10. Draw ROI, detections, density, and timing panels
11. Show on screen
12. Optionally save annotated output video

Key difference from previous phases:
- Phase 2 -> direction-aware counting
- Phase 3 -> adds density estimation + smoothing
- Phase 4 -> adds timing scheduler based on density
"""

import time
from pathlib import Path
import cv2

from pc_app.config import (
    VIDEO_SOURCE,
    SHOW_WINDOW,
    SAVE_OUTPUT_VIDEO,
    OUTPUT_VIDEO_PATH,
    ROI_CONFIG_PATH,
    PCE_WEIGHTS,
    DENSITY_SMOOTHING_ALPHA,
    # PHASE 4:
    # Timing configuration used by the rule-based scheduler
    BASE_GREEN_TIME,
    MIN_GREEN_TIME,
    MAX_GREEN_TIME,
    YELLOW_TIME,
    ALL_RED_TIME,
    DENSITY_EPSILON,
)

from pc_app.vision.detector import RoboflowDetector
from pc_app.vision.roi import load_roi_config, split_detections_by_direction
from pc_app.vision.counter import count_by_class
from pc_app.vision.density import (
    compute_raw_density,
    compute_pce_density,
    EMASmoother,
)

# PHASE 4:
# Import scheduler that converts density into signal timing
from pc_app.control.scheduler import build_signal_plan

from pc_app.vision.visualize import (
    draw_detections,
    draw_bbox_centers,
    draw_polygon,
    draw_status_panel,
)


def main():
    """
    Main loop for Phase 4
    """

    # -------------------------------------------------------------------------
    # Initialize detector (Phase 1)
    # -------------------------------------------------------------------------
    detector = RoboflowDetector()

    # -------------------------------------------------------------------------
    # Open video source
    # -------------------------------------------------------------------------
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

    # -------------------------------------------------------------------------
    # Load ROI configuration (Phase 2)
    # -------------------------------------------------------------------------
    roi_cfg = load_roi_config(ROI_CONFIG_PATH)

    roi_a = roi_cfg["roi_a"]
    roi_b = roi_cfg["roi_b"]

    direction_a_name = roi_cfg["direction_a_name"]
    direction_b_name = roi_cfg["direction_b_name"]

    # -------------------------------------------------------------------------
    # PHASE 3:
    # Initialize EMA smoothers for direction-specific density
    # -------------------------------------------------------------------------
    smoother_a = EMASmoother(alpha=DENSITY_SMOOTHING_ALPHA)
    smoother_b = EMASmoother(alpha=DENSITY_SMOOTHING_ALPHA)

    # -------------------------------------------------------------------------
    # Video writer (same pattern as Phase 1)
    # -------------------------------------------------------------------------
    writer = None

    if SAVE_OUTPUT_VIDEO:
        Path(OUTPUT_VIDEO_PATH).parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # FPS tracking
    # -------------------------------------------------------------------------
    prev_time = time.time()

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break

        # ---------------------------------------------------------------------
        # 1) Detect objects (Phase 1)
        # ---------------------------------------------------------------------
        detections = detector.detect(frame=frame)

        # ---------------------------------------------------------------------
        # 2) Split detections by direction (Phase 2)
        # ---------------------------------------------------------------------
        detections_a, detections_b, detections_outside = split_detections_by_direction(
            detections, roi_a, roi_b
        )

        # ---------------------------------------------------------------------
        # 3) Count objects per direction (Phase 2)
        # ---------------------------------------------------------------------
        counts_a = count_by_class(detections_a)
        counts_b = count_by_class(detections_b)

        # ---------------------------------------------------------------------
        # PHASE 3:
        # Compute raw density and PCE-weighted density
        # ---------------------------------------------------------------------
        raw_density_a = compute_raw_density(detections_a)
        raw_density_b = compute_raw_density(detections_b)

        pce_density_a = compute_pce_density(detections_a, PCE_WEIGHTS)
        pce_density_b = compute_pce_density(detections_b, PCE_WEIGHTS)

        # ---------------------------------------------------------------------
        # PHASE 3:
        # Apply EMA smoothing to reduce frame-to-frame noise
        # ---------------------------------------------------------------------
        smoothed_pce_a = smoother_a.update(pce_density_a)
        smoothed_pce_b = smoother_b.update(pce_density_b)

        # ---------------------------------------------------------------------
        # 4) Compute FPS
        # ---------------------------------------------------------------------
        current_time = time.time()
        dt = current_time - prev_time

        fps = 1.0 / dt if dt > 0 else 0.0
        prev_time = current_time

        # ---------------------------------------------------------------------
        # PHASE 3:
        # Build structured density summary for visualization and logging
        # ---------------------------------------------------------------------
        density_a = {
            "raw_density": raw_density_a,
            "pce_density": pce_density_a,
            "smoothed_pce_density": smoothed_pce_a,
        }

        density_b = {
            "raw_density": raw_density_b,
            "pce_density": pce_density_b,
            "smoothed_pce_density": smoothed_pce_b,
        }

        # ---------------------------------------------------------------------
        # PHASE 4:
        # Convert smoothed density into a signal timing plan
        # ---------------------------------------------------------------------
        signal_plan = build_signal_plan(
            density_a=density_a["smoothed_pce_density"],
            density_b=density_b["smoothed_pce_density"],
            epsilon=DENSITY_EPSILON,
            base_green_time=BASE_GREEN_TIME,
            min_green_time=MIN_GREEN_TIME,
            max_green_time=MAX_GREEN_TIME,
            yellow_time=YELLOW_TIME,
            all_red_time=ALL_RED_TIME,
        )

        # ---------------------------------------------------------------------
        # 5) Draw ROI polygons (Phase 2)
        # ---------------------------------------------------------------------
        frame = draw_polygon(frame, roi_a, direction_a_name, (0, 255, 0))   # A -> green
        frame = draw_polygon(frame, roi_b, direction_b_name, (255, 0, 0))   # B -> blue

        # ---------------------------------------------------------------------
        # 6) Draw detections (Phase 2)
        # ---------------------------------------------------------------------
        frame = draw_detections(frame, detections_a, color=(0, 255, 0))
        frame = draw_detections(frame, detections_b, color=(255, 0, 0))
        frame = draw_detections(frame, detections_outside, color=(128, 128, 128))

        # ---------------------------------------------------------------------
        # 7) Draw bbox centers (debugging ROI assignment)
        # ---------------------------------------------------------------------
        frame = draw_bbox_centers(frame, detections_a, color=(0, 255, 0))
        frame = draw_bbox_centers(frame, detections_b, color=(255, 0, 0))
        frame = draw_bbox_centers(frame, detections_outside, color=(128, 128, 128))

        # ---------------------------------------------------------------------
        # 8) Draw status panel
        # Phase 3 adds density display
        # Phase 4 adds signal timing display
        # ---------------------------------------------------------------------
        frame = draw_status_panel(
            frame=frame,
            fps=fps,
            total_detections=len(detections),
            counts_a=counts_a,
            counts_b=counts_b,
            density_a=density_a,
            density_b=density_b,
            signal_plan=signal_plan,
        )

        # ---------------------------------------------------------------------
        # 9) Initialize writer (first frame)
        # ---------------------------------------------------------------------
        if SAVE_OUTPUT_VIDEO and writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                filename=OUTPUT_VIDEO_PATH,
                fourcc=fourcc,
                fps=20,
                frameSize=(w, h)
            )

        # ---------------------------------------------------------------------
        # 10) Save frame
        # ---------------------------------------------------------------------
        if writer is not None:
            writer.write(frame)

        # ---------------------------------------------------------------------
        # 11) Show window
        # ---------------------------------------------------------------------
        if SHOW_WINDOW:
            cv2.imshow("ATLC Phase 4 - Timing Scheduler", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # ---------------------------------------------------------------------
        # 12) Console output
        # ---------------------------------------------------------------------
        print({
            "total_detections": len(detections),
            "direction_A_counts": counts_a,
            "direction_B_counts": counts_b,
            "direction_A_density": density_a,
            "direction_B_density": density_b,
            "signal_plan": signal_plan,
            "outside_roi": len(detections_outside),
            "fps": round(fps, 2),
        })

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()