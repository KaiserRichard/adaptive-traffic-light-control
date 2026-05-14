"""
main.py

Phase 4/6 full pipeline:
1. Open video
2. Initialize detector backend
3. Load ROI configuration
4. Detect vehicles
5. Split detections by direction
6. Count classes per direction
7. Compute density per direction
8. Smooth density over time using EMA
9. Convert density into signal timing
10. Draw ROI, detections, density, and timing panels
11. Show on screen
12. Optionally save annotated output video

Important FPS note:
- detector benchmark FPS only measures YOLO inference.
- main.py FPS measures the full application loop:
  detection + ROI + density + scheduler + drawing + video writing + display.
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
    BASE_GREEN_TIME,
    MIN_GREEN_TIME,
    MAX_GREEN_TIME,
    YELLOW_TIME,
    ALL_RED_TIME,
    DENSITY_EPSILON,
)

from pc_app.vision.detector_factory import create_detector
from pc_app.vision.roi import load_roi_config, split_detections_by_direction
from pc_app.vision.counter import count_by_class
from pc_app.vision.density import (
    compute_raw_density,
    compute_pce_density,
    EMASmoother,
)

from pc_app.control.scheduler import build_signal_plan

from pc_app.vision.visualize import (
    draw_detections,
    draw_bbox_centers,
    draw_polygon,
    draw_status_panel,
)


# -------------------------------------------------------------------------
# Runtime debug / profiling settings
# -------------------------------------------------------------------------

# Print console log every N frames instead of every frame.
# Printing every frame can significantly reduce FPS.
LOG_EVERY_N_FRAMES = 30

# Print stage timing every N frames.
# This helps identify whether the bottleneck is detection, drawing, writer, etc.
PROFILE_EVERY_N_FRAMES = 30

# If True, draw bbox center points.
# Useful for debugging ROI assignment, but slightly increases drawing overhead.
DRAW_BBOX_CENTERS = True


def main():
    """
    Main full pipeline loop.
    """

    # ---------------------------------------------------------------------
    # Initialize detector
    # ---------------------------------------------------------------------
    detector = create_detector()

    # ---------------------------------------------------------------------
    # Open video source
    # ---------------------------------------------------------------------
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

    # ---------------------------------------------------------------------
    # Load ROI configuration
    # ---------------------------------------------------------------------
    roi_cfg = load_roi_config(ROI_CONFIG_PATH)

    roi_a = roi_cfg["roi_a"]
    roi_b = roi_cfg["roi_b"]

    direction_a_name = roi_cfg["direction_a_name"]
    direction_b_name = roi_cfg["direction_b_name"]

    # ---------------------------------------------------------------------
    # Initialize EMA smoothers
    # ---------------------------------------------------------------------
    smoother_a = EMASmoother(alpha=DENSITY_SMOOTHING_ALPHA)
    smoother_b = EMASmoother(alpha=DENSITY_SMOOTHING_ALPHA)

    # ---------------------------------------------------------------------
    # Video writer
    # ---------------------------------------------------------------------
    writer = None

    if SAVE_OUTPUT_VIDEO:
        Path(OUTPUT_VIDEO_PATH).parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Main loop state
    # ---------------------------------------------------------------------
    frame_index = 0

    # Full-loop FPS is measured from the start of one iteration to the end.
    # This is more realistic than measuring only detection time.
    last_loop_end_time = time.perf_counter()

    print("Main pipeline started")
    print("Video source:", VIDEO_SOURCE)
    print("Save output video:", SAVE_OUTPUT_VIDEO)
    print("Output video path:", OUTPUT_VIDEO_PATH)
    print("Show window:", SHOW_WINDOW)

    while True:
        loop_start_time = time.perf_counter()

        # -----------------------------------------------------------------
        # Read frame
        # -----------------------------------------------------------------
        read_start_time = time.perf_counter()

        ret, frame = cap.read()

        read_end_time = time.perf_counter()

        if not ret:
            break

        frame_index += 1

        # -----------------------------------------------------------------
        # 1) Detect objects
        # -----------------------------------------------------------------
        detect_start_time = time.perf_counter()

        detections = detector.detect(frame=frame)

        detect_end_time = time.perf_counter()

        # -----------------------------------------------------------------
        # 2) ROI split + counts + density + scheduler
        # -----------------------------------------------------------------
        logic_start_time = time.perf_counter()

        detections_a, detections_b, detections_outside = split_detections_by_direction(
            detections, roi_a, roi_b
        )

        counts_a = count_by_class(detections_a)
        counts_b = count_by_class(detections_b)

        raw_density_a = compute_raw_density(detections_a)
        raw_density_b = compute_raw_density(detections_b)

        pce_density_a = compute_pce_density(detections_a, PCE_WEIGHTS)
        pce_density_b = compute_pce_density(detections_b, PCE_WEIGHTS)

        smoothed_pce_a = smoother_a.update(pce_density_a)
        smoothed_pce_b = smoother_b.update(pce_density_b)

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

        logic_end_time = time.perf_counter()

        # -----------------------------------------------------------------
        # 3) Compute full-loop FPS
        # -----------------------------------------------------------------
        # This measures how fast the whole app loop is running.
        now = time.perf_counter()
        full_loop_dt = now - last_loop_end_time
        fps = 1.0 / full_loop_dt if full_loop_dt > 0 else 0.0
        last_loop_end_time = now

        # -----------------------------------------------------------------
        # 4) Draw visualization
        # -----------------------------------------------------------------
        draw_start_time = time.perf_counter()

        frame = draw_polygon(frame, roi_a, direction_a_name, (0, 255, 0))
        frame = draw_polygon(frame, roi_b, direction_b_name, (255, 0, 0))

        frame = draw_detections(frame, detections_a, color=(0, 255, 0))
        frame = draw_detections(frame, detections_b, color=(255, 0, 0))
        frame = draw_detections(frame, detections_outside, color=(128, 128, 128))

        if DRAW_BBOX_CENTERS:
            frame = draw_bbox_centers(frame, detections_a, color=(0, 255, 0))
            frame = draw_bbox_centers(frame, detections_b, color=(255, 0, 0))
            frame = draw_bbox_centers(frame, detections_outside, color=(128, 128, 128))

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

        draw_end_time = time.perf_counter()

        # -----------------------------------------------------------------
        # 5) Initialize writer after first processed frame
        # -----------------------------------------------------------------
        writer_start_time = time.perf_counter()

        if SAVE_OUTPUT_VIDEO and writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                filename=OUTPUT_VIDEO_PATH,
                fourcc=fourcc,
                fps=20,
                frameSize=(w, h),
            )

        if writer is not None:
            writer.write(frame)

        writer_end_time = time.perf_counter()

        # -----------------------------------------------------------------
        # 6) Display window
        # -----------------------------------------------------------------
        display_start_time = time.perf_counter()

        if SHOW_WINDOW:
            cv2.imshow("ATLC Full Pipeline", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        display_end_time = time.perf_counter()

        # -----------------------------------------------------------------
        # 7) End-of-loop profiling
        # -----------------------------------------------------------------
        loop_end_time = time.perf_counter()

        read_ms = (read_end_time - read_start_time) * 1000
        detect_ms = (detect_end_time - detect_start_time) * 1000
        logic_ms = (logic_end_time - logic_start_time) * 1000
        draw_ms = (draw_end_time - draw_start_time) * 1000
        writer_ms = (writer_end_time - writer_start_time) * 1000
        display_ms = (display_end_time - display_start_time) * 1000
        total_ms = (loop_end_time - loop_start_time) * 1000

        # Console output every N frames only.
        if frame_index % LOG_EVERY_N_FRAMES == 0:
            print(
                {
                    "frame": frame_index,
                    "total_detections": len(detections),
                    "direction_A_counts": counts_a,
                    "direction_B_counts": counts_b,
                    "signal_plan": signal_plan,
                    "outside_roi": len(detections_outside),
                    "fps": round(fps, 2),
                }
            )

        # Profiling output every N frames.
        if frame_index % PROFILE_EVERY_N_FRAMES == 0:
            print(
                {
                    "profile_frame": frame_index,
                    "read_ms": round(read_ms, 2),
                    "detect_ms": round(detect_ms, 2),
                    "logic_ms": round(logic_ms, 2),
                    "draw_ms": round(draw_ms, 2),
                    "writer_ms": round(writer_ms, 2),
                    "display_ms": round(display_ms, 2),
                    "total_ms": round(total_ms, 2),
                    "full_loop_fps_est": round(1000.0 / total_ms if total_ms > 0 else 0.0, 2),
                }
            )

    # ---------------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------------
    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()

    print("Main pipeline finished")


if __name__ == "__main__":
    main()