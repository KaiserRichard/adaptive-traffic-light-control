"""
main.py

Phase 5 full pipeline:
1. Open video
2. Initialize detector
3. Load ROI configuration
4. Detect vehicles
5. Split detections by direction
6. Count classes per direction
7. Compute density per direction
8. Smooth density over time (EMA)
9. Convert density into signal timing
10. Send signal plan to MCU via UART
11. Draw ROI, detections, density, timing, and communication status
12. Show on screen
13. Optionally save annotated output video

Key differences from previous phases:
- Phase 2 -> direction-aware counting
- Phase 3 -> density estimation + smoothing
- Phase 4 -> timing scheduler
- Phase 5 -> host-to-MCU communication (UART + ACK tracking)
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
    # PHASE 5:
    MCU_SERIAL_PORT,
    ENABLE_MCU_COMM,
    MCU_BAUD_RATE,
    MCU_SERIAL_TIMEOUT,
    WAIT_FOR_MCU_ACK,
)

from pc_app.vision.detector import RoboflowDetector
from pc_app.vision.roi import load_roi_config, split_detections_by_direction
from pc_app.vision.counter import count_by_class
from pc_app.vision.density import (
    compute_raw_density,
    compute_pce_density,
    EMASmoother,
)
from pc_app.control.scheduler import build_signal_plan

# PHASE 5:
from pc_app.comm.uart_sender import UartPlanSender

from pc_app.vision.visualize import (
    draw_detections,
    draw_bbox_centers,
    draw_polygon,
    draw_status_panel,
)


def main():
    """
    Main loop for Phase 5
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
    smoother_a = EMASmoother(alpha=DENSITY_SMOOTHING_ALPHA)
    smoother_b = EMASmoother(alpha=DENSITY_SMOOTHING_ALPHA)

    # -------------------------------------------------------------------------
    # PHASE 5:
    # Initialize UART sender (only if enabled)
    # -------------------------------------------------------------------------
    sender = None
    if ENABLE_MCU_COMM:
        sender = UartPlanSender(
            port=MCU_SERIAL_PORT,
            baud_rate=MCU_BAUD_RATE,
            timeout=MCU_SERIAL_TIMEOUT,
        )

    # Track plan IDs (used for ACK matching)
    plan_id = 0

    writer = None

    if SAVE_OUTPUT_VIDEO:
        Path(OUTPUT_VIDEO_PATH).parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # FPS tracking
    # -------------------------------------------------------------------------
    prev_time = time.time()

    # PHASE 5: communication status tracking
    last_ack = "none"
    last_plan_id = "none"

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---------------------------------------------------------------------
        # 1) Detect objects (Phase 1)
        # ---------------------------------------------------------------------
        detections = detector.detect(frame=frame)

        # ---------------------------------------------------------------------
        # 2) Split detections (Phase 2)
        # ---------------------------------------------------------------------
        detections_a, detections_b, detections_outside = split_detections_by_direction(
            detections, roi_a, roi_b
        )

        # ---------------------------------------------------------------------
        # 3) Count objects (Phase 2)
        # ---------------------------------------------------------------------
        counts_a = count_by_class(detections_a)
        counts_b = count_by_class(detections_b)

        # ---------------------------------------------------------------------
        # PHASE 3: Density
        # ---------------------------------------------------------------------
        raw_density_a = compute_raw_density(detections_a)
        raw_density_b = compute_raw_density(detections_b)

        pce_density_a = compute_pce_density(detections_a, PCE_WEIGHTS)
        pce_density_b = compute_pce_density(detections_b, PCE_WEIGHTS)

        smoothed_pce_a = smoother_a.update(pce_density_a)
        smoothed_pce_b = smoother_b.update(pce_density_b)

        # ---------------------------------------------------------------------
        # 4) FPS
        # ---------------------------------------------------------------------
        current_time = time.time()
        dt = current_time - prev_time
        fps = 1.0 / dt if dt > 0 else 0.0
        prev_time = current_time

        # ---------------------------------------------------------------------
        # Density summary
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
        # PHASE 4: Build signal plan
        # ---------------------------------------------------------------------
        signal_plan = build_signal_plan(
            density_a=density_a["smoothed_pce_density"],
            density_b=density_b["smoothed_pce_density"],
            base_green=BASE_GREEN_TIME,
            min_green=MIN_GREEN_TIME,
            max_green=MAX_GREEN_TIME,
            yellow_time=YELLOW_TIME,
            all_red_time=ALL_RED_TIME,
            epsilon=DENSITY_EPSILON,
        )

        # ---------------------------------------------------------------------
        # PHASE 5: Send plan to MCU
        # ---------------------------------------------------------------------
        if sender is not None:
            plan_id += 1
            last_plan_id = str(plan_id)

            try:
                ack = sender.send_plan(
                    plan_id=plan_id,
                    signal_plan=signal_plan,
                    wait_for_ack=WAIT_FOR_MCU_ACK,
                )

                if ack is not None:
                    last_ack = str(ack["plan_id"])
                else:
                    last_ack = "not_waited"

            except Exception:
                last_ack = "error"

        # Communication status for UI
        comm_status = {
            "enabled": "yes" if ENABLE_MCU_COMM else "no",
            "last_plan_id": last_plan_id,
            "last_ack": last_ack,
        }

        # ---------------------------------------------------------------------
        # Visualization
        # ---------------------------------------------------------------------
        frame = draw_polygon(frame, roi_a, direction_a_name, (0, 255, 0))
        frame = draw_polygon(frame, roi_b, direction_b_name, (255, 0, 0))

        frame = draw_detections(frame, detections_a, color=(0, 255, 0))
        frame = draw_detections(frame, detections_b, color=(255, 0, 0))
        frame = draw_detections(frame, detections_outside, color=(128, 128, 128))

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
            comm_status=comm_status,
        )

        # ---------------------------------------------------------------------
        # Writer
        # ---------------------------------------------------------------------
        if SAVE_OUTPUT_VIDEO and writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 20, (w, h))

        if writer is not None:
            writer.write(frame)

        # ---------------------------------------------------------------------
        # Display
        # ---------------------------------------------------------------------
        if SHOW_WINDOW:
            cv2.imshow("ATLC Phase 5 - MCU Communication", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # ---------------------------------------------------------------------
        # Console log
        # ---------------------------------------------------------------------
        print({
            "total_detections": len(detections),
            "direction_A_counts": counts_a,
            "direction_B_counts": counts_b,
            "direction_A_density": density_a,
            "direction_B_density": density_b,
            "signal_plan": signal_plan,
            "comm_status": comm_status,
            "outside_roi": len(detections_outside),
            "fps": round(fps, 2),
        })

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    cap.release()

    if writer is not None:
        writer.release()

    if sender is not None:
        sender.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()