"""
main.py

Adaptive Traffic Light Control — Full Pipeline

Current system flow:

    Video / Camera input
    → detector backend
    → ROI split
    → vehicle counting
    → density estimation
    → EMA smoothing
    → adaptive signal scheduler
    → stable signal runtime controller
    → optional UART transmission to MCU
    → visualization with virtual traffic light overlay
    → optional output video

Important Phase 8 change:

    Raw signal_plan is no longer treated as the plan currently being executed.

    Instead:

        raw signal_plan
            = newly computed recommendation from current frame/density

        pending_plan
            = latest recommended plan waiting to be applied

        active_plan
            = plan currently being executed by runtime controller and MCU

        runtime_state
            = current traffic state and countdown

    This prevents the traffic light plan from jumping randomly every frame.

MCU design note:

    The MCU target is device-agnostic.

    Current validated MCU testbed:
        Arduino Uno

    Future candidates:
        ESP32
        STM32
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
    ENABLE_UART,
    UART_PORT,
    UART_BAUD,
    UART_SEND_INTERVAL_FRAMES,
    UART_ACK_TIMEOUT,
    UART_MIN_GREEN_CHANGE,
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
from pc_app.control.uart_sender import UartPlanSender, should_send_plan
from pc_app.control.signal_runtime import SignalRuntimeController

from pc_app.vision.visualize import (
    draw_detections,
    draw_bbox_centers,
    draw_polygon,
    draw_status_panel,
    draw_signal_runtime_panel,
)


LOG_EVERY_N_FRAMES = 30
PROFILE_EVERY_N_FRAMES = 30
DRAW_BBOX_CENTERS = True


def make_default_signal_plan() -> dict:
    """
    Build a safe initial signal plan before the first scheduler result exists.
    """

    return {
        "green_a": BASE_GREEN_TIME,
        "green_b": BASE_GREEN_TIME,
        "yellow": YELLOW_TIME,
        "all_red": ALL_RED_TIME,
    }


def main() -> None:
    """
    Run the full adaptive traffic light pipeline.
    """

    detector = create_detector()

    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

    roi_cfg = load_roi_config(ROI_CONFIG_PATH)

    roi_a = roi_cfg["roi_a"]
    roi_b = roi_cfg["roi_b"]

    direction_a_name = roi_cfg["direction_a_name"]
    direction_b_name = roi_cfg["direction_b_name"]

    smoother_a = EMASmoother(alpha=DENSITY_SMOOTHING_ALPHA)
    smoother_b = EMASmoother(alpha=DENSITY_SMOOTHING_ALPHA)

    writer = None

    if SAVE_OUTPUT_VIDEO:
        Path(OUTPUT_VIDEO_PATH).parent.mkdir(parents=True, exist_ok=True)

    frame_index = 0

    # These are initialized after optional UART startup sync.
    # Reason:
    # Opening serial can reset Arduino and UartPlanSender may wait for boot messages.
    # If we start the runtime clock before that, the overlay countdown can drift.
    last_loop_end_time = None
    last_runtime_update_time = None

    print("Main pipeline started")
    print("Video source:", VIDEO_SOURCE)
    print("Save output video:", SAVE_OUTPUT_VIDEO)
    print("Output video path:", OUTPUT_VIDEO_PATH)
    print("Show window:", SHOW_WINDOW)

    # ---------------------------------------------------------------------
    # Stable runtime controller
    # ---------------------------------------------------------------------
    # This controller simulates the traffic light FSM on the host side.
    # It provides active_plan + countdown for the video overlay.
    # It also gives a stable reference to compare against the MCU LEDs.
    runtime_controller = SignalRuntimeController(
        initial_plan=make_default_signal_plan()
    )

    # ---------------------------------------------------------------------
    # UART sender
    # ---------------------------------------------------------------------
    uart_sender = None
    last_sent_plan = None
    latest_uart_status = "UART: disabled"

    if ENABLE_UART:
        try:
            uart_sender = UartPlanSender(
                port=UART_PORT,
                baud=UART_BAUD,
                ack_timeout=UART_ACK_TIMEOUT,
            )

            latest_uart_status = "UART: enabled"

            print("UART enabled")
            print("UART port:", UART_PORT)
            print("UART baud:", UART_BAUD)
            print("UART send interval frames:", UART_SEND_INTERVAL_FRAMES)

            # -------------------------------------------------------------
            # Phase 8 startup synchronization
            # -------------------------------------------------------------
            # When main.py starts, the MCU may already be running an old FSM cycle.
            # Therefore, immediately send the current active_plan to reset/sync
            # the MCU countdown with the host-side overlay countdown.
            startup_snapshot = runtime_controller.get_snapshot()
            startup_plan = startup_snapshot.active_plan

            startup_result = uart_sender.send_plan(startup_plan)

            if startup_result.success:
                last_sent_plan = {
                    "green_a": startup_plan["green_a"],
                    "green_b": startup_plan["green_b"],
                    "yellow": startup_plan["yellow"],
                    "all_red": startup_plan["all_red"],
                }

                latest_uart_status = (
                    f"UART: startup sync ACK {startup_result.plan_id}, "
                    f"{round(startup_result.latency_ms, 1) if startup_result.latency_ms else 0} ms"
                )

                print(
                    {
                        "uart": "startup_sync_success",
                        "message": startup_result.message,
                        "ack": startup_result.ack_line,
                        "latency_ms": round(startup_result.latency_ms, 2)
                        if startup_result.latency_ms is not None
                        else None,
                    }
                )
            else:
                latest_uart_status = f"UART: startup sync failed ({startup_result.error})"

                print(
                    {
                        "uart": "startup_sync_failed",
                        "message": startup_result.message,
                        "ack_or_last_line": startup_result.ack_line,
                        "error": startup_result.error,
                    }
                )

        except Exception as exc:
            uart_sender = None
            latest_uart_status = "UART: failed to open"
            print(f"[UART/WARN] Failed to open UART port: {exc}")
            print("[UART/WARN] Continuing pipeline without UART.")
    else:
        print("UART disabled")

    # Reset timing baselines after UART startup sync.
    # This keeps the on-screen countdown aligned with the MCU countdown.
    last_runtime_update_time = time.perf_counter()
    last_loop_end_time = time.perf_counter()

    try:
        while True:
            loop_start_time = time.perf_counter()

            # -------------------------------------------------------------
            # Runtime controller update based on real elapsed time
            # -------------------------------------------------------------
            current_time = time.perf_counter()
            runtime_dt = current_time - last_runtime_update_time
            last_runtime_update_time = current_time

            new_cycle_started = runtime_controller.update(runtime_dt)

            # -------------------------------------------------------------
            # Read frame
            # -------------------------------------------------------------
            read_start_time = time.perf_counter()

            ret, frame = cap.read()

            read_end_time = time.perf_counter()

            if not ret:
                break

            frame_index += 1

            # -------------------------------------------------------------
            # Detection
            # -------------------------------------------------------------
            detect_start_time = time.perf_counter()

            detections = detector.detect(frame=frame)

            detect_end_time = time.perf_counter()

            # -------------------------------------------------------------
            # ROI, density, scheduler
            # -------------------------------------------------------------
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

            raw_signal_plan = build_signal_plan(
                density_a=density_a["smoothed_pce_density"],
                density_b=density_b["smoothed_pce_density"],
                epsilon=DENSITY_EPSILON,
                base_green_time=BASE_GREEN_TIME,
                min_green_time=MIN_GREEN_TIME,
                max_green_time=MAX_GREEN_TIME,
                yellow_time=YELLOW_TIME,
                all_red_time=ALL_RED_TIME,
            )

            # Store raw scheduler result as pending plan.
            # It will be applied by SignalRuntimeController only at a safe boundary.
            runtime_controller.update_pending_plan(raw_signal_plan)

            logic_end_time = time.perf_counter()

            runtime_snapshot = runtime_controller.get_snapshot()
            active_plan = runtime_snapshot.active_plan

            # -------------------------------------------------------------
            # UART sending
            # -------------------------------------------------------------
            # Strategy:
            # - Send active_plan, not raw_signal_plan.
            # - This keeps MCU aligned with what the overlay shows.
            # - Try sending periodically or when a new cycle starts.
            uart_start_time = time.perf_counter()

            if uart_sender is not None:
                should_try_uart = (
                    frame_index % UART_SEND_INTERVAL_FRAMES == 0
                    or new_cycle_started
                )

                if should_try_uart and should_send_plan(
                    previous_plan=last_sent_plan,
                    current_plan=active_plan,
                    min_green_change=UART_MIN_GREEN_CHANGE,
                ):
                    result = uart_sender.send_plan(active_plan)

                    if result.success:
                        last_sent_plan = {
                            "green_a": active_plan["green_a"],
                            "green_b": active_plan["green_b"],
                            "yellow": active_plan["yellow"],
                            "all_red": active_plan["all_red"],
                        }

                        latest_uart_status = (
                            f"UART: ACK {result.plan_id}, "
                            f"{round(result.latency_ms, 1) if result.latency_ms else 0} ms"
                        )

                        print(
                            {
                                "uart": "success",
                                "message": result.message,
                                "ack": result.ack_line,
                                "latency_ms": round(result.latency_ms, 2)
                                if result.latency_ms is not None
                                else None,
                            }
                        )
                    else:
                        latest_uart_status = f"UART: failed ({result.error})"

                        print(
                            {
                                "uart": "failed",
                                "message": result.message,
                                "ack_or_last_line": result.ack_line,
                                "error": result.error,
                            }
                        )

            uart_end_time = time.perf_counter()

            # -------------------------------------------------------------
            # Full-loop FPS
            # -------------------------------------------------------------
            now = time.perf_counter()
            full_loop_dt = now - last_loop_end_time
            fps = 1.0 / full_loop_dt if full_loop_dt > 0 else 0.0
            last_loop_end_time = now

            # -------------------------------------------------------------
            # Visualization
            # -------------------------------------------------------------
            draw_start_time = time.perf_counter()

            frame = draw_polygon(frame, roi_a, direction_a_name, (0, 255, 0))
            frame = draw_polygon(frame, roi_b, direction_b_name, (255, 0, 0))

            frame = draw_detections(frame, detections_a, color=(0, 255, 0))
            frame = draw_detections(frame, detections_b, color=(255, 0, 0))
            frame = draw_detections(frame, detections_outside, color=(128, 128, 128))

            if DRAW_BBOX_CENTERS:
                frame = draw_bbox_centers(frame, detections_a, color=(0, 255, 0))
                frame = draw_bbox_centers(frame, detections_b, color=(255, 0, 0))
                frame = draw_bbox_centers(
                    frame,
                    detections_outside,
                    color=(128, 128, 128),
                )

            # Existing status panel still shows raw scheduler output.
            # This is useful for debugging recommendations.
            frame = draw_status_panel(
                frame=frame,
                fps=fps,
                total_detections=len(detections),
                counts_a=counts_a,
                counts_b=counts_b,
                density_a=density_a,
                density_b=density_b,
                signal_plan=raw_signal_plan,
            )

            # New Phase 8 runtime panel shows active execution state.
            # This is the important panel for comparing with MCU LEDs.
            frame = draw_signal_runtime_panel(
                frame=frame,
                runtime_snapshot=runtime_snapshot,
                uart_status=latest_uart_status,
            )

            draw_end_time = time.perf_counter()

            # -------------------------------------------------------------
            # Optional output video writer
            # -------------------------------------------------------------
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

            # -------------------------------------------------------------
            # Optional display window
            # -------------------------------------------------------------
            display_start_time = time.perf_counter()

            if SHOW_WINDOW:
                cv2.imshow("ATLC Full Pipeline", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

            display_end_time = time.perf_counter()

            # -------------------------------------------------------------
            # Profiling
            # -------------------------------------------------------------
            loop_end_time = time.perf_counter()

            read_ms = (read_end_time - read_start_time) * 1000
            detect_ms = (detect_end_time - detect_start_time) * 1000
            logic_ms = (logic_end_time - logic_start_time) * 1000
            uart_ms = (uart_end_time - uart_start_time) * 1000
            draw_ms = (draw_end_time - draw_start_time) * 1000
            writer_ms = (writer_end_time - writer_start_time) * 1000
            display_ms = (display_end_time - display_start_time) * 1000
            total_ms = (loop_end_time - loop_start_time) * 1000

            if frame_index % LOG_EVERY_N_FRAMES == 0:
                print(
                    {
                        "frame": frame_index,
                        "total_detections": len(detections),
                        "direction_A_counts": counts_a,
                        "direction_B_counts": counts_b,
                        "raw_signal_plan": raw_signal_plan,
                        "active_plan": active_plan,
                        "runtime_state": runtime_snapshot.state,
                        "remaining_seconds": runtime_snapshot.remaining_seconds,
                        "outside_roi": len(detections_outside),
                        "fps": round(fps, 2),
                    }
                )

            if frame_index % PROFILE_EVERY_N_FRAMES == 0:
                print(
                    {
                        "profile_frame": frame_index,
                        "read_ms": round(read_ms, 2),
                        "detect_ms": round(detect_ms, 2),
                        "logic_ms": round(logic_ms, 2),
                        "uart_ms": round(uart_ms, 2),
                        "draw_ms": round(draw_ms, 2),
                        "writer_ms": round(writer_ms, 2),
                        "display_ms": round(display_ms, 2),
                        "total_ms": round(total_ms, 2),
                        "full_loop_fps_est": round(
                            1000.0 / total_ms if total_ms > 0 else 0.0,
                            2,
                        ),
                    }
                )

    finally:
        cap.release()

        if writer is not None:
            writer.release()

        if uart_sender is not None:
            uart_sender.close()

        cv2.destroyAllWindows()

        print("Main pipeline finished")


if __name__ == "__main__":
    main()