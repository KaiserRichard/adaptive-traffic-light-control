"""
File:
    demo_uart_dry_run.py

Phase:
    Phase 16.10 - AI-to-MCU UART Integration Preparation

Purpose:
    - Demonstrate dry-run UART framing for generated AI-host PLAN messages.
    - Print the exact bytes that a future serial sender would write.

Responsibilities:
    - Accept mock vehicle counts.
    - Generate a TrafficPlan using Phase 16.9 logic.
    - Serialize and frame the PLAN message.
    - Validate frame/unframe round trip.
    - Print expected future MCU responses.

This file should NOT:
    - Open a serial port.
    - Send real UART messages.
    - Require Raspberry Pi, STM32, or ESP32 hardware.
    - Modify firmware.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

AI_HOST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AI_HOST_DIR))

from plan_generator import PlanTimingConfig, generate_adaptive_plan  # noqa: E402
from plan_protocol import FUTURE_CONTROLLER_RESPONSES, parse_plan, serialize_plan  # noqa: E402
from uart_framing import frame_uart_message, unframe_uart_message, validate_uart_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    """
    Parse dry-run UART demo options.

    Why:

        Mock counts keep Phase 16.10 testable without a camera, Raspberry Pi,
        serial adapter, or MCU board.
    """

    parser = argparse.ArgumentParser(description="Dry-run ATLC AI-host UART PLAN framing.")
    parser.add_argument("--ns-count", type=int, required=True, help="North-south vehicle count.")
    parser.add_argument("--ew-count", type=int, required=True, help="East-west vehicle count.")
    parser.add_argument("--seq", type=int, required=True, help="PLAN sequence number.")
    parser.add_argument("--min-green", type=int, default=10, help="Minimum green time in seconds.")
    parser.add_argument("--max-green", type=int, default=45, help="Maximum green time in seconds.")
    parser.add_argument("--base-cycle-green", type=int, default=45, help="Total green-time budget in seconds.")
    parser.add_argument("--yellow", type=int, default=3, help="Yellow time in seconds.")
    parser.add_argument("--all-red", type=int, default=1, help="All-red clearance time in seconds.")
    parser.add_argument("--max-frame-length", type=int, default=128, help="Maximum UART frame length in bytes.")
    return parser.parse_args()


def main() -> None:
    """
    Run the Phase 16.10 dry-run UART framing demo.

    Why:

        This proves the host can generate a PLAN, serialize it, frame it as
        newline-terminated bytes, and parse it back before any real UART work is
        attempted.
    """

    args = parse_args()
    config = PlanTimingConfig(
        min_green=args.min_green,
        max_green=args.max_green,
        base_cycle_green=args.base_cycle_green,
        yellow=args.yellow,
        all_red=args.all_red,
    )

    plan = generate_adaptive_plan(args.ns_count, args.ew_count, args.seq, config)
    message = serialize_plan(plan)
    frame = frame_uart_message(message, max_length=args.max_frame_length)
    validate_uart_frame(frame, max_length=args.max_frame_length)
    unframed_message = unframe_uart_message(frame, max_length=args.max_frame_length)
    parsed_plan = parse_plan(unframed_message)

    print("=" * 72)
    print("ATLC PHASE 16.10 AI-TO-MCU UART DRY RUN")
    print("=" * 72)
    print(f"NS count:          {args.ns_count}")
    print(f"EW count:          {args.ew_count}")
    print("Generated PLAN:")
    print(message)
    print("UART frame bytes:")
    print(repr(frame))
    print(f"Frame length:      {len(frame)} bytes")
    print(f"Unframed message:  {unframed_message}")
    print(f"Round-trip parse:  {'OK' if parsed_plan == plan else 'FAILED'}")
    print("Dry-run only: no serial port opened.")
    print("Expected future MCU responses:")
    for response in FUTURE_CONTROLLER_RESPONSES:
        print(response.replace("<seq>", str(plan.seq)))
    print("=" * 72)


if __name__ == "__main__":
    main()
