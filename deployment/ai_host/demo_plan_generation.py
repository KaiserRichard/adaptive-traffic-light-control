"""
File:
    demo_plan_generation.py

Phase:
    Phase 16.9 - AI Host PLAN Generation Interface

Purpose:
    - Demonstrate hardware-independent PLAN generation from mock counts.
    - Print a serialized PLAN message suitable for future UART integration.

Responsibilities:
    - Parse mock vehicle counts from the command line.
    - Generate a validated adaptive plan.
    - Demonstrate serialize/parse validation round trip.

This file should NOT:
    - Run YOLO or ONNX inference.
    - Send serial or UART messages.
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
from plan_protocol import parse_plan, serialize_plan, validate_plan  # noqa: E402
from traffic_density import compute_density_ratio  # noqa: E402


def parse_args() -> argparse.Namespace:
    """
    Parse command-line options for the PLAN generation demo.

    Why:

        Mock input makes the planner testable on any development machine
        without requiring live detection, UART, or MCU hardware.
    """

    parser = argparse.ArgumentParser(description="Generate an ATLC AI-host PLAN message from mock vehicle counts.")
    parser.add_argument("--ns-count", type=int, required=True, help="North-south vehicle count.")
    parser.add_argument("--ew-count", type=int, required=True, help="East-west vehicle count.")
    parser.add_argument("--seq", type=int, required=True, help="PLAN sequence number.")
    parser.add_argument("--min-green", type=int, default=10, help="Minimum green time in seconds.")
    parser.add_argument("--max-green", type=int, default=45, help="Maximum green time in seconds.")
    parser.add_argument("--base-cycle-green", type=int, default=45, help="Total green-time budget in seconds.")
    parser.add_argument("--yellow", type=int, default=3, help="Yellow time in seconds.")
    parser.add_argument("--all-red", type=int, default=1, help="All-red clearance time in seconds.")
    return parser.parse_args()


def main() -> None:
    """
    Run the Phase 16.9 PLAN generation demo.

    Why:

        This prints the full AI-host planning boundary: counts, density ratios,
        green allocation, serialized message, and validation round trip.
    """

    args = parse_args()
    config = PlanTimingConfig(
        min_green=args.min_green,
        max_green=args.max_green,
        base_cycle_green=args.base_cycle_green,
        yellow=args.yellow,
        all_red=args.all_red,
    )

    ns_ratio, ew_ratio = compute_density_ratio(args.ns_count, args.ew_count)
    plan = generate_adaptive_plan(args.ns_count, args.ew_count, args.seq, config)
    message = serialize_plan(plan)
    parsed_plan = parse_plan(message)
    validate_plan(parsed_plan)

    print("=" * 72)
    print("ATLC PHASE 16.9 AI HOST PLAN GENERATION DEMO")
    print("=" * 72)
    print(f"NS count:          {args.ns_count}")
    print(f"EW count:          {args.ew_count}")
    print(f"NS density ratio:  {ns_ratio:.3f}")
    print(f"EW density ratio:  {ew_ratio:.3f}")
    print(f"NS green:          {plan.ns_green} s")
    print(f"EW green:          {plan.ew_green} s")
    print(f"Yellow:            {plan.yellow} s")
    print(f"All red:           {plan.all_red} s")
    print("Generated plan:")
    print(message)
    print(f"Round-trip parse:  {'OK' if parsed_plan == plan else 'FAILED'}")
    print("Validation:        OK")
    print("=" * 72)


if __name__ == "__main__":
    main()
