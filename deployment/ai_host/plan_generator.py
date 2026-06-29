"""
File:
    plan_generator.py

Phase:
    Phase 16.9 - AI Host PLAN Generation Interface

Purpose:
    - Convert vehicle counts into a controller-ready TrafficPlan.
    - Allocate green times using a simple deterministic adaptive policy.

Responsibilities:
    - Define safe timing configuration bounds.
    - Use traffic density ratios to assign green time.
    - Return a validated TrafficPlan object.

This file should NOT:
    - Run image inference.
    - Send UART messages.
    - Implement final traffic optimization.
    - Modify firmware.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

try:
    from .plan_protocol import TrafficPlan, validate_plan
    from .traffic_density import compute_density_ratio, validate_vehicle_counts
except ImportError:  # pragma: no cover - supports direct script execution.
    from plan_protocol import TrafficPlan, validate_plan
    from traffic_density import compute_density_ratio, validate_vehicle_counts


@dataclass(frozen=True)
class PlanTimingConfig:
    """
    Timing constraints for AI-host PLAN generation.

    Why:

        The AI host can suggest adaptive timings, but it must respect basic
        guardrails before a later MCU phase validates and executes the plan.
    """

    min_green: int = 10
    max_green: int = 45
    base_cycle_green: int = 45
    yellow: int = 3
    all_red: int = 1


def validate_timing_config(config: PlanTimingConfig) -> None:
    """
    Validate timing bounds before generating a plan.

    Why:

        Impossible bounds, such as a cycle shorter than two minimum greens,
        should fail clearly instead of producing a malformed PLAN message.
    """

    for field_name in ("min_green", "max_green", "base_cycle_green", "yellow", "all_red"):
        value = getattr(config, field_name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} must be an integer.")

    if config.min_green <= 0:
        raise ValueError("min_green must be positive.")
    if config.max_green < config.min_green:
        raise ValueError("max_green must be >= min_green.")
    if config.base_cycle_green < 2 * config.min_green:
        raise ValueError("base_cycle_green must allow both directions to receive min_green.")
    if config.base_cycle_green > 2 * config.max_green:
        raise ValueError("base_cycle_green must not exceed two max_green allocations.")
    if config.yellow <= 0:
        raise ValueError("yellow must be positive.")
    if config.all_red < 0:
        raise ValueError("all_red must be >= 0.")


def round_half_up(value: float) -> int:
    """
    Round halves upward for deterministic timing allocation.

    Why:

        Python's built-in round uses bankers rounding. Traffic timing is easier
        to reason about if .5 always rounds upward.
    """

    return int(floor(value + 0.5))


def split_green_budget(ns_count: int, ew_count: int, config: PlanTimingConfig) -> tuple[int, int]:
    """
    Allocate the green-time budget between north-south and east-west directions.

    Why:

        This policy is intentionally simple: the busier direction receives more
        green time, but the result remains bounded and deterministic. It is a
        control-intent prototype, not final traffic optimization.
    """

    ns_count, ew_count = validate_vehicle_counts(ns_count, ew_count)
    validate_timing_config(config)

    if ns_count == ew_count:
        ns_green = config.base_cycle_green // 2
        ew_green = config.base_cycle_green - ns_green
        return ns_green, ew_green

    ns_ratio, ew_ratio = compute_density_ratio(ns_count, ew_count)
    dominant_ratio = max(ns_ratio, ew_ratio)
    half_budget = config.base_cycle_green / 2
    swing_budget = min(config.max_green - half_budget, half_budget - config.min_green)
    extra_green = round_half_up((dominant_ratio - 0.5) * 2 * swing_budget)
    dominant_green = round_half_up(half_budget + extra_green)
    dominant_green = max(config.min_green, min(config.max_green, dominant_green))
    other_green = config.base_cycle_green - dominant_green
    other_green = max(config.min_green, min(config.max_green, other_green))

    if ns_count > ew_count:
        return dominant_green, other_green
    return other_green, dominant_green


def generate_adaptive_plan(
    ns_count: int,
    ew_count: int,
    seq: int,
    config: PlanTimingConfig | None = None,
) -> TrafficPlan:
    """
    Generate an adaptive TrafficPlan from vehicle counts.

    Why:

        This is the Phase 16.9 bridge between perception-derived counts and the
        future UART integration boundary. The returned object can be serialized
        but is not sent to hardware in this phase.
    """

    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        raise ValueError("seq must be an integer >= 0.")

    timing = config or PlanTimingConfig()
    ns_green, ew_green = split_green_budget(ns_count, ew_count, timing)
    plan = TrafficPlan(
        seq=seq,
        mode="adaptive",
        ns_green=ns_green,
        ew_green=ew_green,
        yellow=timing.yellow,
        all_red=timing.all_red,
    )
    validate_plan(plan)
    return plan
