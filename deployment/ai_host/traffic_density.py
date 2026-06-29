"""
File:
    traffic_density.py

Phase:
    Phase 16.9 - AI Host PLAN Generation Interface

Purpose:
    - Validate simple per-direction vehicle counts.
    - Convert vehicle counts into normalized traffic density ratios.

Responsibilities:
    - Reject invalid count inputs.
    - Handle the zero-vehicle case safely.
    - Provide deterministic density ratios for the plan generator.

This file should NOT:
    - Run YOLO or ONNX inference.
    - Perform ROI assignment.
    - Send UART messages.
    - Modify firmware.
"""

from __future__ import annotations


def validate_vehicle_counts(ns_count: int, ew_count: int) -> tuple[int, int]:
    """
    Validate north-south and east-west vehicle counts.

    Why:

        PLAN generation is a controller-facing step. Invalid negative counts
        should fail early instead of producing unsafe or confusing signal
        timing values.
    """

    for name, value in (("north_south_count", ns_count), ("east_west_count", ew_count)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer.")
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")

    return ns_count, ew_count


def compute_density_ratio(ns_count: int, ew_count: int) -> tuple[float, float]:
    """
    Convert vehicle counts into normalized density ratios.

    Why:

        The timing policy needs a scale-independent signal. A 12:5 count and a
        24:10 count should produce the same allocation pressure even though the
        absolute traffic volume is different.

        If both directions have zero vehicles, the safe neutral result is an
        equal split of 0.5 / 0.5.
    """

    ns_count, ew_count = validate_vehicle_counts(ns_count, ew_count)
    total = ns_count + ew_count
    if total == 0:
        return 0.5, 0.5

    return ns_count / total, ew_count / total


def normalize_count(count: int, reference_count: int) -> float:
    """
    Normalize one count against a reference count.

    Why:

        This helper is useful for future dashboard or logging output where a
        count should be displayed as a 0..1 load estimate. It is not required
        by the current timing policy but keeps normalization behavior explicit.
    """

    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError("count must be an integer.")
    if isinstance(reference_count, bool) or not isinstance(reference_count, int):
        raise ValueError("reference_count must be an integer.")
    if count < 0:
        raise ValueError("count must be non-negative.")
    if reference_count <= 0:
        raise ValueError("reference_count must be positive.")

    return min(1.0, count / reference_count)
