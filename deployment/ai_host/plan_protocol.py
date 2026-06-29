"""
File:
    plan_protocol.py

Phase:
    Phase 16.9 - AI Host PLAN Generation Interface

Purpose:
    - Define a structured traffic PLAN object.
    - Serialize and parse a UART-ready PLAN message string.
    - Validate message fields before a later UART integration phase.

Responsibilities:
    - Keep the host-side PLAN message format explicit.
    - Provide parse/validate round-trip checks for hardware-free testing.
    - Document future ACK / NACK / STATUS / DIAG response shapes.

This file should NOT:
    - Open serial ports.
    - Send UART messages.
    - Parse MCU logs.
    - Modify firmware.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_PLAN_MODES = {"adaptive", "fixed", "fallback"}

FUTURE_CONTROLLER_RESPONSES = (
    "ACK,seq=<seq>",
    "NACK,seq=<seq>,reason=<reason>",
    "STATUS,seq=<seq>,state=<state>,remaining=<seconds>",
    "DIAG,seq=<seq>,...",
)


@dataclass(frozen=True)
class TrafficPlan:
    """
    Structured AI-host traffic plan.

    Why:

        A dataclass makes the contract between density logic and future UART
        integration explicit. The message type is serialized as the first token
        while the remaining fields use readable key-value pairs.
    """

    seq: int
    mode: str
    ns_green: int
    ew_green: int
    yellow: int
    all_red: int
    message_type: str = "PLAN"


def validate_plan(plan: TrafficPlan) -> None:
    """
    Validate a TrafficPlan before serialization or controller handoff.

    Why:

        Phase 16.9 does not send UART, but it should still prevent invalid
        plans from crossing the AI-host interface boundary.
    """

    if plan.message_type != "PLAN":
        raise ValueError("message_type must be PLAN.")
    if isinstance(plan.seq, bool) or not isinstance(plan.seq, int) or plan.seq < 0:
        raise ValueError("seq must be an integer >= 0.")
    if plan.mode not in VALID_PLAN_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_PLAN_MODES)}.")

    for field_name in ("ns_green", "ew_green", "yellow"):
        value = getattr(plan, field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer.")

    if isinstance(plan.all_red, bool) or not isinstance(plan.all_red, int) or plan.all_red < 0:
        raise ValueError("all_red must be an integer >= 0.")


def serialize_plan(plan: TrafficPlan) -> str:
    """
    Serialize a TrafficPlan into a readable PLAN message string.

    Why:

        The key-value format is easy to inspect in logs and easy to adapt later
        when Phase 16.10 prepares real UART sending.
    """

    validate_plan(plan)
    return (
        f"{plan.message_type},"
        f"seq={plan.seq},"
        f"mode={plan.mode},"
        f"ns_green={plan.ns_green},"
        f"ew_green={plan.ew_green},"
        f"yellow={plan.yellow},"
        f"all_red={plan.all_red}"
    )


def parse_plan(message: str) -> TrafficPlan:
    """
    Parse a serialized PLAN message back into a TrafficPlan.

    Why:

        A parse/serialize round trip lets this phase be tested without MCU
        hardware and catches accidental format drift before UART work begins.
    """

    if not message or not isinstance(message, str):
        raise ValueError("message must be a non-empty string.")

    parts = [part.strip() for part in message.strip().split(",") if part.strip()]
    if not parts or parts[0] != "PLAN":
        raise ValueError("message must start with PLAN.")

    fields: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise ValueError(f"invalid key-value field: {part}")
        key, value = part.split("=", 1)
        fields[key] = value

    required = {"seq", "mode", "ns_green", "ew_green", "yellow", "all_red"}
    missing = required - fields.keys()
    unexpected = fields.keys() - required
    if missing:
        raise ValueError(f"missing PLAN field(s): {sorted(missing)}")
    if unexpected:
        raise ValueError(f"unexpected PLAN field(s): {sorted(unexpected)}")

    try:
        plan = TrafficPlan(
            seq=int(fields["seq"]),
            mode=fields["mode"],
            ns_green=int(fields["ns_green"]),
            ew_green=int(fields["ew_green"]),
            yellow=int(fields["yellow"]),
            all_red=int(fields["all_red"]),
        )
    except ValueError as exc:
        raise ValueError(f"invalid numeric PLAN field: {exc}") from exc

    validate_plan(plan)
    return plan
