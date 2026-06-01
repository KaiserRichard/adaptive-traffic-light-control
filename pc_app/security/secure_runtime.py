"""
secure_runtime.py

ATLC Phase 14 - Runtime Security Experiment Layer

Purpose:
    - Create authenticated traffic-light PLAN messages.
    - Verify PLAN message integrity using HMAC-SHA256.
    - Reject replayed PLAN messages using plan_id and nonce tracking.
    - Provide a small security adapter that can later be connected to UART.

This file is intentionally self-contained so it can be tested without MCU
hardware and without modifying pc_app.main.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any


REQUIRED_PLAN_FIELDS = [
    "message_type",
    "plan_id",
    "green_a",
    "green_b",
    "yellow",
    "all_red",
    "timestamp",
    "nonce",
]


@dataclass
class ReplayState:
    """
    Stores replay-protection state.

    highest_plan_id:
        The highest accepted plan_id so far.

    used_nonces:
        A set of already-used nonces.
    """

    highest_plan_id: int = 0
    used_nonces: set[str] = field(default_factory=set)


@dataclass
class VerificationResult:
    """
    Result returned by secure PLAN verification.
    """

    accepted: bool
    reason: str


def canonical_plan_payload(plan: dict[str, Any]) -> str:
    """
    Build a deterministic string representation of the PLAN.

    The MAC itself is excluded because the sender and verifier must compute
    the HMAC over the same original message fields.
    """

    payload = {
        key: plan[key]
        for key in REQUIRED_PLAN_FIELDS
        if key in plan
    }

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )


def compute_plan_mac(
    plan: dict[str, Any],
    secret_key: str,
) -> str:
    """
    Compute HMAC-SHA256 for a PLAN message.

    If any protected field changes, the MAC will change.
    """

    canonical_payload = canonical_plan_payload(plan)

    return hmac.new(
        key=secret_key.encode("utf-8"),
        msg=canonical_payload.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()


def create_secure_plan(
    raw_plan: dict[str, Any],
    secret_key: str,
    plan_id: int,
    timestamp: int | None = None,
    nonce: str | None = None,
) -> dict[str, Any]:
    """
    Convert a normal ATLC signal plan into a secure PLAN message.

    Input raw_plan example:
        {
            "green_a": 40,
            "green_b": 10,
            "yellow": 3,
            "all_red": 1
        }

    Output secure PLAN includes:
        - message_type
        - plan_id
        - timing values
        - timestamp
        - nonce
        - mac
    """

    secure_plan = {
        "message_type": "PLAN",
        "plan_id": int(plan_id),
        "green_a": int(raw_plan["green_a"]),
        "green_b": int(raw_plan["green_b"]),
        "yellow": int(raw_plan["yellow"]),
        "all_red": int(raw_plan["all_red"]),
        "timestamp": int(timestamp if timestamp is not None else time.time()),
        "nonce": nonce if nonce is not None else secrets.token_hex(8),
    }

    secure_plan["mac"] = compute_plan_mac(
        plan=secure_plan,
        secret_key=secret_key,
    )

    return secure_plan


def verify_secure_plan(
    plan: dict[str, Any],
    secret_key: str,
    replay_state: ReplayState,
    now: int | None = None,
    max_clock_skew_seconds: int = 300,
) -> VerificationResult:
    """
    Verify a secure PLAN message.

    Verification order:
        1. Check required fields.
        2. Check MAC field exists.
        3. Check timestamp window.
        4. Check HMAC integrity.
        5. Check plan_id replay protection.
        6. Check nonce replay protection.

    Return examples:
        VerificationResult(True, "OK")
        VerificationResult(False, "INVALID_MAC")
        VerificationResult(False, "REPLAY_OR_OLD_PLAN_ID")
    """

    for field_name in REQUIRED_PLAN_FIELDS:
        if field_name not in plan:
            return VerificationResult(
                accepted=False,
                reason=f"MISSING_FIELD:{field_name}",
            )

    if "mac" not in plan:
        return VerificationResult(
            accepted=False,
            reason="MISSING_MAC",
        )

    current_time = int(now if now is not None else time.time())
    plan_timestamp = int(plan["timestamp"])

    if abs(current_time - plan_timestamp) > max_clock_skew_seconds:
        return VerificationResult(
            accepted=False,
            reason="TIMESTAMP_OUT_OF_WINDOW",
        )

    expected_mac = compute_plan_mac(
        plan=plan,
        secret_key=secret_key,
    )

    if not hmac.compare_digest(
        str(plan["mac"]),
        expected_mac,
    ):
        return VerificationResult(
            accepted=False,
            reason="INVALID_MAC",
        )

    plan_id = int(plan["plan_id"])

    if plan_id <= replay_state.highest_plan_id:
        return VerificationResult(
            accepted=False,
            reason="REPLAY_OR_OLD_PLAN_ID",
        )

    nonce = str(plan["nonce"])

    if nonce in replay_state.used_nonces:
        return VerificationResult(
            accepted=False,
            reason="REPLAYED_NONCE",
        )

    replay_state.highest_plan_id = plan_id
    replay_state.used_nonces.add(nonce)

    return VerificationResult(
        accepted=True,
        reason="OK",
    )


def secure_plan_to_uart_line(plan: dict[str, Any]) -> str:
    """
    Serialize secure PLAN as one JSON line.

    Later, this can be used for UART transport.
    """

    return json.dumps(
        plan,
        sort_keys=True,
        separators=(",", ":"),
    )


def secure_plan_from_uart_line(line: str) -> dict[str, Any]:
    """
    Parse one secure PLAN JSON line.
    """

    return json.loads(line)