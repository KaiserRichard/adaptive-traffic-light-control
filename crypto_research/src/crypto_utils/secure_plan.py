"""
secure_plan.py

HMAC-authenticated PLAN message utilities for the ATLC crypto demo.

Original ATLC UART protocol:

    PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>

This original version is simple, but it has no cryptographic protection.
An attacker could modify or replay a command if they had access to the channel.

Secure protocol idea:

    PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>,<timestamp>,<nonce>,<mac>

Security goals:

    1. Command integrity:
        Detect if green time, yellow time, or all-red time is modified.

    2. Sender authentication:
        Verify that the PLAN was created by a trusted host that knows
        the shared secret key.

    3. Replay protection:
        Reject old PLAN messages using plan_id and nonce tracking.

Cryptographic primitive:

    HMAC-SHA256

Why HMAC:

    - Lightweight.
    - Easy to implement in Python and embedded systems.
    - Suitable for message authentication when both sides share a secret key.
"""

from __future__ import annotations

import hmac
import hashlib
import secrets
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Set, Tuple


def canonical_plan_string(plan: Dict[str, Any]) -> str:
    """
    Convert a PLAN dictionary into a deterministic string.

    Why this is needed:
        HMAC must be computed over exactly the same message on both sender
        and receiver sides.

    If field order changes, the HMAC value changes.

    Example output:
        PLAN|1|40|10|3|1|1710000000|abc123
    """

    fields = [
        "message_type",
        "plan_id",
        "green_a",
        "green_b",
        "yellow",
        "all_red",
        "timestamp",
        "nonce",
    ]

    return "|".join(str(plan[field]) for field in fields)


def compute_hmac_sha256(secret_key: bytes, message: str) -> str:
    """
    Compute HMAC-SHA256 for a message.

    Parameters:
        secret_key:
            Shared secret known by trusted sender and receiver.

        message:
            Canonical string representation of the PLAN command.

    Returns:
        Hexadecimal HMAC digest.

    Security meaning:
        Anyone without the secret key cannot generate a valid MAC
        for a modified PLAN message.
    """

    return hmac.new(
        secret_key,
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def create_secure_plan(
    secret_key: bytes,
    plan_id: int,
    green_a: int,
    green_b: int,
    yellow: int,
    all_red: int,
    timestamp: int | None = None,
    nonce: str | None = None,
) -> Dict[str, Any]:
    """
    Create a secure authenticated PLAN message.

    Workflow:
        1. Build PLAN fields.
        2. Add timestamp.
        3. Add random nonce.
        4. Compute HMAC over the canonical PLAN string.
        5. Attach MAC to the message.

    Parameters:
        secret_key:
            Shared HMAC key.

        plan_id:
            Increasing command sequence number.

        green_a / green_b / yellow / all_red:
            Traffic-light timing values.

        timestamp:
            Unix timestamp. If None, current time is used.

        nonce:
            Random one-time value. If None, generated automatically.

    Returns:
        Dictionary containing secure PLAN fields and MAC.
    """

    if timestamp is None:
        timestamp = int(time.time())

    if nonce is None:
        # 8 bytes = 16 hex characters.
        # Enough for this research demo.
        nonce = secrets.token_hex(8)

    plan = {
        "message_type": "PLAN",
        "plan_id": int(plan_id),
        "green_a": int(green_a),
        "green_b": int(green_b),
        "yellow": int(yellow),
        "all_red": int(all_red),
        "timestamp": int(timestamp),
        "nonce": nonce,
    }

    # Compute MAC after all authenticated fields have been set.
    plan["mac"] = compute_hmac_sha256(
        secret_key=secret_key,
        message=canonical_plan_string(plan),
    )

    return plan


@dataclass
class ReplayProtector:
    """
    Replay protection state.

    This class stores which PLAN messages have already been accepted.

    Demo policy:
        - plan_id must strictly increase.
        - nonce must not repeat.

    Why:
        If an attacker records a valid PLAN message and sends it again later,
        the HMAC may still be valid. Replay protection rejects that old message.
    """

    last_plan_id: int = 0
    seen_nonces: Set[str] = field(default_factory=set)

    def check_and_update(self, plan_id: int, nonce: str) -> Tuple[bool, str]:
        """
        Check whether a PLAN message is fresh.

        Returns:
            (True, "OK") if the message is fresh.

            (False, reason) if the message is replayed or old.
        """

        if plan_id <= self.last_plan_id:
            return False, "REPLAY_OR_OLD_PLAN_ID"

        if nonce in self.seen_nonces:
            return False, "REPLAYED_NONCE"

        # Update replay state only after the message passes checks.
        self.last_plan_id = plan_id
        self.seen_nonces.add(nonce)

        return True, "OK"


def verify_secure_plan(
    secret_key: bytes,
    plan: Dict[str, Any],
    replay_protector: ReplayProtector,
    max_time_skew_sec: int = 300,
    now: int | None = None,
) -> Tuple[bool, str]:
    """
    Verify a secure PLAN message.

    Verification steps:
        1. Check required fields.
        2. Check message type.
        3. Check timestamp freshness.
        4. Recompute HMAC.
        5. Compare received MAC with expected MAC.
        6. Check replay protection using plan_id and nonce.

    Parameters:
        secret_key:
            Shared HMAC key.

        plan:
            PLAN dictionary received by the verifier.

        replay_protector:
            State object that remembers accepted plan IDs and nonces.

        max_time_skew_sec:
            Allowed timestamp difference.
            Default 300 seconds = 5 minutes.

        now:
            Optional current time for testing.

    Returns:
        (True, "OK") if accepted.
        (False, reason) if rejected.

    Important:
        MAC comparison uses hmac.compare_digest() to avoid timing attacks.
    """

    if now is None:
        now = int(time.time())

    required_fields = [
        "message_type",
        "plan_id",
        "green_a",
        "green_b",
        "yellow",
        "all_red",
        "timestamp",
        "nonce",
        "mac",
    ]

    for field_name in required_fields:
        if field_name not in plan:
            return False, f"MISSING_FIELD_{field_name}"

    if plan["message_type"] != "PLAN":
        return False, "INVALID_MESSAGE_TYPE"

    timestamp = int(plan["timestamp"])

    # Reject messages that are too old or too far in the future.
    if abs(now - timestamp) > max_time_skew_sec:
        return False, "STALE_OR_FUTURE_TIMESTAMP"

    expected_mac = compute_hmac_sha256(
        secret_key=secret_key,
        message=canonical_plan_string(plan),
    )

    # Secure comparison prevents subtle timing-leak style attacks.
    if not hmac.compare_digest(expected_mac, str(plan["mac"])):
        return False, "INVALID_MAC"

    replay_ok, replay_reason = replay_protector.check_and_update(
        plan_id=int(plan["plan_id"]),
        nonce=str(plan["nonce"]),
    )

    if not replay_ok:
        return False, replay_reason

    return True, "OK"