"""
attack_simulator.py

ATLC Phase 14 - Runtime Security Attack Simulator

Purpose:
    - Simulate attacker behavior against secure ATLC PLAN messages.
    - Validate that HMAC-protected PLAN messages reject tampering.
    - Validate replay protection using plan_id and nonce.
    - Validate timestamp-window protection.
    - Generate tamper-evident audit-log evidence.
    - Generate report-ready JSON and Markdown summaries.

This file belongs in experiments/ because it is an experiment/test artifact,
not production runtime code.

Expected dependency:
    pc_app/security/secure_runtime.py

Expected functions/classes from secure_runtime.py:
    - ReplayState
    - create_secure_plan
    - verify_secure_plan
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from pc_app.security.secure_runtime import (
    ReplayState,
    create_secure_plan,
    verify_secure_plan,
)


OUTPUT_DIR = Path("outputs/security/phase14_runtime_security")

AUDIT_LOG_PATH = OUTPUT_DIR / "secure_runtime_audit_log.jsonl"
TAMPERED_AUDIT_LOG_PATH = OUTPUT_DIR / "tampered_secure_runtime_audit_log.jsonl"

SUMMARY_JSON_PATH = OUTPUT_DIR / "attack_simulation_summary.json"
SUMMARY_MD_PATH = OUTPUT_DIR / "attack_simulation_summary.md"

SECRET_KEY = "phase14_demo_shared_hmac_key"

FIXED_NOW = 1779958555


class HashChainAuditLog:
    """
    Minimal hash-chain JSONL audit logger.

    Each line stores:
        - timestamp
        - event_type
        - payload
        - previous_hash
        - current_hash

    If an old line is modified, verification fails.
    """

    def __init__(
        self,
        log_path: Path,
    ) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _hash_entry(
        self,
        entry: dict[str, Any],
    ) -> str:
        encoded = json.dumps(
            entry,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        return hashlib.sha256(encoded).hexdigest()

    def _get_last_hash(self) -> str:
        if not self.log_path.exists():
            return "0" * 64

        last_line = None

        with self.log_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                if line.strip():
                    last_line = line.strip()

        if last_line is None:
            return "0" * 64

        last_entry = json.loads(last_line)

        return str(last_entry["current_hash"])

    def append(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        previous_hash = self._get_last_hash()

        entry_without_current_hash = {
            "timestamp": int(time.time()),
            "event_type": event_type,
            "payload": payload,
            "previous_hash": previous_hash,
        }

        current_hash = self._hash_entry(
            entry_without_current_hash
        )

        entry = {
            **entry_without_current_hash,
            "current_hash": current_hash,
        }

        with self.log_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    entry,
                    sort_keys=True,
                )
                + "\n"
            )

        return entry

    def verify(self) -> dict[str, Any]:
        if not self.log_path.exists():
            return {
                "valid": False,
                "entries": 0,
                "reason": "LOG_NOT_FOUND",
            }

        previous_hash = "0" * 64
        entries = 0

        with self.log_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                entry = json.loads(line)

                if entry.get("previous_hash") != previous_hash:
                    return {
                        "valid": False,
                        "entries": entries,
                        "reason": (
                            "BROKEN_PREVIOUS_HASH_AT_LINE:"
                            f"{line_number}"
                        ),
                    }

                stored_hash = entry.get("current_hash")

                entry_without_current_hash = {
                    "timestamp": entry["timestamp"],
                    "event_type": entry["event_type"],
                    "payload": entry["payload"],
                    "previous_hash": entry["previous_hash"],
                }

                computed_hash = self._hash_entry(
                    entry_without_current_hash
                )

                if stored_hash != computed_hash:
                    return {
                        "valid": False,
                        "entries": entries,
                        "reason": (
                            "BROKEN_CURRENT_HASH_AT_LINE:"
                            f"{line_number}"
                        ),
                    }

                previous_hash = str(stored_hash)
                entries += 1

        return {
            "valid": True,
            "entries": entries,
            "reason": "OK",
        }


def make_demo_raw_plan() -> dict[str, int]:
    """
    Create a normal ATLC signal plan before security wrapping.
    """

    return {
        "green_a": 40,
        "green_b": 10,
        "yellow": 3,
        "all_red": 1,
    }


def tamper_with_green_time(
    secure_plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Simulate an attacker changing green_a after MAC generation.
    """

    tampered_plan = dict(secure_plan)
    tampered_plan["green_a"] = 80

    return tampered_plan


def forge_mac(
    secure_plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Simulate an attacker replacing the MAC with a fake value.
    """

    forged_plan = dict(secure_plan)
    forged_plan["mac"] = "0" * 64

    return forged_plan


def tamper_with_timestamp(
    secure_plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Simulate an attacker modifying the timestamp.
    """

    timestamp_plan = dict(secure_plan)
    timestamp_plan["timestamp"] = FIXED_NOW + 1000

    return timestamp_plan


def create_tampered_audit_log_copy() -> dict[str, Any]:
    """
    Copy the valid audit log, modify one old event, and verify that
    the hash-chain check catches the modification.
    """

    shutil.copyfile(
        AUDIT_LOG_PATH,
        TAMPERED_AUDIT_LOG_PATH,
    )

    lines = TAMPERED_AUDIT_LOG_PATH.read_text(
        encoding="utf-8",
    ).splitlines()

    if not lines:
        return {
            "tampered_log_path": str(TAMPERED_AUDIT_LOG_PATH),
            "verification": {
                "valid": False,
                "entries": 0,
                "reason": "NO_LINES_TO_TAMPER",
            },
        }

    first_entry = json.loads(lines[0])
    first_entry["payload"]["accepted"] = False

    lines[0] = json.dumps(
        first_entry,
        sort_keys=True,
    )

    TAMPERED_AUDIT_LOG_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    tampered_log = HashChainAuditLog(
        TAMPERED_AUDIT_LOG_PATH
    )

    return {
        "tampered_log_path": str(TAMPERED_AUDIT_LOG_PATH),
        "verification": tampered_log.verify(),
    }


def build_markdown_summary(
    summary: dict[str, Any],
) -> str:
    """
    Build a report-ready Markdown summary.
    """

    return f"""# ATLC Phase 14 Runtime Security Attack Simulation

## Purpose

This experiment validates the Phase 14 runtime security layer for ATLC.

It simulates attacks against authenticated traffic-light PLAN messages.

## Test Results

| Test | Expected | Actual |
|---|---:|---:|
| Valid PLAN accepted | True | {summary["valid_plan_accepted"]} |
| Tampered PLAN rejected | True | {summary["tampered_plan_rejected"]} |
| Replay attack rejected | True | {summary["replay_rejected"]} |
| Forged MAC rejected | True | {summary["forged_mac_rejected"]} |
| Timestamp tamper rejected | True | {summary["timestamp_tamper_rejected"]} |
| Audit log valid | True | {summary["audit_log_valid"]} |
| Tampered audit log rejected | True | {summary["tampered_audit_log_rejected"]} |

## Rejection Reasons

| Attack | Reason |
|---|---|
| Tampered PLAN | {summary["tampered_plan_reason"]} |
| Replay attack | {summary["replay_reason"]} |
| Forged MAC | {summary["forged_mac_reason"]} |
| Timestamp tamper | {summary["timestamp_tamper_reason"]} |
| Tampered audit log | {summary["tampered_audit_log_reason"]} |

## Evidence Files

- outputs/security/phase14_runtime_security/secure_runtime_audit_log.jsonl
- outputs/security/phase14_runtime_security/tampered_secure_runtime_audit_log.jsonl
- outputs/security/phase14_runtime_security/attack_simulation_summary.json
- outputs/security/phase14_runtime_security/attack_simulation_summary.md

## Engineering Interpretation

The experiment shows that the ATLC control plan can be protected against:

- unauthorized green-time modification
- replayed old PLAN messages
- forged MAC values
- stale timestamp attacks
- post-run audit-log tampering

This phase is still an experiment layer. It is not yet integrated into
pc_app.main or the UART sender.
"""


def run_attack_simulation() -> dict[str, Any]:
    """
    Run all Phase 14 attack simulations and write evidence files.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if AUDIT_LOG_PATH.exists():
        AUDIT_LOG_PATH.unlink()

    if TAMPERED_AUDIT_LOG_PATH.exists():
        TAMPERED_AUDIT_LOG_PATH.unlink()

    audit_log = HashChainAuditLog(
        AUDIT_LOG_PATH
    )

    raw_plan = make_demo_raw_plan()

    secure_plan = create_secure_plan(
        raw_plan=raw_plan,
        secret_key=SECRET_KEY,
        plan_id=1,
        timestamp=FIXED_NOW,
        nonce="phase14nonce0001",
    )

    replay_state = ReplayState()

    valid_result = verify_secure_plan(
        plan=secure_plan,
        secret_key=SECRET_KEY,
        replay_state=replay_state,
        now=FIXED_NOW,
    )

    audit_log.append(
        event_type="VALID_PLAN_TEST",
        payload={
            "accepted": valid_result.accepted,
            "reason": valid_result.reason,
        },
    )

    tampered_plan = tamper_with_green_time(
        secure_plan
    )

    tamper_result = verify_secure_plan(
        plan=tampered_plan,
        secret_key=SECRET_KEY,
        replay_state=ReplayState(),
        now=FIXED_NOW,
    )

    audit_log.append(
        event_type="TAMPERED_PLAN_TEST",
        payload={
            "accepted": tamper_result.accepted,
            "reason": tamper_result.reason,
        },
    )

    replay_result = verify_secure_plan(
        plan=secure_plan,
        secret_key=SECRET_KEY,
        replay_state=replay_state,
        now=FIXED_NOW,
    )

    audit_log.append(
        event_type="REPLAY_ATTACK_TEST",
        payload={
            "accepted": replay_result.accepted,
            "reason": replay_result.reason,
        },
    )

    forged_plan = forge_mac(
        secure_plan
    )

    forged_result = verify_secure_plan(
        plan=forged_plan,
        secret_key=SECRET_KEY,
        replay_state=ReplayState(),
        now=FIXED_NOW,
    )

    audit_log.append(
        event_type="FORGED_MAC_TEST",
        payload={
            "accepted": forged_result.accepted,
            "reason": forged_result.reason,
        },
    )

    timestamp_plan = tamper_with_timestamp(
        secure_plan
    )

    timestamp_result = verify_secure_plan(
        plan=timestamp_plan,
        secret_key=SECRET_KEY,
        replay_state=ReplayState(),
        now=FIXED_NOW,
    )

    audit_log.append(
        event_type="TIMESTAMP_TAMPER_TEST",
        payload={
            "accepted": timestamp_result.accepted,
            "reason": timestamp_result.reason,
        },
    )

    audit_verification = audit_log.verify()
    tampered_audit_result = create_tampered_audit_log_copy()

    summary = {
        "valid_plan_accepted": valid_result.accepted,
        "valid_plan_reason": valid_result.reason,
        "tampered_plan_rejected": not tamper_result.accepted,
        "tampered_plan_reason": tamper_result.reason,
        "replay_rejected": not replay_result.accepted,
        "replay_reason": replay_result.reason,
        "forged_mac_rejected": not forged_result.accepted,
        "forged_mac_reason": forged_result.reason,
        "timestamp_tamper_rejected": not timestamp_result.accepted,
        "timestamp_tamper_reason": timestamp_result.reason,
        "audit_log_valid": audit_verification["valid"],
        "audit_log_reason": audit_verification["reason"],
        "tampered_audit_log_rejected": (
            not tampered_audit_result["verification"]["valid"]
        ),
        "tampered_audit_log_reason": (
            tampered_audit_result["verification"]["reason"]
        ),
    }

    SUMMARY_JSON_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    SUMMARY_MD_PATH.write_text(
        build_markdown_summary(summary),
        encoding="utf-8",
    )

    return summary


def main() -> None:
    summary = run_attack_simulation()

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()