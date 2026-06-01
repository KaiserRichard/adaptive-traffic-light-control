"""
test_secure_runtime.py

ATLC Phase 14 - Secure Runtime Test Cases

Purpose:
    - Validate secure PLAN protection.
    - Validate replay protection.
    - Validate timestamp-window protection.
    - Validate audit-log integrity.
    - Produce report-ready evidence files.

This intentionally runs without pytest.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.attack_simulator import (
    SUMMARY_JSON_PATH,
    SUMMARY_MD_PATH,
    run_attack_simulation,
)


def assert_true(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    summary = run_attack_simulation()

    # --------------------------------------------------
    # Valid PLAN
    # --------------------------------------------------

    assert_true(
        summary["valid_plan_accepted"] is True,
        "Valid PLAN should be accepted.",
    )

    # --------------------------------------------------
    # Tampering attack
    # --------------------------------------------------

    assert_true(
        summary["tampered_plan_rejected"] is True,
        "Tampered PLAN should be rejected.",
    )

    assert_true(
        summary["tampered_plan_reason"] == "INVALID_MAC",
        "Tampered PLAN should fail with INVALID_MAC.",
    )

    # --------------------------------------------------
    # Replay attack
    # --------------------------------------------------

    assert_true(
        summary["replay_rejected"] is True,
        "Replay attack should be rejected.",
    )

    assert_true(
        summary["replay_reason"] == "REPLAY_OR_OLD_PLAN_ID",
        "Replay attack should fail with REPLAY_OR_OLD_PLAN_ID.",
    )

    # --------------------------------------------------
    # Forged MAC attack
    # --------------------------------------------------

    assert_true(
        summary["forged_mac_rejected"] is True,
        "Forged MAC should be rejected.",
    )

    assert_true(
        summary["forged_mac_reason"] == "INVALID_MAC",
        "Forged MAC should fail with INVALID_MAC.",
    )

    # --------------------------------------------------
    # Timestamp attack
    # --------------------------------------------------

    assert_true(
        summary["timestamp_tamper_rejected"] is True,
        "Timestamp attack should be rejected.",
    )

    assert_true(
        summary["timestamp_tamper_reason"]
        == "TIMESTAMP_OUT_OF_WINDOW",
        "Timestamp attack should fail with TIMESTAMP_OUT_OF_WINDOW.",
    )

    # --------------------------------------------------
    # Audit log
    # --------------------------------------------------

    assert_true(
        summary["audit_log_valid"] is True,
        "Original audit log should be valid.",
    )

    assert_true(
        summary["tampered_audit_log_rejected"] is True,
        "Tampered audit log should be rejected.",
    )

    # --------------------------------------------------
    # Output artifacts
    # --------------------------------------------------

    assert_true(
        SUMMARY_JSON_PATH.exists(),
        "Summary JSON should exist.",
    )

    assert_true(
        SUMMARY_MD_PATH.exists(),
        "Summary markdown should exist.",
    )

    print(
        "\n[PASS] Phase 14 Secure Runtime Tests\n"
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()