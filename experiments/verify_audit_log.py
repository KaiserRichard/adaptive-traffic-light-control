"""
verify_audit_log.py

ATLC Phase 14 - Audit Log Verification Utility

Purpose:
    - Verify a secure runtime audit log file.
    - Detect manual tampering in copied audit logs.
    - Provide a simple terminal demo for the hacker-modified-log scenario.

Example:
    python -m experiments.verify_audit_log \
      --log outputs/security/phase14_runtime_security/secure_runtime_audit_log.jsonl

    python -m experiments.verify_audit_log \
      --log outputs/security/phase14_runtime_security/hacker_modified_log.jsonl

Expected:
    - Original audit log should be valid.
    - Hacker-modified audit log should be invalid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ZERO_HASH = "0" * 64

HASH_FIELD_CANDIDATES = [
    "entry_hash",
    "current_hash",
    "record_hash",
    "hash",
]

PREVIOUS_HASH_FIELD_CANDIDATES = [
    "previous_hash",
    "prev_hash",
    "previous_entry_hash",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify ATLC Phase 14 hash-chain audit logs."
    )

    parser.add_argument(
        "--log",
        required=True,
        help="Path to the audit log JSONL file.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every verification step.",
    )

    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Audit log not found: {path}")

    entries: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            if not isinstance(entry, dict):
                raise ValueError(
                    f"Line {line_number} is not a JSON object."
                )

            entries.append(entry)

    return entries


def get_first_existing_key(
    entry: dict[str, Any],
    candidates: list[str],
) -> str | None:
    for key in candidates:
        if key in entry:
            return key

    return None


def canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def compute_entry_hash(
    entry_without_hash: dict[str, Any],
) -> str:
    canonical = canonical_json(entry_without_hash)

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def verify_entries(
    entries: list[dict[str, Any]],
    verbose: bool = False,
) -> tuple[bool, str]:
    if not entries:
        return False, "EMPTY_LOG"

    previous_computed_hash = ZERO_HASH

    for index, entry in enumerate(entries):
        line_number = index + 1

        hash_key = get_first_existing_key(
            entry,
            HASH_FIELD_CANDIDATES,
        )

        if hash_key is None:
            return False, f"MISSING_HASH_FIELD_AT_LINE_{line_number}"

        previous_hash_key = get_first_existing_key(
            entry,
            PREVIOUS_HASH_FIELD_CANDIDATES,
        )

        stored_hash = str(entry.get(hash_key, ""))

        if not stored_hash:
            return False, f"EMPTY_HASH_AT_LINE_{line_number}"

        entry_without_hash = dict(entry)
        entry_without_hash.pop(hash_key, None)

        computed_hash = compute_entry_hash(entry_without_hash)

        if previous_hash_key is not None:
            stored_previous_hash = str(
                entry.get(previous_hash_key, "")
            )

            if index == 0:
                expected_previous_hash = ZERO_HASH
            else:
                expected_previous_hash = previous_computed_hash

            if stored_previous_hash != expected_previous_hash:
                return (
                    False,
                    f"PREVIOUS_HASH_MISMATCH_AT_LINE_{line_number}",
                )

        if stored_hash != computed_hash:
            return False, f"HASH_MISMATCH_AT_LINE_{line_number}"

        if verbose:
            print(
                f"[OK] line={line_number} "
                f"hash={stored_hash[:12]}..."
            )

        previous_computed_hash = computed_hash

    return True, "OK"


def print_result(
    log_path: Path,
    valid: bool,
    reason: str,
    entries: int,
) -> None:
    print("Audit log verification result")
    print("-----------------------------")
    print(f"Log path: {log_path}")
    print(f"Entries: {entries}")
    print(f"Audit log valid: {valid}")
    print(f"Reason: {reason}")


def main() -> None:
    args = parse_args()

    log_path = Path(args.log)

    entries = load_jsonl(log_path)

    valid, reason = verify_entries(
        entries=entries,
        verbose=args.verbose,
    )

    print_result(
        log_path=log_path,
        valid=valid,
        reason=reason,
        entries=len(entries),
    )

    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()