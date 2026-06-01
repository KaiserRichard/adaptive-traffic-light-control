"""
audit_runtime.py

ATLC Phase 14 - Tamper-Evident Runtime Audit Log

Purpose:
    - Store security events in JSONL format.
    - Chain each event using SHA-256.
    - Detect post-run log modification.

This is an experiment-layer audit logger. It does not replace the normal
runtime CSV/JSONL logger from pc_app.control.system_logger.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


class HashChainAuditLog:
    """
    Simple hash-chained JSONL audit log.

    Each entry contains:
        - timestamp
        - event_type
        - payload
        - previous_hash
        - current_hash
    """

    def __init__(
        self,
        log_path: str | Path,
    ) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _hash_payload(
        self,
        entry: dict[str, Any],
    ) -> str:
        encoded = json.dumps(
            entry,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        return hashlib.sha256(encoded).hexdigest()

    def _last_hash(self) -> str:
        if not self.log_path.exists():
            return "0" * 64

        last_line = None

        with self.log_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                if line.strip():
                    last_line = line

        if last_line is None:
            return "0" * 64

        last_entry = json.loads(last_line)
        return str(last_entry["current_hash"])

    def append(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        previous_hash = self._last_hash()

        entry_without_hash = {
            "timestamp": int(time.time()),
            "event_type": event_type,
            "payload": payload,
            "previous_hash": previous_hash,
        }

        current_hash = self._hash_payload(entry_without_hash)

        entry = {
            **entry_without_hash,
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
                        "reason": f"BROKEN_PREVIOUS_HASH_AT_LINE:{line_number}",
                    }

                stored_hash = entry.get("current_hash")

                entry_without_hash = {
                    "timestamp": entry["timestamp"],
                    "event_type": entry["event_type"],
                    "payload": entry["payload"],
                    "previous_hash": entry["previous_hash"],
                }

                computed_hash = self._hash_payload(entry_without_hash)

                if stored_hash != computed_hash:
                    return {
                        "valid": False,
                        "entries": entries,
                        "reason": f"BROKEN_CURRENT_HASH_AT_LINE:{line_number}",
                    }

                previous_hash = stored_hash
                entries += 1

        return {
            "valid": True,
            "entries": entries,
            "reason": "OK",
        }