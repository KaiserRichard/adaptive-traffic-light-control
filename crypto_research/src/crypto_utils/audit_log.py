"""
audit_log.py

Tamper-evident audit logging for the ATLC crypto demo.

Purpose:
    The ATLC system may generate important security events, such as:
        - system start
        - PLAN accepted
        - PLAN rejected
        - invalid MAC detected
        - replay attack rejected
        - report encrypted
        - operator action

    These logs should not be silently modified after they are written.

Hash-chain concept:

    entry_hash_i = SHA256(canonical_entry_i || previous_hash)

In this implementation, each log entry stores:
    - timestamp
    - event_type
    - payload
    - previous_hash
    - entry_hash

If an attacker modifies an old log entry, its hash changes.
Then the next entry's previous_hash no longer matches.
This makes tampering detectable.

Important:
    This is tamper-evident, not tamper-proof.
    It helps detect modification, but it does not physically prevent deletion.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from crypto_research.src.crypto_utils.file_io import append_jsonl, ensure_parent


# The first log entry has no previous entry, so it uses a fixed genesis hash.
GENESIS_HASH = "0" * 64


def sha256_text(text: str) -> str:
    """
    Compute SHA-256 hash of a text string.

    Used for:
        - hashing canonical JSON entries
        - building hash-chain links
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(data: Dict[str, Any]) -> str:
    """
    Convert dictionary into deterministic JSON.

    Why:
        Hashing requires stable representation.
        The same data must always produce the same string.

    sort_keys=True:
        Ensures dictionary keys are always ordered.

    separators=(",", ":"):
        Removes unnecessary spaces so formatting differences do not change hash.
    """

    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


class HashChainAuditLog:
    """
    JSONL-based tamper-evident audit log.

    Each line in the log file is one JSON object.

    Example event:
        {
            "timestamp": "...",
            "event_type": "PLAN_ACCEPTED",
            "payload": {"plan_id": 1},
            "previous_hash": "...",
            "entry_hash": "..."
        }

    Main methods:
        append():
            Add a new audit event.

        verify():
            Check whether the full hash chain is still valid.
    """

    def __init__(self, path: Path):
        """
        Initialize audit log file path.

        The file is not overwritten.
        New entries are appended to the existing JSONL file.
        """

        self.path = path
        ensure_parent(path)

    def _read_entries(self) -> List[Dict[str, Any]]:
        """
        Read all existing JSONL audit entries.

        Returns:
            List of dictionaries.

        Empty file or missing file:
            Returns an empty list.
        """

        if not self.path.exists():
            return []

        entries = []

        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if line:
                    entries.append(json.loads(line))

        return entries

    def _latest_hash(self) -> str:
        """
        Return the hash of the latest audit entry.

        If the log is empty, return GENESIS_HASH.
        """

        entries = self._read_entries()

        if not entries:
            return GENESIS_HASH

        return str(entries[-1]["entry_hash"])

    def append(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Append a new audit event to the hash chain.

        Workflow:
            1. Read latest hash.
            2. Build entry body with previous_hash.
            3. Compute entry_hash.
            4. Write full entry to JSONL.
            5. Return the written entry.

        Parameters:
            event_type:
                Short event name, for example:
                    PLAN_ACCEPTED
                    INVALID_MAC
                    REPLAY_REJECTED

            payload:
                Event details as a dictionary.

        Returns:
            Full audit entry including entry_hash.
        """

        previous_hash = self._latest_hash()

        entry_body = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event_type": event_type,
            "payload": payload,
            "previous_hash": previous_hash,
        }

        # Hash only the entry body first.
        # entry_hash is added after hashing.
        entry_hash = sha256_text(canonical_json(entry_body))

        full_entry = {
            **entry_body,
            "entry_hash": entry_hash,
        }

        append_jsonl(self.path, full_entry)

        return full_entry

    def verify(self) -> Dict[str, Any]:
        """
        Verify the complete audit log hash chain.

        Verification rules:
            1. First entry must point to GENESIS_HASH.
            2. Each entry's previous_hash must equal the previous entry_hash.
            3. Each entry_hash must match the recomputed hash of its body.

        Returns:
            {
                "valid": True/False,
                "entries": number_of_entries,
                "reason": "OK" or error reason
            }

        Possible failure reasons:
            PREVIOUS_HASH_MISMATCH:
                A link between entries is broken.

            ENTRY_HASH_MISMATCH:
                The content of an entry was modified.
        """

        entries = self._read_entries()
        previous_hash = GENESIS_HASH

        for index, entry in enumerate(entries):
            expected_previous = previous_hash

            if entry.get("previous_hash") != expected_previous:
                return {
                    "valid": False,
                    "failed_index": index,
                    "reason": "PREVIOUS_HASH_MISMATCH",
                }

            entry_body = {
                "timestamp": entry["timestamp"],
                "event_type": entry["event_type"],
                "payload": entry["payload"],
                "previous_hash": entry["previous_hash"],
            }

            expected_hash = sha256_text(canonical_json(entry_body))

            if entry.get("entry_hash") != expected_hash:
                return {
                    "valid": False,
                    "failed_index": index,
                    "reason": "ENTRY_HASH_MISMATCH",
                }

            previous_hash = entry["entry_hash"]

        return {
            "valid": True,
            "entries": len(entries),
            "reason": "OK",
        }