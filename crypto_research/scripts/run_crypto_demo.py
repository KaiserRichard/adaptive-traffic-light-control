"""
run_crypto_demo.py

Runs the complete ATLC cryptographic security demonstration.

From project root:
    python -m crypto_research.scripts.run_crypto_demo
"""

from __future__ import annotations

import copy
import json
import secrets
import time
from pathlib import Path

from crypto_research.src.crypto_utils.audit_log import HashChainAuditLog
from crypto_research.src.crypto_utils.crypto_envelope import (
    hybrid_encrypt_file,
    hybrid_decrypt_file,
)
from crypto_research.src.crypto_utils.file_io import append_jsonl, write_json
from crypto_research.src.crypto_utils.secure_plan import (
    ReplayProtector,
    create_secure_plan,
    verify_secure_plan,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CRYPTO_ROOT = PROJECT_ROOT / "crypto_research"

INPUT_FILE = CRYPTO_ROOT / "inputs" / "atlc_runtime_sample.txt"

ENCRYPTED_DIR = CRYPTO_ROOT / "outputs" / "encrypted"
DECRYPTED_DIR = CRYPTO_ROOT / "outputs" / "decrypted"
LOG_DIR = CRYPTO_ROOT / "outputs" / "logs"
REPORT_DIR = CRYPTO_ROOT / "outputs" / "reports"

ENCRYPTED_DATA = ENCRYPTED_DIR / "encrypted_atlc_data.bin"
ENCRYPTED_KEY = ENCRYPTED_DIR / "encrypted_aes_key.bin"
METADATA_JSON = ENCRYPTED_DIR / "encryption_metadata.json"

PRIVATE_KEY = ENCRYPTED_DIR / "demo_rsa_private_key.pem"
PUBLIC_KEY = ENCRYPTED_DIR / "demo_rsa_public_key.pem"

DECRYPTED_DATA = DECRYPTED_DIR / "decrypted_atlc_data.txt"

DEMO_LOG = LOG_DIR / "crypto_demo_log.jsonl"
AUDIT_LOG = LOG_DIR / "audit_log_hash_chain.jsonl"
SUMMARY_JSON = REPORT_DIR / "crypto_demo_summary.json"


def log_event(event_type: str, payload: dict) -> None:
    append_jsonl(
        DEMO_LOG,
        {
            "timestamp": int(time.time()),
            "event_type": event_type,
            "payload": payload,
        },
    )


def ensure_input_exists() -> None:
    if INPUT_FILE.exists():
        return

    INPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    INPUT_FILE.write_text(
        "ATLC fallback crypto demo input.\n"
        "mode=AUTO\n"
        "phase=A_GREEN\n"
        "green_a=40\n"
        "green_b=10\n",
        encoding="utf-8",
    )


def run_hybrid_encryption_demo() -> dict:
    result = hybrid_encrypt_file(
        input_path=INPUT_FILE,
        encrypted_data_path=ENCRYPTED_DATA,
        encrypted_key_path=ENCRYPTED_KEY,
        metadata_path=METADATA_JSON,
        private_key_path=PRIVATE_KEY,
        public_key_path=PUBLIC_KEY,
    )

    decrypt_result = hybrid_decrypt_file(
        encrypted_data_path=ENCRYPTED_DATA,
        encrypted_key_path=ENCRYPTED_KEY,
        metadata_path=METADATA_JSON,
        private_key_path=PRIVATE_KEY,
        output_path=DECRYPTED_DATA,
    )

    output = {
        "encrypted_data_path": str(result.encrypted_data_path),
        "encrypted_key_path": str(result.encrypted_key_path),
        "metadata_path": str(result.metadata_path),
        "decrypted_data_path": str(DECRYPTED_DATA),
        "plaintext_sha256": result.plaintext_sha256,
        "decrypted_sha256": result.decrypted_sha256,
        "encryption_ms": result.encryption_ms,
        "decryption_ms": result.decryption_ms,
        "integrity_ok": result.integrity_ok and decrypt_result["integrity_ok"],
    }

    log_event("HYBRID_ENCRYPTION_DEMO", output)

    return output


def run_secure_plan_demo() -> dict:
    secret_key = secrets.token_bytes(32)
    replay_protector = ReplayProtector()

    secure_plan = create_secure_plan(
        secret_key=secret_key,
        plan_id=1,
        green_a=40,
        green_b=10,
        yellow=3,
        all_red=1,
    )

    valid_ok, valid_reason = verify_secure_plan(
        secret_key=secret_key,
        plan=secure_plan,
        replay_protector=replay_protector,
    )

    tampered_plan = copy.deepcopy(secure_plan)
    tampered_plan["green_a"] = 99

    tamper_ok, tamper_reason = verify_secure_plan(
        secret_key=secret_key,
        plan=tampered_plan,
        replay_protector=replay_protector,
    )

    replay_ok, replay_reason = verify_secure_plan(
        secret_key=secret_key,
        plan=secure_plan,
        replay_protector=replay_protector,
    )

    output = {
        "valid_plan_accepted": valid_ok,
        "valid_plan_reason": valid_reason,
        "tampered_plan_accepted": tamper_ok,
        "tampered_plan_reason": tamper_reason,
        "replayed_plan_accepted": replay_ok,
        "replayed_plan_reason": replay_reason,
        "secure_plan_example": secure_plan,
    }

    log_event("SECURE_PLAN_DEMO", output)

    return output


def run_audit_log_demo() -> dict:
    if AUDIT_LOG.exists():
        AUDIT_LOG.unlink()

    audit_log = HashChainAuditLog(AUDIT_LOG)

    audit_log.append(
        "SYSTEM_START",
        {
            "module": "crypto_research",
            "mode": "demo",
        },
    )

    audit_log.append(
        "PLAN_ACCEPTED",
        {
            "plan_id": 1,
            "green_a": 40,
            "green_b": 10,
        },
    )

    audit_log.append(
        "TAMPER_TEST_REJECTED",
        {
            "reason": "INVALID_MAC",
        },
    )

    verification = audit_log.verify()

    output = {
        "audit_log_path": str(AUDIT_LOG),
        "verification": verification,
    }

    log_event("AUDIT_LOG_DEMO", output)

    return output


def main() -> None:
    ensure_input_exists()

    for folder in [ENCRYPTED_DIR, DECRYPTED_DIR, LOG_DIR, REPORT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    if DEMO_LOG.exists():
        DEMO_LOG.unlink()

    hybrid_result = run_hybrid_encryption_demo()
    secure_plan_result = run_secure_plan_demo()
    audit_result = run_audit_log_demo()

    summary = {
        "project": "ATLC Cryptographic Security Demo",
        "hybrid_encryption": hybrid_result,
        "secure_plan": secure_plan_result,
        "audit_log": audit_result,
        "overall_status": {
            "encryption_integrity_ok": hybrid_result["integrity_ok"],
            "valid_plan_accepted": secure_plan_result["valid_plan_accepted"],
            "tampered_plan_rejected": not secure_plan_result["tampered_plan_accepted"],
            "replay_rejected": not secure_plan_result["replayed_plan_accepted"],
            "audit_log_valid": audit_result["verification"]["valid"],
        },
    }

    write_json(SUMMARY_JSON, summary)

    print(json.dumps(summary["overall_status"], indent=2))
    print("Saved summary:", SUMMARY_JSON)


if __name__ == "__main__":
    main()
