"""
generate_crypto_report.py

Generates a Markdown report from the crypto demo summary.

From project root:
    python -m crypto_research.scripts.generate_crypto_report
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CRYPTO_ROOT = PROJECT_ROOT / "crypto_research"

SUMMARY_JSON = CRYPTO_ROOT / "outputs" / "reports" / "crypto_demo_summary.json"
REPORT_MD = CRYPTO_ROOT / "outputs" / "reports" / "crypto_demo_report.md"


def main() -> None:
    if not SUMMARY_JSON.exists():
        raise FileNotFoundError(
            "crypto_demo_summary.json not found. Run:\n"
            "python -m crypto_research.scripts.run_crypto_demo"
        )

    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))

    status = summary["overall_status"]
    hybrid = summary["hybrid_encryption"]
    secure_plan = summary["secure_plan"]
    audit = summary["audit_log"]

    report = f"""# ATLC Cryptographic Security Demo Report

## 1. Purpose

This report summarizes the cryptographic security demonstration for the Adaptive Traffic Light Control system.

The demo protects three important ATLC security areas:

- stored runtime data
- traffic-light PLAN command messages
- audit events

## 2. Hybrid Encryption Result

The demo uses AES-GCM to encrypt ATLC runtime data and RSA-OAEP to wrap the AES session key.

| Item | Result |
|---|---|
| Encrypted data | {hybrid["encrypted_data_path"]} |
| Encrypted AES key | {hybrid["encrypted_key_path"]} |
| Decrypted data | {hybrid["decrypted_data_path"]} |
| Encryption time | {hybrid["encryption_ms"]:.3f} ms |
| Decryption time | {hybrid["decryption_ms"]:.3f} ms |
| Integrity check | {hybrid["integrity_ok"]} |

## 3. Secure PLAN Message Result

The demo uses HMAC-SHA256 to authenticate traffic-light PLAN messages.

| Test | Result |
|---|---|
| Valid PLAN accepted | {secure_plan["valid_plan_accepted"]} |
| Tampered PLAN accepted | {secure_plan["tampered_plan_accepted"]} |
| Tampered PLAN reason | {secure_plan["tampered_plan_reason"]} |
| Replayed PLAN accepted | {secure_plan["replayed_plan_accepted"]} |
| Replayed PLAN reason | {secure_plan["replayed_plan_reason"]} |

## 4. Audit Log Result

The demo uses a SHA-256 hash chain to make audit logs tamper-evident.

| Item | Result |
|---|---|
| Audit log path | {audit["audit_log_path"]} |
| Log valid | {audit["verification"]["valid"]} |
| Entries | {audit["verification"].get("entries", "N/A")} |
| Reason | {audit["verification"]["reason"]} |

## 5. Overall Security Status

| Security Check | Passed |
|---|---|
| Encryption integrity OK | {status["encryption_integrity_ok"]} |
| Valid PLAN accepted | {status["valid_plan_accepted"]} |
| Tampered PLAN rejected | {status["tampered_plan_rejected"]} |
| Replay rejected | {status["replay_rejected"]} |
| Audit log valid | {status["audit_log_valid"]} |

## 6. Conclusion

The demonstration shows that the ATLC system can be extended with a cryptographic security layer.

AES-GCM protects stored runtime data, RSA-OAEP protects the AES session key, HMAC-SHA256 authenticates PLAN messages, nonce and plan_id values help reject replay attacks, and SHA-256 hash chaining makes audit logs tamper-evident.

This module is currently a research demonstration and can later be integrated into the production runtime under pc_app/security.
"""

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(report, encoding="utf-8")

    print("Saved report:", REPORT_MD)


if __name__ == "__main__":
    main()
