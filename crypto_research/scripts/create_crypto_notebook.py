"""
create_crypto_notebook.py

Creates the Jupyter Notebook required for the cryptography assignment.

From project root:
    python -m crypto_research.scripts.create_crypto_notebook
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "crypto_research"
    / "notebooks"
    / "atlc_crypto_security_demo.ipynb"
)


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    cells = [
        markdown_cell(
            """# ATLC Cryptographic Security Demonstration

This notebook demonstrates a cryptographic security layer for the Adaptive Traffic Light Control system.

The goal is to protect sensitive runtime data, traffic-light control messages, and audit logs using modern cryptographic techniques.
"""
        ),
        markdown_cell(
            """## 1. Security Motivation

The ATLC system processes traffic data and generates signal-control decisions. These assets should be protected because unauthorized modification or replay of traffic-light commands can create unsafe behavior.

Sensitive assets include:

- runtime traffic logs
- traffic-light PLAN messages
- operator control commands
- exported reports
- audit events
"""
        ),
        markdown_cell(
            """## 2. Cryptographic Architecture

This notebook demonstrates a hybrid security design:

- AES-GCM encrypts ATLC runtime data.
- RSA-OAEP wraps the AES key.
- HMAC-SHA256 authenticates PLAN messages.
- plan_id, timestamp, and nonce provide replay protection.
- SHA-256 hash chaining protects audit logs from silent tampering.

This follows the hybrid cryptography architecture: AES provides fast encryption for large data, while RSA protects the small AES session key.
"""
        ),
        code_cell(
            """from pathlib import Path
import json

PROJECT_ROOT = Path.cwd()
print("Project root:", PROJECT_ROOT)
"""
        ),
        markdown_cell(
            """## 3. Run End-to-End Crypto Demo

The script below runs the complete cryptographic demonstration:

1. Encrypt ATLC runtime sample data using AES-GCM.
2. Wrap the AES key using RSA-OAEP.
3. Decrypt and verify file integrity.
4. Create and verify a secure PLAN message using HMAC-SHA256.
5. Simulate tamper and replay attacks.
6. Create and verify a hash-chained audit log.
"""
        ),
        code_cell(
            """!python -m crypto_research.scripts.run_crypto_demo
"""
        ),
        markdown_cell(
            """## 4. Generate Markdown Report

The next cell generates a report-ready Markdown summary from the demo outputs.
"""
        ),
        code_cell(
            """!python -m crypto_research.scripts.generate_crypto_report
"""
        ),
        markdown_cell(
            """## 5. Inspect Summary JSON

This cell loads the generated crypto demo summary and displays the main security checks.
"""
        ),
        code_cell(
            """summary_path = PROJECT_ROOT / "crypto_research" / "outputs" / "reports" / "crypto_demo_summary.json"

with summary_path.open("r", encoding="utf-8") as file:
    summary = json.load(file)

summary["overall_status"]
"""
        ),
        markdown_cell(
            """## 6. Inspect Secure PLAN Example

The secure PLAN message contains the original timing fields plus timestamp, nonce, and MAC.

The MAC is computed over a canonical PLAN string. If any field is modified, verification fails.
"""
        ),
        code_cell(
            """summary["secure_plan"]["secure_plan_example"]
"""
        ),
        markdown_cell(
            """## 7. Inspect Audit Log Verification

The audit log uses a SHA-256 hash chain. Each entry includes the hash of the previous entry. If an attacker changes an old log entry, verification fails.
"""
        ),
        code_cell(
            """summary["audit_log"]["verification"]
"""
        ),
        markdown_cell(
            """## 8. Conclusion

The simulation demonstrates that the ATLC system can be extended with practical cryptographic protections.

The results show:

- encrypted runtime data can be decrypted correctly
- file integrity is verified using SHA-256
- valid PLAN messages are accepted
- tampered PLAN messages are rejected
- replayed PLAN messages are rejected
- audit logs can be verified using hash chaining

This notebook is a research prototype. Future integration can move stable security modules into pc_app/security and connect secure PLAN authentication to the UART communication layer.
"""
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.x",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Saved notebook:", NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
