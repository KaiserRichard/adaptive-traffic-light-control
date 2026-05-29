# ATLC Cryptographic Security Research

## Purpose

This research module demonstrates how cryptographic mechanisms can protect the Adaptive Traffic Light Control system.

The module is designed for a cryptography course assignment and follows a Jupyter Notebook-based workflow.

## Security Goals

The ATLC system contains several sensitive assets:

    runtime traffic logs
    traffic-light PLAN messages
    operator control commands
    traffic reports
    audit events

This research module demonstrates protection for those assets using:

    AES-GCM for data confidentiality and integrity
    RSA-OAEP for AES key wrapping
    HMAC-SHA256 for PLAN message authentication
    nonce / timestamp / plan_id for replay protection
    SHA-256 hash chain for tamper-evident audit logs

## Folder Structure

    crypto_research/
      notebooks/
        atlc_crypto_security_demo.ipynb

      src/crypto_utils/
        crypto_envelope.py
        secure_plan.py
        audit_log.py
        file_io.py

      scripts/
        run_crypto_demo.py
        generate_crypto_report.py
        create_crypto_notebook.py

      inputs/
        atlc_runtime_sample.txt

      outputs/
        encrypted/
        decrypted/
        logs/
        reports/
        figures/

## Main Notebook

    crypto_research/notebooks/atlc_crypto_security_demo.ipynb

The notebook includes:

    1. Introduction and cryptographic principles
    2. ATLC threat model
    3. Hybrid AES-GCM + RSA-OAEP encryption demo
    4. Secure PLAN message demo with HMAC-SHA256
    5. Replay and tamper attack simulation
    6. Tamper-evident audit log demo
    7. Runtime and integrity result analysis

## Run Demo Script

From project root:

    python -m crypto_research.scripts.run_crypto_demo

## Generate Notebook

From project root:

    python -m crypto_research.scripts.create_crypto_notebook

## Expected Outputs

    crypto_research/outputs/encrypted/encrypted_atlc_data.bin
    crypto_research/outputs/encrypted/encrypted_aes_key.bin
    crypto_research/outputs/encrypted/encryption_metadata.json
    crypto_research/outputs/decrypted/decrypted_atlc_data.txt
    crypto_research/outputs/logs/crypto_demo_log.jsonl
    crypto_research/outputs/logs/audit_log_hash_chain.jsonl
    crypto_research/outputs/reports/crypto_demo_summary.json
    crypto_research/outputs/reports/crypto_demo_report.md

## Important Note

This is a research/demo module. It does not directly control the MCU yet.

Future integration can move stable code into:

    pc_app/security/

and connect secure PLAN authentication to:

    pc_app/control/uart_sender.py
