# ATLC Phase 14 - Runtime Security and Cryptography Demo

This folder contains the cryptography research and notebook evidence for Phase 14 of the ATLC project.

## Purpose

Phase 14 focuses on applying cryptographic mechanisms to protect an Adaptive Traffic Light Control system against runtime security risks.

The main security goals are:

- protect ATLC runtime data,
- authenticate traffic-light PLAN messages,
- reject tampered control messages,
- reject replayed old messages,
- detect forged MAC values,
- detect invalid timestamps,
- detect post-run audit-log tampering.

## Main Notebook

The main submission notebook is:

```text
crypto_research/notebooks/atlc_phase14_security_runtime_demo.ipynb

This notebook contains both Markdown explanation and executable test cells. It demonstrates:

ATLC security context,
PLAN message authentication using HMAC-SHA256,
replay protection using plan_id, timestamp, and nonce,
runtime attack simulation,
audit-log verification,
hacker-modified log detection,
evidence generated from the ATLC software pipeline.
Related Runtime Security Code

The notebook uses the following project modules:

pc_app/security/secure_runtime.py
pc_app/security/audit_runtime.py
experiments/attack_simulator.py
experiments/test_secure_runtime.py
experiments/verify_audit_log.py

These files are placed in the main repository because they test and verify the ATLC runtime system directly.

Evidence Outputs

Generated evidence files are stored in:

outputs/security/phase14_runtime_security/
docs/security/figures/

Important outputs include:

attack_simulation_summary.json
attack_simulation_summary.md
secure_runtime_audit_log.jsonl
tampered_secure_runtime_audit_log.jsonl
hacker_modified_log.jsonl
Scope

This phase is a software-based simulation. It does not require hardware traffic lights or UART testing.

The ATLC runtime can be executed using a sample video, while the security layer is verified through attack simulation scripts and the notebook.