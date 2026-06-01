# ATLC Phase 14 — Runtime Security Experiment Layer

## 1. Phase Purpose

Phase 14 introduces an experimental security layer for the ATLC project.

The goal is to protect traffic-light control messages from:

```text
tampering
replay attacks
forged control commands
timestamp manipulation
audit-log modification
This phase does not yet modify the production runtime pipeline in `pc_app.main`.

Instead, it provides a controlled experiment environment for testing secure traffic-light PLAN messages.

---

## 2. Why Security Is Needed

The ATLC runtime produces traffic-light timing plans such as:

```
green_a = 40
green_b = 10
yellow = 3
all_red = 1
```

If these plans are transmitted to a microcontroller over UART, an attacker could theoretically attempt to:

```
change green_a or green_b
replay an old PLAN message
forge a fake PLAN message
modify runtime logs after the experiment
```

For a traffic-control system, even simple message tampering can create unsafe behavior.

Therefore, Phase 14 adds security experiments before hardware-level integration.

---

## 3. Implemented Security Components

## 3.1 HMAC-SHA256 PLAN Authentication

The file:

```
pc_app/security/secure_runtime.py
```

creates authenticated PLAN messages.

Each secure PLAN includes:

```
message_type
plan_id
green_a
green_b
yellow
all_red
timestamp
nonce
mac
```

The `mac` field is computed using HMAC-SHA256.

If an attacker modifies any protected field, the MAC verification fails.

---

## 3.2 Replay Protection

Replay protection is implemented using:

```
plan_id
nonce
```

The verifier tracks:

```
highest_plan_id
used_nonces
```

A PLAN is rejected if:

```
plan_id <= highest_plan_id
```

or if the nonce was already used.

This prevents an attacker from recording a valid old PLAN and sending it again later.

---

## 3.3 Timestamp Window Check

The verifier rejects PLAN messages outside an allowed timestamp window.

This reduces the risk of delayed or stale messages being accepted.

---

## 3.4 Tamper-Evident Audit Log

The file:

```
pc_app/security/audit_runtime.py
```

implements a hash-chained JSONL audit log.

Each log entry stores:

```
previous_hash
current_hash
```

If any previous event is edited, the hash chain becomes invalid.

This provides tamper-evident experiment evidence.

---

## 4. Attack Simulation

The file:

```
pc_app/security/attack_simulator.py
```

simulates a malicious actor attempting to attack the control path.

The simulation includes:

| Attack | Expected Defense |
| --- | --- |
| Modify green time | Rejected by HMAC |
| Replay old PLAN | Rejected by plan_id |
| Forge fake MAC | Rejected by HMAC |
| Modify timestamp | Rejected by timestamp window |
| Modify audit log | Rejected by hash-chain verification |

---

## 5. Testcase

The file:

```
experiments/test_secure_runtime.py
```

validates the security experiment.

The test passes only if:

```
valid PLAN is accepted
tampered PLAN is rejected
replayed PLAN is rejected
forged MAC is rejected
timestamp attack is rejected
tampered audit log is rejected
```

---

## 6. Output Evidence

The experiment generates:

```
outputs/security/phase14_runtime_security/secure_runtime_audit_log.jsonl
outputs/security/phase14_runtime_security/tampered_secure_runtime_audit_log.jsonl
outputs/security/phase14_runtime_security/attack_simulation_summary.json
outputs/security/phase14_runtime_security/attack_simulation_summary.md
```

These files can be used as report evidence.

---

## 7. Engineering Interpretation

Phase 14 demonstrates that ATLC control messages can be protected before being sent to a microcontroller.

The implemented security layer provides:

```
integrity
authentication
anti-replay protection
tamper-evident logging
attack simulation evidence
```

This is a meaningful security improvement over a plain UART protocol where control messages are sent without authentication.

---

## 8. Current Scope Limitation

This phase is still experimental.

It is not yet connected to:

```
pc_app.main
uart_sender.py
firmware MCU verification
real hardware ACK authentication
```

Those integrations should only be added after the experiment layer is stable.

---

## 9. Recommended Next Step

The next engineering step is to connect this secure PLAN format to the UART layer.

Recommended future work:

```
secure PLAN serialization
UART sender integration
MCU-side MAC verification
authenticated ACK messages
replay-state persistence
hardware-in-the-loop attack tests
```

For now, Phase 14 should be treated as a validated crypto/security experiment layer.

```

---

# 7. What this phase proves

After these files are added, Phase 14 proves this:

```text
An attacker cannot change green_a / green_b without detection.
An attacker cannot replay an old valid PLAN without detection.
An attacker cannot forge a fake MAC without the secret key.
An attacker cannot silently modify the audit log after the run.
```