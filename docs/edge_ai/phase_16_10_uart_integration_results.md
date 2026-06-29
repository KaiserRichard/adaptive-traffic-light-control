# Phase 16.10 UART Integration Results

## Status

```text
UART framing and dry-run handoff prepared.
No real UART hardware communication yet.
No MCU firmware integration yet.
No end-to-end hardware demo yet.
```

## Interpreter Used

```text
.venv/bin/python
```

## Commands Run

Phase 16.9 verification:

```bash
.venv/bin/python deployment/ai_host/demo_plan_generation.py --ns-count 12 --ew-count 5 --seq 1
.venv/bin/python deployment/ai_host/demo_plan_generation.py --ns-count 5 --ew-count 12 --seq 2
.venv/bin/python deployment/ai_host/demo_plan_generation.py --ns-count 0 --ew-count 0 --seq 3
.venv/bin/python deployment/ai_host/demo_plan_generation.py --ns-count 10 --ew-count 10 --seq 4
```

Phase 16.10 dry-run UART framing:

```bash
.venv/bin/python deployment/ai_host/demo_uart_dry_run.py --ns-count 12 --ew-count 5 --seq 1
.venv/bin/python deployment/ai_host/demo_uart_dry_run.py --ns-count 5 --ew-count 12 --seq 2
.venv/bin/python deployment/ai_host/demo_uart_dry_run.py --ns-count 0 --ew-count 0 --seq 3
.venv/bin/python deployment/ai_host/demo_uart_dry_run.py --ns-count 10 --ew-count 10 --seq 4
```

## Example Outputs

### Case 1 - NS Heavier

```text
Generated PLAN:
PLAN,seq=1,mode=adaptive,ns_green=28,ew_green=17,yellow=3,all_red=1
UART frame bytes:
b'PLAN,seq=1,mode=adaptive,ns_green=28,ew_green=17,yellow=3,all_red=1\n'
Frame length:      68 bytes
Unframed message:  PLAN,seq=1,mode=adaptive,ns_green=28,ew_green=17,yellow=3,all_red=1
Round-trip parse:  OK
Dry-run only: no serial port opened.
Expected future MCU responses:
ACK,seq=1
NACK,seq=1,reason=<reason>
STATUS,seq=1,state=<state>,remaining=<seconds>
DIAG,seq=1,...
```

### Case 2 - EW Heavier

```text
Generated PLAN:
PLAN,seq=2,mode=adaptive,ns_green=17,ew_green=28,yellow=3,all_red=1
UART frame bytes:
b'PLAN,seq=2,mode=adaptive,ns_green=17,ew_green=28,yellow=3,all_red=1\n'
Frame length:      68 bytes
Unframed message:  PLAN,seq=2,mode=adaptive,ns_green=17,ew_green=28,yellow=3,all_red=1
Round-trip parse:  OK
Dry-run only: no serial port opened.
Expected future MCU responses:
ACK,seq=2
NACK,seq=2,reason=<reason>
STATUS,seq=2,state=<state>,remaining=<seconds>
DIAG,seq=2,...
```

### Case 3 - Empty Intersection

```text
Generated PLAN:
PLAN,seq=3,mode=adaptive,ns_green=22,ew_green=23,yellow=3,all_red=1
UART frame bytes:
b'PLAN,seq=3,mode=adaptive,ns_green=22,ew_green=23,yellow=3,all_red=1\n'
Frame length:      68 bytes
Unframed message:  PLAN,seq=3,mode=adaptive,ns_green=22,ew_green=23,yellow=3,all_red=1
Round-trip parse:  OK
Dry-run only: no serial port opened.
Expected future MCU responses:
ACK,seq=3
NACK,seq=3,reason=<reason>
STATUS,seq=3,state=<state>,remaining=<seconds>
DIAG,seq=3,...
```

### Case 4 - Equal Counts

```text
Generated PLAN:
PLAN,seq=4,mode=adaptive,ns_green=22,ew_green=23,yellow=3,all_red=1
UART frame bytes:
b'PLAN,seq=4,mode=adaptive,ns_green=22,ew_green=23,yellow=3,all_red=1\n'
Frame length:      68 bytes
Unframed message:  PLAN,seq=4,mode=adaptive,ns_green=22,ew_green=23,yellow=3,all_red=1
Round-trip parse:  OK
Dry-run only: no serial port opened.
Expected future MCU responses:
ACK,seq=4
NACK,seq=4,reason=<reason>
STATUS,seq=4,state=<state>,remaining=<seconds>
DIAG,seq=4,...
```

## What Was Validated

- Phase 16.9 PLAN generation demo is executable.
- PLAN strings can be framed as newline-terminated ASCII bytes.
- UART frames can be validated without opening serial ports.
- UART frames can be unframed and parsed back into `TrafficPlan` objects.

## What Was Not Validated

- Real UART transmission.
- Raspberry Pi serial configuration.
- STM32 / ESP32 parser compatibility.
- ACK / NACK / STATUS / DIAG hardware responses.
- MCU traffic FSM execution.

## Limitations

- No hardware was required or used.
- No `pyserial` dependency was added.
- No serial port was opened.
- The key-value PLAN format may still need adapter/parser alignment with firmware in a later phase.

## Next Phase

```text
Phase 17 / later integration - MCU-side validation and real serial testing
```
