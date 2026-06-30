# ATLC Common Firmware Layer

## Purpose

This folder contains hardware-independent controller logic for the Adaptive Traffic Light Control project.

The common layer is intended to be shared conceptually by:

- the existing ESP32 FreeRTOS prototype.
- the future STM32F103C8T6 PCB firmware.
- host-side tests that run without embedded hardware.

## Current Status

```text
Hardware-independent C layer added.
Host-side tests added.
No ESP32 firmware migration has been performed.
No STM32 hardware build has been created.
No STM32 hardware validation has been performed.
```

## Design Rules

- Plain C only.
- No Arduino dependency.
- No ESP-IDF dependency.
- No STM32 HAL/LL/CMSIS dependency.
- No FreeRTOS dependency.
- No dynamic allocation.
- Bounded strings and explicit result codes.
- Canonical ATLC protocol only.

## Canonical Protocol

The common layer accepts this host-to-controller message:

```text
PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>
```

It formats controller responses:

```text
ACK,<plan_id>
NACK,<plan_id>,<reason>
STATUS,<plan_id>,<state>,<remaining_seconds>,<health>
```

Deprecated key/value variants such as `PLAN,seq=...`, `ns_green=...`, or `ew_green=...` are rejected.

## Timing Policy

The timing bounds mirror the current ESP32 FreeRTOS prototype:

| Field | Minimum | Maximum | Notes |
| --- | --- | --- | --- |
| `green_a` | 10 s | 45 s | adaptive green for direction A |
| `green_b` | 10 s | 45 s | adaptive green for direction B |
| `yellow` | 3 s | 3 s | fixed yellow interval |
| `all_red` | 1 s | 1 s | fixed all-red interval |

If the ESP32 policy changes later, update this common layer and tests in the same commit.

## Source Layout

```text
include/
    atlc_plan.h
    atlc_protocol.h
    atlc_fsm.h
    atlc_status.h

src/
    atlc_plan.c
    atlc_protocol.c
    atlc_fsm.c
    atlc_status.c

tests/
    test_protocol_parser.py
    test_plan_validator.py
    test_fsm_sequence.py
```

## Test Commands

Run from the repository root:

```bash
python3 firmware/common/tests/test_protocol_parser.py
python3 firmware/common/tests/test_plan_validator.py
python3 firmware/common/tests/test_fsm_sequence.py
```

The tests compile small temporary host executables with the system C compiler. They do not use STM32 hardware, ST-LINK, FreeRTOS, Arduino, or STM32 vendor files.

## Hardware Boundary

This layer does not:

- initialize GPIO.
- configure UART.
- drive LEDs.
- refresh the seven-segment display.
- create FreeRTOS tasks.
- select a clock tree.
- flash firmware.

Those responsibilities remain in board-specific firmware.
