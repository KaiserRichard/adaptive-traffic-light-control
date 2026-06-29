# AI Host PLAN Generation and UART Dry-Run Interface

## Purpose

This folder contains the AI-host-side planning and UART preparation boundary for the Adaptive Traffic Light Control project.

It converts mock vehicle counts, or future ROI-based traffic counts, into a structured PLAN message. Phase 16.10 then prepares that PLAN message as newline-terminated ASCII bytes for future UART transport from the Raspberry Pi AI host to the MCU controller.

## Current Status

```text
PLAN generation interface prepared.
UART dry-run framing prepared.
No real UART hardware communication yet.
No MCU firmware integration yet.
No end-to-end hardware demo yet.
```

## Architecture

```text
vehicle counts / density input
        |
        v
TrafficPlan generation
        |
        v
PLAN serialization
        |
        v
newline-terminated UART frame
        |
        v
dry-run validation
        |
        v
future serial transport
        |
        v
MCU controller
```

## Files

```text
traffic_density.py
    Validates vehicle counts and computes simple density ratios.

plan_generator.py
    Converts direction-specific counts into bounded green-time allocations.

plan_protocol.py
    Defines the TrafficPlan data structure, PLAN serialization, parsing, and validation.

demo_plan_generation.py
    Runs a hardware-independent Phase 16.9 PLAN generation demo.

uart_framing.py
    Builds and validates newline-terminated ASCII UART frames.

demo_uart_dry_run.py
    Runs a Phase 16.10 dry-run demo without opening a serial port.
```

## PLAN Generation Demo

```bash
.venv/bin/python deployment/ai_host/demo_plan_generation.py \
  --ns-count 12 \
  --ew-count 5 \
  --seq 1
```

Expected PLAN output:

```text
PLAN,seq=1,mode=adaptive,ns_green=28,ew_green=17,yellow=3,all_red=1
```

## UART Dry-Run Demo

```bash
.venv/bin/python deployment/ai_host/demo_uart_dry_run.py \
  --ns-count 12 \
  --ew-count 5 \
  --seq 1
```

Expected UART frame bytes:

```text
b'PLAN,seq=1,mode=adaptive,ns_green=28,ew_green=17,yellow=3,all_red=1\n'
```

The dry-run demo validates the frame, restores the original PLAN message, parses it back into a `TrafficPlan`, and confirms that no serial port was opened.

## Message Format

PLAN message:

```text
PLAN,seq=1,mode=adaptive,ns_green=28,ew_green=17,yellow=3,all_red=1
```

UART frame:

```text
PLAN,seq=1,mode=adaptive,ns_green=28,ew_green=17,yellow=3,all_red=1\n
```

The frame is ASCII text terminated by exactly one newline character. This keeps the future MCU parser simple and makes host-side logs easy to inspect.

## Future MCU Responses

Expected future controller response formats:

```text
ACK,seq=1
NACK,seq=1,reason=<reason>
STATUS,seq=1,state=<state>,remaining=<seconds>
DIAG,seq=1,...
```

These response formats are documented for integration planning only. Phase 16.10 does not read real MCU responses.

## What This Does Not Do

```text
Does not run YOLO/ONNX inference directly.
Does not open a serial port.
Does not send real UART messages.
Does not require Raspberry Pi, STM32, or ESP32 hardware.
Does not modify MCU firmware.
Does not validate real UART wiring.
```

## Connection to Phase 16

```text
Phase 16.9 creates validated PLAN messages.
Phase 16.10 prepares those messages for future UART transport using dry-run framing.
```

The AI host remains responsible for perception, future ROI counting, density estimation, adaptive timing decisions, and preparing controller-ready PLAN messages.

## Connection to Phase 17

```text
Phase 17 will implement or validate the MCU-side receiver, parser, ACK/NACK/STATUS responses, and traffic-light FSM execution.
```

Phase 16.10 deliberately stops before MCU-side behavior so that the AI-host boundary stays testable without hardware.

## Next Step

```text
Write a Phase 16 closure report, then continue MCU-side integration in Phase 17.
```
