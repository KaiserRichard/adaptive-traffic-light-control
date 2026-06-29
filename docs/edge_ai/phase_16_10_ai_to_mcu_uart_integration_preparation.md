# Phase 16.10 AI-to-MCU UART Integration Preparation

## Goal

Prepare the host-side UART integration boundary between the Raspberry Pi AI host and the ESP32 / STM32 controller.

This phase does not send real UART messages. It creates a dry-run framing layer that can be tested without Raspberry Pi, STM32, ESP32, serial adapters, or firmware changes.

## Scope

Included:

- UART frame construction for serialized PLAN messages
- newline-terminated ASCII framing
- frame validation
- frame/unframe round-trip testing
- dry-run demo script
- expected future MCU response documentation

Not included:

- opening serial ports
- using `pyserial`
- sending real UART messages
- receiving ACK / NACK from hardware
- modifying STM32 firmware
- modifying ESP32 firmware
- implementing MCU-side receiving
- implementing FreeRTOS FSM behavior

## Why This Phase Exists

Phase 16.9 created a validated PLAN message:

```text
PLAN,seq=1,mode=adaptive,ns_green=28,ew_green=17,yellow=3,all_red=1
```

Phase 16.10 prepares that message for future MCU communication:

```text
TrafficPlan object
        |
        v
serialize_plan(plan)
        |
        v
UART frame construction
        |
        v
newline-terminated ASCII message
        |
        v
dry-run output
        |
        v
future serial write
```

The point is to validate the host-side message boundary before real UART hardware is connected.

## Architecture

```text
AI Host PLAN generation
        |
        v
PLAN message serialization
        |
        v
UART framing
        |
        v
dry-run UART sender
        |
        v
future serial transport
        |
        v
MCU controller later
        |
        v
ACK / NACK / STATUS / DIAG
```

## UART Framing Concept

Frame format:

```text
PLAN,seq=1,mode=adaptive,ns_green=28,ew_green=17,yellow=3,all_red=1\n
```

Rules:

- message must be non-empty
- message must be ASCII
- message must not contain embedded newline characters
- frame must end with exactly one newline
- frame length must be bounded
- frame type must be one of `PLAN`, `ACK`, `NACK`, `STATUS`, or `DIAG`

## Message Boundary

`deployment/ai_host/uart_framing.py` provides:

```python
frame_uart_message(message: str) -> bytes
unframe_uart_message(frame: bytes) -> str
validate_uart_frame(frame: bytes, max_length: int = 128) -> None
```

This gives Phase 16.10 a testable boundary without depending on a serial transport package.

## Dry-Run Sender Concept

The dry-run demo:

1. accepts mock vehicle counts
2. generates a Phase 16.9 `TrafficPlan`
3. serializes the PLAN message
4. frames it as UART bytes
5. validates the frame
6. unframes and parses the message back
7. prints expected future MCU responses

Command:

```bash
.venv/bin/python deployment/ai_host/demo_uart_dry_run.py \
  --ns-count 12 \
  --ew-count 5 \
  --seq 1
```

## Expected MCU Responses

Future controller responses should remain simple and line-oriented:

```text
ACK,seq=1
NACK,seq=1,reason=...
STATUS,seq=1,state=...,remaining=...
DIAG,seq=1,...
```

Phase 16.10 does not implement response parsing from real hardware.

## Why No Real UART Is Opened Yet

Real UART depends on:

- Raspberry Pi hardware
- serial adapter or GPIO UART configuration
- selected baud rate
- MCU firmware parser compatibility
- ACK / NACK timeout policy
- safe handling when hardware is disconnected

Those belong in a later hardware integration step. This phase validates the host message before the transport layer is introduced.

## Connection To Phase 16.9

Phase 16.9 generates and validates the `TrafficPlan` and PLAN string.

Phase 16.10 takes that PLAN string and prepares newline-terminated UART bytes.

## Connection To Phase 17 STM32 / ESP32 Validation

Phase 17 should validate the MCU-side parser, ACK / NACK behavior, and traffic FSM execution. Phase 16.10 does not change MCU firmware.

## Limitations

- No hardware UART validation.
- No serial package dependency.
- No ACK / NACK round trip.
- No MCU parser compatibility guarantee yet.
- The Phase 16.9 key-value format may require a firmware parser update or adapter later.

## Next Step

Prepare a real UART integration phase only after confirming the host-side frame format is accepted by the selected MCU parser contract.
