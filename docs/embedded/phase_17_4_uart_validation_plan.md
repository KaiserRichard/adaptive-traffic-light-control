# Phase 17.4 - UART Link Validation Plan

## Goal

Define how to validate the UART link between the Raspberry Pi AI host and the STM32F103C8T6 PCB without implementing UART firmware in this phase.

## Scope

Included:

- UART electrical safety requirements.
- TX/RX crossing checklist.
- Message framing plan.
- Relationship between `PLAN`, `ACK`, `NACK`, and `STATUS`.
- Loopback and staged host/controller validation plan.

Not included:

- STM32 UART firmware implementation.
- Raspberry Pi runtime integration.
- FreeRTOS FSM port.
- End-to-end AI-to-hardware demo.

## Planned Link Roles

```text
Raspberry Pi AI Host
    sends PLAN messages

STM32 Controller
    sends ACK / STATUS messages
```

More complete response set:

```text
STM32 Controller
    sends ACK for accepted plans
    sends NACK for rejected plans
    sends STATUS for periodic controller state after that feature is ported
```

## UART Settings

Recommended default:

```text
115200 baud
8 data bits
no parity
1 stop bit
newline-terminated ASCII
```

This keeps the STM32 plan compatible with the existing ESP32 prototype.

## Protocol Compatibility

The STM32 should preserve the existing ATLC concept:

```text
PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>
ACK,<plan_id>
NACK,<plan_id>,<reason>
STATUS,<plan_id>,<state>,<remaining_seconds>,<health>
```

Validation rules to preserve:

- Reject malformed messages.
- Reject green times outside configured limits.
- Reject yellow/all-red values outside configured limits.
- ACK only after a valid plan is accepted into controller ownership.
- Use STATUS to report active controller state.

## Electrical Preconditions

Before any Pi-to-STM32 UART test:

```text
[ ] STM32 3.3 V rail validated.
[ ] Raspberry Pi UART pins confirmed as 3.3 V logic.
[ ] GND reference connected.
[ ] TX/RX crossing confirmed.
[ ] UART header orientation confirmed.
[ ] No 5 V USB-UART adapter connected directly.
```

## Staged Validation

### Stage 1 - Serial Adapter or Pi Loopback

Purpose:

- Verify the serial host setup before involving the STM32.

Expected result:

- Sent characters echo correctly at 115200 8N1.

### Stage 2 - STM32 UART Echo

Purpose:

- Verify USART1 RX/TX on the STM32.

Expected result:

- STM32 receives text and sends deterministic echo or response.

### Stage 3 - PLAN Parser Smoke Test

Purpose:

- Verify the STM32 can parse protocol lines without driving real traffic logic.

Example inputs:

```text
PLAN,17,25,15,3,1
PLAN,19,2,15,3,1
PLAN,abc
HELLO
```

Expected future outputs:

```text
ACK,17
NACK,19,GREEN_A_OUT_OF_RANGE
NACK,-1,MALFORMED_PLAN
NACK,-1,UNKNOWN_COMMAND
```

### Stage 4 - STATUS Reporting Smoke Test

Purpose:

- Verify that the host can read periodic controller status messages.

Expected future output:

```text
STATUS,17,A_GREEN,12,OK
```

This stage depends on future firmware work and is not implemented now.

## Failure Symptoms

- No received bytes: wrong serial device, missing ground, TX/RX not crossed, disabled UART, wrong firmware pin config.
- Garbled bytes: wrong baud, unstable ground, signal integrity issue, voltage mismatch.
- Repeated resets: power rail issue, regulator thermal issue, firmware fault.
- ACK without behavior change: may be normal if plan apply is deferred to safe FSM boundary.
- STATUS mismatch: FSM state ownership or reporting snapshot issue.

## Debug Checklist

```text
[ ] Confirm Pi UART is enabled.
[ ] Disable Pi login console on the selected UART if needed.
[ ] Confirm serial device path.
[ ] Confirm 115200 8N1.
[ ] Confirm STM32 PA9 is TX1 and PA10 is RX1.
[ ] Confirm UART cable crossing.
[ ] Confirm common ground.
[ ] Use a logic analyzer if terminal output is ambiguous.
[ ] Capture raw transcripts for report evidence.
```

## What Counts as Phase 17.4 Complete

For this documentation phase:

```text
[x] UART roles documented.
[x] Electrical safety checks documented.
[x] Message framing plan documented.
[x] PLAN / ACK / NACK / STATUS relationship documented.
[x] Loopback and staged validation plan documented.
[ ] UART tested on hardware.
[ ] UART firmware implemented.
[ ] Raspberry Pi integration executed.
```

Hardware validation remains pending.
