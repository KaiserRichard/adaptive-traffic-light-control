# Phase 17 - STM32 PCB Integration Track

## Goal

Start a parallel STM32 PCB hardware integration track while keeping the Phase 16 Raspberry Pi / Edge AI deployment track isolated.

This phase is documentation-first and offline-safe. It does not flash hardware, require STM32CubeIDE, require a real STM32 board, or claim hardware validation.

## System Transition

```text
ESP32 prototype
        |
        v
STM32F103C8T6 PCB controller
        |
        v
UART PLAN messages from Raspberry Pi AI host
        |
        v
FreeRTOS traffic light FSM
        |
        v
LED / 7-segment / PCB outputs
```

## Relationship to Existing ESP32 Prototype

The ESP32 FreeRTOS controller remains the behavior reference for:

- UART command parsing.
- validated `SignalPlan` ownership.
- queue-based task separation.
- ACK/NACK response behavior.
- STATUS reporting concept.
- host timeout watchdog and fallback behavior.
- traffic light FSM sequencing.

The STM32 work should port these concepts only after the pin map, toolchain path, and hardware bring-up procedure are reviewed.

## Planned STM32 FreeRTOS Task Plan

Planned task ownership for the future STM32 port:

| Planned task/module | Responsibility | Source reference |
| --- | --- | --- |
| UART receive task | Read newline-terminated host messages from USART1 | ESP32 `TaskUARTReceive` concept |
| Plan parser task | Parse and validate `PLAN` messages | ESP32 parser/protocol concept |
| Plan queue | Transfer validated plans to FSM owner | ESP32 queue concept |
| Traffic FSM task | Own active plan and light state transitions | ESP32 FSM concept |
| Status reporter | Emit periodic `STATUS` messages | Phase 15.8 concept |
| Watchdog/fallback logic | Detect host timeout and apply safe fallback | Phase 15.9 concept |
| GPIO driver layer | Own STM32 pin writes for LEDs and display | New STM32-specific module |

This is a plan, not an implemented STM32 firmware architecture.

## UART Protocol Relationship

Planned protocol direction:

```text
Raspberry Pi AI Host
    sends PLAN messages

STM32 Controller
    sends ACK / NACK / STATUS messages
```

Existing message concept:

```text
PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>
ACK,<plan_id>
NACK,<plan_id>,<reason>
STATUS,<plan_id>,<state>,<remaining_seconds>,<health>
```

Phase 17.4 keeps UART validation separate from full AI integration. A UART link must be electrically and logically validated before the Raspberry Pi YOLO pipeline is allowed to control the STM32 board.

## Traffic Light FSM Relationship

The STM32 should eventually execute the same state model already documented for the ESP32 prototype:

```text
STATE_A_GREEN
STATE_A_YELLOW
STATE_ALL_RED_AFTER_A
STATE_B_GREEN
STATE_B_YELLOW
STATE_ALL_RED_AFTER_B
```

The STM32 pin map currently does not label which LED is direction A/B red/yellow/green. The FSM output mapping is therefore blocked until the PCB LED ownership is verified.

## Watchdog / Fallback Concept

The controller should not depend on the host being healthy forever. The future STM32 firmware should preserve this behavior:

- Track the last valid `PLAN` time.
- Continue executing a safe fallback plan if host messages stop.
- Report a health state such as `HOST_TIMEOUT` in `STATUS`.
- Recover when valid host messages return.

This is a planned firmware requirement, not a completed STM32 implementation.

## Future Firmware Port

The future STM32 firmware port should not start by copying all ESP32 code directly. First review:

1. STM32 pin mapping.
2. GPIO current and output constraints.
3. USART1 wiring and baud rate.
4. FreeRTOS/HAL/LL/CMSIS toolchain choice.
5. Build system and debug workflow on MacBook.
6. Minimal blink and UART echo behavior.

Only after that should Phase 17.5 port the FreeRTOS traffic light FSM.

## Files Added in Phase 17.1-17.4

- Hardware documentation and pin map under `docs/hardware/stm32_pcb/`.
- Phase overview and UART validation docs under `docs/embedded/`.
- Documentation-first firmware skeleton under `firmware/stm32_f103_traffic_light/`.
- Phase ledger for continuation by another assistant.

## Status Labels

| Item | Status |
| --- | --- |
| STM32 hardware validation | Pending hardware validation |
| STM32 firmware build | Planned |
| STM32 firmware flash | Not attempted |
| UART Pi-to-STM32 test | Planned |
| FreeRTOS STM32 FSM port | Not started |
| End-to-end AI host demo | Not started |
