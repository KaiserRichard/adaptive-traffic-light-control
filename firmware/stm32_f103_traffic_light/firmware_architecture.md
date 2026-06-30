# STM32 Firmware Architecture Note

## Purpose

Document the intended STM32 firmware module architecture before implementation.

This file describes future modules and ownership boundaries. It does not implement firmware and does not create buildable source or header files.

## Current Status

```text
Architecture documented.
Hardware-independent STM32 scaffold .c/.h files created later.
No buildable firmware created.
No FreeRTOS port started.
No UART firmware implemented.
No hardware validation completed.
```

The scaffold files are host-safe stubs only. They do not configure STM32 peripherals or prove hardware behavior.

## Future Firmware Layers

Planned layers:

```text
Application behavior
    traffic_fsm
    safety/fallback
    protocol message handling

Board services
    gpio_outputs
    seven_segment
    uart_link
    timers/timebase

MCU support
    startup file
    linker script
    CMSIS/HAL/LL support
    board_config
```

Early bring-up should use a simple loop. FreeRTOS should be added later only after the board, build, and basic IO are validated.

## Planned Source Files

| Future file | Responsibility | Create now? |
| --- | --- | --- |
| `src/main.c` | Initialize system, GPIO/UART/timers, then run simple bring-up loop or start scheduler later | No |
| `src/gpio_outputs.c` | Control traffic light LEDs and status LED through a board abstraction | No |
| `src/seven_segment.c` | Control dual 7-segment display after current/polarity validation | No |
| `src/uart_link.c` | Handle byte-level UART send/receive and buffering | No |
| `src/protocol.c` | Parse `PLAN` messages and format `ACK`/`NACK`/`STATUS` messages | No |
| `src/traffic_fsm.c` | Execute safe traffic-light state transitions | No |
| `src/safety.c` | Handle timeout, invalid plan, fallback mode, and health state | No |

## Planned Header Files

| Future file | Responsibility | Create now? |
| --- | --- | --- |
| `include/board_config.h` | Board-level pin map, UART instance, baud rate, timing constants, validation notes | No |
| `include/gpio_outputs.h` | GPIO output API for LEDs/status LED | No |
| `include/seven_segment.h` | 7-segment display API | No |
| `include/uart_link.h` | UART byte/line transport API | No |
| `include/protocol.h` | Protocol structures and parse/format functions | No |
| `include/traffic_fsm.h` | FSM states, plan structures, and transition API | No |
| `include/safety.h` | Watchdog/fallback API and health states | No |

## Hardware Abstraction Idea

STM32 hardware details should be isolated behind small modules:

- `board_config.h` owns pin names and board constants.
- `gpio_outputs` owns traffic/status LED writes.
- `seven_segment` owns display multiplexing only after electrical validation.
- `uart_link` owns USART byte transport.
- `protocol`, `traffic_fsm`, and `safety` should avoid direct register or HAL calls where practical.

This keeps host-side tests possible for protocol/FSM logic later.

## GPIO Output Module

Future `gpio_outputs.c/.h` should:

- initialize traffic light GPIO outputs.
- initialize PC13 status LED only after active polarity is confirmed.
- expose named functions such as `traffic_outputs_apply(...)`.
- avoid hard-coding final A/B lane ownership until the pin map is reviewed.
- default outputs to a safe state during initialization.

Do not implement this module until the pin map and output polarity are reviewed.

## UART Module

Future `uart_link.c/.h` should:

- configure USART1 on PA9/PA10 if confirmed.
- use 115200 baud, 8N1 unless a later decision changes this.
- receive newline-terminated ASCII messages.
- enforce bounded line buffers.
- provide non-blocking or timeout-bounded send behavior.

Do not implement this module until the build stack and UART test approach are reviewed.

## Protocol Parser Module

Future `protocol.c/.h` should preserve the ATLC message concept:

```text
PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>
ACK,<plan_id>
NACK,<plan_id>,<reason>
STATUS,<plan_id>,<state>,<remaining_seconds>,<health>
```

The parser should:

- reject malformed messages.
- reject out-of-range timing values.
- keep validation reason strings stable enough for host-side debugging.
- be testable without STM32 hardware where practical.

## Traffic FSM Module

Future `traffic_fsm.c/.h` should own:

- active signal plan.
- pending signal plan.
- safe cycle-boundary plan application.
- state timing.
- output state requested from `gpio_outputs`.

Planned states remain aligned with the ESP32 prototype:

```text
STATE_A_GREEN
STATE_A_YELLOW
STATE_ALL_RED_AFTER_A
STATE_B_GREEN
STATE_B_YELLOW
STATE_ALL_RED_AFTER_B
```

Do not port the FSM in Phase 17.2.2.

## Safety / Fallback Module

Future `safety.c/.h` should own:

- host timeout tracking.
- fallback plan selection.
- health status such as `OK` and `HOST_TIMEOUT`.
- invalid plan handling policy.

The safety module should not depend on the Raspberry Pi being continuously healthy.

## Display Module

Future `seven_segment.c/.h` should:

- remain disabled until display polarity and current are verified.
- support multiplexing only after PA3/PA4 drive method is reviewed.
- avoid high-current continuous segment patterns during early tests.
- keep countdown display separate from FSM state ownership.

## Main Loop / FreeRTOS Transition Plan

Recommended bring-up sequence:

1. Minimal `main.c` with system init and a simple loop.
2. PC13 blink only after board/pin validation.
3. Single GPIO output smoke tests.
4. UART echo or line receive test.
5. Protocol parser smoke test.
6. Traffic FSM in a simple cooperative loop.
7. FreeRTOS task split only after basic behavior is stable.

Future FreeRTOS ownership can mirror the ESP32 concept:

```text
UART receive task
    -> raw message queue
Plan parser task
    -> validated plan queue
Traffic FSM task
    -> GPIO/display outputs
Status reporter
    -> UART STATUS output
Safety/watchdog
    -> fallback behavior
```

## What Is Intentionally Not Implemented Yet

- No `.c` implementation files.
- No `.h` API headers.
- No `board_config.h`.
- No CMake build file.
- No startup file.
- No linker script.
- No CMSIS/HAL/LL source.
- No UART driver.
- No protocol parser.
- No traffic FSM.
- No FreeRTOS tasks.
- No binary artifacts.

The architecture is intentionally documented first so implementation can stay small, testable, and tied to verified hardware facts.
