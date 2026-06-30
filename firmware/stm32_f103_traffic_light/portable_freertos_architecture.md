# Phase 17.2.4 - Portable FreeRTOS Architecture Proposal

## Purpose

Document how the existing ESP32 FreeRTOS traffic light controller can guide a future STM32F103C8T6 firmware architecture without moving source files yet.

This document identifies reusable application concepts, board-specific code, and the proposed future split between shared firmware logic and board support layers.

## Current Status

```text
Documentation-only architecture proposal.
ESP32 firmware was reviewed as the reference implementation.
No ESP32 source files were moved or refactored.
An initial firmware/common/ layer was added later as hardware-independent C.
No ESP32-to-common source migration has happened.
No STM32 FreeRTOS port was implemented.
No buildable STM32 firmware was created.
No hardware validation was performed.
```

## Why ESP32 Phase 15 Is Still Valuable

The ESP32 FreeRTOS controller is the current behavior reference for the embedded side of ATLC. It already demonstrates the controller boundary that the STM32 target should preserve:

```text
UART input
    -> RawMessage queue
    -> PLAN parser and validator
    -> SignalPlan queue
    -> local traffic light FSM
    -> GPIO outputs
    -> ACK / NACK / STATUS / DIAG telemetry
```

The ESP32 remains useful because it is a fast FreeRTOS validation testbed. It can be used to review task ownership, queue flow, protocol behavior, safe plan application, and watchdog/fallback logic before those ideas are ported to a more hardware-constrained STM32 PCB.

## Why STM32 Should Not Start From Zero

Starting the STM32 firmware from zero would risk losing working design decisions from the ESP32 prototype:

- implemented separation between UART receive, parsing, and FSM execution.
- bounded raw message buffers.
- documented plan timing ranges.
- `ACK` / `NACK` response behavior.
- safe plan application only at a full-cycle boundary.
- host timeout and fallback concept.
- periodic `STATUS` and diagnostic telemetry concept.

The STM32 implementation should reuse the architecture, not blindly copy ESP32-specific APIs.

## Target Architecture Overview

Future target concept:

```text
Raspberry Pi / AI host
    sends canonical PLAN lines

STM32 board transport layer
    receives newline-terminated UART bytes

Portable protocol layer
    parses and validates PLAN messages

Portable controller layer
    queues accepted SignalPlan objects
    applies pending plans only at safe FSM boundaries
    detects host timeout

STM32 board output layer
    drives verified traffic LED pins
    drives 7-segment display only after electrical validation
    reports ACK / NACK / STATUS / DIAG through UART
```

## Recommended Future firmware/common Structure

Initial common layer now exists as:

```text
firmware/
├── common/
│   ├── include/
│   ├── src/
│   └── tests/
├── esp32_freertos_traffic_light/
│   └── ESP32 board-specific implementation
└── stm32_f103_traffic_light/
    └── STM32 PCB board-specific implementation
```

No ESP32 source migration has happened. A larger folder split under `common/` should wait until the shared-code boundary is proven useful.

## What Should Go Into common/

Future `firmware/common/` candidates:

| Area | Candidate content | Reason |
| --- | --- | --- |
| `messages/` | `RawMessage`, `ParsedPlanFields`, `SignalPlan`, `TrafficState`, `ControllerStatus` | Shared data contracts between parser, FSM, and telemetry |
| `protocol/` | PLAN command detection, field parsing, range validation, reason codes | Must remain compatible with the Phase 16 AI host PLAN generation |
| `core/` | state names, next-state function, state duration calculation, default plan policy | Mostly independent of ESP32 or STM32 peripheral APIs |
| `services/` | status snapshot concept, fallback/health concepts | Useful if separated from board-specific logging and timer implementation |
| `tasks/` | task ownership pattern and queue flow documentation | Reusable architecture, but not necessarily direct source reuse |

## What Should Remain ESP32-Specific

The following should remain inside `firmware/esp32_freertos_traffic_light/` unless deliberately abstracted later:

- `platformio.ini`.
- Arduino framework startup with `setup()` / `loop()`.
- `Serial.begin`, `Serial.print`, and `Serial.onReceive`.
- ESP32 GPIO pin numbers in `app_config.h`.
- `pinMode()` and `digitalWrite()` output control.
- ESP32 Arduino FreeRTOS critical section details such as `portMUX_TYPE`.
- PlatformIO build and upload workflow.
- ESP32 diagnostics details such as heap and task stack reporting implementation.
- ESP32-specific task stack sizes and priorities until remeasured on STM32.

## What Should Become STM32-Specific

The STM32 board layer should eventually own:

- USART peripheral selection and initialization, likely USART1 on PA9/PA10 if verified.
- GPIO initialization for verified traffic LED pins.
- PC13 status LED behavior after polarity validation.
- 7-segment segment and digit control after current/polarity validation.
- board clock setup.
- FreeRTOS port integration for Cortex-M3.
- interrupt handlers and NVIC configuration.
- startup file, linker script, and CMSIS/HAL/LL support.
- SWD/debug constraints.
- board-specific `board_config.h` pin mapping and validation notes.

## Application Logic vs Board Support Layer

Future firmware should separate two categories:

Application logic:

- PLAN data structures.
- parser and validator.
- traffic FSM state sequence.
- pending/active plan ownership.
- safe boundary plan apply rule.
- fallback and health state policy.

Board support layer:

- UART byte transport.
- GPIO writes.
- timer/tick source.
- critical section implementation.
- logging transport.
- 7-segment multiplexing.
- reset/clock/startup details.

This separation keeps the ATLC behavior portable while allowing ESP32 and STM32 to use different hardware APIs.

## FreeRTOS Task Portability

The ESP32 task model is a strong reference:

```text
TaskUARTReceive
    reads UART bytes and publishes RawMessage

TaskPlanParser
    parses RawMessage, validates SignalPlan, sends ACK/NACK

TaskTrafficFSM
    owns active plan, pending plan, current state, fallback behavior

TaskStatusReporter
    publishes STATUS from a snapshot

Diagnostics timer/service
    publishes DIAG when supported
```

The ownership model is portable. The exact task stack sizes, priorities, ISR callback mechanism, and timer/critical-section code are board-specific and must be reviewed for STM32.

## Queue / Message Portability

Reusable concept:

```text
rawMessageQueue: RawMessage from UART task to parser task
planQueue: SignalPlan from parser task to FSM task
```

Future STM32 concerns:

- queue depth may need adjustment for lower RAM.
- message sizes must remain bounded.
- blocking times must be reviewed so UART responses do not disturb FSM timing.
- queue overflow handling must remain explicit.

## Protocol Portability

The canonical protocol relationship should remain:

```text
AI host -> controller:
    PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>

controller -> AI host:
    ACK,<plan_id>
    NACK,<plan_id>,<reason>
    STATUS,<plan_id>,<state>,<remaining_seconds>,<health>
```

The STM32 port must not introduce a second PLAN format. Protocol compatibility with the Phase 16 AI host must be preserved.

Portable candidates:

- command prefix detection.
- field count validation.
- numeric conversion policy.
- timing range validation.
- stable reason strings such as `GREEN_A_OUT_OF_RANGE`.

Board-specific pieces:

- UART send function.
- UART receive buffering.
- line termination handling under real UART interrupt/DMA/polling strategy.

## FSM Portability

The state sequence is portable:

```text
STATE_A_GREEN
STATE_A_YELLOW
STATE_ALL_RED_AFTER_A
STATE_B_GREEN
STATE_B_YELLOW
STATE_ALL_RED_AFTER_B
```

The rule that a pending plan is applied only when the FSM returns to `STATE_A_GREEN` should remain. This prevents frame-by-frame AI decisions from directly disrupting red/yellow/all-red ordering.

Board-specific pieces:

- mapping FSM states to actual STM32 GPIO pins.
- active-high vs active-low output behavior.
- safe default state during reset.
- verified lane/color ownership.

## Logging and Diagnostics Portability

The concept is portable:

- one-line machine-readable logs.
- nonblocking/bounded telemetry.
- `STATUS` for controller state.
- `DIAG` for runtime health when supported.

The implementation is board-specific:

- ESP32 uses Arduino `Serial` and a FreeRTOS mutex.
- ESP32 status snapshot uses `portMUX_TYPE`.
- diagnostics use ESP32/FreeRTOS heap and high-water mark APIs.
- STM32 may need a different critical section and may initially skip DIAG until UART and scheduler behavior are stable.

## UART Abstraction Idea

Future application code should not call `Serial.print()` or STM32 HAL UART APIs directly.

Proposed abstraction concept:

```text
uart_transport_send_line(const char *line)
uart_transport_try_receive_line(RawMessage *message)
```

ESP32 implementation can wrap Arduino `Serial`.

STM32 implementation can later wrap USART polling, interrupt-driven receive, or DMA receive after bring-up.

## GPIO Output Abstraction Idea

Future FSM code should request symbolic output states instead of writing board pins directly.

Proposed abstraction concept:

```text
traffic_outputs_apply(TrafficState state)
traffic_outputs_all_off()
traffic_outputs_init_safe()
```

ESP32 implementation can wrap `pinMode()` and `digitalWrite()`.

STM32 implementation can later wrap HAL/LL/CMSIS GPIO writes after pin mapping is verified.

## Seven-Segment Display Abstraction Idea

The dual 7-segment display should stay outside the core FSM. It should subscribe to current countdown/status information rather than own traffic timing.

Future abstraction concept:

```text
seven_segment_init_safe()
seven_segment_set_number(uint8_t value)
seven_segment_service_tick()
seven_segment_blank()
```

STM32 implementation must wait for:

- common-anode/common-cathode verification.
- segment current calculation.
- PA3/PA4 digit drive review.
- refresh timing decision.

## Risks of Copying ESP32 Code Directly Into STM32

- Arduino APIs do not exist in a bare STM32 CMake project.
- ESP32 pin numbers do not match STM32 PCB pins.
- ESP32 is more capable than STM32F103C8T6 in RAM/flash/CPU headroom.
- ESP32 dual-core critical section code does not directly map to Cortex-M3.
- `Serial.onReceive()` has no direct equivalent unless a specific STM32 UART driver strategy is chosen.
- PlatformIO project assumptions do not match the proposed STM32 CMake/Make path.
- Directly porting output code before pin validation could drive the wrong LED or overload GPIO.

## What Should Not Be Moved Yet

Do not move or refactor in this phase:

- ESP32 source files.
- ESP32 headers.
- ESP32 PlatformIO project files.
- protocol source into `firmware/common/`.
- FSM source into `firmware/common/`.
- STM32 `.c` / `.h` implementation files.
- FreeRTOS kernel files.
- CMake build files.

## Review Gates Before Real Refactor

Before creating `firmware/common/` or moving source:

```text
[ ] Approve the common/ folder boundary.
[ ] Decide C vs C++ strategy for shared code.
[ ] Decide whether ESP32 remains Arduino C++ while STM32 starts in C.
[ ] Write host-side tests for protocol parsing where possible.
[ ] Confirm canonical PLAN/ACK/NACK/STATUS message contract.
[ ] Identify board support interfaces for UART, GPIO, time, logging, and critical sections.
[ ] Confirm STM32 pin mapping and output polarity remain pending until hardware validation.
[ ] Create a rollback-friendly commit before any source migration.
```

## Recommended Next Step

Recommended next Phase 17 step:

```text
Execute Phase 17.3.2 - Controlled Power Rail Validation before any firmware/common migration.
```

The portable `firmware/common/` idea should remain future optional work unless the user explicitly approves source movement and a verification plan for both ESP32 and STM32 targets.
