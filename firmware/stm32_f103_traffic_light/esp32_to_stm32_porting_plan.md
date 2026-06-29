# Phase 17.2.4 - ESP32 to STM32 Porting Plan

## Purpose

Plan how the ATLC embedded controller can evolve from the ESP32 FreeRTOS prototype toward a future STM32F103C8T6 PCB implementation while keeping both board targets useful.

This plan is documentation-only. It does not move files, create a common firmware package, or implement STM32 code.

## Current Status

```text
ESP32 FreeRTOS firmware exists as the behavior reference.
STM32 firmware folder is documentation-first.
No STM32 FreeRTOS port has started.
No common/ source migration has started.
No STM32 buildable project exists.
```

## ESP32 Role as Rapid FreeRTOS Testbed

The ESP32 should remain the rapid validation target for application behavior:

- FreeRTOS task decomposition.
- queue-based message flow.
- PLAN parser behavior.
- ACK/NACK behavior.
- safe pending-plan application.
- host timeout and fallback behavior.
- STATUS and DIAG telemetry concepts.

The goal is not to delete ESP32. The goal is to use ESP32 and STM32 as two board targets sharing one application architecture.

## STM32 Role as PCB Target

The STM32F103C8T6 PCB is the intended custom controller target. It should eventually own:

- deterministic local traffic-light FSM execution.
- verified PCB GPIO outputs.
- USART link to the Raspberry Pi AI host.
- ACK/NACK/STATUS responses.
- watchdog/fallback behavior.
- 7-segment display support if the electrical design is validated.

STM32 hardware work must proceed only after power, SWD, pin mapping, and UART validation are planned and executed safely.

## What Can Be Reused Conceptually

The following ESP32 ideas should be reused:

- `RawMessage` as a bounded line-oriented input object.
- `SignalPlan` as the validated timing plan.
- parser task separated from UART receive.
- FSM task as the owner of active/pending plans.
- safe plan application at a cycle boundary.
- default/fallback plan.
- host timeout health state.
- machine-readable `ACK`, `NACK`, `STATUS`, and `DIAG` lines.
- timer-driven status reporting that does not calculate FSM behavior inside the timer callback.

## What Can Be Reused as Source Later

Candidate source-level reuse, after review:

| ESP32 area | Reuse potential | Required cleanup |
| --- | --- | --- |
| message structs | High | remove unnecessary Arduino include if possible |
| protocol parsing | Medium to high | split parser/validator from Serial logging and ACK/NACK transport |
| FSM state helpers | Medium to high | split pure state logic from GPIO writes |
| default plan and timing limits | High | move limits to shared config or protocol policy |
| status formatting concept | Medium | separate snapshot logic from ESP32 critical sections |

No source reuse should happen until the common boundary and build/test strategy are approved.

## What Must Be Rewritten for STM32

STM32-specific rewrites will be required for:

- startup code and linker script.
- clock initialization.
- GPIO initialization and writes.
- USART initialization and receive/transmit.
- interrupt handlers.
- FreeRTOS port configuration.
- critical sections if shared status snapshots are used.
- diagnostic data source.
- board-specific pin and peripheral constants.
- 7-segment refresh driver.

## ESP32-Specific APIs to Isolate

Current ESP32 code uses these APIs that should not leak into common logic:

- `Arduino.h`.
- `Serial.begin`, `Serial.print`, `Serial.println`, `Serial.available`, `Serial.read`.
- `Serial.onReceive`.
- `pinMode`.
- `digitalWrite`.
- `delay`.
- PlatformIO Arduino framework.
- ESP32 `portMUX_TYPE` critical section behavior.

## STM32 HAL / LL / CMSIS Replacement Areas

Future STM32 implementation must decide whether each area uses CMSIS-only, STM32 HAL, STM32 LL, or a mixed policy:

| Area | STM32 replacement decision needed |
| --- | --- |
| GPIO | HAL GPIO, LL GPIO, or CMSIS register access |
| UART | HAL UART, LL USART, interrupt-driven register code, or later DMA |
| timebase | SysTick, FreeRTOS tick, hardware timer, or HAL tick |
| critical sections | FreeRTOS critical sections / Cortex-M interrupt mask policy |
| diagnostics | FreeRTOS heap/stack APIs and optional telemetry formatting |
| display refresh | timer-driven multiplex or FSM service tick |

Do not mix random vendor files without a documented source/version policy.

## FreeRTOS API Similarity

The high-level FreeRTOS APIs are conceptually portable:

- `xTaskCreate`.
- `xQueueCreate`.
- `xQueueSendToBack`.
- `xQueueReceive`.
- `vTaskDelay`.
- software timers.
- task notifications.

The exact configuration is not portable without review:

- heap implementation.
- stack sizes.
- tick rate.
- interrupt priority rules.
- timer task priority/stack.
- queue memory usage.
- critical section behavior.

## Board Configuration Strategy

Future STM32 `board_config.h` should own only board-level facts and validation status:

- MCU target name.
- UART instance and pins after validation.
- traffic LED pins after lane/color mapping.
- status LED pin and polarity after validation.
- 7-segment segment/digit pins after polarity/current review.
- default baud rate.
- safe timing constants if they are board-specific.

The application should not hard-code STM32 pins outside the board layer.

## Protocol Compatibility Requirement

The STM32 port must remain compatible with the existing canonical PLAN concept:

```text
PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>
ACK,<plan_id>
NACK,<plan_id>,<reason>
STATUS,<plan_id>,<state>,<remaining_seconds>,<health>
```

Do not introduce a second PLAN format during the port. The Raspberry Pi AI host and ESP32 testbed should remain aligned with the same message contract.

## UART Bring-Up Path

Recommended future path:

1. Validate STM32 board power and SWD.
2. Build a minimal non-flashing or blink firmware only after the toolchain is approved.
3. Implement a minimal UART echo or line receiver.
4. Test UART with USB-UART adapter or Raspberry Pi loopback.
5. Add protocol parsing only after byte-level UART is stable.
6. Add ACK/NACK responses.
7. Add STATUS only after FSM state ownership is in place.

Do not start with full AI host integration.

## GPIO Bring-Up Path

Recommended future path:

1. Keep all outputs safe/off during reset/init.
2. Test PC13 status LED only after polarity is verified.
3. Test traffic LED pins one at a time.
4. Record physical lane/color mapping.
5. Only then bind FSM symbolic states to STM32 output pins.

Do not port `applyTrafficOutputs()` directly until STM32 pin ownership is verified.

## Seven-Segment Bring-Up Path

Recommended future path:

1. Confirm common-anode/common-cathode behavior.
2. Confirm segment resistor values.
3. Confirm whether PA3/PA4 can safely drive digit common pins.
4. Test one segment and one digit at a time.
5. Select a conservative multiplex refresh rate.
6. Keep display code separate from FSM timing ownership.

## Testing Strategy Across Boards

ESP32 testbed:

- continue testing FreeRTOS task behavior and protocol flow.
- use serial logs to validate parser/FSM behavior.
- preserve the canonical message contract.

Host-side tests:

- test parser and validation logic without hardware where possible.
- test PLAN/ACK/NACK/STATUS formatting.
- test boundary cases for timing ranges.

STM32 target:

- start with compile-only.
- then power/SWD/blink.
- then GPIO.
- then UART.
- then parser.
- then FSM.
- then AI host integration.

## Risk Table

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Copying ESP32 Arduino code directly | STM32 build will fail or hide wrong assumptions | define board support interfaces first |
| Moving too much into common/ too early | ESP32 may regress and STM32 may still not build | use small reviewable migration steps |
| Changing PLAN format | breaks Phase 16 AI host compatibility | keep canonical protocol unchanged |
| Porting GPIO before pin validation | wrong LEDs or unsafe current paths | wait for bring-up evidence |
| Treating ESP32 timing as STM32 timing | stack/RAM/tick behavior may differ | remeasure on STM32 after real port |
| Adding FreeRTOS before basic IO | harder hardware debugging | validate power, SWD, blink, GPIO, UART first |

## Step-by-Step Future Migration Plan

1. Approve the future `firmware/common/` boundary.
2. Decide C/C++ strategy for shared code.
3. Add host-side protocol tests before moving parser logic.
4. Extract message and protocol definitions into a small common candidate.
5. Keep ESP32 build green after each migration.
6. Add STM32 board support skeleton only after the toolchain is approved.
7. Add compile-only STM32 build.
8. Validate STM32 power and SWD.
9. Bring up STM32 GPIO and UART.
10. Port FSM only after board IO and UART basics are proven.

## What Is Intentionally Not Done Yet

- No source files moved.
- No `firmware/common/` created.
- No ESP32 refactor.
- No STM32 source implementation.
- No STM32 build system.
- No FreeRTOS kernel integration for STM32.
- No UART hardware driver.
- No GPIO hardware driver.
- No hardware validation.

This plan keeps the ESP32 useful while preparing the STM32 target without pretending the hardware-dependent port is complete.
