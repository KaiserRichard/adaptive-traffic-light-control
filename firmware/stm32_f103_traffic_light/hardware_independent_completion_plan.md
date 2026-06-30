# Phase 17 Hardware-Independent Firmware and Software Completion Plan

## Purpose

Define what can be completed before the STM32 PCB is available and what must remain blocked by hardware evidence.

This plan follows the Phase 17.2 closure decision while allowing safe software preparation that does not require power, ST-LINK, flashing, UART hardware, GPIO hardware, or a real board.

## Current Status

```text
Phase 17.1 documentation is complete.
Phase 17.2 firmware planning is closed.
Phase 17.3.1 pre-power inspection checklist is complete.
Phase 17.3.2 controlled power rail validation procedure is complete, execution pending.
Phase 17.3.3 ST-LINK attach/read-ID procedure is complete, execution pending.
No physical STM32 hardware is available in this environment.
```

## Can Be Completed Now Without Hardware

| Area | Offline-completable work | Status after this phase |
| --- | --- | --- |
| Portable protocol parser | Parse canonical `PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>` and reject malformed/deprecated formats | Implemented in `firmware/common/` |
| Plan validator | Enforce ESP32-aligned timing bounds and stable reason strings | Implemented in `firmware/common/` |
| FSM model | Represent traffic states, state durations, next-state order, and safe pending-plan boundary | Implemented in `firmware/common/` |
| Status model | Represent and format `STATUS,<plan_id>,<state>,<remaining_seconds>,<health>` | Implemented in `firmware/common/` |
| Host-side tests | Run parser, validator, FSM, ACK/NACK, STATUS, and PC app format tests without hardware | Implemented as Python standard-library scripts |
| STM32 board abstraction stubs | Define future GPIO, UART, task, and port-status module boundaries without HAL dependency | Implemented as honest stubs |
| Documentation cleanup | Update README and ledger so hardware-independent completion is visible but not overclaimed | Completed |

## Must Wait for Hardware

| Area | Why it must wait |
| --- | --- |
| power validation | Requires real PCB, power source, current-limit setup, multimeter, and measured VCC rails |
| ST-LINK read-ID | Requires validated power, SWD probe, cable orientation, and approved tool |
| flashing | Requires ST-LINK attach/read-ID and selected firmware toolchain |
| GPIO electrical validation | Requires powered board and one-output-at-a-time measurement/observation |
| UART electrical validation | Requires board power, UART firmware, host wiring, and raw serial transcript |
| seven-segment validation | Requires display polarity/current measurements and refresh testing |
| FreeRTOS runtime validation | Requires buildable STM32 firmware and working board support |
| AI host to STM32 hardware demo | Requires validated power, SWD, firmware, UART, GPIO, and PLAN/ACK behavior |

## Firmware Boundary Decision

The new `firmware/common/` layer is portable application logic only.

It does not:

- initialize STM32 clocks.
- configure GPIO.
- configure USART1.
- start FreeRTOS.
- drive traffic LEDs.
- drive the seven-segment display.
- flash or erase hardware.

The STM32 folder receives board-layer stubs only. They define future responsibilities but do not touch hardware registers or HAL calls.

## ESP32 Compatibility Decision

The ESP32 FreeRTOS firmware remains the behavior reference.

This phase does not:

- move ESP32 files.
- refactor ESP32 source.
- change ESP32 PlatformIO configuration.
- replace the ESP32 parser/FSM with common code.

Future migration can be done only after tests protect the behavior and the user approves the shared-code boundary.

## Timing Policy

The common validator mirrors the current ESP32 timing policy:

```text
green_a: 10..45 seconds
green_b: 10..45 seconds
yellow:  exactly 3 seconds
all_red: exactly 1 second
```

If the ESP32 policy changes, update the common layer and tests in the same commit.

## Validation Commands

Run from the repository root:

```bash
python3 firmware/common/tests/test_protocol_parser.py
python3 firmware/common/tests/test_plan_validator.py
python3 firmware/common/tests/test_fsm_sequence.py
python3 pc_app/tests/test_plan_protocol_format.py
```

These tests prove host-side software behavior only. They do not prove STM32 hardware behavior.

## Next Hardware Step

When the PCB is available, continue with:

```text
Phase 17.3.2 - Controlled Power Rail Validation
```

Do not proceed to ST-LINK read-ID, blink, UART, FreeRTOS, or AI host integration until each previous hardware gate passes.
