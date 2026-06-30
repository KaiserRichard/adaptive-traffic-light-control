# STM32F103 Traffic Light Firmware Skeleton

## Purpose

This folder is the planned firmware location for the STM32F103C8T6 traffic light controller PCB.

It now contains hardware-independent board-layer scaffolding only. No generated HAL files, CubeMX `.ioc`, startup file, linker script, binary artifacts, or flashing scripts are included.

## Status

```text
Documentation skeleton created.
Target MCU identified.
Build/flash toolchain not selected.
No firmware has been built, flashed, or tested.
No buildable STM32 firmware has been created yet.
No flashable binary exists.
Hardware-independent STM32 scaffold stubs have been added.
Portable common protocol/FSM/status logic exists under firmware/common/.
Phase 17.2.3 proposes the future CMake scaffold.
Phase 17.2.4 closes the firmware planning phase and moves the recommended next step to Phase 17.3 hardware bring-up.
```

## Target MCU

```text
STM32F103C8T6
Core: ARM Cortex-M3
Planned board role: ATLC traffic light controller PCB
```

## Planned Firmware Architecture

Future STM32 firmware should preserve the controller boundary already proven in the ESP32 prototype:

```text
USART1 receive
    -> raw message buffer / queue
    -> PLAN parser and validator
    -> validated plan queue
    -> traffic light FSM task
    -> GPIO LED / 7-segment drivers
    -> ACK / NACK / STATUS responses
```

## Current Scaffold Modules

Current hardware-independent scaffold layout:

```text
src/
    main.c
    board_gpio.c
    board_uart.c
    app_tasks.c
    stm32_port_status.c

include/
    board_config.h
    board_gpio.h
    board_uart.h
    app_tasks.h
    stm32_port_status.h
```

These files are not a real STM32 hardware implementation. They intentionally avoid STM32 HAL, CMSIS, LL, startup code, linker scripts, and FreeRTOS includes. Hardware-dependent functions return pending/not-implemented status.

The portable hardware-independent application logic is in [../common/README.md](../common/README.md).

Detailed design notes:

- [hardware_independent_completion_plan.md](hardware_independent_completion_plan.md) - what was completed before hardware and what remains gated.
- [phase_17_2_closure_decision.md](phase_17_2_closure_decision.md) - Phase 17.2 streamlining and closure decision.
- [build_scaffold_design.md](build_scaffold_design.md) - future non-flashing build scaffold design.
- [cmake_scaffold_proposal.md](cmake_scaffold_proposal.md) - future CMake scaffold proposal.
- [firmware_architecture.md](firmware_architecture.md) - intended future module boundaries.
- [minimal_build_checklist.md](minimal_build_checklist.md) - gates before the first real compile attempt.
- [portable_freertos_architecture.md](portable_freertos_architecture.md) - optional future portable FreeRTOS reference based on the ESP32 implementation.
- [esp32_to_stm32_porting_plan.md](esp32_to_stm32_porting_plan.md) - optional future ESP32-to-STM32 migration reference.

## GPIO Dependency on Pin Mapping

Firmware must not hard-code final LED or 7-segment ownership until [../../docs/hardware/stm32_pcb/stm32f103c8t6_pin_mapping.md](../../docs/hardware/stm32_pcb/stm32f103c8t6_pin_mapping.md) is reviewed.

Known schematic-observed output groups:

- Traffic LEDs: PB9, PB8, PB7, PB6, PB5, PB3.
- 7-segment segments: PA8, PB15, PB14, PB13, PB12, PA12, PA11.
- 7-segment digit/common control: PA3, PA4.
- Status LED candidate: PC13.

Traffic direction and color ownership remain TBD.

## UART Protocol Dependency

The STM32 UART link should stay compatible with the existing ATLC protocol concept:

```text
Raspberry Pi AI host -> STM32: PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>
STM32 -> Raspberry Pi AI host: ACK,<plan_id>
STM32 -> Raspberry Pi AI host: NACK,<plan_id>,<reason>
STM32 -> Raspberry Pi AI host: STATUS,<plan_id>,<state>,<remaining_seconds>,<health>
```

USART1 pins observed in schematic images:

- PA9 as `TX1`.
- PA10 as `RX1`.

TX/RX crossing with Raspberry Pi is not validated yet.

## FreeRTOS Porting Plan

The ESP32 FreeRTOS firmware remains the reference testbed for task ownership, queue flow, protocol parsing, safe FSM behavior, watchdog/fallback behavior, and STATUS/DIAG telemetry.

The STM32 target should receive a board-specific implementation later. A small hardware-independent `firmware/common/` layer now exists for protocol parsing, plan validation, FSM state progression, and status formatting.

No ESP32 refactor or STM32 FreeRTOS implementation has been done yet.

Phase 17.2 is now closed as a documentation/planning phase. The next real track is Phase 17.3 hardware bring-up, beginning with pre-power inspection and measurement. `firmware/common/` migration is future optional work, not a blocker for Phase 17.3.

FreeRTOS should be added only after:

1. Command-line build path is selected.
2. Minimal blink builds without STM32CubeIDE.
3. SWD programming/debug is verified on real hardware.
4. UART loopback or controlled host test is validated.
5. GPIO pin ownership is reviewed.

Do not start the FreeRTOS traffic light FSM port in this phase.

## Build/Flash Tools Pending

Candidate command-line tool options are documented in [toolchain_plan.md](toolchain_plan.md).

The current MacBook toolchain state is documented in [toolchain_inspection.md](toolchain_inspection.md).

Build-scaffold planning is documented in [build_scaffold_design.md](build_scaffold_design.md).

CMake scaffold planning is documented in [cmake_scaffold_proposal.md](cmake_scaffold_proposal.md).

Current phase requirements:

```text
STM32CubeIDE required now? No
ST-LINK required now? No
Real PCB required now? No
```

## What Is Intentionally Not Implemented Yet

- No STM32CubeIDE project.
- No CubeMX `.ioc` file.
- No generated HAL project.
- No buildable CMake project.
- No CMake toolchain file.
- No startup files.
- No linker script.
- No `.elf`, `.bin`, or `.hex` artifacts.
- No flashing scripts.
- No real STM32 UART firmware.
- No real STM32 LED GPIO driver.
- No real STM32 7-segment driver.
- No ESP32-to-common source migration.
- No ESP32 firmware refactor.
- No FreeRTOS FSM port.
- No end-to-end AI host demo.

## Verification for This Phase

This folder is complete for the current documentation-first STM32 preparation work if:

```text
[ ] README documents purpose and status.
[ ] phase_17_2_closure_decision.md records the closure decision and next step.
[ ] toolchain_plan.md describes command-line options.
[ ] toolchain_inspection.md records installed/missing tools.
[ ] build_scaffold_design.md explains the future build scaffold.
[ ] cmake_scaffold_proposal.md explains the future CMake scaffold.
[ ] firmware_architecture.md explains the planned firmware modules.
[ ] minimal_build_checklist.md defines the first compile gates.
[ ] portable_freertos_architecture.md explains future common/ and board-layer boundaries.
[ ] esp32_to_stm32_porting_plan.md explains the ESP32-to-STM32 migration path.
[ ] bringup_plan.md documents firmware bring-up sequence.
[ ] hardware_independent_completion_plan.md separates offline-complete work from hardware-gated work.
[ ] src/README.md and include/README.md clarify that source/header files are scaffolds only.
[ ] No build artifacts or generated vendor files are committed.
```
