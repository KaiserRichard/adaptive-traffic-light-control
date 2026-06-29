# STM32F103 Traffic Light Firmware Skeleton

## Purpose

This folder is the planned firmware location for the STM32F103C8T6 traffic light controller PCB.

It is intentionally documentation-first for Phase 17. No STM32 firmware source, generated HAL files, CubeMX `.ioc`, binary artifacts, or flashing scripts are included yet.

## Status

```text
Documentation skeleton created.
Target MCU identified.
Build/flash toolchain not selected.
No firmware has been built, flashed, or tested.
No buildable STM32 firmware has been created yet.
No flashable binary exists.
Phase 17.2.3 proposes the future CMake scaffold.
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

## Expected Modules

Potential module layout for a later implementation:

```text
src/
    main.c or main.cpp
    board_pins.c
    gpio_outputs.c
    uart_link.c
    protocol.c
    traffic_fsm.c
    status_reporter.c
    watchdog_fallback.c

include/
    board_pins.h
    gpio_outputs.h
    uart_link.h
    protocol.h
    traffic_fsm.h
    status_reporter.h
    watchdog_fallback.h
```

This layout is not committed as source code yet because the build system and HAL/LL/CMSIS choice are still pending.

Detailed design notes:

- [build_scaffold_design.md](build_scaffold_design.md) - future non-flashing build scaffold design.
- [cmake_scaffold_proposal.md](cmake_scaffold_proposal.md) - future CMake scaffold proposal.
- [firmware_architecture.md](firmware_architecture.md) - intended future module boundaries.
- [minimal_build_checklist.md](minimal_build_checklist.md) - gates before the first real compile attempt.

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
- No UART firmware.
- No LED GPIO driver.
- No 7-segment driver.
- No FreeRTOS FSM port.
- No end-to-end AI host demo.

## Verification for This Phase

This folder is complete for the current documentation-first STM32 preparation work if:

```text
[ ] README documents purpose and status.
[ ] toolchain_plan.md describes command-line options.
[ ] toolchain_inspection.md records installed/missing tools.
[ ] build_scaffold_design.md explains the future build scaffold.
[ ] cmake_scaffold_proposal.md explains the future CMake scaffold.
[ ] firmware_architecture.md explains the planned firmware modules.
[ ] minimal_build_checklist.md defines the first compile gates.
[ ] bringup_plan.md documents firmware bring-up sequence.
[ ] src/README.md and include/README.md clarify that source/header files are intentionally absent.
[ ] No build artifacts or generated vendor files are committed.
```
