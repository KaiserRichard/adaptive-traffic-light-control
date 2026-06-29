# Phase 17.2.2 - Minimal STM32 Build Scaffold Design

## Purpose

Design the future STM32F103C8T6 firmware build scaffold before creating a buildable project.

This is a documentation-only design review. It defines the intended folder structure, minimum files, build-system direction, and approval gates for a later non-flashing compile attempt.

## Current Status

```text
Documentation-only scaffold design.
No buildable STM32 firmware has been created.
No CMake project has been created.
No startup file or linker script has been added.
No binary artifact exists.
No hardware has been flashed or validated.
```

## Target MCU

```text
MCU:          STM32F103C8T6
Core:         ARM Cortex-M3
FPU:          None
Logic level:  3.3 V MCU
PCB role:     ATLC traffic light controller
```

## Why Non-Flashing Scaffold Design Comes First

A real STM32 build requires more than `main.c`. The project needs a selected cross-compiler, a startup file, a linker script, device headers, reset/clock assumptions, and a clear policy for vendor-generated files.

Designing this first prevents:

- committing unreviewed CubeMX/CubeIDE generated files.
- inventing startup/linker files without a known source.
- hard-coding PCB pins before hardware validation.
- mixing compile-only work with flashing or board bring-up.
- claiming firmware readiness before the toolchain exists.

## Recommended Folder Structure

Future buildable structure, subject to review:

```text
firmware/stm32_f103_traffic_light/
├── README.md
├── build_scaffold_design.md
├── firmware_architecture.md
├── minimal_build_checklist.md
├── toolchain_inspection.md
├── toolchain_plan.md
├── bringup_plan.md
├── CMakeLists.txt                  # future, not created yet
├── cmake/
│   └── toolchain-arm-none-eabi.cmake # future, not created yet
├── startup/
│   └── startup_stm32f103c8tx.s       # future, source must be documented
├── linker/
│   └── STM32F103C8Tx_FLASH.ld        # future, source must be documented
├── include/
│   ├── README.md
│   ├── board_config.h                # future, after pin review
│   ├── gpio_outputs.h                # future
│   ├── protocol.h                    # future
│   ├── safety.h                      # future
│   ├── seven_segment.h               # future
│   ├── traffic_fsm.h                 # future
│   └── uart_link.h                   # future
└── src/
    ├── README.md
    ├── main.c                        # future
    ├── gpio_outputs.c                # future
    ├── protocol.c                    # future
    ├── safety.c                      # future
    ├── seven_segment.c               # future
    ├── traffic_fsm.c                 # future
    └── uart_link.c                   # future
```

Only the documentation files exist in this phase.

## Future Minimum Build Files

A first real compile attempt will eventually need:

| File / directory | Purpose | Create now? | Notes |
| --- | --- | --- | --- |
| `CMakeLists.txt` | Top-level build definition | No | Wait until toolchain/source policy is approved |
| `cmake/toolchain-arm-none-eabi.cmake` | Tells CMake to use `arm-none-eabi-gcc` | No | Requires installed cross-compiler |
| `startup/startup_stm32f103c8tx.s` | Vector table and reset entry | No | Must come from a traceable ST/CMSIS source |
| `linker/STM32F103C8Tx_FLASH.ld` | Flash/RAM memory map | No | Must match STM32F103C8T6 memory size |
| `include/board_config.h` | Board pin definitions and clock assumptions | No | Depends on pin mapping review |
| `src/main.c` | Minimal firmware entry point | No | Should start as blink or no-op compile test |
| CMSIS device headers | Core/device register definitions | No | Source/version must be documented |
| HAL/LL sources, if selected | Peripheral abstraction | No | Strategy not selected yet |

## Build System Recommendation

Recommended first build path:

```text
CMake + Make first
```

Reason:

- CMake is already installed.
- GNU Make is already installed.
- Ninja is missing.
- `arm-none-eabi-gcc` is missing.
- A Make-compatible CMake workflow is the lowest-friction next review step.

Future compile-only command shape, not active yet:

```bash
cmake -S . -B build -G "Unix Makefiles" \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-arm-none-eabi.cmake
cmake --build build
```

Do not run this until `arm-none-eabi-gcc`, startup files, linker script, and source files exist.

## Files Intentionally Not Created Yet

- `CMakeLists.txt`.
- `cmake/toolchain-arm-none-eabi.cmake`.
- startup assembly file.
- linker script.
- `main.c`.
- module `.c` files.
- module `.h` files.
- CMSIS headers.
- HAL or LL sources.
- FreeRTOS kernel files.
- `.elf`, `.bin`, `.hex`, `.map`, `.o`, `.a`, or build directory.
- STM32CubeIDE project files.
- CubeMX `.ioc` file.

## Dependency on Pin Mapping

The future `board_config.h` must depend on:

- [STM32F103C8T6 pin mapping](../../docs/hardware/stm32_pcb/stm32f103c8t6_pin_mapping.md)
- real PCB review.
- traffic LED lane/color identification.
- 7-segment polarity and current validation.
- UART TX/RX crossing confirmation.

Current pin rows are schematic-observed or marked `Needs verification`. Firmware must not treat them as hardware-validated.

## Dependency on Toolchain Installation

The first real build requires approved installation of:

- `arm-none-eabi-gcc`.
- Arm binutils.
- optionally `arm-none-eabi-gdb`.
- optionally Ninja if the project later chooses Ninja over Make.

OpenOCD, STM32CubeProgrammer, ST-LINK, and real PCB access are not needed for a compile-only build, but are needed later for flashing/debugging.

## Dependency on STM32 Startup / Linker Files

The startup file and linker script are safety-critical build inputs:

- startup file defines the vector table, reset handler, weak interrupt handlers, and C runtime entry.
- linker script defines flash/RAM memory regions and where code/data/stack/heap live.

Do not hand-write these casually. The future source should be one of:

- ST CMSIS device package.
- STM32CubeF1 package.
- a clearly licensed template reviewed against STM32F103C8T6 memory layout.

## What Can Be Reviewed Now

- Folder structure.
- Build-system choice.
- Generated-files policy.
- Future source/module boundaries.
- Whether CMake + Make is acceptable for the MacBook workflow.
- What must be installed later.
- What firmware files should depend on verified hardware data.

## What Must Wait

- Installing `arm-none-eabi-gcc`.
- Creating a buildable CMake project.
- Adding startup/linker files.
- Adding CMSIS/HAL/LL code.
- Creating `main.c`.
- Compiling firmware.
- Flashing firmware.
- Running blink, UART, LED, 7-segment, or FreeRTOS tests.

## Risks

- Committing generated vendor files before deciding the source policy.
- Creating a build scaffold that cannot be reproduced on another machine.
- Selecting pins in `board_config.h` before physical PCB validation.
- Treating compile success as hardware validation.
- Letting early blink scaffolding grow into an unreviewed firmware architecture.

## Next Step

Recommended next safe step:

```text
Phase 17.2.3 - Draft non-buildable CMake scaffold proposal
```

This keeps the project installation-free while allowing review of exact future file names, CMake variables, compiler flags, generated-file policy, and build commands before any real toolchain or hardware dependency is introduced.
