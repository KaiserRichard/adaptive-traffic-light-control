# Phase 17.2.3 - STM32 CMake Scaffold Proposal

## Purpose

Propose the future STM32 CMake build scaffold for `firmware/stm32_f103_traffic_light/` without creating a buildable firmware project yet.

This document explains what future files should contain, why they are needed, and what must be reviewed before they are created.

## Current Status

```text
Proposal-only.
No CMake project has been created.
No startup file has been added.
No linker script has been added.
No CMSIS/HAL/LL vendor files have been added.
No STM32 firmware has been compiled.
No hardware has been flashed or validated.
```

## Why This Is Proposal-Only

The repository is not ready for a real STM32 build scaffold because:

- `arm-none-eabi-gcc` is not installed.
- startup file source and license are not selected.
- linker script source and memory layout are not reviewed.
- CMSIS/HAL/LL strategy is not selected.
- generated-files policy is not finalized.
- board pin mapping remains pending hardware validation.

Creating real build files now would make the repository look more complete than it is. This phase keeps the work reviewable without fake completeness.

## Target MCU Assumptions

```text
Target MCU:          STM32F103C8T6
Core:                ARM Cortex-M3
FPU:                 None
Nominal logic level: 3.3 V
Role:                deterministic traffic-light controller
```

These are target assumptions for planning only. They do not prove PCB hardware validation.

## Recommended Future CMake Structure

Recommended future build strategy:

```text
CMake + Unix Makefiles + arm-none-eabi-gcc
```

Reason:

- CMake is already installed.
- GNU Make is already installed.
- Ninja is not installed.
- `arm-none-eabi-gcc` is not installed yet but is the recommended future cross-compiler.
- Make-compatible CMake is the lowest-friction first STM32 compile path on this MacBook.

## Future File Tree

Proposed future tree:

```text
firmware/stm32_f103_traffic_light/
├── CMakeLists.txt                         # future
├── cmake/
│   └── toolchain-arm-none-eabi.cmake      # future
├── startup/
│   └── startup_stm32f103c8tx.s            # future, source must be verified
├── linker/
│   └── STM32F103C8Tx_FLASH.ld             # future, memory layout must be verified
├── include/
│   ├── board_config.h                     # future, after pin map review
│   ├── gpio_outputs.h                     # future
│   ├── uart_link.h                        # future
│   ├── protocol.h                         # future
│   ├── traffic_fsm.h                      # future
│   ├── safety.h                           # future
│   └── seven_segment.h                    # future
└── src/
    ├── main.c                             # future
    ├── gpio_outputs.c                     # future
    ├── uart_link.c                        # future
    ├── protocol.c                         # future
    ├── traffic_fsm.c                      # future
    ├── safety.c                           # future
    └── seven_segment.c                    # future
```

This tree is a proposal. It is not fully created in this phase.

## Explanation of Each Future File

| Future file | Purpose | Review required before creation |
| --- | --- | --- |
| `CMakeLists.txt` | Defines the firmware target, source files, include paths, compile flags, and link options | Toolchain, source layout, startup/linker policy |
| `cmake/toolchain-arm-none-eabi.cmake` | Forces CMake to use the ARM embedded cross-compiler instead of the macOS host compiler | Cross-compiler install path and supported CMake behavior |
| `startup/startup_stm32f103c8tx.s` | Provides vector table, reset handler, weak interrupt handlers, and C runtime entry path | Traceable source, license, STM32F103xB compatibility |
| `linker/STM32F103C8Tx_FLASH.ld` | Defines FLASH/RAM layout and output sections for the linked image | STM32F103C8T6 memory size and stack/heap policy |
| `include/board_config.h` | Central board configuration: pins, UART instance, baud rate, status flags | Hardware pin mapping and validation notes |
| `include/gpio_outputs.h` / `src/gpio_outputs.c` | Traffic light LED and status LED output API | LED lane/color mapping and polarity |
| `include/uart_link.h` / `src/uart_link.c` | Byte-level UART transport | USART1 pin validation and UART test plan |
| `include/protocol.h` / `src/protocol.c` | PLAN parsing and ACK/NACK/STATUS formatting | Protocol contract with host and test strategy |
| `include/traffic_fsm.h` / `src/traffic_fsm.c` | Safe traffic-light state transitions | FSM review; not Phase 17.2.3 |
| `include/safety.h` / `src/safety.c` | Timeout and fallback behavior | Host timeout policy |
| `include/seven_segment.h` / `src/seven_segment.c` | Dual 7-segment output layer | Display polarity and current validation |

## Toolchain File Proposal

Future `cmake/toolchain-arm-none-eabi.cmake` should eventually:

- set the C compiler to `arm-none-eabi-gcc`.
- set the ASM compiler to `arm-none-eabi-gcc`.
- set `OBJCOPY` to `arm-none-eabi-objcopy`.
- set `OBJDUMP` to `arm-none-eabi-objdump`.
- set `SIZE` to `arm-none-eabi-size`.
- set Cortex-M3 target flags.
- avoid the native macOS host compiler.
- avoid accidental native macOS builds.

Likely target/compiler flags:

```text
-mcpu=cortex-m3
-mthumb
-ffunction-sections
-fdata-sections
-Wall
-Wextra
```

Additional flags should be reviewed later:

```text
-Os or -Og
-g3
-MMD
-MP
-fno-common
```

Do not create the actual toolchain file until `arm-none-eabi-gcc` installation and CMake behavior are approved.

## Top-Level CMakeLists Proposal

Future `CMakeLists.txt` should eventually contain:

- minimum CMake version.
- project name, likely `stm32_f103_traffic_light`.
- C standard, likely C11.
- target name, likely `stm32_f103_traffic_light`.
- source list.
- include directories.
- startup file reference.
- linker script reference.
- compile options.
- link options.
- post-build size command.
- optional `.bin` / `.hex` generation later.

Future CMake should not:

- flash hardware.
- download dependencies.
- assume STM32CubeIDE.
- assume generated CubeMX files exist.
- build native macOS binaries by accident.

Do not create actual `CMakeLists.txt` yet.

## Startup File Policy

A startup file is required before a real STM32 firmware link.

It provides:

- vector table.
- `Reset_Handler`.
- weak interrupt handlers.
- C runtime entry path.
- stack pointer initialization.

Policy:

- source must be traceable.
- license must be compatible with repository use.
- file must match STM32F103xB / STM32F103C8T6 class.
- interrupt names must match the selected CMSIS/HAL/LL strategy.
- do not hand-write from memory.

Do not add a startup file in this phase.

## Linker Script Policy

A linker script is required before producing a real `.elf`.

It defines:

- FLASH and RAM layout.
- `.isr_vector`.
- `.text`.
- `.rodata`.
- `.data`.
- `.bss`.
- stack and heap placement.

Policy:

- memory layout must match STM32F103C8T6.
- flash/RAM sizes must be verified against the selected device class.
- stack/heap assumptions must be documented.
- output sections must be reviewed before first compile.

Do not add a linker script in this phase.

## CMSIS / HAL / LL Policy

### Option 1 - CMSIS-Only

Pros:

- minimal dependency footprint.
- explicit register-level learning.
- easier to see what firmware touches.

Cons:

- more boilerplate for clocks, GPIO, UART, timers.
- higher risk of register mistakes.
- slower to bring up peripherals.

Repository impact:

- fewer vendor files, but more project-owned low-level code.

Learning value:

- high for Cortex-M and STM32 fundamentals.

Risk:

- medium to high during hardware bring-up if not reviewed carefully.

### Option 2 - STM32 HAL

Pros:

- common STM32 ecosystem path.
- many examples available.
- faster peripheral bring-up.

Cons:

- larger dependency set.
- generated files can create repository churn.
- can hide hardware details that matter during learning.

Repository impact:

- likely adds STM32CubeF1 HAL/CMSIS vendor files or a documented external dependency.

Learning value:

- good for practical STM32 development, lower for register-level details.

Risk:

- medium; main risk is uncontrolled generated code and unclear versioning.

### Option 3 - STM32 LL

Pros:

- closer to registers than HAL.
- lighter than HAL.
- still supported by ST ecosystem.

Cons:

- less beginner-friendly than HAL.
- still needs vendor source/version policy.
- examples may be less direct than HAL examples.

Repository impact:

- likely adds selected LL/CMSIS files with documented source.

Learning value:

- good balance between practical control and lower-level understanding.

Risk:

- medium; requires more STM32-specific review than HAL.

### Recommendation for This Project

Start with documentation and a compile-only scaffold proposal.

Before adding vendor files, choose one strategy:

```text
CMSIS-only, STM32 HAL, or STM32 LL
```

Do not mix random vendor files into the repository without:

- source package name.
- source version.
- license notes.
- regeneration or update instructions.
- rationale for selected files.

## Generated Files Policy

Do not commit generated CubeMX/CubeIDE files until the project decides a generation policy.

If generated files are used later, record:

- STM32Cube version.
- `.ioc` source.
- exact generation settings.
- generated file list.
- regeneration instructions.
- which files are allowed to be edited manually.

Current phase decision:

```text
No generated files.
No STM32CubeIDE project.
No .ioc file.
No generated HAL project.
```

## Source / Header Policy

Future source and header files should:

- be small and module-owned.
- keep board-specific pins in `board_config.h`.
- keep protocol structures testable without STM32 hardware where practical.
- avoid hidden global state.
- document ownership for timers, queues, and shared status if FreeRTOS is added later.

Do not create `.c` or `.h` implementation files in Phase 17.2.3.

## Build Artifact Policy

The following should stay untracked:

```text
build/
*.elf
*.bin
*.hex
*.map
*.o
*.a
CMakeFiles/
CMakeCache.txt
```

Current `.gitignore` status:

- `build/` is already ignored.
- `firmware/**/*.bin` is already ignored.
- `firmware/**/*.elf` is already ignored.
- `firmware/**/*.hex` appears to need cleanup because the current line is `firmware/**/*.hex.vscode/`.
- `*.map`, `*.o`, `*.a`, `CMakeFiles/`, and `CMakeCache.txt` are not clearly covered for STM32 builds.

Recommendation:

- propose a future `.gitignore` update before the first real build.
- do not modify `.gitignore` in this proposal-only phase.

## No-Flashing Policy

CMake build commands should compile and link only.

Flashing should remain separate and require:

- real PCB power validation.
- SWD header orientation check.
- ST-LINK or approved debug probe.
- OpenOCD or STM32CubeProgrammer CLI.
- explicit user approval.

No future `cmake --build` target should silently flash hardware.

## Review Gates Before Implementation

Do not create the real CMake scaffold until:

```text
[ ] User approves CMake + Make path.
[ ] `arm-none-eabi-gcc` install is approved.
[ ] startup file source is selected.
[ ] linker script source is selected.
[ ] CMSIS/HAL/LL strategy is selected.
[ ] generated-files policy is documented.
[ ] `.gitignore` artifact policy is fixed or accepted.
[ ] pin map dependency is acknowledged.
[ ] first build is scoped as non-flashing.
```

## Risks

- accidentally creating a native macOS CMake build instead of an ARM cross-build.
- committing generated vendor code without source/version policy.
- using a startup file for the wrong STM32F103 memory class.
- linking with an incorrect FLASH/RAM layout.
- treating CMake success as hardware validation.
- letting early build scaffolding imply the UART/FSM code is ready.
- committing build artifacts because ignore rules are incomplete.

## Recommended Next Step

Recommended next Phase 17 step:

```text
Phase 17.2.4 - Approve STM32 build strategy and artifact policy
```

This should decide:

- CMake + Make vs another workflow.
- CMSIS/HAL/LL direction.
- startup/linker source.
- whether to update `.gitignore`.
- whether tool installation is approved.

Do not install tools or create buildable firmware until that decision is made.
