# STM32 Command-Line Toolchain Plan

## Status

```text
Planned.
No STM32 build system is configured yet.
No toolchain is required for the current documentation-only Phase 17 work.
```

STM32CubeIDE is not required for this phase.

## Constraints

- Development machine: MacBook.
- No STM32CubeIDE requirement.
- No real board access assumed.
- No flashing required for Phase 17.1-17.4.
- No generated HAL/CubeMX project files in this pass.
- No `.elf`, `.bin`, or `.hex` artifacts committed.

## Option A - STM32CubeCLT

STM32CubeCLT is ST's command-line toolchain package. It can provide a supported path for building and programming without opening STM32CubeIDE.

Potential benefits:

- Official ST command-line tooling.
- Can work with STM32CubeProgrammer workflows.
- Familiar path if HAL-based firmware is later generated or maintained.

Risks / questions:

- Need to verify macOS installation path and license/package requirements.
- Need to decide whether HAL, LL, or CMSIS-only code will be used.
- Generated project files should not be added unless the source and review workflow are clear.

Possible future commands:

```bash
STM32_Programmer_CLI --version
arm-none-eabi-gcc --version
cmake --version
```

These are examples for a future environment check, not commands required by this phase.

## Option B - arm-none-eabi-gcc + CMake / Make

This option uses the GNU Arm Embedded toolchain with a small command-line build system.

Potential benefits:

- Portable and transparent.
- Works well for repository-controlled builds.
- Avoids IDE lock-in.
- Can be paired with Ninja or Make.

Required future decisions:

- Source of startup file.
- Source of linker script.
- HAL vs LL vs CMSIS-only peripheral access.
- FreeRTOS kernel integration method.
- Whether tests can be split into host-side protocol tests and target firmware builds.

Possible future commands:

```bash
arm-none-eabi-gcc --version
cmake -S . -B build -G Ninja
cmake --build build
```

These commands should not be added as a required workflow until the project skeleton exists.

## Option C - PlatformIO if Later Selected

PlatformIO could support STM32 targets and may be convenient because the ESP32 prototype already uses PlatformIO.

Potential benefits:

- Familiar repository pattern.
- Dependency/toolchain management.
- Works from command line.

Risks / questions:

- Framework choice must be reviewed.
- Generated dependency state should not be committed.
- Need to confirm STM32F103C8T6 board definition or custom board configuration.

Possible future commands:

```bash
pio run
pio run --target upload
pio device monitor -b 115200
```

These are not active Phase 17 requirements.

## Flashing / Debugging Options

Future options:

- STM32CubeProgrammer CLI.
- OpenOCD.
- ST-LINK.

Current phase:

```text
STM32CubeProgrammer required now? No
OpenOCD required now? No
ST-LINK required now? No
Real PCB required now? No
```

## Recommended Next Toolchain Step

Before implementing STM32 source code, run a toolchain selection review:

```text
Goal:
    choose STM32CubeCLT vs arm-none-eabi-gcc/CMake vs PlatformIO

Inputs:
    MacBook install constraints
    desired HAL/LL/CMSIS layer
    FreeRTOS integration plan
    debug/programming method

Output:
    one documented build command
    one documented clean command
    one documented non-flashing compile-only test
    one documented future flashing command
```

Do not create a full firmware project until that decision is made.
