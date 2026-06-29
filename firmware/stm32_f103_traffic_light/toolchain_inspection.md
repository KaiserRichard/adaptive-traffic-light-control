# Phase 17.2.1 - STM32 Toolchain Inspection

## Purpose

Inspect the current MacBook command-line environment for STM32 firmware development readiness without installing tools, flashing hardware, or creating a full firmware project.

This report supports the next safe STM32 firmware step: a minimal non-flashing build scaffold review.

## Inspection Time

```text
2026-06-29 09:59:37 +07
```

## Machine / OS Summary

```text
ProductName:    macOS
ProductVersion: 15.7.7
BuildVersion:   24G720
Kernel:         Darwin 24.6.0
Architecture:   x86_64
```

## Tool Availability Table

| Tool | Command Checked | Status | Version / Output Summary | Needed Now? | Notes |
| --- | --- | --- | --- | --- | --- |
| Homebrew | `brew --version` | Installed | Homebrew 6.0.2 at `/usr/local/bin/brew` | No | Useful later for installing command-line tools only after approval |
| CMake | `cmake --version` | Installed | CMake 4.2.3 at `/usr/local/bin/cmake` | No | Useful for a future non-flashing build scaffold |
| Ninja | `ninja --version` | Missing / not installed | Command not found | No | Nice to have; Make can be used initially |
| Make | `make --version` | Installed | GNU Make 3.81 at `/usr/bin/make` | No | Available fallback generator/build tool |
| Arm GCC | `arm-none-eabi-gcc --version` | Missing / not installed | Command not found | No | Required later for real STM32 cross-compilation |
| Arm GDB | `arm-none-eabi-gdb --version` | Missing / not installed | Command not found | No | Required later for source-level embedded debugging |
| OpenOCD | `openocd --version` | Missing / not installed | Command not found | No | Required later only if OpenOCD/ST-LINK debug path is selected |
| STM32CubeProgrammer CLI | `STM32_Programmer_CLI --version` | Missing / not installed | Command not found | No | Required later only for ST flashing/programming workflow |
| PlatformIO | `pio --version` | Missing / not installed | Command not found | No | Optional later path; not selected for this phase |
| Python 3 | `python3 --version` | Installed | Python 3.11.0rc2 at `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3` | No | Useful for host-side scripts and future test tooling |

## Tools Installed

- Homebrew 6.0.2.
- CMake 4.2.3.
- GNU Make 3.81.
- Python 3.11.0rc2.

## Tools Missing

- Ninja.
- `arm-none-eabi-gcc`.
- `arm-none-eabi-gdb`.
- OpenOCD.
- STM32CubeProgrammer CLI.
- PlatformIO.

## Toolchain Options Compared

### Option A - STM32CubeCLT-Based Command-Line Workflow

Pros:

- Official ST command-line direction.
- Can align with STM32CubeProgrammer and ST-provided components.
- Avoids STM32CubeIDE GUI as the daily workflow.

Cons:

- Not installed now.
- Installation/source of generated support files must be reviewed.
- Can encourage generated project churn if not controlled.

### Option B - arm-none-eabi-gcc + CMake/Ninja + OpenOCD

Pros:

- Transparent, repository-controlled command-line workflow.
- Good fit for MacBook development.
- Separates compile-only verification from later flashing/debugging.
- Avoids STM32CubeIDE dependency.

Cons:

- Arm GCC, GDB, Ninja, and OpenOCD are not installed now.
- Requires deliberate choices for startup file, linker script, HAL/LL/CMSIS layer, and FreeRTOS integration.

### Option C - PlatformIO

Pros:

- Familiar because the ESP32 prototype uses PlatformIO.
- Can manage toolchains and board packages.
- Simple command-line UX once configured.

Cons:

- Not installed now.
- STM32F103C8T6 board definition and framework choice still need review.
- May hide lower-level STM32 startup/linker details that are useful for learning and bring-up.

### Option D - STM32CubeIDE GUI Workflow

Pros:

- Common STM32 beginner path.
- Integrates project generation, build, flash, and debug.

Cons:

- Not required for this project stage.
- Adds GUI dependency.
- Can generate large project files before the repository is ready for them.
- Not aligned with the current MacBook command-line workflow constraint.

## Recommended Toolchain Path

Recommended direction:

```text
Use Option B later:
    arm-none-eabi-gcc + CMake/Make first,
    then add Ninja and OpenOCD only when approved and needed.
```

Reasoning:

- CMake and Make are already installed, so the next safe step can design a minimal compile-only scaffold without requiring flashing.
- A repository-owned CMake layout is easier to review than a generated IDE project.
- It keeps STM32CubeIDE optional rather than mandatory.
- It supports the learning goal: understand startup, linker, board pins, protocol modules, and FreeRTOS integration explicitly.

## Toolchain Decision

For now:

```text
documentation + inspection only
```

Next:

```text
minimal non-flashing build scaffold review
```

Later:

```text
install selected tools only after user approval
```

Much later:

```text
flash and debug real STM32 hardware only after power, SWD, and board safety checks
```

## Why STM32CubeIDE Is Not Required Now

STM32CubeIDE is not required because Phase 17.2.1 does not build, flash, debug, or generate STM32 firmware. The current work only records environment readiness and updates project documentation.

Future STM32 firmware can start from a command-line workflow that uses a small CMake/Make scaffold and `arm-none-eabi-gcc`, while keeping flashing and debugging optional until the real PCB and ST-LINK workflow are validated.

## Install Later Only After User Approval

Do not install anything automatically.

Potential future installs, after approval:

- `arm-none-eabi-gcc` and related binutils.
- `arm-none-eabi-gdb`.
- Ninja, if selected over Make.
- OpenOCD, if selected for ST-LINK debug.
- STM32CubeProgrammer CLI, if selected for flashing/programming.
- PlatformIO, only if the project intentionally chooses PlatformIO for STM32.
- STM32CubeCLT, only if the project chooses ST's command-line package.

## What Can Be Done Before Hardware Arrives

- Decide HAL vs LL vs CMSIS-only style.
- Design a minimal compile-only source layout.
- Keep protocol parsing code hardware-independent.
- Plan host-side unit tests for `PLAN` parsing.
- Define board pin constants from the pin map, marked pending validation.

## What Requires ST-LINK / Real PCB

- SWD target detection.
- Flashing firmware.
- PC13 blink validation.
- Traffic LED validation.
- 7-segment validation.
- UART electrical validation.
- FreeRTOS timing behavior on the real MCU.

## Next Safe Step

Recommended next Phase 17 step:

```text
Phase 17.2.2 - Minimal non-flashing STM32 build scaffold design review
```

That step should still avoid flashing hardware and should not add generated CubeMX/CubeIDE files.
