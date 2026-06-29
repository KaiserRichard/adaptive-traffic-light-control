# Phase 17.2.3 - CMake Scaffold Proposal Result

## Goal

Create a documentation-only proposal for the future STM32 CMake build scaffold.

## Scope

Included:

- future CMake scaffold structure.
- toolchain file responsibilities.
- top-level `CMakeLists.txt` responsibilities.
- startup file policy.
- linker script policy.
- CMSIS/HAL/LL comparison.
- generated-files policy.
- build artifact policy.
- no-flashing policy.

Not included:

- no real CMake project.
- no startup file.
- no linker script.
- no CMSIS/HAL/LL vendor files.
- no `.c` or `.h` implementation.
- no firmware build.
- no flashing.
- no hardware validation.

## Files Created

- `firmware/stm32_f103_traffic_light/cmake_scaffold_proposal.md`
- `firmware/stm32_f103_traffic_light/docs/phase_17_2_3_cmake_scaffold_proposal.md`

## Files Modified

- `firmware/stm32_f103_traffic_light/README.md`
- `docs/embedded/phase_17_ledger.md`

## What Was Proposed

The proposed future build scaffold is:

```text
CMake + Unix Makefiles + arm-none-eabi-gcc
```

Future files were described but not created:

- `CMakeLists.txt`.
- `cmake/toolchain-arm-none-eabi.cmake`.
- startup assembly file.
- linker script.
- `board_config.h`.
- future module `.c` and `.h` files.

## What Was Intentionally Not Created

- No buildable CMake project.
- No CMake toolchain file.
- No startup file.
- No linker script.
- No generated CubeMX/CubeIDE files.
- No CMSIS/HAL/LL vendor files.
- No source/header implementation.
- No build artifacts.

## Why This Phase Stays Documentation-Only

The repository is not ready for a real STM32 build because:

- `arm-none-eabi-gcc` is not installed.
- startup/linker sources have not been selected.
- CMSIS/HAL/LL strategy is not chosen.
- generated-file policy is not decided.
- pin mapping is not hardware-validated.

## What Must Happen Before a Real CMake Project

```text
[ ] Approve CMake + Make path.
[ ] Approve tool installation plan.
[ ] Decide CMSIS/HAL/LL strategy.
[ ] Select startup file source and license.
[ ] Select linker script source and memory layout.
[ ] Decide generated-files policy.
[ ] Confirm build artifact ignore policy.
```

## What Must Happen Before First Compile

```text
[ ] Install approved `arm-none-eabi-gcc`.
[ ] Add reviewed startup file.
[ ] Add reviewed linker script.
[ ] Add minimal source files.
[ ] Add CMake scaffold.
[ ] Keep first compile non-flashing.
[ ] Record expected `.elf` output path.
```

## What Must Happen Before Flashing

```text
[ ] Validate PCB power rails.
[ ] Confirm SWD header orientation.
[ ] Confirm BOOT0 and NRST behavior.
[ ] Select OpenOCD or STM32CubeProgrammer CLI.
[ ] Confirm ST-LINK or approved probe.
[ ] Receive explicit approval to flash.
```

## Connection to Phase 16 PLAN Generation

Phase 16 prepares the Raspberry Pi / host-side AI path that will eventually generate `PLAN` messages.

The STM32 firmware scaffold must preserve a clean boundary:

```text
Raspberry Pi AI Host
    sends PLAN messages

STM32 Controller
    validates plans
    executes local FSM
    returns ACK/NACK/STATUS
```

No host-side Phase 16 files were modified in this phase.

## Connection to Phase 17 UART Validation

The future CMake scaffold must leave room for a `uart_link` module and a `protocol` module, but UART hardware code should wait until:

- USART1 PA9/PA10 mapping is reviewed.
- Pi TX/RX crossing is confirmed.
- 3.3 V logic and shared ground are validated.
- a minimal build path exists.

## Next Step

Recommended next step:

```text
Phase 17.2.4 - Approve STM32 build strategy and artifact policy
```

That phase should decide whether to approve CMake + Make, how to handle CMSIS/HAL/LL sources, and whether to update `.gitignore` before the first real build.
