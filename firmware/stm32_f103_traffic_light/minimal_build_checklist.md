# STM32 Minimal Build Checklist

## Purpose

Define the approval and file checklist required before the first real STM32F103C8T6 compile attempt.

This is not a build log. No build has been run.

## Checklist Before First Real Compile

```text
[ ] Selected toolchain approved.
[ ] `arm-none-eabi-gcc` installed.
[ ] Arm binutils available.
[ ] CMake generator selected.
[ ] Startup file source identified.
[ ] Linker script source identified.
[ ] STM32F103C8T6 flash/RAM memory layout confirmed.
[ ] System clock strategy selected.
[ ] CMSIS/HAL/LL strategy selected.
[ ] Generated files policy decided.
[ ] Pin mapping reviewed.
[ ] `board_config.h` created.
[ ] Minimal `main.c` created.
[ ] Build command documented.
[ ] Expected output artifact documented.
[ ] No flashing during first compile.
```

## Toolchain Requirements

Minimum future compile-only tools:

- CMake.
- Make or Ninja.
- `arm-none-eabi-gcc`.
- Arm binutils.

Already present from Phase 17.2.1:

- CMake 4.2.3.
- GNU Make 3.81.

Missing as of Phase 17.2.1:

- `arm-none-eabi-gcc`.
- `arm-none-eabi-gdb`.
- Ninja.
- OpenOCD.
- STM32CubeProgrammer CLI.
- PlatformIO.

## Startup File Checklist

Before adding startup code:

```text
[ ] Source package identified.
[ ] License acceptable for repository use.
[ ] File matches STM32F103C8T6 / STM32F103xB class.
[ ] Vector table reviewed.
[ ] Reset handler path understood.
[ ] C runtime initialization path understood.
[ ] Interrupt handler naming compatible with selected CMSIS/HAL/LL strategy.
```

## Linker Script Checklist

Before adding a linker script:

```text
[ ] Flash size confirmed for target part.
[ ] RAM size confirmed for target part.
[ ] Stack location reviewed.
[ ] Heap policy reviewed.
[ ] Vector table location reviewed.
[ ] Output sections understood.
[ ] Artifact names documented.
```

## Board Configuration Checklist

Before adding `board_config.h`:

```text
[ ] PC13 LED polarity verified or clearly marked pending validation.
[ ] Traffic LED lane/color mapping verified or marked TBD.
[ ] 7-segment polarity/current path verified or display disabled.
[ ] USART1 PA9/PA10 mapping reviewed.
[ ] UART header TX/RX crossing reviewed.
[ ] SWD pins reserved.
[ ] BOOT0 and NRST notes documented.
```

## First Compile Scope

The first real compile should be intentionally narrow:

```text
Goal:
    prove the command-line build can compile and link a minimal STM32 image.

Allowed:
    minimal main
    startup file
    linker script
    CMSIS device headers
    compile/link command

Not allowed:
    flashing
    UART protocol implementation
    FreeRTOS FSM port
    traffic light output assumptions
    hardware validation claims
```

## Expected Output Artifact

A future first build may produce:

```text
build/stm32_f103_traffic_light.elf
```

Optional later conversion artifacts:

```text
build/stm32_f103_traffic_light.bin
build/stm32_f103_traffic_light.hex
```

These artifacts must not be committed.

## Do Not Proceed to Real Build Until

```text
[ ] Toolchain path is approved.
[ ] Startup/linker source is known.
[ ] Generated files policy is decided.
[ ] Pin mapping is reviewed.
[ ] README status is updated.
[ ] Build output paths are ignored or otherwise kept out of commits.
[ ] The first compile is explicitly scoped as non-flashing.
```

## Failure Symptoms During Future Compile

- `arm-none-eabi-gcc: command not found`: cross-compiler not installed.
- undefined `_estack` or memory symbols: linker script/startup mismatch.
- undefined `Reset_Handler`: startup file or entry point mismatch.
- missing `SystemInit`: CMSIS/system file strategy mismatch.
- wrong architecture flags: Cortex-M3 flags not set correctly.
- binary too large: memory map or linked sources not appropriate.

## Next Safe Step

Before compiling, draft the exact non-buildable CMake scaffold proposal and review the future file sources.
