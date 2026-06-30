# STM32 Source Directory

## Status

```text
Hardware-independent scaffold source files added.
No STM32 HAL/CMSIS/LL source is used.
No hardware-validated STM32 implementation exists yet.
```

This directory now contains host-safe STM32 board-layer stubs. They define future responsibilities but do not configure clocks, GPIO, UART, FreeRTOS, interrupts, or hardware registers.

## Current Source Files

Current scaffold files:

```text
main.c
board_gpio.c
board_uart.c
app_tasks.c
stm32_port_status.c
```

Do not add generated HAL source, startup files, linker scripts, or FreeRTOS kernel files without documenting their source and licensing.

## Implementation Gate

Before adding source files:

```text
[ ] Select command-line toolchain.
[ ] Confirm STM32F103C8T6 package and clock/reset assumptions.
[ ] Review pin mapping.
[ ] Decide HAL vs LL vs CMSIS-only approach.
[ ] Define compile-only build command.
[ ] Keep flashing optional and separate from build.
```
