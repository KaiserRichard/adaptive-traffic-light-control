# STM32 Source Directory

## Status

```text
Intentionally empty except for this README.
No STM32 source files are implemented in Phase 17.2.
```

This directory is reserved for future STM32 firmware source after the command-line toolchain and hardware pin map are reviewed.

## Future Source Files

Possible future files:

```text
main.c or main.cpp
board_pins.c
gpio_outputs.c
uart_link.c
protocol.c
traffic_fsm.c
status_reporter.c
watchdog_fallback.c
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
