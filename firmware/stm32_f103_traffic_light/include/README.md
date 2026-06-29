# STM32 Include Directory

## Status

```text
Intentionally empty except for this README.
No STM32 header files are implemented in Phase 17.2.
```

This directory is reserved for future STM32 firmware public/internal headers after the build system and module boundaries are reviewed.

## Future Header Files

Possible future files:

```text
board_pins.h
gpio_outputs.h
uart_link.h
protocol.h
traffic_fsm.h
status_reporter.h
watchdog_fallback.h
```

## Header Design Rules

Future headers should:

- Avoid hidden global state.
- Keep hardware pin definitions in one board-specific location.
- Keep protocol structures independent from STM32 HAL types when practical.
- Document task ownership for any FreeRTOS queue, mutex, timer, or shared state.
- Keep Raspberry Pi protocol compatibility clear.

Do not expose unstable APIs before the firmware architecture is reviewed.
