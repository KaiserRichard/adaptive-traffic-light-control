# STM32 Include Directory

## Status

```text
Hardware-independent scaffold headers added.
No STM32 HAL/CMSIS/LL headers are used.
No hardware-validated STM32 implementation exists yet.
```

This directory now defines board-layer scaffold APIs and compile-time status for future STM32 firmware work.

## Current Header Files

Current scaffold files:

```text
board_config.h
board_gpio.h
board_uart.h
app_tasks.h
stm32_port_status.h
```

These headers are not final board support APIs. They exist to define the future ownership boundary while power, SWD, GPIO, UART, and display validation remain pending.

## Header Design Rules

Future headers should:

- Avoid hidden global state.
- Keep hardware pin definitions in one board-specific location.
- Keep protocol structures independent from STM32 HAL types when practical.
- Document task ownership for any FreeRTOS queue, mutex, timer, or shared state.
- Keep Raspberry Pi protocol compatibility clear.

Do not expose unstable APIs before the firmware architecture is reviewed.
