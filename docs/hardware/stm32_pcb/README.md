# STM32F103C8T6 PCB Hardware Integration

## Purpose

This folder documents the STM32F103C8T6 controller PCB path for the Adaptive Traffic Light Control project.

The PCB is intended to replace or extend the ESP32 FreeRTOS prototype after the hardware has been inspected, powered safely, programmed through SWD, and validated over UART.

## Current Status

```text
Status:
    Schematic documentation and integration planning started.
    Final hardware validation pending.
```

No board flashing, UART testing, LED testing, SWD testing, or end-to-end Raspberry Pi integration has been completed in this phase.

## PCB Role in ATLC

The STM32 PCB is the planned real-time traffic light controller. It should receive timing plans from the AI host, validate the command format, execute a traffic light finite-state machine, drive PCB outputs, and report status back to the host.

The current ESP32 implementation remains the firmware behavior reference until the STM32 firmware port is reviewed and implemented.

## System Architecture

```text
Camera / video input
    -> Raspberry Pi AI host
    -> YOLO detection, ROI counting, density estimation
    -> adaptive green-time planner
    -> UART PLAN messages
    -> STM32F103C8T6 PCB controller
    -> FreeRTOS traffic light FSM
    -> traffic light LEDs / dual 7-segment display / expansion outputs
```

## Schematic Blocks

Schematic block images currently exist as `.jpg` files in this folder. A `schematics/` subfolder has been added for documentation notes, but the expected `.png` schematic exports are not present yet.

| Block | Current schematic evidence | Status |
| --- | --- | --- |
| Power input and 3.3 V regulation | [01_power_usb_5v_3v3_regulator.jpg](01_power_usb_5v_3v3_regulator.jpg) | Documentation started |
| Raspberry Pi UART header | [02_uart_pi_header.jpg](02_uart_pi_header.jpg) | Documentation started |
| STM32F103C8T6 MCU core | [03_stm32f103c8t6_mcu_core.jpg](03_stm32f103c8t6_mcu_core.jpg) | Documentation started |
| Traffic light LED outputs | [04_traffic_light_led_outputs.jpg](04_traffic_light_led_outputs.jpg) | Documentation started |
| Dual 7-segment display | [05_dual_7segment_display.jpg](05_dual_7segment_display.jpg) | Documentation started |
| Expansion headers | [06_expansion_headers.jpg](06_expansion_headers.jpg) | Documentation started |
| SWD programming/debug header | [07_swd_programming_header.jpg](07_swd_programming_header.jpg) | Documentation started |

## Hardware Block Explanation

The detailed hardware explanation is in [stm32_hardware_blocks_explained.md](stm32_hardware_blocks_explained.md).

That document supports:

- Phase 17 hardware understanding.
- PCB bring-up preparation.
- microprocessor and analog/digital electronics explanation.
- future STM32 firmware `board_config` planning.

It is still schematic-level documentation. It does not claim power, SWD, UART, GPIO, LED, display, or full board validation.

## Raspberry Pi AI Host vs STM32 Controller Responsibilities

| Area | Raspberry Pi AI host | STM32 PCB controller |
| --- | --- | --- |
| Perception | Runs camera/video input, YOLO inference, ROI counting, density estimation | Not responsible |
| Planning | Computes green-time plans | Receives already computed plans |
| UART transmit | Sends `PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>` | Receives and validates `PLAN` messages |
| UART response | Receives controller responses | Sends `ACK`, `NACK`, and future `STATUS` messages |
| Real-time execution | Not the owner of light timing once a plan is accepted | Owns traffic light FSM timing |
| Safety fallback | Should continue sending valid plans and monitor responses | Should use watchdog/fallback behavior if host messages stop |
| Hardware outputs | Not directly driving traffic LEDs | Drives traffic light LEDs, 7-segment display, and future expansion outputs |

## Power and Voltage Notes

Observed from the power schematic image:

- USB Micro-B VBUS appears to feed a 5 V rail through an SS34 diode.
- An AMS1117-3.3V regulator appears to generate `VCC3V3`.
- Input/output capacitors are present in the schematic block.
- A power LED is connected from `VCC3V3` through a resistor to ground.
- A PC13 status LED path is also shown.

Needs verification:

- Actual input voltage range and connector intent.
- AMS1117 thermal margin under worst-case load.
- Whether the USB connector is power-only or also intended for data.
- Whether 3.3 V rail current is sufficient for the STM32, LEDs, 7-segment display, and any expansion loads.

## UART Notes

Observed from the UART and MCU schematic images:

- The UART header exposes `VCC3V3`, `GND`, `TX1`, and `RX1`.
- The MCU core image labels `TX1` on PA9 and `RX1` on PA10.
- These names are consistent with STM32 USART1 pins, but the Raspberry Pi TX/RX crossing must still be checked against the physical connector orientation.

Planned default UART settings:

```text
115200 baud
8 data bits
no parity
1 stop bit
newline-terminated ASCII messages
3.3 V logic only
```

## SWD Programming/Debug Notes

Observed from the SWD header schematic image:

- Header signals include `VCC3V3`, `SWDIO`, `SWCLK`, and `GND`.
- The MCU core image labels `SWDIO` on PA13 and `SWCLK` on PA14.

Needs verification:

- Physical pin 1 orientation on the actual PCB.
- Whether NRST is available on any connector or test pad.
- ST-LINK/OpenOCD compatibility after BOOT0 and power rails are verified.

## LED and 7-Segment Output Notes

Observed from schematic images:

- Six traffic light LED outputs are shown on PB9, PB8, PB7, PB6, PB5, and PB3 through 1k resistors.
- The schematic image does not label which outputs correspond to direction A/B red/yellow/green. Firmware signal ownership is therefore TBD.
- Dual 7-segment segment lines are shown on PA8, PB15, PB14, PB13, PB12, PA12, and PA11 through 220 ohm resistors.
- Digit select/common lines are shown on PA3 and PA4.

Needs verification:

- Display polarity and current path.
- Whether PA3/PA4 directly drive digit common pins or require transistor drivers.
- Total GPIO current across traffic LEDs and 7-segment display.

## Bring-Up Summary

Recommended order:

1. Visual inspection before power.
2. Power validation with current-limited supply.
3. 3.3 V rail and ground checks.
4. SWD connection check.
5. Minimal blink firmware only after power/SWD are safe.
6. GPIO traffic LED smoke test.
7. 7-segment display smoke test.
8. UART loopback and Raspberry Pi UART validation.
9. Only then review the FreeRTOS FSM port.

See [stm32_pcb_bringup_plan.md](stm32_pcb_bringup_plan.md).

## Known Risks

- Pin mapping is only partially confirmed from schematic images.
- Traffic light direction/color ownership is not labeled in the current LED schematic image.
- UART connector labels do not by themselves prove Raspberry Pi TX/RX crossing.
- 7-segment current budget may exceed safe GPIO limits if multiple segments are driven without multiplexing discipline.
- AMS1117 regulator thermal margin needs real load calculation and measurement.
- PC13 LED behavior on STM32F103 boards can be inverted or weak depending on board wiring; verify this specific PCB.

## Next Steps

1. Review [stm32f103c8t6_pin_mapping.md](stm32f103c8t6_pin_mapping.md) against the schematic source and PCB layout.
2. Read [stm32_hardware_blocks_explained.md](stm32_hardware_blocks_explained.md) before board bring-up or firmware `board_config` planning.
3. Follow [stm32_pcb_review_notes.md](stm32_pcb_review_notes.md) before applying power.
4. Use [stm32_pcb_bringup_plan.md](stm32_pcb_bringup_plan.md) for hardware bring-up.
5. Use [stm32_uart_pi_validation_plan.md](stm32_uart_pi_validation_plan.md) before any AI host integration.
6. Keep Phase 17 isolated from Phase 16 until a shared UART interface update is explicitly approved.
