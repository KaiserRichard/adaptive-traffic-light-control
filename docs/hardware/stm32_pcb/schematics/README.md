# STM32 PCB Schematic Blocks

## Status

```text
Documentation started.
Expected schematic image folder exists.
Expected .png schematic exports are not present in this folder yet.
Current schematic evidence is stored as .jpg files one level up.
```

This README records what each schematic block should prove during review. It does not replace the original schematic source or PCB netlist.

## Expected Image Files

| Expected file in this folder | Current available reference | Status |
| --- | --- | --- |
| `01_power_usb_5v_3v3_regulator.png` | [../01_power_usb_5v_3v3_regulator.jpg](../01_power_usb_5v_3v3_regulator.jpg) | JPG exists; PNG TODO |
| `02_uart_pi_header.png` | [../02_uart_pi_header.jpg](../02_uart_pi_header.jpg) | JPG exists; PNG TODO |
| `03_stm32f103c8t6_mcu_core.png` | [../03_stm32f103c8t6_mcu_core.jpg](../03_stm32f103c8t6_mcu_core.jpg) | JPG exists; PNG TODO |
| `04_traffic_light_led_outputs.png` | [../04_traffic_light_led_outputs.jpg](../04_traffic_light_led_outputs.jpg) | JPG exists; PNG TODO |
| `05_dual_7segment_display.png` | [../05_dual_7segment_display.jpg](../05_dual_7segment_display.jpg) | JPG exists; PNG TODO |
| `06_expansion_headers.png` | [../06_expansion_headers.jpg](../06_expansion_headers.jpg) | JPG exists; PNG TODO |
| `07_swd_programming_header.png` | [../07_swd_programming_header.jpg](../07_swd_programming_header.jpg) | JPG exists; PNG TODO |

## Block Review Notes

### 01 - Power USB 5 V and 3.3 V Regulator

What this block shows:

- USB Micro-B power input.
- SS34 diode on the 5 V path.
- AMS1117-3.3V regulator.
- Input/output capacitors.
- Power LED and PC13 LED paths.

Why it matters:

- The STM32, UART interface, SWD programmer, and IO outputs depend on a stable 3.3 V rail.
- Regulator and power-entry mistakes can damage the board before firmware testing starts.

What must be verified later:

- Input source and voltage range.
- Regulator output voltage.
- Regulator thermal margin.
- Whether USB is power-only.
- LED resistor values and polarity.

### 02 - UART Raspberry Pi Header

What this block shows:

- Header H1 with `VCC3V3`, `RX1`, `TX1`, and `GND`.
- A decoupling capacitor near the header.

Why it matters:

- The Raspberry Pi AI host will send `PLAN` messages to the STM32 through this interface.
- The STM32 will send `ACK`, `NACK`, and later `STATUS` messages back.

What must be verified later:

- Physical connector pin 1.
- Raspberry Pi TX/RX crossing.
- 3.3 V logic level only.
- Shared ground.
- Whether the header's `VCC3V3` pin is only a reference or intended to power anything.

### 03 - STM32F103C8T6 MCU Core

What this block shows:

- STM32F103C8T6 symbol.
- VDD/VSS decoupling.
- BOOT0 pulldown.
- PA9/PA10 UART labels.
- PA13/PA14 SWD labels.
- PC13 status LED net.
- NRST/RST net.

Why it matters:

- This block is the source of truth for MCU pin assignment before firmware can safely be written.

What must be verified later:

- Package pinout against the STM32F103C8T6 datasheet.
- BOOT0 low at reset.
- NRST accessibility.
- Oscillator/reset requirements.
- Whether all VDD/VSS pins are connected and decoupled correctly.

### 04 - Traffic Light LED Outputs

What this block shows:

- Six LED outputs on PB9, PB8, PB7, PB6, PB5, and PB3.
- Each output has a 1k series resistor and LED to GND.

Why it matters:

- These are the final visible traffic light outputs for the embedded controller.

What must be verified later:

- Actual LED color and lane ownership.
- GPIO active-high behavior.
- Current per LED.
- Safe initial firmware pattern that does not mislead observers.

### 05 - Dual 7-Segment Display

What this block shows:

- HDSM-577G dual 7-segment display.
- Segment lines on PA8, PB15, PB14, PB13, PB12, PA12, and PA11.
- Digit/common lines on PA3 and PA4.
- 220 ohm segment resistors.

Why it matters:

- The display can show countdown/status information, but it also creates a GPIO current and multiplexing constraint.

What must be verified later:

- Common-anode/common-cathode behavior.
- Segment polarity.
- Digit-drive current.
- Whether transistor drivers are required.
- Whether decimal point is connected.

### 06 - Expansion Headers

What this block shows:

- Multiple GPIO groups broken out to 1x10 headers.
- `VCC5V`, `VCC3V3`, and `GND` header pins.
- UART, LED, and 7-segment related pins also appear on expansion headers.

Why it matters:

- Expansion access is useful for probing and future features, but it can also create pin conflicts.

What must be verified later:

- Header orientation.
- Which pins are already committed to existing PCB loads.
- Whether any expansion pin is 5 V tolerant for the intended use.
- External load limits.

### 07 - SWD Programming Header

What this block shows:

- Header H2 with `VCC3V3`, `SWDIO`, `SWCLK`, and `GND`.

Why it matters:

- SWD is the preferred path for programming and debugging STM32 firmware without STM32CubeIDE.

What must be verified later:

- Pin 1 orientation.
- ST-LINK wiring.
- Target voltage sense behavior.
- Whether NRST is available elsewhere.
- OpenOCD or STM32CubeProgrammer connection after power validation.
