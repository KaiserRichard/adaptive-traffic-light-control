# STM32F103C8T6 PCB Pin Mapping

## Status

```text
Documentation started.
Pin mapping is based on visible schematic block images in docs/hardware/stm32_pcb/.
Final schematic review and hardware validation pending.
```

Do not treat this table as a validated PCB netlist. Any signal marked `Needs verification` or `TBD / verify from schematic` must be checked against the original schematic source, PCB layout, and real board before firmware depends on it.

## Pin Mapping Table

| Signal | STM32 Pin | Direction | Connected To | Purpose | Firmware Role | Validation Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `VCC3V3` | VDD pins | Power input | AMS1117-3.3V output rail | MCU and peripheral 3.3 V supply | Board power prerequisite | Schematic observed; pending hardware validation | Verify rail voltage before connecting programmer or Pi |
| `GND` | VSS pins | Power return | USB ground, regulator ground, headers | Common reference | Board power prerequisite | Schematic observed; pending hardware validation | Pi UART requires shared ground |
| `BOOT0` | BOOT0 | Input | R3 10k to GND | Boot mode select | Normal boot should keep BOOT0 low | Schematic observed; pending hardware validation | Verify pull-down value and soldering |
| `NRST` / `RST` | NRST / TBD verify exact schematic exposure | Input | Reset circuit / test pad / header TBD | MCU reset | Debug/reset support | Needs verification | Confirm whether NRST is accessible on SWD header, reset button, test pad, or only internal net |
| Status LED | PC13 | Output | R2 1k -> L2 -> GND | Board/user status LED | Candidate minimal blink output | Schematic observed; pending hardware validation | PC13 electrical behavior should be verified before relying on it |
| SWDIO | PA13 | Bidirectional debug | SWD header H2 pin 3 | SWD programming/debug | Debug interface | Schematic observed; pending hardware validation | Verify H2 physical pin 1 orientation |
| SWCLK | PA14 | Debug input | SWD header H2 pin 2 | SWD clock | Debug interface | Schematic observed; pending hardware validation | Verify H2 physical pin 1 orientation |
| SWD reference voltage | N/A | Power reference | SWD header H2 pin 4 `VCC3V3` | Programmer target voltage sense | Debug interface | Schematic observed; pending hardware validation | Do not back-power unintentionally from programmer |
| SWD ground | N/A | Power return | SWD header H2 pin 1 `GND` | Programmer ground reference | Debug interface | Schematic observed; pending hardware validation | Ground must be connected before SWD |
| USART1 TX / `TX1` | PA9 | Output | UART header H1 pin 2; expansion header | STM32 transmit to host | Future ACK/NACK/STATUS TX | Schematic observed; pending TX/RX crossing check | Confirm this connects to Raspberry Pi RXD, not TXD |
| USART1 RX / `RX1` | PA10 | Input | UART header H1 pin 3; expansion header | STM32 receive from host | Future PLAN RX | Schematic observed; pending TX/RX crossing check | Confirm this connects to Raspberry Pi TXD, not RXD |
| UART header 3.3 V | N/A | Power/reference | UART header H1 pin 4 `VCC3V3` | Logic reference or limited power reference | Hardware support only | Schematic observed; pending hardware validation | Do not assume Pi should be powered from this pin |
| UART header ground | N/A | Power return | UART header H1 pin 1 `GND` | UART ground reference | Hardware support only | Schematic observed; pending hardware validation | Must share GND with Pi |
| Traffic LED output 1 | PB9 | Output | R5 1k -> L3 -> GND | Traffic LED output | TBD lane/color assignment | Schematic observed; pending hardware validation | LED appears green in schematic image; lane ownership not labeled |
| Traffic LED output 2 | PB8 | Output | R6 1k -> L4 -> GND | Traffic LED output | TBD lane/color assignment | Schematic observed; pending hardware validation | LED appears red in schematic image; lane ownership not labeled |
| Traffic LED output 3 | PB7 | Output | R7 1k -> L5 -> GND | Traffic LED output | TBD lane/color assignment | Schematic observed; pending hardware validation | LED appears yellow in schematic image; lane ownership not labeled |
| Traffic LED output 4 | PB6 | Output | R8 1k -> L6 -> GND | Traffic LED output | TBD lane/color assignment | Schematic observed; pending hardware validation | LED appears green in schematic image; lane ownership not labeled |
| Traffic LED output 5 | PB5 | Output | R9 1k -> L7 -> GND | Traffic LED output | TBD lane/color assignment | Schematic observed; pending hardware validation | LED appears red in schematic image; lane ownership not labeled |
| Traffic LED output 6 | PB3 | Output | R10 1k -> L8 -> GND | Traffic LED output | TBD lane/color assignment | Schematic observed; pending hardware validation | LED appears yellow in schematic image; lane ownership not labeled |
| 7-segment segment `a` | PA8 | Output | 220 ohm resistor -> HDSM-577G segment `a` | Dual 7-segment display | Future multiplexed display output | Schematic observed; pending hardware validation | Verify display polarity and current path |
| 7-segment segment `b` | PB15 | Output | 220 ohm resistor -> HDSM-577G segment `b` | Dual 7-segment display | Future multiplexed display output | Schematic observed; pending hardware validation | Verify display polarity and current path |
| 7-segment segment `c` | PB14 | Output | 220 ohm resistor -> HDSM-577G segment `c` | Dual 7-segment display | Future multiplexed display output | Schematic observed; pending hardware validation | Verify display polarity and current path |
| 7-segment segment `d` | PB13 | Output | 220 ohm resistor -> HDSM-577G segment `d` | Dual 7-segment display | Future multiplexed display output | Schematic observed; pending hardware validation | Verify display polarity and current path |
| 7-segment segment `e` | PB12 | Output | 220 ohm resistor -> HDSM-577G segment `e` | Dual 7-segment display | Future multiplexed display output | Schematic observed; pending hardware validation | Verify display polarity and current path |
| 7-segment segment `f` | PA12 | Output | 220 ohm resistor -> HDSM-577G segment `f` | Dual 7-segment display | Future multiplexed display output | Schematic observed; pending hardware validation | PA12 may also appear on expansion header |
| 7-segment segment `g` | PA11 | Output | 220 ohm resistor -> HDSM-577G segment `g` | Dual 7-segment display | Future multiplexed display output | Schematic observed; pending hardware validation | PA11 may also appear on expansion header |
| 7-segment digit/common A1 | PA3 | Output | HDSM-577G A1 / pin 10 | Digit select/common control | Future multiplexed display output | Schematic observed; pending hardware validation | Verify whether direct GPIO drive is safe |
| 7-segment digit/common A2 | PA4 | Output | HDSM-577G A2 / pin 5 | Digit select/common control | Future multiplexed display output | Schematic observed; pending hardware validation | Verify whether direct GPIO drive is safe |
| 7-segment decimal point | TBD / verify from schematic | Output if populated | 7-segment DP segment if used | Optional display decimal point | Optional display output | Needs verification | Do not use until segment mapping, polarity, and current budget are confirmed |
| Expansion header group 1 | PB9, PB8, PB7, PB6, PB5, PB3, PA15, PA1, PA2 | Bidirectional/TBD | 1x10 expansion header with `VCC5V` | Expansion access | Not assigned | Schematic observed; pending hardware validation | Some pins are already used by traffic LEDs |
| Expansion header group 2 | PA3, PA4, PA5, PA6, PA7, PB0, PB1, PB10, PB11 | Bidirectional/TBD | 1x10 expansion header with `GND` | Expansion access | Not assigned | Schematic observed; pending hardware validation | Some pins are already used by 7-segment digit control |
| Expansion header group 3 | PA12, PA11, PA8, PB15, PB14, PB13, PB12, `RX1`, `TX1` | Bidirectional/TBD | 1x10 expansion header with `VCC3V3` | Expansion access | Not assigned | Schematic observed; pending hardware validation | Many pins are already used by UART or 7-segment display |

## Firmware Mapping Caution

The ESP32 prototype uses named outputs such as `A_RED`, `A_YELLOW`, `A_GREEN`, `B_RED`, `B_YELLOW`, and `B_GREEN`. The STM32 schematic image shows six LED outputs but does not label them by lane or color ownership. Do not port the FSM output table until the hardware owner verifies the intended mapping.

## Validation Checklist

```text
[ ] Confirm STM32 package pinout against the official STM32F103C8T6 datasheet.
[ ] Confirm schematic image pin labels against original schematic source.
[ ] Confirm PCB routing against layout/netlist.
[ ] Confirm physical connector orientation.
[ ] Confirm traffic LED color and lane assignment.
[ ] Confirm 7-segment display polarity and current path.
[ ] Confirm UART TX/RX crossing with Raspberry Pi.
[ ] Confirm SWD header orientation and target voltage.
[ ] Confirm BOOT0 low at reset.
[ ] Confirm NRST access strategy.
```
