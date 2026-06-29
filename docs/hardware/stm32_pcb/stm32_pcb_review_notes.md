# STM32 PCB Review Notes

## Status

```text
Documentation started.
This is a pre-bring-up review checklist.
No hardware validation has been performed in this phase.
```

Use this checklist before applying power, connecting ST-LINK, or connecting the Raspberry Pi UART header.

## Power Input

- Status: Needs verification.
- Confirm the intended input source: USB Micro-B 5 V, external 5 V header, or both.
- Confirm that the SS34 diode orientation matches the intended current path.
- Check for shorts between `VCC5V`, `VCC3V3`, and `GND` before power.
- Use a current-limited bench supply for first power if available.

## 3.3 V Regulation

- Status: Needs verification.
- The schematic shows an AMS1117-3.3V regulator generating `VCC3V3`.
- Measure 3.3 V without the Raspberry Pi attached.
- Confirm that the regulator output remains within tolerance under expected LED and display load.
- Confirm input and output capacitors are populated correctly.

## AMS1117 Thermal Margin

- Status: Needs verification.
- AMS1117 dissipation depends on input voltage and load current.
- Estimate power before long tests:

```text
P_dissipation = (V_in - 3.3 V) * I_load
```

- A 5 V input gives about 1.7 V drop across the regulator.
- LED and 7-segment current can dominate the load if too many outputs are on at once.
- Do not claim thermal margin until measured or calculated with real current.

## Decoupling Capacitors

- Status: Needs verification.
- The schematic images show 100 nF-style capacitors near MCU supply rails and regulator nodes.
- Confirm physical placement near STM32 VDD/VSS pins.
- Confirm capacitor values and solder quality.

## BOOT0 Pulldown

- Status: Schematic observed; pending hardware validation.
- The MCU core image shows BOOT0 pulled to GND through R3 10k.
- Verify BOOT0 voltage is low during reset.
- A floating BOOT0 can prevent normal flash boot.

## NRST Availability

- Status: Needs verification.
- The MCU core image labels an `RST`/NRST net.
- The SWD header image does not visibly include NRST.
- Confirm whether NRST is available on a test pad, reset button, header, or only as a net.
- Debug workflows are easier and more reliable if NRST is accessible.

## SWD Header

- Status: Schematic observed; pending hardware validation.
- The SWD image shows `VCC3V3`, `SWDIO`, `SWCLK`, and `GND`.
- Verify physical pin 1 orientation before connecting ST-LINK.
- Confirm ST-LINK target voltage sensing does not back-power the board.
- Do not connect SWD until 3.3 V and GND are confirmed.

## UART TX/RX Crossing

- Status: Needs verification.
- The schematic labels STM32 `TX1` and `RX1`, with PA9 as `TX1` and PA10 as `RX1`.
- Correct Pi connection should be:

```text
STM32 TX1 -> Raspberry Pi RXD
STM32 RX1 <- Raspberry Pi TXD
GND       -> Raspberry Pi GND
```

- Header labels alone do not prove the cable is crossed correctly.
- Confirm with schematic, connector orientation, and continuity tests.

## Raspberry Pi 3.3 V UART Safety

- Status: Needs verification.
- Raspberry Pi GPIO UART is 3.3 V logic only.
- Do not connect 5 V UART signals.
- Do not power one board through a signal line while the other board is off.
- Use a shared ground reference.

## Traffic Light LED Current Limiting

- Status: Schematic observed; pending hardware validation.
- The traffic LED block shows 1k resistors in series with each LED.
- Verify actual resistor values on PCB.
- Confirm LED polarity.
- Confirm GPIO drive direction: STM32 pin high appears to source current into LED to GND.

## 7-Segment Current Budget

- Status: Needs verification.
- The dual 7-segment block shows 220 ohm segment resistors.
- Multiple segments can be active at once, so total current must be checked.
- Confirm whether digit/common pins PA3 and PA4 can safely drive the display directly.
- Prefer multiplexing and conservative duty cycle until current is measured.

## STM32 GPIO Current Limits

- Status: Needs verification.
- Check the official STM32F103C8T6 datasheet for per-pin and total package current limits.
- Avoid turning on every LED and every 7-segment segment continuously during first firmware tests.
- Treat schematic resistor values as necessary but not sufficient proof of safe current.

## USB Power-Only Note

- Status: Needs verification.
- The USB Micro-B schematic shows VBUS and GND usage.
- USB D+/D- are not shown as connected in the current image.
- Treat USB as power-only unless the full schematic proves otherwise.

## Ground Reference

- Status: Required.
- STM32 PCB, Raspberry Pi, SWD programmer, and measurement instruments must share a safe ground reference where required.
- Avoid connecting UART without common ground.
- Avoid multiple power paths that create unintended current through ground or IO pins.

## Expansion Headers

- Status: Schematic observed; pending hardware validation.
- Many expansion header pins are also used by LEDs, UART, or 7-segment display.
- Do not assign expansion functions in firmware until conflicts are reviewed.
- Label connector orientation in bring-up photos or notes.

## Mechanical / Connector Orientation

- Status: Needs verification.
- Mark pin 1 on UART, SWD, and expansion headers before connecting cables.
- Check for mirrored header orientation between schematic symbol and physical PCB.
- Confirm connector pitch and cable keying.
- Photograph the board before first power and before each wiring change.

## Review Outcome Template

```text
Review date:
Reviewer:
PCB revision:
Schematic revision/source:

Power checks:
SWD checks:
UART checks:
LED/display checks:
Risks found:
Decision:
    [ ] Safe to power with current limit
    [ ] Hold for rework
    [ ] More schematic/layout review required
```
