# Phase 17.3.1 - STM32 PCB Pre-Power Inspection and ST-LINK Non-Detection Triage

## Purpose

Provide a power-off inspection checklist and triage worksheet for the first STM32 PCB bring-up step.

This document is written for the current hardware-team symptom:

```text
Hardware team reported that ST-LINK does not detect the STM32 chip.
Root cause not confirmed yet.
```

The goal is to collect evidence before applying power, connecting ST-LINK again, connecting Raspberry Pi UART, or attempting firmware work.

## Current Symptom

Reported symptom:

```text
ST-LINK does not recognize / detect the STM32F103C8T6.
```

Interpretation:

- The symptom is not yet a firmware problem.
- The root cause may be power, ground, SWD wiring, reset, boot mode, soldering, orientation, or physical connector mismatch.
- Do not debug UART, FreeRTOS, PLAN protocol, LEDs, or seven-segment display until power and SWD basics are understood.

## Scope

Included:

- schematic evidence review.
- power-off visual inspection.
- power-off resistance and continuity checks.
- SWD header, SWDIO, SWCLK, GND, and 3.3 V reference checks.
- NRST and BOOT0 inspection.
- possible root-cause triage for ST-LINK non-detection.

Not included:

- no firmware build.
- no firmware flashing.
- no chip erase.
- no UART test.
- no GPIO test.
- no FreeRTOS work.
- no Raspberry Pi connection.
- no claim that hardware is validated.

## Safety Rules

```text
[ ] Do not flash firmware in this phase.
[ ] Do not erase the chip in this phase.
[ ] Do not connect Raspberry Pi UART in this phase.
[ ] Do not connect ST-LINK until power/header orientation is reviewed.
[ ] Do not power the board if resistance checks suggest a rail short.
[ ] Do not let ST-LINK back-power the board unintentionally.
[ ] Record evidence before changing wiring or reworking solder.
```

## Required Tools

- PCB under inspection.
- Schematic block images from `docs/hardware/stm32_pcb/`.
- Multimeter with continuity and resistance modes.
- Magnifier, microscope, or phone macro camera.
- Good lighting.
- Optional: board photos, PCB layout, and original schematic source.
- Later only, after decision gates: current-limited bench supply and ST-LINK.

## Documents Reviewed

- `docs/embedded/phase_17_ledger.md`
- `docs/hardware/stm32_pcb/README.md`
- `docs/hardware/stm32_pcb/stm32f103c8t6_pin_mapping.md`
- `docs/hardware/stm32_pcb/stm32_pcb_review_notes.md`
- `docs/hardware/stm32_pcb/stm32_pcb_bringup_plan.md`
- `docs/hardware/stm32_pcb/stm32_uart_pi_validation_plan.md`
- `docs/hardware/stm32_pcb/stm32_hardware_blocks_explained.md`
- `firmware/stm32_f103_traffic_light/phase_17_2_closure_decision.md`

## Known Relevant Hardware Blocks

| Block | Repository evidence | Relevance to ST-LINK detection |
| --- | --- | --- |
| Power input | `01_power_usb_5v_3v3_regulator.jpg` | ST-LINK cannot attach reliably without valid board power/reference |
| 3.3 V regulator | SS34 + AMS1117-3.3 in power schematic | Must provide stable `VCC3V3` for MCU and SWD reference |
| MCU core | `03_stm32f103c8t6_mcu_core.jpg` | Contains VDD/VSS, BOOT0, NRST, SWDIO, SWCLK, UART labels |
| SWD header | `07_swd_programming_header.jpg` | Exposes GND, SWCLK, SWDIO, and VCC3V3 reference |
| BOOT0 | MCU schematic shows BOOT0 through R3 10k to GND | Floating/wrong BOOT0 can affect boot/debug behavior |
| NRST | MCU schematic shows `RST`/NRST net | Reset access may be needed for recovery, but SWD header does not visibly expose NRST |
| UART header | `02_uart_pi_header.jpg` | Not used in this phase; keep disconnected |
| Traffic LEDs | `04_traffic_light_led_outputs.jpg` | Not relevant until power/SWD basics are solved |
| 7-segment display | `05_dual_7segment_display.jpg` | Not relevant until power/SWD basics are solved |

## ST-LINK Non-Detection Triage Map

```text
ST-LINK does not detect STM32
    -> Is 3.3 V present and stable?
    -> Is ST-LINK GND connected to board GND?
    -> Is SWDIO routed to PA13?
    -> Is SWCLK routed to PA14?
    -> Is the SWD header pinout correct for the actual ST-LINK cable?
    -> Is the ST-LINK target-voltage reference connected to board 3.3 V?
    -> Is NRST accessible and normally high when powered?
    -> Is BOOT0 pulled to the intended default state?
    -> Is STM32 soldered and oriented correctly?
    -> Is there a short on 3V3/GND, SWDIO, SWCLK, or reset?
```

Primary triage blocks:

- Power Block.
- MCU Block.
- SWD / NRST Block.

Do not debug UART, FreeRTOS, PLAN protocol, LEDs, or seven-segment display until these blocks are understood.

## Power-Off Visual Inspection Checklist

```text
[ ] Photograph top side of PCB.
[ ] Photograph bottom side of PCB.
[ ] Inspect USB Micro connector solder joints.
[ ] Inspect SS34 diode orientation.
[ ] Inspect AMS1117-3.3 orientation.
[ ] Inspect regulator input/output pins for solder bridges.
[ ] Inspect input/output capacitors near the regulator.
[ ] Inspect STM32 orientation / pin 1.
[ ] Inspect all STM32 pins for solder bridges.
[ ] Inspect all STM32 pins for lifted pins or cold joints.
[ ] Inspect BOOT0 resistor R3 / 10k path if visible.
[ ] Inspect NRST route / reset circuit if visible.
[ ] Inspect SWD header orientation and pin numbering.
[ ] Inspect SWD header solder joints.
[ ] Inspect for visible copper damage, scratches, or flux bridges.
```

## Power-Off Continuity / Resistance Checklist

Use resistance mode for rail checks. Use continuity mode only where a direct connection is expected.

```text
[ ] Measure resistance between VCC3V3 and GND before power.
[ ] Measure resistance between VCC5V and GND before power.
[ ] Check continuity from SWD header GND to board GND.
[ ] Check continuity from SWD header VCC3V3 reference to board VCC3V3.
[ ] Check continuity from SWD header SWDIO to STM32 PA13.
[ ] Check continuity from SWD header SWCLK to STM32 PA14.
[ ] Check for accidental continuity between SWDIO and GND.
[ ] Check for accidental continuity between SWCLK and GND.
[ ] Check for accidental continuity between SWDIO and SWCLK.
[ ] Check whether NRST is exposed on the SWD header, reset button, or test pad.
```

Do not apply power if either rail appears near-short to ground.

## SWD Header Inspection Checklist

Repository schematic evidence for H2:

```text
H2 pin 1: GND
H2 pin 2: SWCLK
H2 pin 3: SWDIO
H2 pin 4: VCC3V3 reference
```

Inspection:

```text
[ ] Confirm physical pin 1 on the PCB header.
[ ] Confirm the installed header is not rotated or mirrored.
[ ] Confirm ST-LINK cable pinout matches the board header order.
[ ] Confirm ST-LINK GND would connect to board GND.
[ ] Confirm ST-LINK SWCLK would connect to board SWCLK.
[ ] Confirm ST-LINK SWDIO would connect to board SWDIO.
[ ] Confirm ST-LINK target voltage/reference would connect to VCC3V3.
[ ] Confirm ST-LINK is not expected to power the PCB unless explicitly reviewed.
```

Common failure mode:

```text
Schematic pin order correct, but physical cable/header orientation reversed.
```

## NRST / BOOT0 Inspection Checklist

NRST:

```text
[ ] Locate NRST / RST net on schematic and PCB if possible.
[ ] Confirm whether NRST is available on SWD header, test pad, reset button, or only internal net.
[ ] Check for visible solder issue around NRST pin.
[ ] Later powered check: verify NRST is normally high.
[ ] Later powered check: verify NRST is not stuck low.
```

BOOT0:

```text
[ ] Inspect BOOT0 pull-down resistor R3 / 10k.
[ ] Confirm BOOT0 resistor is populated.
[ ] Confirm no visible short pulls BOOT0 high.
[ ] Later powered check: verify BOOT0 is low by default.
```

Why this matters:

- NRST stuck low can prevent normal SWD attach.
- NRST access can help recovery if attach is unstable.
- BOOT0 floating or wrong can send the MCU into an unexpected boot mode.

## MCU Orientation and Soldering Checklist

```text
[ ] Confirm STM32 package orientation against schematic/PCB pin 1 marker.
[ ] Inspect pin 1 area with magnification.
[ ] Inspect PA13 / SWDIO pin area.
[ ] Inspect PA14 / SWCLK pin area.
[ ] Inspect NRST pin area.
[ ] Inspect BOOT0 pin area.
[ ] Inspect VDD and VSS pin areas.
[ ] Inspect for bridges between adjacent pins.
[ ] Inspect for unconnected/lifted pins.
[ ] Clean flux residue if it may cause leakage or visual ambiguity.
```

## Power Block Inspection Checklist

```text
[ ] Confirm USB Micro connector is mechanically sound.
[ ] Confirm USB VBUS path to SS34 diode input.
[ ] Confirm SS34 diode orientation matches intended current direction.
[ ] Confirm AMS1117 input pin receives post-diode 5 V rail later.
[ ] Confirm AMS1117 output pin connects to VCC3V3 rail.
[ ] Confirm regulator ground pin connects to board GND.
[ ] Confirm C1/C2/C3 style capacitors are populated and not shorted.
[ ] Confirm power LED path does not short VCC3V3 to GND.
[ ] Confirm PC13 LED path does not short PC13 to GND or VCC3V3.
```

No powered voltage is claimed in this phase. Powered checks must wait for the controlled power validation step.

## UART Header Do-Not-Connect Warning

Do not connect Raspberry Pi UART during this phase.

Reasons:

- ST-LINK non-detection must be triaged at power/SWD/reset/boot level first.
- A Raspberry Pi connection can introduce another power/ground path.
- UART wiring cannot fix SWD detection.
- Raspberry Pi GPIO UART is 3.3 V only and must not be exposed to uncertain board power states.

## Measurement Table Template

Power-off checks:

| Check | Expected | Measured | Pass/Fail | Notes |
| --- | --- | --- | --- | --- |
| 3V3 to GND resistance, power off | Not near 0 ohm | TBD | TBD | |
| 5V to GND resistance, power off | Not near 0 ohm | TBD | TBD | |
| SWD header GND to board GND | Continuity | TBD | TBD | |
| SWD header 3V3 to board 3V3 | Continuity | TBD | TBD | |
| SWDIO header to PA13 | Continuity | TBD | TBD | |
| SWCLK header to PA14 | Continuity | TBD | TBD | |
| SWDIO to GND, power off | No short | TBD | TBD | |
| SWCLK to GND, power off | No short | TBD | TBD | |
| SWDIO to SWCLK, power off | No short | TBD | TBD | |

Later powered checks, not part of this phase:

| Check | Expected | Measured | Pass/Fail | Notes |
| --- | --- | --- | --- | --- |
| 3V3 rail voltage | Approximately 3.3 V | TBD | TBD | power-on only |
| 5V rail voltage | Approximately intended input/post-diode rail | TBD | TBD | power-on only |
| NRST idle voltage | Approximately 3.3 V | TBD | TBD | power-on only |
| BOOT0 idle voltage | LOW by default | TBD | TBD | power-on only |
| ST-LINK target voltage reading | Approximately 3.3 V | TBD | TBD | ST-LINK attach step only |

## Photo Evidence Checklist

```text
[ ] Full PCB top side.
[ ] Full PCB bottom side.
[ ] USB Micro connector.
[ ] SS34 diode.
[ ] AMS1117 regulator.
[ ] STM32 package orientation and pin 1.
[ ] Close-up of STM32 PA13 / PA14 side.
[ ] Close-up of NRST / BOOT0 area if visible.
[ ] SWD header top view with pin 1 marked.
[ ] SWD cable orientation photo before connecting.
[ ] Multimeter resistance reading for VCC3V3 to GND.
[ ] Multimeter resistance reading for VCC5V to GND.
```

## Possible Root Causes

| Root cause | Evidence to check | Notes |
| --- | --- | --- |
| No 3.3 V rail | Later powered voltage check | Do not power until resistance checks pass |
| 3.3 V/GND short | Power-off resistance near 0 ohm | Stop and inspect/rework |
| ST-LINK GND missing | Continuity from H2 GND to board GND | Required for SWD logic reference |
| SWDIO/SWCLK swapped | Continuity to PA13/PA14 and cable orientation | Common header/cable issue |
| Header mirrored | Physical pin 1 mismatch | Compare PCB silkscreen, schematic, cable |
| Target voltage reference missing | H2 VCC3V3 continuity | ST-LINK may report no target voltage |
| NRST stuck low | Later powered NRST voltage | Can block normal operation/attach |
| BOOT0 wrong/floating | R3 inspection and later powered BOOT0 voltage | May alter boot mode |
| MCU rotated or poorly soldered | Visual inspection | Check pin 1 and fine-pitch solder bridges |
| Wrong MCU or damaged MCU | Marking/orientation/current symptoms | Only consider after simpler checks |

## Decision Gate Before Applying Power

Apply controlled power only if:

```text
[ ] No visible solder bridge or reversed critical component.
[ ] VCC3V3 to GND is not near 0 ohm.
[ ] VCC5V to GND is not near 0 ohm.
[ ] USB/diode/regulator orientation is understood.
[ ] STM32 orientation is understood.
[ ] SWD header orientation is understood.
```

Do not apply power if:

```text
[ ] Any rail appears shorted.
[ ] Regulator or diode orientation is uncertain.
[ ] STM32 orientation is uncertain.
[ ] A solder bridge is visible on MCU or regulator pins.
```

## Decision Gate Before Connecting ST-LINK

Connect ST-LINK only after:

```text
[ ] Power-off rail checks pass.
[ ] Controlled power validation confirms VCC3V3 is present and safe.
[ ] SWD header pin 1 and cable orientation are confirmed.
[ ] ST-LINK GND, SWDIO, SWCLK, and target-voltage reference mapping are confirmed.
[ ] Back-powering risk is understood.
[ ] NRST/BOOT0 status is reviewed or marked as a known risk.
```

First ST-LINK action should be non-flashing target detection/read-ID only. Do not erase or program the device.

## What Is Not Tested Yet

- 3.3 V rail voltage.
- 5 V rail voltage.
- AMS1117 thermal behavior.
- ST-LINK target detection.
- SWD attach.
- NRST voltage.
- BOOT0 voltage.
- UART connection.
- Raspberry Pi integration.
- traffic LED outputs.
- 7-segment display.
- STM32 firmware build.
- STM32 firmware flash.

## Next Step

If power-off inspection passes:

```text
Phase 17.3.2 - Controlled Power Rail Validation
```

If power-off inspection fails:

```text
Fix PCB/soldering/header issue before applying power or connecting ST-LINK.
```
