# Phase 17.3.2 - Controlled Power Rail Validation

## Purpose

Define the controlled power-up procedure for the STM32F103C8T6 PCB after the Phase 17.3.1 pre-power inspection.

This phase is intended to answer one narrow question:

```text
Can the PCB generate safe and stable power rails before ST-LINK, UART, GPIO, or firmware tests are attempted?
```

## Current Status

```text
Procedure documented.
No powered measurement has been performed in this repository.
Power rail validation is pending hardware-team execution.
```

Do not treat this document as evidence that the board has powered successfully. It is a worksheet for collecting that evidence later.

## Entry Gate

Do not apply power unless Phase 17.3.1 inspection has passed or the hardware reviewer explicitly accepts the remaining risk.

Required pre-power evidence:

```text
[ ] No visible solder bridge on regulator, MCU, USB connector, or power rails.
[ ] STM32 orientation / pin 1 has been checked.
[ ] SS34 diode orientation has been checked.
[ ] AMS1117-3.3 orientation has been checked.
[ ] VCC3V3 to GND resistance is not near 0 ohm.
[ ] VCC5V to GND resistance is not near 0 ohm.
[ ] SWD header orientation is understood, even if ST-LINK is not connected yet.
[ ] Raspberry Pi UART is disconnected.
[ ] ST-LINK is disconnected.
```

If any item is unknown, stop and return to Phase 17.3.1.

## Scope

Included:

- controlled 5 V input application.
- current-limit setup.
- VCC5V and VCC3V3 measurement.
- ground-reference checks.
- AMS1117 temperature observation.
- power LED/status LED observation.
- evidence table for pass/fail decision.

Not included:

- ST-LINK attach.
- read-ID.
- firmware flashing.
- blink firmware.
- GPIO traffic LED test.
- 7-segment display test.
- Raspberry Pi UART connection.
- FreeRTOS or PLAN protocol work.

## Safety Rules

```text
[ ] Use current-limited power if available.
[ ] Keep Raspberry Pi UART disconnected.
[ ] Keep ST-LINK disconnected during the first controlled power check.
[ ] Do not power the board from two sources at once.
[ ] Do not touch or move probe tips across fine-pitch pins while powered.
[ ] Stop immediately on smoke, smell, unexpected heating, or current-limit trip.
[ ] Record measurements before changing hardware.
```

## Required Tools

- Multimeter.
- Current-limited bench supply if available.
- USB 5 V source only if current-limited supply is unavailable and pre-power checks are clean.
- Good lighting and magnification.
- Optional thermal camera or non-contact thermometer.
- Schematic block: `01_power_usb_5v_3v3_regulator.jpg`.
- MCU block: `03_stm32f103c8t6_mcu_core.jpg`.

## Known Power Block Evidence

Observed from repository schematic images:

| Item | Schematic evidence | Validation status |
| --- | --- | --- |
| 5 V input | USB Micro-B VBUS path | Pending measurement |
| Reverse/drop protection | SS34 diode in input path | Orientation must be checked physically |
| 3.3 V regulator | AMS1117-3.3 | Output voltage and thermal behavior pending |
| 3.3 V rail | `VCC3V3` | Pending measurement |
| Input/output capacitors | Capacitors near regulator nodes | Placement/value/soldering pending |
| Power indicator LED | LED path from `VCC3V3` through resistor | Behavior pending |
| MCU supply | STM32 VDD/VSS to `VCC3V3`/GND | Pending measurement |

## Recommended Power Setup

Preferred setup:

```text
Bench supply:
    Voltage: 5.0 V
    Current limit: start conservatively, then adjust only with reviewer approval
    Output: off before wiring
```

Connection:

```text
Bench supply +5 V -> intended VCC5V / USB 5 V input path
Bench supply GND  -> board GND
```

Notes:

- If powering through USB Micro-B, confirm that the source and cable are suitable and that no data connection is being relied on.
- Do not connect ST-LINK target voltage, Raspberry Pi UART, USB-UART, LEDs, or expansion loads during first power.
- The exact safe current limit depends on the assembled board. Record the chosen value and reason.

## First Power Procedure

```text
[ ] Confirm supply output is off.
[ ] Connect GND first.
[ ] Connect +5 V to the intended input path.
[ ] Set current limit.
[ ] Place multimeter ground probe on a known board GND point.
[ ] Turn supply on.
[ ] Observe current immediately.
[ ] Measure VCC5V relative to GND.
[ ] Measure VCC3V3 relative to GND.
[ ] Observe power LED behavior if present.
[ ] Observe AMS1117 temperature after short intervals.
[ ] Turn supply off if anything is abnormal.
```

## Measurement Table

| Check | Expected | Measured | Pass/Fail | Notes |
| --- | --- | --- | --- | --- |
| Input source voltage | Approximately intended 5 V input | TBD | TBD | |
| VCC5V after input path | Near expected post-diode/input rail | TBD | TBD | Depends on SS34 drop and measurement point |
| VCC3V3 regulator output | Approximately 3.3 V | TBD | TBD | Must be safe before ST-LINK |
| Board current at first power | Reasonable and not current-limit tripping | TBD | TBD | Record current-limit setting |
| GND reference between measurement points | Continuity / same reference | TBD | TBD | Power off if uncertain |
| Power LED behavior | Consistent with schematic | TBD | TBD | Do not use LED alone as voltage proof |
| AMS1117 temperature after 10 s | No rapid heating | TBD | TBD | Use caution |
| AMS1117 temperature after 60 s | Stable / not overheating | TBD | TBD | Stop if heating rapidly |

## Optional Ripple / Stability Checks

If an oscilloscope is available:

| Check | Expected | Measured | Pass/Fail | Notes |
| --- | --- | --- | --- | --- |
| VCC3V3 ripple at idle | No obvious excessive ripple | TBD | TBD | Scope bandwidth/probe setup noted |
| VCC3V3 dip during reset or LED changes | Not tested yet | TBD | TBD | Later phase only |

Oscilloscope checks are useful but not required for the first pass if a multimeter confirms a safe DC rail and the board current is reasonable.

## Failure Symptoms and Immediate Actions

| Symptom | Possible cause | Immediate action |
| --- | --- | --- |
| Current limit trips immediately | Rail short, reversed part, solder bridge | Power off, return to inspection |
| VCC3V3 is 0 V | Regulator issue, missing input, solder fault | Power off, inspect input/regulator path |
| VCC3V3 is too high | Regulator wrong part/orientation/fault | Power off immediately |
| AMS1117 heats rapidly | Short or excessive load | Power off immediately |
| Power LED off but VCC3V3 valid | LED orientation/resistor issue | Record, do not assume board failure |
| Smoke/smell | Severe electrical fault | Power off and do not reapply power |

## Decision Gate Before ST-LINK

Proceed to Phase 17.3.3 only if:

```text
[ ] VCC3V3 is measured near 3.3 V.
[ ] VCC5V/input rail behavior is understood.
[ ] Board current is reasonable and stable.
[ ] AMS1117 does not heat rapidly.
[ ] No visible abnormal behavior occurs.
[ ] Power source and ground reference are documented.
[ ] Photos and measurements are saved.
```

Do not connect ST-LINK if:

```text
[ ] VCC3V3 is missing, too high, or unstable.
[ ] Current draw is unexplained.
[ ] The regulator heats rapidly.
[ ] GND reference is uncertain.
[ ] SWD header orientation is still uncertain.
```

## Evidence Package

Before marking Phase 17.3.2 complete, collect:

```text
[ ] Photo of power wiring.
[ ] Photo or note of current-limit setting.
[ ] Multimeter reading for VCC5V.
[ ] Multimeter reading for VCC3V3.
[ ] Current draw reading.
[ ] Regulator temperature observation.
[ ] Notes on power LED behavior.
[ ] Reviewer/date/PCB revision.
```

## What Is Not Tested Yet

- ST-LINK target detection.
- SWD read-ID.
- chip erase or flashing.
- firmware build.
- blink firmware.
- traffic LED outputs.
- 7-segment display.
- UART with Raspberry Pi.
- FreeRTOS behavior.

## Next Step

If controlled power validation passes:

```text
Phase 17.3.3 - ST-LINK Attach / Read-ID Validation
```

If controlled power validation fails:

```text
Stop hardware bring-up and repair the power fault before connecting ST-LINK or any host device.
```
