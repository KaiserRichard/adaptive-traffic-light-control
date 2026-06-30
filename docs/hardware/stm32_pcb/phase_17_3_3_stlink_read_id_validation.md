# Phase 17.3.3 - ST-LINK Attach / Read-ID Validation

## Purpose

Define the first non-flashing SWD validation procedure for the STM32F103C8T6 PCB.

This phase is intended to answer one narrow question:

```text
Can an SWD probe detect the STM32 target and read basic target identity information without erasing or flashing the chip?
```

## Current Status

```text
Procedure documented.
ST-LINK attach has not been performed in this repository.
Read-ID validation is pending real hardware and approved tools.
```

Do not treat this document as evidence that ST-LINK works. It is a controlled validation plan and result worksheet.

## Entry Gate

Do not connect ST-LINK unless Phase 17.3.2 controlled power validation has passed.

Required evidence:

```text
[ ] VCC3V3 measured near 3.3 V.
[ ] Board current is reasonable and stable.
[ ] AMS1117 does not heat rapidly.
[ ] SWD header orientation is confirmed.
[ ] ST-LINK cable orientation is confirmed.
[ ] ST-LINK GND, SWDIO, SWCLK, and target-voltage reference mapping are confirmed.
[ ] Raspberry Pi UART is disconnected.
```

If any item is unknown, stop and return to Phase 17.3.1 or Phase 17.3.2.

## Scope

Included:

- SWD cable/header inspection.
- target-voltage reference check.
- non-flashing target attach.
- read-ID or equivalent target identity check.
- capture of raw tool output.
- failure triage if target is not detected.

Not included:

- firmware flashing.
- chip erase.
- option byte modification.
- unlocking protected flash.
- blink firmware.
- UART testing.
- GPIO output testing.
- FreeRTOS porting.

## Safety Rules

```text
[ ] Do not erase the chip.
[ ] Do not flash firmware.
[ ] Do not modify option bytes.
[ ] Do not connect Raspberry Pi UART.
[ ] Do not back-power the board unintentionally from ST-LINK.
[ ] Do not continue if target voltage is missing or wrong.
[ ] Capture exact command/tool output.
```

## Required Tools

Hardware:

- Powered STM32 PCB that passed Phase 17.3.2.
- ST-LINK or compatible SWD probe.
- Multimeter.
- Confirmed SWD cable or jumper wires.

Software, later only after approval:

- STM32CubeProgrammer CLI, or
- OpenOCD with STM32F1 target support, or
- another documented SWD read-ID tool.

No tool installation is performed by this document.

## Known SWD Header Evidence

Observed from repository schematic images:

```text
H2 pin 1: GND
H2 pin 2: SWCLK
H2 pin 3: SWDIO
H2 pin 4: VCC3V3 reference
```

Observed MCU core labels:

```text
PA13: SWDIO
PA14: SWCLK
NRST/RST net: present in MCU schematic, but not visibly exposed on H2
BOOT0: pulled down through R3 10k in schematic
```

Validation status:

```text
Schematic observed.
Physical header orientation and cable mapping still require real-board confirmation.
```

## Connection Checklist

```text
[ ] Board is powered from the validated power source.
[ ] ST-LINK GND connects to board GND.
[ ] ST-LINK SWDIO connects to H2 SWDIO.
[ ] ST-LINK SWCLK connects to H2 SWCLK.
[ ] ST-LINK target voltage/reference connects to board VCC3V3 reference.
[ ] ST-LINK is not powering the board unless explicitly reviewed.
[ ] NRST strategy is documented if attach is unstable.
[ ] BOOT0 default state is low or pending powered measurement.
```

## Non-Flashing Attach Procedure

1. Power the board using the validated Phase 17.3.2 setup.
2. Measure VCC3V3 once more before connecting ST-LINK.
3. Turn off power if wiring changes are needed.
4. Connect ST-LINK GND, SWDIO, SWCLK, and VCC3V3 reference.
5. Turn on board power.
6. Confirm ST-LINK reports target voltage near 3.3 V if the tool exposes it.
7. Run a connect/read-ID command only.
8. Save the full command and raw output.
9. Disconnect safely if the target is not detected.

## Example Commands for Later

These are examples for a future approved toolchain. Do not run them until the tool is installed and hardware entry gates have passed.

STM32CubeProgrammer CLI example:

```bash
STM32_Programmer_CLI -c port=SWD mode=UR
```

OpenOCD example:

```bash
openocd -f interface/stlink.cfg -f target/stm32f1x.cfg \
  -c "init; targets; mdw 0xE0042000 1; shutdown"
```

Notes:

- Exact command syntax depends on installed tool version and ST-LINK model.
- A successful result should identify an STM32F1-class target or read the debug ID register.
- Do not add erase, write, program, reset-run, or option-byte commands in this phase.

## Expected Result

Passing result:

```text
[ ] Tool detects target voltage near 3.3 V.
[ ] Tool connects over SWD.
[ ] Tool reports STM32F1 / STM32F103-class target identity or readable debug ID.
[ ] No erase or flash operation is executed.
[ ] Board remains powered and stable.
```

The exact reported device name can vary by tool. Record the raw output rather than rewriting it.

## Measurement / Output Table

| Check | Expected | Observed | Pass/Fail | Notes |
| --- | --- | --- | --- | --- |
| VCC3V3 before ST-LINK | Approximately 3.3 V | TBD | TBD | |
| ST-LINK target voltage reading | Approximately 3.3 V | TBD | TBD | If tool reports it |
| ST-LINK GND continuity | Confirmed | TBD | TBD | Power off check |
| SWDIO mapping | H2 SWDIO to PA13 | TBD | TBD | Power off check if needed |
| SWCLK mapping | H2 SWCLK to PA14 | TBD | TBD | Power off check if needed |
| Tool command | Read/connect only | TBD | TBD | Paste exact command |
| Tool output | Target detected/readable ID | TBD | TBD | Attach raw log |
| Board stability during attach | No brownout/reset loop | TBD | TBD | |

## Failure Triage

| Failure symptom | Likely area | Checks |
| --- | --- | --- |
| No target voltage | Power/reference | VCC3V3, H2 pin 4, ST-LINK Vref wiring |
| Cannot connect, target voltage valid | SWD wiring | SWDIO/SWCLK swap, header mirrored, PA13/PA14 solder |
| Connect unstable | Reset/power | NRST stuck low, brownout, regulator heating |
| Tool sees wrong/unknown device | Tool config or wiring | STM32F1 target config, SWD speed, cable mapping |
| Board current changes sharply on attach | Power path | Back-powering, wrong Vref, accidental short |
| Attach only works with reset | NRST/recovery | Document reset strategy; do not flash yet |

## Recovery Notes

- If NRST is available, a future "connect under reset" workflow may help recovery.
- The current SWD header image does not visibly expose NRST, so confirm whether a reset button, test pad, or header is available.
- BOOT0 should be low for normal boot. If BOOT0 behavior is uncertain, measure it during the powered validation step.
- Do not use mass erase as a diagnostic shortcut in this phase.

## Evidence Package

Before marking Phase 17.3.3 complete, collect:

```text
[ ] Photo of ST-LINK wiring.
[ ] Photo/diagram of H2 pin 1 orientation.
[ ] Tool name and version.
[ ] Exact connect/read-ID command.
[ ] Raw command output.
[ ] Target voltage reading.
[ ] VCC3V3 measurement during attach.
[ ] Notes on reset/BOOT0 state.
[ ] Reviewer/date/PCB revision.
```

## What Is Not Tested Yet

- firmware flashing.
- chip erase.
- blink firmware.
- traffic LED outputs.
- 7-segment display.
- UART link.
- Raspberry Pi integration.
- STM32 FreeRTOS port.

## Decision Gate Before Blink Firmware

Proceed to Phase 17.3.4 only if:

```text
[ ] Phase 17.3.2 controlled power validation passed.
[ ] ST-LINK target attach/read-ID passed.
[ ] Tool version and raw output are recorded.
[ ] No power instability is observed during attach.
[ ] Firmware toolchain selection is approved.
[ ] Startup/linker/vendor-file policy is approved.
```

Do not start blink firmware if read-ID fails or power behavior is unstable.

## Next Step

If ST-LINK attach/read-ID validation passes:

```text
Phase 17.3.4 - Minimal Blink Firmware
```

If ST-LINK attach/read-ID validation fails:

```text
Return to power, SWD wiring, NRST, BOOT0, soldering, and header-orientation triage before any firmware work.
```
