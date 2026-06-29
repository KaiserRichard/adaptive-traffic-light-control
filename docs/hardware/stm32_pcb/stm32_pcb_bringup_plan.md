# STM32 PCB Bring-Up Plan

## Status

```text
Planned.
No step in this bring-up procedure has been executed in Phase 17.
Hardware validation pending.
```

This document is a safe bring-up procedure for the STM32F103C8T6 PCB. It is written to avoid damaging the board and to keep evidence traceable.

## Phase 0 - Visual Inspection

Goal:

- Catch obvious assembly, soldering, orientation, and connector problems before power is applied.

Tools needed:

- Good lighting.
- Magnifier or microscope.
- Multimeter.
- Schematic images and PCB photo if available.

Procedure:

1. Photograph both sides of the PCB.
2. Inspect STM32 pin alignment and solder bridges.
3. Check regulator, diode, capacitors, LEDs, display, and headers for orientation.
4. Confirm pin 1 markings for UART, SWD, and expansion headers.
5. Measure resistance between `VCC3V3` and `GND`.
6. Measure resistance between `VCC5V` and `GND`.

Expected result:

- No visible solder bridges.
- No obvious reversed polarized parts.
- No near-short between power rails and ground.

Failure symptoms:

- Solder bridge on STM32 pins.
- Regulator or diode installed backward.
- Very low resistance between power and ground.
- Connector orientation unclear.

Do not continue if:

- Any power rail appears shorted.
- SWD or UART pin orientation cannot be identified.
- The board has visible damage.

Notes:

- Record photos and resistance readings before power.

## Phase 1 - Power Validation

Goal:

- Confirm that the board generates a stable 3.3 V rail safely.

Tools needed:

- Current-limited bench supply if available.
- Multimeter.
- Optional oscilloscope for ripple/transient checks.

Procedure:

1. Do not connect Raspberry Pi or ST-LINK yet.
2. Power the board through the intended 5 V input with current limit enabled.
3. Measure `VCC5V`, `VCC3V3`, and ground.
4. Check AMS1117 temperature by cautious touch or thermal camera after short intervals.
5. Confirm power LED behavior if present.

Expected result:

- `VCC3V3` measures near 3.3 V.
- Current draw is reasonable for an unprogrammed board.
- Regulator does not heat rapidly.

Failure symptoms:

- Current limit immediately triggers.
- 3.3 V rail missing or too high.
- Regulator overheats.
- Smoke, smell, or visible LED/display abnormality.

Do not continue if:

- 3.3 V is out of tolerance.
- Current draw is unexplained.
- Regulator overheats.

Notes:

- Do not power the Raspberry Pi from the STM32 board unless explicitly reviewed.

## Phase 2 - SWD Connection

Goal:

- Confirm that the STM32 can be detected through SWD.

Tools needed:

- ST-LINK or compatible SWD probe.
- STM32CubeProgrammer CLI or OpenOCD in a future setup.
- Multimeter.

Procedure:

1. Verify SWD header orientation.
2. Connect `GND`, `SWDIO`, `SWCLK`, and target voltage reference.
3. Avoid back-powering the board from the programmer.
4. Attempt a non-flashing connect/read-ID operation.
5. Record tool version and command output.

Expected result:

- Tool detects the STM32 target.
- Target voltage is reported correctly.
- No firmware flashing is needed for this phase.

Failure symptoms:

- Cannot connect.
- Target voltage reads 0 V or incorrect.
- SWDIO/SWCLK swapped.
- BOOT0/reset issue prevents stable connection.

Do not continue if:

- SWD wiring is uncertain.
- Target voltage is wrong.
- The board resets or browns out during connection.

Notes:

- NRST access may be useful if SWD attach is unreliable.

## Phase 3 - Minimal Blink Firmware

Goal:

- Prove that the MCU can run a minimal firmware image and toggle one safe output.

Tools needed:

- Selected command-line build toolchain.
- SWD programmer.
- Multimeter or oscilloscope.

Procedure:

1. Select a minimal blink target, likely PC13 only after confirming schematic behavior.
2. Build firmware from a documented command-line project.
3. Flash only after SWD and power are validated.
4. Observe LED or measure pin toggling.

Expected result:

- The selected output toggles at the expected interval.
- Board remains powered and stable.

Failure symptoms:

- Firmware does not start.
- SWD flash fails.
- PC13 LED polarity is unexpected.
- Clock/reset setup is wrong.

Do not continue if:

- Power becomes unstable.
- Flash/debug workflow is not repeatable.
- The output pin cannot be confirmed.

Notes:

- This phase is not implemented in the current documentation-only work.

## Phase 4 - GPIO Traffic Light Output Test

Goal:

- Identify and validate each traffic light LED output one at a time.

Tools needed:

- Minimal GPIO firmware.
- Multimeter or visual inspection.
- Pin mapping table.

Procedure:

1. Drive PB9, PB8, PB7, PB6, PB5, and PB3 one at a time.
2. Keep duty cycle conservative.
3. Record observed LED color and physical position.
4. Map each pin to final traffic signal meaning only after observation.

Expected result:

- Each intended LED turns on individually.
- No unintended LEDs turn on.
- Pin-to-color mapping is documented.

Failure symptoms:

- Wrong LED turns on.
- Multiple LEDs turn on unexpectedly.
- LED does not turn on.
- GPIO pin conflicts with debug or boot behavior.

Do not continue if:

- LED current is too high.
- Mapping is ambiguous.
- Any output appears shorted.

Notes:

- Do not name A/B red/yellow/green ownership until verified.

## Phase 5 - 7-Segment Display Test

Goal:

- Validate display polarity, segment pins, digit/common pins, and current budget.

Tools needed:

- Minimal GPIO firmware.
- Multimeter.
- Optional oscilloscope for multiplex timing.

Procedure:

1. Test one segment on one digit at a time.
2. Verify segment pins PA8, PB15, PB14, PB13, PB12, PA12, and PA11.
3. Verify digit/common control pins PA3 and PA4.
4. Record whether the display is common-anode or common-cathode in practice.
5. Measure or estimate peak and average current.

Expected result:

- Each segment can be controlled predictably.
- Digit selection works.
- Current remains within safe limits.

Failure symptoms:

- All segments glow dimly.
- Wrong digit lights.
- GPIO pin overheats or voltage droops.
- Display requires drive current beyond direct GPIO capability.

Do not continue if:

- Current budget is unsafe.
- Display polarity is not understood.
- PA3/PA4 drive method appears inadequate.

Notes:

- A future driver may need transistor stages if direct GPIO drive is unsafe.

## Phase 6 - UART Loopback / Raspberry Pi UART Test

Goal:

- Validate the UART electrical path before integrating AI planning messages.

Tools needed:

- USB-UART adapter or Raspberry Pi UART.
- Logic analyzer optional.
- Serial terminal.
- Multimeter for ground and voltage checks.

Procedure:

1. Verify STM32 TX1/RX1 pins and header orientation.
2. Confirm 3.3 V logic levels.
3. Start with loopback or USB-UART test before Raspberry Pi.
4. Use 115200 baud, 8N1, newline-terminated ASCII.
5. Validate STM32 receive and transmit separately.

Expected result:

- Bytes sent to STM32 are received correctly.
- STM32 responses are readable by host.
- No framing errors at 115200 baud.

Failure symptoms:

- No received data.
- Garbled serial output.
- TX/RX swapped.
- Missing ground reference.
- 5 V signal accidentally used.

Do not continue if:

- UART voltage level is not confirmed.
- TX/RX crossing is uncertain.
- Pi ground is not connected safely.

Notes:

- Full AI host integration is not part of this phase.

## Phase 7 - FreeRTOS Traffic Light FSM Port

Goal:

- Port the ESP32 FreeRTOS controller behavior to STM32 after hardware basics are validated.

Tools needed:

- Selected STM32 toolchain.
- FreeRTOS integration plan.
- Verified pin mapping.
- UART validation results.

Procedure:

1. Review task ownership and queue design.
2. Port protocol structures in a hardware-independent way.
3. Add GPIO abstraction for STM32 outputs.
4. Add FSM task and safe plan apply behavior.
5. Add STATUS and watchdog/fallback behavior.

Expected result:

- STM32 executes a local traffic FSM from accepted plans.
- GPIO output mapping matches verified hardware.
- Controller remains safe if host stops sending plans.

Failure symptoms:

- Race conditions on shared state.
- Plans applied mid-cycle unsafely.
- UART responses block timing.
- GPIO output mapping is wrong.

Do not continue if:

- Pin mapping is unverified.
- UART link is unstable.
- Build/debug workflow is unreliable.

Notes:

- This is Phase 17.5 or later and is intentionally not started now.

## Phase 8 - AI Host PLAN Integration

Goal:

- Connect Raspberry Pi AI host planning output to STM32 controller after UART and firmware are validated.

Tools needed:

- Raspberry Pi deployment environment.
- Validated STM32 firmware.
- Serial logging.
- Test traffic input or recorded video.

Procedure:

1. Run AI host pipeline in a controlled mode.
2. Send bounded, validated `PLAN` messages.
3. Verify STM32 `ACK`/`NACK` behavior.
4. Monitor `STATUS` messages.
5. Compare host plan and physical light state timing.

Expected result:

- Host sends plans.
- STM32 validates and executes plans locally.
- STATUS output matches FSM state.

Failure symptoms:

- Host sends plans too frequently.
- ACK received but plan not applied yet.
- STATUS mismatches physical output.
- Watchdog triggers due to host overload.

Do not continue if:

- UART validation is incomplete.
- STM32 FSM port is incomplete.
- AI host deployment is unstable.

Notes:

- This is Phase 17.6 or later and requires careful review and hardware validation.
