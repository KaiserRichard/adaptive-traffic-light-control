# STM32 PCB Hardware Blocks Explained

## Purpose

Explain the STM32F103C8T6 PCB hardware blocks from a microprocessor and analog/digital electronics perspective.

This document supports Phase 17 hardware understanding, PCB bring-up preparation, firmware `board_config` planning, and future STM32 firmware review.

## Current Status

```text
Schematic-level explanation.
Final schematic, PCB layout, and physical hardware validation pending.
No measurements have been performed in this phase.
Do not treat this document as hardware validation evidence.
```

When exact pins or connections are uncertain, verify from schematic, PCB layout, and real hardware before firmware depends on them.

## Power Block

### Purpose

The power block provides the voltage rails required by the STM32 and board peripherals.

Known schematic-level context:

- USB Micro input.
- SS34 Schottky diode.
- `VCC5V` rail.
- AMS1117-3.3 regulator.
- `VCC3V3` rail.
- decoupling capacitors.
- power indicator LED path.

### Low-Level Electrical Idea

The STM32F103C8T6 is a 3.3 V microcontroller. The board therefore needs a regulated 3.3 V rail. A 5 V input cannot be applied directly to STM32 VDD pins.

The SS34 diode likely provides input polarity/current path protection or rail isolation. Because it is a Schottky diode, it has a forward voltage drop. The regulator input voltage is therefore lower than raw USB VBUS by the diode drop.

The AMS1117-3.3 is a linear regulator. It converts excess voltage into heat:

```text
P_dissipation = (V_in - 3.3 V) * I_load
```

LEDs, 7-segment display current, expansion loads, and the STM32 itself all contribute to `I_load`.

### Signals / Components

- USB 5 V input.
- SS34 diode.
- AMS1117-3.3 regulator.
- input/output capacitors.
- `VCC5V`.
- `VCC3V3`.
- `GND`.
- power indicator LED and resistor.

### What Firmware Depends On

Firmware depends on a stable 3.3 V rail before any GPIO, UART, SWD, or FreeRTOS behavior can be trusted.

If the power rail sags, firmware symptoms can look like software bugs:

- random resets.
- failed SWD attach.
- corrupted UART.
- display flicker.
- unstable GPIO outputs.

### What Can Go Wrong

- short between `VCC3V3` and `GND`.
- reversed diode or regulator.
- missing or poorly soldered capacitors.
- AMS1117 overheating under LED/display load.
- dropout if regulator input is too low after diode drop.
- unintended back-powering through UART or SWD reference pins.

### Bring-Up Checks

```text
[ ] Measure resistance from VCC3V3 to GND before power.
[ ] Power from a current-limited 5 V source if available.
[ ] Measure VCC5V and VCC3V3 with no Pi or ST-LINK attached.
[ ] Check regulator temperature after short power intervals.
[ ] Verify no peripheral output is drawing unexpected current.
```

### Debug Symptoms

- current limit trips immediately.
- 3.3 V missing or too high.
- regulator heats rapidly.
- STM32 cannot be detected over SWD.
- UART text is corrupted even at correct baud rate.

### Why This Matters for Microprocessor Systems

Digital logic only works correctly when supply voltage and ground reference are stable. A microcontroller firmware bug cannot be diagnosed reliably until power rails are confirmed safe.

## MCU Block

### Purpose

The MCU block contains the STM32F103C8T6 and the minimum connections needed for boot, reset, programming, UART, and GPIO output.

Known schematic-level context:

- STM32F103C8T6 MCU.
- VDD / GND.
- BOOT0 pull-down.
- NRST / RST net, exact external access needs verification.
- SWDIO / SWCLK.
- UART TX/RX.
- GPIO outputs.
- decoupling capacitors.

### Low-Level Electrical Idea

The STM32 samples boot configuration pins during reset. BOOT0 should normally be held low so the MCU boots from user flash. If BOOT0 floats or is high unintentionally, firmware in flash may not start.

NRST resets the MCU. External reset access is useful for debug recovery, especially when firmware misconfigures clocks, GPIO, or low-power modes.

Decoupling capacitors near VDD/VSS pins provide local transient current when internal logic switches.

### Signals / Components

- `VCC3V3` to VDD pins.
- `GND` to VSS pins.
- BOOT0 pull-down resistor, observed as 10k in current notes.
- `NRST` / `RST` net, verify exposure.
- PA13 `SWDIO`.
- PA14 `SWCLK`.
- PA9 `TX1`, verify as USART1 TX.
- PA10 `RX1`, verify as USART1 RX.
- GPIO outputs for LEDs/display.

### What Firmware Depends On

Firmware depends on:

- correct boot mode.
- valid reset behavior.
- stable clock configuration.
- correct pin ownership.
- reserved SWD pins remaining usable during debug.

### What Can Go Wrong

- BOOT0 not held low.
- NRST inaccessible, making recovery harder.
- SWD pins reused accidentally.
- wrong GPIO mode on output pins.
- no decoupling or poor soldering near supply pins.
- firmware assumes pin mappings that have not been validated.

### Bring-Up Checks

```text
[ ] Verify BOOT0 is low at reset.
[ ] Confirm NRST access strategy.
[ ] Confirm PA13/PA14 are not repurposed.
[ ] Confirm VDD/GND soldering and decoupling.
[ ] Confirm MCU package orientation.
```

### Debug Symptoms

- MCU boots to system bootloader instead of user flash.
- SWD cannot attach.
- board works only after manual reset.
- firmware appears to run but outputs are on unexpected pins.

### Why This Matters for Microprocessor Systems

The MCU core block defines whether firmware can start, be recovered, and communicate with debug tools. Poor boot/reset/debug design can block all later firmware work.

## UART-to-Pi Block

### Purpose

The UART-to-Pi block is the planned communication boundary between the Raspberry Pi AI host and the STM32 controller.

Planned relationship:

```text
Raspberry Pi AI host
    sends PLAN messages

STM32 controller
    sends ACK / NACK / STATUS messages
```

### Low-Level Electrical Idea

UART is asynchronous serial communication. Both sides must agree on baud rate, frame format, voltage levels, and ground reference.

For this project:

```text
115200 baud
8 data bits
no parity
1 stop bit
newline-terminated ASCII
3.3 V logic
```

UART TX and RX must cross:

```text
STM32 TX1 -> Raspberry Pi RXD
STM32 RX1 <- Raspberry Pi TXD
GND       -> Raspberry Pi GND
```

### Signals / Components

- UART header `GND`.
- UART header `VCC3V3`.
- `TX1`, observed on PA9.
- `RX1`, observed on PA10.
- Raspberry Pi GPIO UART pins, exact Pi-side pinout to verify.

### What Firmware Depends On

Firmware depends on:

- USART instance and pins.
- baud rate.
- receive buffering strategy.
- line termination.
- bounded parser input.
- nonblocking or timeout-bounded transmit.

### What Can Go Wrong

- TX/RX not crossed.
- missing shared ground.
- Pi UART disabled or used by console.
- wrong baud rate.
- 5 V UART adapter connected.
- one board back-powered through UART pins.
- parser receives partial or overlong lines.

### Bring-Up Checks

```text
[ ] Confirm 3.3 V logic only.
[ ] Confirm GND connection.
[ ] Confirm TX/RX crossing.
[ ] Start with loopback or USB-UART test.
[ ] Test valid and invalid PLAN lines only after minimal UART firmware exists.
```

### Debug Symptoms

- no serial data.
- garbled text.
- random characters.
- STM32 receives but Pi does not, or opposite direction only.
- ACK received but physical outputs do not change immediately.

### Why This Matters for Microprocessor Systems

UART is both an electrical interface and a software protocol boundary. Electrical validation must come before assuming protocol or FSM problems.

## Traffic LED Block

### Purpose

The traffic LED block provides visible red/yellow/green outputs for the traffic controller prototype.

Known schematic-level context:

- STM32 GPIO outputs.
- six LED channels.
- PB9, PB8, PB7, PB6, PB5, PB3 observed.
- 1k series resistors observed.

Exact lane/color ownership remains `verify from schematic` and pending hardware validation.

### Low-Level Electrical Idea

A GPIO pin can source or sink only limited current. A series resistor limits LED current:

```text
I_LED ~= (V_GPIO - V_LED_forward) / R
```

With a 3.3 V GPIO and 1k resistor, current is likely modest, but actual LED forward voltage and resistor value should be verified.

### Signals / Components

- PB9, PB8, PB7, PB6, PB5, PB3, pending final mapping.
- series resistors, observed as 1k.
- red/yellow/green LEDs.
- ground return.

### What Firmware Depends On

Firmware depends on:

- which pin maps to A/B red/yellow/green.
- active-high or active-low behavior.
- safe off state.
- per-pin and total GPIO current limits.

### What Can Go Wrong

- wrong lane/color mapping.
- output active polarity opposite of assumption.
- too much LED current.
- output conflicts with debug/boot or expansion header use.
- multiple LEDs turn on due to solder bridge or wrong firmware mapping.

### Bring-Up Checks

```text
[ ] Test one LED output at a time.
[ ] Record physical LED color and position.
[ ] Measure or estimate current.
[ ] Keep default state safe/off.
[ ] Do not bind FSM outputs until mapping is verified.
```

### Debug Symptoms

- wrong LED lights.
- no LED lights.
- more than one LED lights.
- LED remains dimly on.
- 3.3 V rail droops when LEDs are active.

### Why This Matters for Microprocessor Systems

GPIO output code directly controls external loads. Firmware must respect electrical current limits and verified signal ownership.

## Seven-Segment Display Block

### Purpose

The dual 7-segment display can show countdown or state information after the electrical design is validated.

Known schematic-level context:

- segment pins.
- digit select/common pins.
- segment resistors.
- multiplexing likely required.
- PA8, PB15, PB14, PB13, PB12, PA12, PA11 observed for segments.
- PA3 and PA4 observed for digit/common control.

### Low-Level Electrical Idea

A 7-segment display contains multiple LEDs. A dual display shares segment lines and uses digit select/common pins. Firmware usually multiplexes digits:

```text
enable digit 1, drive segments
short delay
disable digit 1
enable digit 2, drive segments
short delay
repeat quickly
```

If refresh is too slow, flicker appears. If digit switching is not blanked correctly, ghosting appears. If too many segments draw current at once, GPIO or regulator limits may be exceeded.

### Signals / Components

- segment `a` through `g`, pending final electrical verification.
- optional decimal point, verify from schematic.
- PA3 / PA4 digit select or common control, pending drive method review.
- 220 ohm segment resistors observed.
- display part and common-anode/common-cathode behavior to verify.

### What Firmware Depends On

Firmware depends on:

- common-anode vs common-cathode.
- active polarity for segment and digit pins.
- current budget.
- refresh rate.
- whether direct GPIO drive is safe or transistor drivers are required.

### What Can Go Wrong

- display polarity assumed incorrectly.
- PA3/PA4 cannot safely drive total digit current.
- ghosting due to no blanking between digits.
- flicker due to low refresh rate.
- high peak current during multiplexing.
- display code interferes with traffic FSM timing.

### Bring-Up Checks

```text
[ ] Identify common-anode/common-cathode behavior.
[ ] Test one segment on one digit.
[ ] Verify digit select pins.
[ ] Estimate peak and average current.
[ ] Choose conservative refresh timing.
[ ] Keep display disabled until safe.
```

### Debug Symptoms

- wrong segments light.
- both digits light together.
- display flickers.
- dim ghost segments appear.
- GPIO pins heat or voltage droops.

### Why This Matters for Microprocessor Systems

Multiplexed displays combine timing, current, and GPIO ownership. They should be brought up after basic power, SWD, and simple GPIO are stable.

## SWD / NRST Block

### Purpose

The SWD block allows programming and debugging the STM32.

Known schematic-level context:

- SWDIO.
- SWCLK.
- GND.
- 3.3 V reference.
- NRST recovery path needs verification.

### Low-Level Electrical Idea

SWD is the ARM Serial Wire Debug interface. The debug probe needs:

```text
SWDIO
SWCLK
GND
target voltage reference
optional NRST
```

Target voltage reference tells the programmer the board voltage. It should not be used to unintentionally power the board unless the workflow explicitly supports that.

NRST helps recover the MCU when firmware prevents normal debug attach.

### Signals / Components

- PA13 `SWDIO`.
- PA14 `SWCLK`.
- SWD header `GND`.
- SWD header `VCC3V3`.
- `NRST` if available on header/test pad/reset circuit, verify from schematic.

### What Firmware Depends On

Firmware development depends on SWD for:

- target detection.
- flashing.
- halt/run debug.
- recovery from bad firmware.
- inspecting registers and memory.

### What Can Go Wrong

- SWD header orientation reversed.
- SWDIO/SWCLK swapped.
- missing ground.
- target voltage reference wrong.
- BOOT0/reset state prevents attach.
- firmware reconfigures debug pins.
- NRST unavailable during recovery.

### Bring-Up Checks

```text
[ ] Verify board power before connecting ST-LINK.
[ ] Verify header pin 1 orientation.
[ ] Connect GND first.
[ ] Confirm target voltage reference.
[ ] Attempt non-flashing connect/read-ID before any flashing.
[ ] Confirm NRST access strategy.
```

### Debug Symptoms

- cannot connect to target.
- target voltage reads 0 V.
- intermittent SWD attach.
- flashing starts but fails.
- board resets during debug.

### Why This Matters for Microprocessor Systems

SWD is the recovery path. Power and SWD should be validated before UART, display, or FreeRTOS work because they are required to diagnose every later firmware problem.

## Bring-Up Ordering Rationale

Recommended ordering:

```text
power rails
    -> SWD / reset
    -> minimal blink
    -> one GPIO output at a time
    -> 7-segment electrical validation
    -> UART electrical validation
    -> protocol parser
    -> FreeRTOS FSM
    -> AI host integration
```

This order keeps risk low. A UART or FreeRTOS failure is hard to debug if power, reset, SWD, and simple GPIO behavior are not already trusted.
