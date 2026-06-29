# STM32 UART to Raspberry Pi Validation Plan

## Status

```text
Planned.
UART has not been tested in Phase 17.
No UART firmware is implemented in this phase.
```

## Goal

Validate the physical and protocol-level UART link between the Raspberry Pi AI host and the STM32F103C8T6 controller PCB before using it for traffic light control.

## Raspberry Pi UART Role

The Raspberry Pi AI host is planned to:

- Run camera/video input.
- Run YOLO inference and ROI counting.
- Compute adaptive signal plans.
- Send `PLAN` messages to the STM32 over UART.
- Receive `ACK`, `NACK`, and `STATUS` messages from the STM32.

## STM32 UART Role

The STM32 controller is planned to:

- Receive newline-terminated `PLAN` messages.
- Parse and validate timing fields.
- Send `ACK` for accepted plans.
- Send `NACK` for malformed or unsafe plans.
- Send `STATUS` messages after status reporting is implemented.
- Execute traffic timing locally after accepting a plan.

## TX/RX Crossing Check

Observed schematic labels:

```text
STM32 PA9  -> TX1
STM32 PA10 -> RX1
UART header exposes TX1 and RX1
```

Required crossing:

```text
STM32 TX1 -> Raspberry Pi RXD
STM32 RX1 <- Raspberry Pi TXD
GND       -> Raspberry Pi GND
```

Validation actions:

```text
[ ] Confirm header pin 1 orientation.
[ ] Confirm H1 pinout on PCB silkscreen or layout.
[ ] Confirm Pi UART pins used.
[ ] Check continuity if possible.
[ ] Confirm cable crossing before powering both boards together.
```

## 3.3 V Logic Safety

- Raspberry Pi UART GPIO is 3.3 V only.
- STM32F103 USART1 pins are used at 3.3 V in this design.
- Do not connect 5 V UART adapters directly to STM32 or Pi UART pins.
- Do not power one board through UART signal pins.

## GND Reference Requirement

The Pi and STM32 must share a ground reference for UART to work.

Required connection:

```text
Raspberry Pi GND <-> STM32 PCB GND
```

Failure without common ground:

- Random characters.
- No receive.
- Unstable logic thresholds.
- Possible IO stress.

## Baud Rate Recommendation

Planned default:

```text
Baud:      115200
Data bits: 8
Parity:    none
Stop bits: 1
Framing:   newline-terminated ASCII
```

This matches the ESP32 prototype's `SERIAL_BAUD_RATE = 115200`.

## Message Framing Plan

Use one message per line:

```text
PLAN,17,25,15,3,1\n
ACK,17\n
NACK,19,GREEN_A_OUT_OF_RANGE\n
STATUS,17,A_GREEN,12,OK\n
```

Receiver behavior should:

- Buffer until newline.
- Reject overlong messages.
- Reject malformed fields.
- Avoid blocking the FSM on serial output.

## PLAN / ACK / STATUS Relationship

Planned direction:

```text
Raspberry Pi AI Host
    sends PLAN messages

STM32 Controller
    sends ACK / STATUS messages
```

Important semantic rule:

- `ACK` means the STM32 parsed, validated, and accepted the plan into controller ownership.
- `ACK` does not necessarily mean the FSM has already switched to the new plan.
- `STATUS` is the better message for observing current active state.

## Loopback Test Plan

Goal:

- Prove serial equipment and wiring assumptions before involving both boards.

Procedure:

1. Test USB-UART adapter loopback by shorting adapter TX to RX.
2. Open serial terminal at 115200 8N1.
3. Type sample lines and confirm echo.
4. If using Raspberry Pi UART, run a Pi-side loopback test before connecting STM32.

Expected result:

- Sent characters return exactly.

Failure symptoms:

- No echo.
- Garbled text.
- Wrong baud rate.
- Wrong serial device.

## Pi-to-STM32 Test Plan

Goal:

- Verify STM32 receives host messages.

Procedure:

1. Use minimal STM32 UART firmware only after Phase 17.2 build planning and board bring-up.
2. Send simple newline-terminated text from Pi.
3. Observe STM32 debug output or response.
4. Send valid and invalid `PLAN` lines.

Expected result:

- STM32 receives complete lines.
- Valid `PLAN` can be parsed.
- Invalid input is rejected.

Failure symptoms:

- STM32 receives nothing.
- Messages are truncated.
- Parser accepts unsafe values.
- Input overruns buffer.

## STM32-to-Pi Test Plan

Goal:

- Verify Raspberry Pi receives controller responses.

Procedure:

1. Make STM32 send a boot banner or echo response.
2. Confirm Pi serial terminal receives clean text.
3. Confirm `ACK`, `NACK`, and later `STATUS` parse correctly on the Pi side.

Expected result:

- Pi reads complete newline-terminated STM32 messages.

Failure symptoms:

- No response.
- Corrupted characters.
- Pi UART disabled or claimed by console.
- TX/RX reversed.

## Failure Symptoms

- No received bytes: likely wrong serial device, disabled Pi UART, missing ground, TX/RX not crossed, or STM32 USART pin configuration error.
- Garbled bytes: likely wrong baud rate, unstable ground, voltage mismatch, or signal integrity issue.
- STM32 resets during traffic: possible power rail issue, firmware fault, watchdog reset, or accidental back-powering.
- Pi logs nothing from STM32: possible PA9/TX1 wiring issue, Pi RX configuration issue, or console/device conflict.
- STM32 receives nothing from Pi: possible PA10/RX1 wiring issue, Pi TX configuration issue, or missing newline framing.
- ACK appears but physical outputs do not change immediately: may be expected if a future FSM applies plans only at a safe cycle boundary.

## Debug Checklist

```text
[ ] Both devices use 115200 8N1.
[ ] Pi UART console is disabled if using primary UART.
[ ] Pi serial port name is correct.
[ ] STM32 USART1 pins PA9/PA10 are configured correctly.
[ ] STM32 and Pi share GND.
[ ] TX/RX are crossed.
[ ] No 5 V UART signal is connected.
[ ] Message includes newline.
[ ] Parser buffer length is bounded.
[ ] Logic analyzer confirms signal transitions if terminal output is unclear.
```

## What Counts as Validation Complete

UART validation is complete only when all of the following are recorded:

```text
[ ] Hardware wiring photo or diagram.
[ ] Exact Pi serial device name.
[ ] Exact baud/settings.
[ ] STM32 firmware commit hash.
[ ] Pi-side test command.
[ ] Raw serial transcript showing valid PLAN and ACK.
[ ] Raw serial transcript showing invalid PLAN and NACK.
[ ] STATUS transcript after status reporting is implemented.
[ ] Notes on any errors and fixes.
```

None of these validation items have been completed in this phase.
