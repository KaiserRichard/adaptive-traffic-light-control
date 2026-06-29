# STM32 Firmware Bring-Up Plan

## Status

```text
Planned.
No firmware has been built, flashed, or tested in Phase 17.2.
```

This plan describes the firmware sequence after the PCB passes hardware bring-up checks. It intentionally avoids requiring STM32CubeIDE.

## Firmware Bring-Up Sequence

### Step 1 - Select Toolchain

Goal:

- Choose one command-line-friendly toolchain path before writing source files.

Acceptable options:

- STM32CubeCLT.
- `arm-none-eabi-gcc` with CMake or Make.
- PlatformIO, only if explicitly selected later.

Exit criteria:

```text
[ ] Toolchain version recorded.
[ ] Build command recorded.
[ ] Flash/debug command recorded, if hardware is available.
[ ] No STM32CubeIDE dependency introduced.
```

### Step 2 - Minimal Board Definition

Goal:

- Define MCU, clock assumptions, linker script source, startup source, and pin names.

Do not proceed until:

- STM32F103C8T6 package/pinout is confirmed.
- Clock source and reset configuration are reviewed.
- BOOT0 and SWD access are verified from hardware documentation.

### Step 3 - Minimal Blink

Goal:

- Toggle only a safe status LED candidate, likely PC13, after confirming the schematic and board behavior.

Expected output:

- PC13 LED blinks at a visible rate.

Failure symptoms:

- Cannot connect through SWD.
- Board resets repeatedly.
- 3.3 V rail droops.
- PC13 behavior is inverted or no LED changes.

### Step 4 - Traffic LED Smoke Test

Goal:

- Drive each traffic LED output one at a time with conservative timing.

Candidate pins from schematic image:

```text
PB9, PB8, PB7, PB6, PB5, PB3
```

Do not use final lane/color names until the mapping is verified.

### Step 5 - 7-Segment Smoke Test

Goal:

- Verify segment polarity, digit control, and current budget.

Candidate pins from schematic image:

```text
Segments: PA8, PB15, PB14, PB13, PB12, PA12, PA11
Digits:   PA3, PA4
```

Start with one digit and one segment at low duty cycle.

### Step 6 - UART Echo

Goal:

- Verify USART1 RX/TX path independent of AI planning logic.

Planned default:

```text
115200 baud, 8N1, newline-terminated ASCII
```

Do not connect the Raspberry Pi until TX/RX crossing and 3.3 V logic safety are verified.

### Step 7 - Protocol Parser

Goal:

- Parse `PLAN` commands and respond with `ACK` or `NACK`.

This step should reuse the protocol semantics from the ESP32 prototype but adapt implementation details to the selected STM32 stack.

### Step 8 - FreeRTOS Controller Port

Goal:

- Port task ownership, queues, FSM, status reporting, and watchdog behavior.

Status:

- Not started in Phase 17.2.
- Requires review before implementation.

## What Counts as Firmware Bring-Up Complete

Firmware bring-up should not be considered complete until:

```text
[ ] Minimal blink runs on real hardware.
[ ] SWD programming/debug works reliably.
[ ] Each traffic LED output is identified and documented.
[ ] 7-segment polarity and current are verified.
[ ] USART1 link is validated with loopback or controlled host test.
[ ] Toolchain commands are recorded.
[ ] Commit hash and test notes are recorded.
```
