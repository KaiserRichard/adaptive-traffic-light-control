# Phase 17 Hardware-Gated Completion Review

## Purpose

Summarize what Phase 17 can honestly complete offline and what must remain blocked until physical STM32 PCB measurements are available.

This document prevents two failure modes:

- pretending hardware has been validated from documentation alone.
- creating more planning documents when the next real risk is measurement evidence.

## Current Branch Status

```text
Branch: phase17-stm32-pcb-integration
Phase 17.1: documentation complete.
Phase 17.2: firmware planning closed.
Phase 17.3.1: pre-power inspection checklist complete.
Phase 17.3.2: controlled power rail validation procedure complete; execution pending.
Phase 17.3.3: ST-LINK attach/read-ID procedure complete; execution pending.
```

## Sequential Gate Review

| Roadmap item | Offline-completable work | Hardware-dependent work | Current decision |
| --- | --- | --- | --- |
| 17.3.2 Controlled Power Rail Validation | Procedure, measurement tables, pass/fail gates | Actual VCC5V/VCC3V3/current/thermal measurements | Procedure complete; execution pending |
| 17.3.3 ST-LINK Attach / Read-ID Validation | Non-flashing attach plan, command examples, evidence checklist | ST-LINK connection, target voltage reading, read-ID output | Procedure complete; execution pending |
| 17.3.4 Minimal Blink Firmware | Existing build/toolchain policy and future blink gate | Toolchain install, startup/linker source, SWD flash, observed blink | Blocked by 17.3.2 and 17.3.3 |
| 17.3.5 Traffic LED Validation | Pin map and bring-up plan | One-pin-at-a-time GPIO test and physical LED mapping | Blocked by blink and hardware validation |
| 17.3.6 Seven-Segment Validation | Pin map and current-risk notes | Polarity/current/multiplex validation | Blocked by GPIO and current checks |
| 17.4 UART Hardware Validation | UART validation plan and protocol relationship | Physical Pi/STM32 or adapter UART transcript | Blocked by power, SWD, and UART firmware |
| 17.5 STM32 FreeRTOS Port | Portable architecture and ESP32-to-STM32 porting notes | Real STM32 build, board support, FreeRTOS runtime validation | Not started; blocked by earlier gates |
| 17.6 AI Host Integration | PLAN/ACK/STATUS compatibility notes | End-to-end host-to-STM32 hardware demo | Not started; blocked by UART and firmware |
| 17.7 Repository Cleanup | README/ledger/status audit | Final cleanup after hardware evidence | Offline audit completed for current hardware-gated state |
| 17 Final Closure | Clear merge readiness decision | Validated hardware/firmware evidence if claiming full integration | Not hardware-complete |

## What Is Complete

- STM32 PCB documentation and pin mapping are prepared.
- Schematic images are referenced.
- Hardware review notes exist.
- Pre-power inspection and ST-LINK non-detection triage exists.
- Controlled power rail validation procedure exists.
- ST-LINK read-ID validation procedure exists.
- STM32 firmware folder is documented as documentation-first.
- Command-line toolchain inspection is documented.
- CMake + Make future build direction is proposed only.
- ESP32 remains the FreeRTOS behavior reference.
- UART validation plan exists and keeps the canonical `PLAN` / `ACK` / `NACK` / `STATUS` relationship.

## What Remains Hardware-Dependent

- VCC3V3 measurement.
- VCC5V/input rail measurement.
- board current draw.
- AMS1117 thermal behavior.
- ST-LINK target-voltage reading.
- SWD attach/read-ID.
- minimal firmware flash.
- blink observation.
- traffic LED pin-to-color/lane mapping.
- 7-segment polarity/current/multiplex behavior.
- UART electrical validation.
- end-to-end Raspberry Pi to STM32 behavior.

## Firmware Status

```text
No buildable STM32 firmware exists.
No CMake project exists.
No startup file exists.
No linker script exists.
No CMSIS/HAL/LL vendor files are committed.
No STM32 source implementation exists.
No firmware/common/ migration has happened.
No ESP32 firmware refactor has happened.
```

This is intentional. The repository should not create STM32 firmware until power and SWD evidence exists and the toolchain/vendor-file policy is approved.

## Protocol Status

Canonical protocol remains:

```text
AI host -> controller:
    PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>

controller -> AI host:
    ACK,<plan_id>
    NACK,<plan_id>,<reason>
    STATUS,<plan_id>,<state>,<remaining_seconds>,<health>
```

No second PLAN format should be introduced.

## Repository Cleanup Decision

Keep:

- Phase 17.2 planning docs as references.
- `portable_freertos_architecture.md` and `esp32_to_stm32_porting_plan.md` as optional future references.
- STM32 hardware block explanation because it supports bring-up and report writing.
- Phase 17 ledger as the continuation source of truth.

Do not create now:

- `firmware/common/`.
- buildable STM32 firmware.
- more Phase 17.2 planning subphases.
- duplicate protocol documents.
- fake hardware results.

## Merge Readiness Decision

This branch is not a fully validated STM32 hardware integration.

It may be merge-ready only as an offline documentation and bring-up-preparation branch if the maintainer wants `main` to contain:

- STM32 PCB documentation.
- bring-up procedures.
- hardware-gated validation worksheets.
- honest pending-status documentation.

It is not merge-ready if the merge criterion is:

- powered STM32 PCB validation.
- working ST-LINK.
- buildable/flashed STM32 firmware.
- UART hardware validation.
- FreeRTOS STM32 port.
- AI host to STM32 hardware demo.

## Next Required Evidence

The next engineering step is not more firmware planning.

It is:

```text
Execute Phase 17.3.2 - Controlled Power Rail Validation on real hardware.
```

If Phase 17.3.2 passes, execute:

```text
Phase 17.3.3 - ST-LINK Attach / Read-ID Validation
```

If either fails, repair the hardware path before firmware work.

## What Must Not Be Claimed

Do not claim:

- STM32 board powered successfully.
- 3.3 V rail is valid.
- ST-LINK detects the chip.
- firmware builds.
- firmware flashes.
- blink works.
- traffic LEDs are mapped.
- 7-segment display works.
- UART is tested.
- STM32 FreeRTOS is ported.
- Raspberry Pi controls STM32 hardware.
- Phase 17 hardware integration is complete.
