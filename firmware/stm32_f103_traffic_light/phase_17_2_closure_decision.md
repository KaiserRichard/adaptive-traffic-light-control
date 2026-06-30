# Phase 17.2 Closure Decision

## Purpose

Close the STM32 firmware planning portion of Phase 17 without adding more redundant documentation or starting hardware-dependent firmware work.

This document audits the existing Phase 17.2 outputs, records the final planning decisions, and defines the minimum safe path into Phase 17.3 hardware bring-up.

## Why Phase 17.2 Needed Streamlining

Phase 17.2 successfully kept the STM32 work offline-safe, but it started to produce several overlapping planning documents before any real PCB bring-up or STM32 toolchain installation.

The project now has enough planning material to avoid premature implementation. More firmware-planning subphases would add maintenance cost without reducing the main remaining risk: unvalidated hardware.

## Existing Phase 17.2 Outputs

| Document | Main purpose | Unique value | Overlap | Decision |
| --- | --- | --- | --- | --- |
| `toolchain_plan.md` | Compare command-line STM32 toolchain options | Records allowed future tool options | Overlaps with inspection and CMake proposal | Keep |
| `toolchain_inspection.md` | Record MacBook tool availability | Concrete installed/missing tool evidence | Overlaps with build checklist | Keep |
| `build_scaffold_design.md` | Describe future non-flashing build scaffold | Explains minimum future files and gates | Overlaps with CMake proposal | Keep as background |
| `firmware_architecture.md` | Describe future STM32 modules | Captures module boundaries without code | Overlaps with portable FreeRTOS notes | Keep |
| `minimal_build_checklist.md` | Define first compile prerequisites | Good gate list before real build | Overlaps with CMake proposal | Keep |
| `cmake_scaffold_proposal.md` | Propose future CMake structure | Most detailed CMake/startup/linker policy | Overlaps with scaffold design | Keep as canonical build proposal |
| `docs/phase_17_2_3_cmake_scaffold_proposal.md` | Summarize Phase 17.2.3 result | Useful continuation note | Overlaps with ledger | Keep, do not expand |
| `portable_freertos_architecture.md` | Propose future portable RTOS boundary | Useful later when refactoring shared logic | Overlaps with architecture and porting plan | Keep as optional reference |
| `esp32_to_stm32_porting_plan.md` | Discuss ESP32-to-STM32 migration | Useful later for common/source migration | Overlaps with portable architecture | Keep as optional reference |
| `README.md` | Firmware folder entry point | Good status and navigation page | Overlaps with all detailed docs | Keep concise |
| `phase_17_ledger.md` | Cross-phase continuation ledger | Required next-assistant context | Can become too long | Clean next-step guidance |

## Redundancy Audit Summary

The main duplication is among:

- `build_scaffold_design.md`
- `minimal_build_checklist.md`
- `cmake_scaffold_proposal.md`
- `toolchain_plan.md`
- `toolchain_inspection.md`
- firmware `README.md`
- Phase 17 ledger

The overlap is acceptable because each document answers a different review question:

- toolchain availability.
- future build file shape.
- future CMake policy.
- first compile gates.
- firmware module boundaries.

The redundant part is not technical content; it is the repeated next-step recommendation to create more planning phases. That recommendation is now replaced by a transition to Phase 17.3 hardware bring-up.

## Documents Kept

All existing Phase 17.2 documents are kept because they are already committed or useful as references.

No files are deleted in this closure phase.

## Documents Merged or Cleaned

No standalone documents are merged physically.

The cleanup is navigational:

- firmware README points to this closure decision.
- Phase 17 ledger records Phase 17.2 as closed.
- next-step guidance moves from more planning to Phase 17.3.1 pre-power inspection execution.

## Documents Not Created

This closure phase intentionally does not create more large planning documents.

The previously proposed large portable FreeRTOS and hardware-block documentation work is reduced to optional reference material because existing docs already cover most planning needs. In this repository, those reference documents already exist from local commit `6d39017`; they are not expanded here.

Future optional work:

- portable `firmware/common/` migration design after hardware bring-up or explicit approval.
- deeper hardware-block explanations inside existing hardware docs only if a report or bring-up review needs them.
- artifact-policy cleanup before the first real STM32 build.

## Final Firmware Architecture Direction

The future STM32 firmware direction remains:

```text
USART receive
    -> bounded PLAN parser
    -> validated plan queue
    -> local traffic light FSM
    -> GPIO / display board layer
    -> ACK / NACK / STATUS telemetry
```

This direction was documented enough to close Phase 17.2. A later hardware-independent preparation step added STM32 scaffold stubs, but no real hardware implementation exists yet.

## Final Build-System Direction

The build-system direction is:

```text
CMake + Make proposal only
```

No real `CMakeLists.txt`, CMake toolchain file, build directory, or generated artifact exists.

## Final Toolchain Direction

The toolchain direction is:

```text
arm-none-eabi-gcc later, after explicit approval
```

Current MacBook status:

- CMake is installed.
- GNU Make is installed.
- `arm-none-eabi-gcc` is not installed.
- OpenOCD / STM32CubeProgrammer / ST-LINK workflow is not required for this closure phase.

## Final ESP32 / STM32 Role Decision

ESP32 role:

- Keep as the FreeRTOS reference/testbed.
- Do not delete, refactor, or move ESP32 firmware during Phase 17.2 closure.

STM32 role:

- Future custom PCB controller target.
- Hardware bring-up comes before STM32 FreeRTOS porting.

## Final common/ Decision

`firmware/common/` is a future concept only.

Decision:

```text
Do not create firmware/common/ now.
Do not migrate ESP32 files now.
Do not make common/ migration a blocker for Phase 17.3.
```

Shared-code extraction can be reviewed later after hardware bring-up or explicit approval.

## What Is Still Not Implemented

- No STM32 firmware source implementation.
- No STM32 build system.
- No CMake project.
- No startup file.
- No linker script.
- No CMSIS/HAL/LL vendor files.
- No FreeRTOS STM32 port.
- No UART hardware driver.
- No GPIO output driver.
- No `firmware/common/` migration.
- No firmware build.
- No flashing.
- No STM32 hardware validation.

## Why Phase 17.2 Can Close

Phase 17.2 can close because the repository already documents:

- STM32 target MCU and firmware folder intent.
- command-line toolchain options and current MacBook tool availability.
- preferred future CMake + Make direction.
- first compile prerequisites.
- startup/linker/vendor-file policy.
- planned STM32 firmware module boundaries.
- dependency on verified pin mapping.
- relationship to ESP32 FreeRTOS reference behavior.

The next useful risk-reducing work is not another planning document. It is Phase 17.3 hardware bring-up execution, starting with safe inspection and power checks.

## Gate Checklist Before Phase 17.3

Before executing Phase 17.3.1:

```text
[ ] Working tree clean.
[ ] Phase 17.2 closure committed.
[ ] User has the physical PCB or board photos available.
[ ] Schematic images and pin map available for review.
[ ] No Raspberry Pi connected to STM32 UART yet.
[ ] No ST-LINK connected until power/header orientation is reviewed.
[ ] No firmware flashing planned in the first pre-power step.
```

## Phase 17.3 Entry Plan

Minimum safe Phase 17.3.1 at the time of Phase 17.2 closure:

```text
Phase 17.3.1 - STM32 PCB pre-power inspection execution
```

This step has since been documented in `docs/hardware/stm32_pcb/phase_17_3_1_pre_power_inspection.md`.

Scope:

- inspect PCB visually.
- verify connector orientation.
- inspect regulator/diode/capacitor polarity.
- measure resistance between rails and ground.
- record photos and findings.
- decide whether it is safe to proceed to controlled power validation.

Do not build or flash firmware in Phase 17.3.1.

## What Must Not Be Claimed Yet

Do not claim:

- STM32 board powered successfully.
- SWD connection works.
- STM32 firmware builds.
- STM32 firmware flashes.
- UART is tested.
- traffic LEDs are tested.
- 7-segment display is tested.
- ESP32 code has been migrated to common code.
- STM32 FreeRTOS FSM port is implemented.
- Raspberry Pi controls STM32 hardware.

## Recommended Next Step

Recommended next Phase 17 step:

```text
Phase 17.3.2 - Controlled Power Rail Validation
```

Phase 17.3.1 is already documented. The next step should collect real power-rail evidence and should not build, flash, or port firmware.
