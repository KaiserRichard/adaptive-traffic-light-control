# Phase 17 Ledger - STM32 PCB Integration

## Current Phase 17 Branch Name

```text
phase17-stm32-pcb-integration
```

## Base Branch

```text
main
```

## Subphases Completed in This Run

| Subphase | Status | Notes |
| --- | --- | --- |
| Phase 17.1 - STM32 PCB Documentation and Pin Mapping | Completed as offline documentation | Hardware validation pending |
| Phase 17.2 - STM32 Firmware Skeleton and Command-Line Build Plan | Completed as documentation-first skeleton | No source or build system implemented |
| Phase 17.2.1 - Command-Line STM32 Toolchain Inspection and README Synchronization | Completed as offline inspection/documentation | No tools installed; no firmware project created |
| Phase 17.3 - STM32 PCB Bring-Up Procedure | Completed as planned procedure | No bring-up step executed |
| Phase 17.4 - UART Link Validation Plan | Completed as planned procedure | UART not tested |

## Files Created by Subphase

### Phase 17.1

- `docs/hardware/stm32_pcb/README.md`
- `docs/hardware/stm32_pcb/stm32f103c8t6_pin_mapping.md`
- `docs/hardware/stm32_pcb/stm32_pcb_review_notes.md`
- `docs/hardware/stm32_pcb/schematics/README.md`
- `docs/embedded/phase_17_stm32_pcb_integration.md`

### Phase 17.2

- `firmware/stm32_f103_traffic_light/README.md`
- `firmware/stm32_f103_traffic_light/bringup_plan.md`
- `firmware/stm32_f103_traffic_light/toolchain_plan.md`
- `firmware/stm32_f103_traffic_light/src/README.md`
- `firmware/stm32_f103_traffic_light/include/README.md`

### Phase 17.2.1

- `firmware/stm32_f103_traffic_light/toolchain_inspection.md`
- `README.md`
- `docs/embedded/phase_17_ledger.md`

### Phase 17.3

- `docs/hardware/stm32_pcb/stm32_pcb_bringup_plan.md`

### Phase 17.4

- `docs/hardware/stm32_pcb/stm32_uart_pi_validation_plan.md`
- `docs/embedded/phase_17_4_uart_validation_plan.md`

### Ledger

- `docs/embedded/phase_17_ledger.md`

## Phase 17.2.1 Notes

Dirty-tree resolution summary:

- Two Phase 17 documentation files contained unresolved conflict markers.
- `docs/hardware/stm32_pcb/stm32_uart_pi_validation_plan.md` was cleaned to keep one conservative `Failure Symptoms` section.
- `docs/hardware/stm32_pcb/stm32f103c8t6_pin_mapping.md` was cleaned to keep conservative `NRST` and 7-segment decimal point rows.
- Cleanup was committed separately as `docs: resolve Phase 17 STM32 documentation conflicts`.

What was inspected:

- `brew --version`
- `cmake --version`
- `ninja --version`
- `make --version`
- `arm-none-eabi-gcc --version`
- `arm-none-eabi-gdb --version`
- `openocd --version`
- `STM32_Programmer_CLI --version`
- `pio --version`
- `python3 --version`
- macOS version and architecture

Tools found:

- Homebrew 6.0.2.
- CMake 4.2.3.
- GNU Make 3.81.
- Python 3.11.0rc2.

Tools missing:

- Ninja.
- `arm-none-eabi-gcc`.
- `arm-none-eabi-gdb`.
- OpenOCD.
- STM32CubeProgrammer CLI.
- PlatformIO.

Recommended toolchain path:

```text
Use a command-line-friendly workflow first.
Prefer arm-none-eabi-gcc + CMake/Make for the next minimal non-flashing build scaffold review.
Do not require STM32CubeIDE yet.
Install embedded Arm/debug tools only after explicit approval.
```

README update summary:

- Root README now distinguishes Phase 16 Edge AI/Raspberry Pi progress from Phase 17 STM32 PCB integration progress.
- README now documents that ONNX export, ONNX Runtime image/video inference, PyTorch vs ONNX comparison, quantization experiment, benchmark strategy/reporting path, and Raspberry Pi deployment guide are complete or prepared.
- README now documents that STM32 PCB docs, pin mapping, bring-up planning, firmware skeleton docs, UART validation planning, and toolchain inspection are prepared.
- README explicitly avoids claiming Raspberry Pi benchmark completion, STM32 hardware validation, UART testing, firmware flashing, TFLite implementation, or final hardware demo completion.

What not to install yet:

- Do not install `arm-none-eabi-gcc`, GDB, Ninja, OpenOCD, STM32CubeProgrammer, STM32CubeCLT, STM32CubeIDE, or PlatformIO without user approval.

Next recommended Phase 17 step:

```text
Phase 17.2.2 - Minimal non-flashing STM32 build scaffold design review
```

## Files Intentionally Not Created

- STM32CubeIDE project files.
- CubeMX `.ioc` file.
- Generated STM32 HAL project.
- Startup files.
- Linker script.
- CMakeLists.txt or Makefile.
- STM32 firmware source files.
- `.elf`, `.bin`, or `.hex` build artifacts.
- Flashing scripts.
- Phase 17.5 FreeRTOS FSM port files.
- Phase 17.6 end-to-end AI host hardware demo files.

## Known Hardware Blocks

Observed from existing schematic image files:

- STM32F103C8T6 MCU core.
- USB Micro-B 5 V input path.
- AMS1117-3.3V regulator.
- Raspberry Pi UART header.
- SWD programming/debug header.
- Six traffic light LED outputs.
- Dual 7-segment display.
- Expansion headers.
- BOOT0 pulldown.
- PC13 status LED path.

## Confirmed Information

Confirmed only from repository files and visible schematic images:

- Existing schematic images are under `docs/hardware/stm32_pcb/` as `.jpg` files.
- `stm32f103c8.pdf` exists in the same folder.
- `TX1` appears on PA9.
- `RX1` appears on PA10.
- `SWDIO` appears on PA13.
- `SWCLK` appears on PA14.
- PC13 has a visible LED path through a resistor to GND.
- BOOT0 appears pulled down by a 10k resistor.
- Traffic LED outputs appear on PB9, PB8, PB7, PB6, PB5, and PB3 through 1k resistors.
- Dual 7-segment segment lines appear on PA8, PB15, PB14, PB13, PB12, PA12, and PA11 through 220 ohm resistors.
- Dual 7-segment digit/common lines appear on PA3 and PA4.
- The existing ESP32 firmware uses 115200 baud and the `PLAN` / `ACK` / `NACK` / `STATUS` concept.

## Assumptions

- STM32F103C8T6 is the target MCU for the PCB.
- USART1 is intended for Raspberry Pi communication because PA9/PA10 are labeled `TX1`/`RX1`.
- The Raspberry Pi will remain the AI host and planner.
- STM32 will own real-time traffic light execution after plan acceptance.
- The ESP32 FreeRTOS implementation remains the behavior reference.
- Phase 16 Raspberry Pi deployment work remains isolated from Phase 17 hardware integration work.

## TBD Items

- Confirm final schematic source and revision.
- Confirm PCB revision.
- Confirm UART header physical pin 1 and cable orientation.
- Confirm Raspberry Pi TX/RX crossing.
- Confirm exact traffic light LED lane/color assignment.
- Confirm dual 7-segment common-anode/common-cathode behavior.
- Confirm total GPIO current budget.
- Confirm AMS1117 thermal margin.
- Confirm NRST accessibility.
- Confirm SWD header orientation.
- Confirm selected STM32 command-line toolchain after reviewing Phase 17.2.1 inspection results.
- Confirm whether HAL, LL, or CMSIS-only firmware style will be used.

## Risks

- Overclaiming hardware validation from schematic images.
- Incorrect LED lane/color mapping if firmware uses PB pins without physical verification.
- UART TX/RX reversal.
- 5 V UART exposure to Raspberry Pi or STM32 pins.
- Insufficient regulator thermal margin.
- 7-segment current exceeding safe GPIO limits.
- Adding generated STM32 vendor files before choosing a maintainable build path.
- Installing embedded toolchains before the build/debug path is approved.
- Mixing Phase 16 Edge AI deployment changes into the Phase 17 PCB branch.

## What Not to Overclaim

Do not claim:

- STM32 board powered successfully.
- SWD connection works.
- Firmware builds.
- Firmware flashes.
- PC13 blink works.
- Traffic LEDs are tested.
- 7-segment display is tested.
- UART link is tested.
- Raspberry Pi controls STM32.
- FreeRTOS FSM has been ported.
- End-to-end AI host to STM32 demo is complete.

## Recommended Next Subphase

Recommended safe next step:

```text
Phase 17.2.2 - Minimal non-flashing STM32 build scaffold design review
```

Alternative if hardware is available:

```text
Phase 17.3.1 - pre-power visual inspection and measurement worksheet
```

Do not start Phase 17.5 until the user approves the firmware architecture, pin ownership, and build system direction.

## Exact Suggested Prompt for Phase 17.5 or Next Safe Step

Preferred next safe prompt:

```text
Continue Phase 17 with an offline-safe Phase 17.2.1 toolchain decision review.

Do not flash hardware.
Do not require STM32CubeIDE.
Inspect the Phase 17 ledger, STM32 pin mapping, firmware toolchain plan, and existing ESP32 FreeRTOS firmware.
Recommend one command-line STM32 build approach for MacBook.
If and only if justified, create a minimal non-flashing build scaffold, but do not add generated HAL/CubeMX files and do not implement the traffic FSM yet.
Keep the work on branch phase17-stm32-pcb-integration.
```

Future Phase 17.5 review prompt:

```text
Review readiness for Phase 17.5 FreeRTOS Traffic Light FSM Port to STM32.

Do not implement yet.
Inspect docs/embedded/phase_17_ledger.md, docs/hardware/stm32_pcb/stm32f103c8t6_pin_mapping.md, firmware/stm32_f103_traffic_light/toolchain_plan.md, and firmware/esp32_freertos_traffic_light/.
Identify the exact modules that would need to be ported, unresolved pin/toolchain risks, test strategy, rollback strategy, and what hardware validation evidence is required before coding.
```

## Review Checklist for the User

```text
[ ] Confirm the schematic images match the current PCB revision.
[ ] Confirm whether the .jpg schematic images should be copied/exported into schematics/ as .png later.
[ ] Review pin mapping table for any incorrect inferred signal.
[ ] Mark the six traffic LEDs by physical color and lane.
[ ] Confirm whether PC13 LED is present and active-high.
[ ] Confirm UART header orientation before wiring Pi.
[ ] Decide command-line STM32 toolchain direction.
[ ] Decide whether to use HAL, LL, or CMSIS-only code.
[ ] Keep Phase 16 and Phase 17 branches separate.
```

## Learning Map for the User

Must Know:

- STM32 boot pins and BOOT0 behavior.
- SWD wiring: SWDIO, SWCLK, GND, target voltage.
- UART TX/RX crossing and 3.3 V logic safety.
- GPIO current limits and why 7-segment multiplexing matters.
- Difference between accepting a `PLAN` and applying it at a safe FSM boundary.

Good to Know:

- STM32CubeCLT vs arm-none-eabi-gcc/CMake vs PlatformIO tradeoffs.
- HAL vs LL vs CMSIS-only firmware layers.
- OpenOCD vs STM32CubeProgrammer.
- FreeRTOS task ownership on microcontrollers.

Skip For Now:

- Full AI-to-STM32 hardware demo.
- Advanced low-power modes.
- DMA UART receive.
- RTOS tracing.
- Production PCB revision control.

## PHASE_17_CONTEXT_FOR_NEXT_ASSISTANT

```text
Branch:
    phase17-stm32-pcb-integration

Completed:
    Phase 17.1, Phase 17.2, Phase 17.2.1, Phase 17.3, and Phase 17.4 as offline documentation/planning work.

Created files:
    docs/hardware/stm32_pcb/README.md
    docs/hardware/stm32_pcb/stm32f103c8t6_pin_mapping.md
    docs/hardware/stm32_pcb/stm32_pcb_review_notes.md
    docs/hardware/stm32_pcb/schematics/README.md
    docs/embedded/phase_17_stm32_pcb_integration.md
    firmware/stm32_f103_traffic_light/README.md
    firmware/stm32_f103_traffic_light/bringup_plan.md
    firmware/stm32_f103_traffic_light/toolchain_plan.md
    firmware/stm32_f103_traffic_light/toolchain_inspection.md
    firmware/stm32_f103_traffic_light/src/README.md
    firmware/stm32_f103_traffic_light/include/README.md
    docs/hardware/stm32_pcb/stm32_pcb_bringup_plan.md
    docs/hardware/stm32_pcb/stm32_uart_pi_validation_plan.md
    docs/embedded/phase_17_4_uart_validation_plan.md
    docs/embedded/phase_17_ledger.md

Hardware status:
    Schematic documentation and bring-up planning started. Final hardware validation pending.

Firmware status:
    Documentation-first STM32 firmware skeleton only. No source files, build system, or generated project files.

UART status:
    Validation plan written. UART not tested. UART firmware not implemented.

Toolchain status:
    Host inspected. Homebrew, CMake, Make, and Python are installed. Ninja, arm-none-eabi-gcc/gdb, OpenOCD, STM32CubeProgrammer CLI, and PlatformIO are missing. Recommended next direction is arm-none-eabi-gcc + CMake/Make after approval.

Do not overclaim:
    No power validation, SWD validation, blink, UART test, LED test, 7-segment test, FSM port, or AI-to-STM32 demo has been completed.

Next recommended step:
    Phase 17.2.2 minimal non-flashing STM32 build scaffold design review, or Phase 17.3.1 pre-power inspection worksheet if hardware is available.
```
