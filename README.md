# Adaptive Traffic Light Control System

End-to-end Adaptive Traffic Light Control (ATLC) project combining computer vision, Edge AI deployment, embedded control, UART communication, and hardware integration planning.

This repository is not only an AI notebook. It is structured as a full engineering system:

```text
camera/video input -> YOLO vehicle detection -> traffic logic -> AI-to-MCU protocol -> embedded traffic light controller -> hardware output
```

## Project Overview

The project builds an adaptive traffic light control system using:

- camera-based vehicle detection
- YOLO vehicle classification and bounding boxes
- ROI counting and traffic density estimation
- adaptive green-time planning
- UART communication between an AI host and an MCU controller
- FreeRTOS-based embedded controller execution
- STM32F103C8T6 PCB integration planning

The long-term deployment target is an AI host, such as a Raspberry Pi, connected over UART to a microcontroller-based traffic light controller.

Current high-level status:

```text
The project currently has a validated ONNX Runtime deployment path and a prepared Raspberry Pi deployment guide.
The STM32 PCB integration track has documentation, pin mapping, bring-up planning, UART validation planning, command-line toolchain inspection, pre-power inspection triage, hardware-gated power/SWD validation worksheets, a host-tested common firmware layer, and STM32 board-layer scaffold stubs.
Real Raspberry Pi benchmark validation and STM32 hardware validation remain pending.
```

## Why This Project Matters

Traffic-light control is a practical embedded AI problem: perception, timing logic, communication, firmware reliability, and hardware constraints all matter. This project demonstrates that workflow end to end:

- train and validate a vehicle detector
- export the model for deployment
- verify ONNX Runtime inference on image and video inputs
- compare PyTorch and ONNX behavior
- send timing plans to a real-time embedded controller path
- document the transition from ESP32 prototype to STM32 PCB hardware

The project emphasizes traceable engineering decisions, reproducible commands, and honest status tracking.

## System Architecture

```text
Camera / Video
      |
      v
YOLO Vehicle Detection
      |
      v
ROI Counting / Traffic Density
      |
      v
Adaptive Green-Time Planner
      |
      v
UART PLAN Message
      |
      v
ESP32 / STM32 Controller
      |
      v
Traffic Light FSM
      |
      v
LED / PCB Hardware Output
```

Planned hardware split:

```text
Raspberry Pi AI Host
      <-> UART
STM32F103C8T6 Controller PCB
```

## Key Features

- YOLO-based vehicle detection
- custom vehicle detection training workflow
- ONNX export and ONNX Runtime validation
- ONNX Runtime image and video inference
- PyTorch vs ONNX comparison workflow
- letterbox preprocessing correction for deployment consistency
- ROI counting and traffic density pipeline
- adaptive green-time planning
- FreeRTOS-based embedded controller prototype
- UART `PLAN` / `ACK` / `STATUS` protocol path
- traffic light finite-state machine
- ESP32 controller implementation path
- STM32F103C8T6 PCB schematic integration planning
- future Raspberry Pi AI host deployment path

## Current Status

| Area | Status | Notes |
| --- | --- | --- |
| YOLO training | Completed | Custom vehicle detection workflow exists under `yolo_research/` |
| Local AI pipeline | Completed / prototype | Video input, YOLO detection, ROI counting, density estimation, adaptive timing |
| ONNX export | Completed | YOLO `.pt` exported to `deployment/onnx/atlc_yolo26n_custom.onnx` |
| ONNX validation | Completed | ONNX checker and ONNX Runtime session load passed |
| ONNX image inference | Completed | Letterbox preprocessing and box restoration are implemented |
| ONNX video inference | Completed | Basic video smoke test completed with CPUExecutionProvider |
| PyTorch vs ONNX comparison | Completed | Direct-resize deployment bug found and fixed with letterbox preprocessing |
| ONNX quantization experiment | Completed / experiment | Phase 16.5 dynamic ONNX quantization smoke test exists; not a final deployment decision |
| Edge AI benchmark report | Prepared | Benchmark strategy/reporting path exists; Raspberry Pi benchmark remains pending |
| Raspberry Pi AI deployment | Prepared | Deployment guide exists; real Raspberry Pi validation remains pending |
| FreeRTOS controller | Completed / prototype | ESP32 FreeRTOS controller path with tasks, queues, UART parser, FSM |
| UART protocol | Completed / prototype | `PLAN`, `ACK` / `NACK`, status reporting, watchdog/fallback concepts |
| Hardware-independent common firmware | Completed / host-tested | Plain-C PLAN parser, validator, FSM model, STATUS formatter, and tests under `firmware/common/` |
| STM32 PCB integration | Documentation prepared / hardware-gated | Pin mapping, bring-up plan, firmware skeleton docs, UART validation plan, toolchain inspection, pre-power triage, power-rail worksheet, and ST-LINK read-ID worksheet are prepared |
| STM32 firmware scaffold | Prepared / not hardware-validated | Board GPIO, UART, task, and status stubs exist; no HAL/CMSIS/LL hardware implementation |
| STM32 command-line toolchain | Inspected | STM32CubeIDE is not required at this stage; embedded Arm tools are not installed yet |
| Final STM32 hardware validation | Pending | ST-LINK non-detection has been reported by the hardware team; root cause is not confirmed |

## Demo / Visual Results

### Vehicle Detection Result

![Runtime annotated traffic frame](docs/security/figures/fig_phase14_runtime_annotated_video_frame.png)

### Runtime Evidence

![Runtime FPS evidence](docs/security/figures/runtime_fps.png)

### Letterbox Preprocessing Fix

Phase 16.4 found an important deployment bug during the PyTorch vs ONNX comparison:

```text
Bug:
    The first ONNX Runtime path used direct resize preprocessing.

Impact:
    PyTorch / Ultralytics and ONNX Runtime produced different detections.
    Several obvious motorbikes were missed in the ONNX result.

Fix:
    The ONNX path now uses Ultralytics-style letterbox preprocessing.
    Bounding boxes are restored with ratio, pad_x, and pad_y.

Result:
    ONNX detections became visually close to the PyTorch reference path.
```

![Letterbox preprocessing and bounding-box restoration](docs/edge_ai/figures/letterbox_box_restoration.png)

### STM32 PCB Schematics

Status: schematic documentation and bring-up planning started. Final hardware validation is not claimed complete.

![STM32F103C8T6 MCU core schematic](docs/hardware/stm32_pcb/03_stm32f103c8t6_mcu_core.jpg)

![UART Raspberry Pi header schematic](docs/hardware/stm32_pcb/02_uart_pi_header.jpg)

![Traffic light LED outputs schematic](docs/hardware/stm32_pcb/04_traffic_light_led_outputs.jpg)

Additional schematic blocks:

- [Power input and 3.3 V regulation](docs/hardware/stm32_pcb/01_power_usb_5v_3v3_regulator.jpg)
- [Dual 7-segment display](docs/hardware/stm32_pcb/05_dual_7segment_display.jpg)
- [Expansion headers](docs/hardware/stm32_pcb/06_expansion_headers.jpg)
- [SWD programming header](docs/hardware/stm32_pcb/07_swd_programming_header.jpg)

## Edge AI Deployment

The trained YOLO model was exported from PyTorch / Ultralytics `.pt` format to ONNX and validated with ONNX Runtime.

Generated model artifacts such as `.pt` and `.onnx` files are intentionally not committed to GitHub. The repository keeps the export, validation, inference, and comparison scripts so the deployment artifacts can be reproduced locally.

Implemented deployment workflows:

- `deployment/onnx/export_onnx.py` - YOLO `.pt` to ONNX export
- `deployment/onnx/validate_onnx.py` - ONNX checker and ONNX Runtime load validation
- `deployment/onnx/infer_onnx_image.py` - ONNX Runtime image inference
- `deployment/onnx/infer_onnx_video.py` - ONNX Runtime video inference
- `deployment/compare/compare_pytorch_onnx_image.py` - PyTorch vs ONNX image comparison

Important engineering finding:

```text
Initial ONNX path:
    Direct resize preprocessing

Observed issue:
    Missed detections compared with PyTorch / Ultralytics

Root cause:
    Ultralytics inference preserves aspect ratio with letterbox preprocessing,
    while the initial ONNX path stretched the image directly to 640 x 640.

Fix:
    Letterbox preprocessing plus letterbox-aware box restoration:

        x1 = (x1 - pad_x) / ratio
        y1 = (y1 - pad_y) / ratio
        x2 = (x2 - pad_x) / ratio
        y2 = (y2 - pad_y) / ratio

Result:
    PyTorch and ONNX visual alignment improved significantly.
```

This is an important deployment lesson: model export alone is not enough. The runtime preprocessing and postprocessing must match the training/reference inference path, otherwise a valid ONNX model can still produce poor real-world detections.

Related docs:

- [Phase 16.1 ONNX export](docs/edge_ai/phase_16_1_onnx_export.md)
- [Phase 16.2 ONNX image inference](docs/edge_ai/phase_16_2_onnx_image_inference.md)
- [Phase 16.3 ONNX video inference](docs/edge_ai/phase_16_3_onnx_video_inference.md)
- [Phase 16.4 PyTorch vs ONNX comparison](docs/edge_ai/phase_16_4_pytorch_vs_onnx_comparison.md)
- [Phase 16.5 ONNX quantization](docs/edge_ai/phase_16_5_onnx_quantization.md)
- [Phase 16.5 quantization results](docs/edge_ai/phase_16_5_quantization_results.md)
- [Raspberry Pi YOLO deployment plan](docs/deployment/raspberry_pi_yolo.md)

## Raspberry Pi Deployment Path

The Raspberry Pi track prepares the AI host side of the final system:

```text
Raspberry Pi camera/video input
    -> local YOLO inference
    -> ROI counting and density estimation
    -> adaptive timing plan
    -> UART PLAN message to ESP32/STM32 controller
```

Status:

- ONNX Runtime image/video inference has been validated on the development machine.
- PyTorch vs ONNX comparison found and fixed a preprocessing mismatch.
- ONNX dynamic quantization has been tested as an experiment, not adopted as a final deployment decision.
- Raspberry Pi deployment steps are documented.
- Raspberry Pi hardware benchmark and TFLite deployment are not claimed complete.

## Embedded Controller

The embedded controller receives timing commands from the AI host, validates them, executes a traffic light finite-state machine, and reports status/diagnostics.

Implemented or documented controller concepts:

- FreeRTOS tasks
- queues for command ownership
- UART receive task
- validated signal plan queue
- traffic light FSM task
- software timer status reporting
- ACK / NACK protocol
- host timeout watchdog and safe fallback concept
- mutex-protected serial logging
- task notification cleanup

Main embedded prototype:

```text
firmware/esp32_freertos_traffic_light/
```

Hardware-independent portable controller logic:

```text
firmware/common/
```

The common layer is plain C and host-tested. It covers the canonical `PLAN` parser, plan validation, FSM state sequence, pending-plan boundary behavior, and `ACK` / `NACK` / `STATUS` formatting. It does not depend on Arduino, ESP-IDF, STM32 HAL, CMSIS, LL, or FreeRTOS.

Relevant docs:

- [FreeRTOS queue warmup](docs/embedded/Phase_15_1_freertos_queue_warmup.md)
- [Traffic FSM task](docs/embedded/phase_15_5_traffic_fsm_task.md)
- [ACK / NACK protocol](docs/embedded/phase_15_6_ack_nack_protocol.md)
- [Host timeout watchdog](docs/embedded/phase_15_9_host_timeout_watchdog.md)
- [Runtime diagnostics](docs/embedded/phase_15_10_runtime_diagnostics.md)

## STM32 PCB Hardware Integration

The project is being extended from software simulation and ESP32/FreeRTOS prototyping toward an STM32F103C8T6-based controller PCB.

Current status:

```text
Hardware received / schematic documentation prepared / hardware-gated bring-up procedures prepared
Physical power, SWD, UART, firmware, and GPIO validation remain pending
Hardware-independent common firmware tests pass on the host
```

Schematic blocks under documentation:

- power input and 3.3 V regulation
- STM32F103C8T6 MCU core
- UART interface to Raspberry Pi
- SWD programming header
- traffic light LED outputs
- dual 7-segment display
- expansion headers

Planned STM32 work:

- execute controlled power rail validation on real hardware
- execute non-flashing ST-LINK attach/read-ID validation
- approve toolchain/vendor-file policy before any STM32 build
- implement future pin-safe blink firmware only after power and SWD pass
- validate traffic LED and 7-segment outputs only after blink/GPIO basics
- validate UART hardware before AI host integration
- run end-to-end Raspberry Pi to STM32 demo only after hardware validation

Prepared STM32 documentation:

- [STM32 PCB overview](docs/hardware/stm32_pcb/README.md)
- [STM32F103C8T6 pin mapping](docs/hardware/stm32_pcb/stm32f103c8t6_pin_mapping.md)
- [STM32 PCB bring-up plan](docs/hardware/stm32_pcb/stm32_pcb_bringup_plan.md)
- [STM32 pre-power inspection checklist](docs/hardware/stm32_pcb/phase_17_3_1_pre_power_inspection.md)
- [STM32 controlled power rail validation worksheet](docs/hardware/stm32_pcb/phase_17_3_2_controlled_power_rail_validation.md)
- [STM32 ST-LINK read-ID validation worksheet](docs/hardware/stm32_pcb/phase_17_3_3_stlink_read_id_validation.md)
- [STM32 UART to Raspberry Pi validation plan](docs/hardware/stm32_pcb/stm32_uart_pi_validation_plan.md)
- [ATLC common firmware layer](firmware/common/README.md)
- [STM32 firmware skeleton](firmware/stm32_f103_traffic_light/README.md)
- [STM32 hardware-independent completion plan](firmware/stm32_f103_traffic_light/hardware_independent_completion_plan.md)
- [STM32 toolchain inspection](firmware/stm32_f103_traffic_light/toolchain_inspection.md)
- [Phase 17 hardware-gated completion review](docs/embedded/phase_17_hardware_gated_completion_review.md)

Important status limits:

- STM32CubeIDE is not required at this stage.
- STM32 power rail validation has not been executed.
- ST-LINK read-ID validation has not been executed.
- STM32 board-layer source files are scaffold stubs only.
- STM32 firmware has not been flashed.
- STM32 UART has not been tested.
- STM32 PCB hardware validation is pending.

## Repository Structure

```text
.
├── deployment/
│   ├── compare/
│   └── onnx/
├── docs/
│   ├── edge_ai/
│   ├── embedded/
│   ├── hardware/
│   │   └── stm32_pcb/
│   └── security/
├── experiments/
├── firmware/
│   ├── esp32_freertos_traffic_light/
│   ├── stm32_f103_traffic_light/
│   └── esp32_traffic_light/
├── pc_app/
│   ├── control/
│   ├── security/
│   └── vision/
├── results/
│   ├── comparison/
│   ├── onnx_image_predictions/
│   └── onnx_video_predictions/
└── yolo_research/
    ├── configs/
    ├── scripts/
    └── src/
```

## Technical Stack

| Area | Tools / Technologies |
| --- | --- |
| Computer vision | OpenCV, Ultralytics YOLO |
| Model training | Python, PyTorch, YOLO training scripts |
| Edge AI deployment | ONNX, ONNX Runtime |
| Comparison / validation | PyTorch vs ONNX image comparison, visual inspection, smoke tests |
| Application logic | ROI counting, density estimation, adaptive green-time planning |
| Embedded firmware | C++, FreeRTOS concepts, PlatformIO for ESP32 prototype, planned STM32 command-line toolchain |
| MCU path | ESP32 prototype, STM32F103C8T6 controller PCB path |
| Communication | UART, `PLAN` / `ACK` / `STATUS` protocol |
| Target AI host | Raspberry Pi deployment path prepared; hardware benchmark pending |
| Repository workflow | Git, GitHub, phase-based documentation |

## How To Run

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Inspect available Edge AI commands:

```bash
python deployment/onnx/infer_onnx_image.py --help
python deployment/onnx/infer_onnx_video.py --help
python deployment/compare/compare_pytorch_onnx_image.py --help
```

Example ONNX image inference:

```bash
python deployment/onnx/infer_onnx_image.py \
  --model deployment/onnx/atlc_yolo26n_custom.onnx \
  --image yolo_research/datasets/atlc_2000/images/test/09150440_jpg.rf.DvAoBPo7uxkzXD4hgu8H.jpg \
  --output results/onnx_image_predictions/09150440_onnx.jpg \
  --imgsz 640 \
  --conf 0.25 \
  --providers CPUExecutionProvider
```

Example PyTorch vs ONNX comparison:

```bash
python deployment/compare/compare_pytorch_onnx_image.py \
  --pt-model yolo_research/outputs/runs/atlc_yolo26n_custom/weights/best.pt \
  --onnx-model deployment/onnx/atlc_yolo26n_custom.onnx \
  --image yolo_research/datasets/atlc_2000/images/test/09150440_jpg.rf.DvAoBPo7uxkzXD4hgu8H.jpg \
  --output /tmp/atlc_phase16_4_comparison_fixed.jpg \
  --imgsz 640 \
  --conf 0.25 \
  --providers CPUExecutionProvider \
  --runs 5
```

For the full local adaptive traffic pipeline, see `pc_app/` and the deployment docs under `docs/deployment/`.

## Development Roadmap

| Phase | Status | Description |
| --- | --- | --- |
| Phase 16.5 | Completed / experiment | ONNX Quantization Experiment |
| Phase 16.6 | Prepared | Edge AI Benchmark Report / benchmark strategy; Raspberry Pi benchmark pending |
| Phase 16.7 | Prepared | Raspberry Pi Deployment Path; TFLite not implemented |
| Phase 16.9 | Planned | AI Host `PLAN` Generation Interface |
| Phase 16.10 | Planned | AI-to-MCU Integration Preparation |
| Phase 17.1 | Completed / documentation | STM32 PCB Documentation and Pin Mapping |
| Phase 17.2 | Completed / documentation skeleton | STM32 firmware skeleton and command-line build plan |
| Phase 17.2.1 | Completed / inspection | Command-line STM32 toolchain inspection and README synchronization |
| Phase 17.3 | Hardware-gated | STM32 PCB Bring-Up Procedure; execution requires real board measurements |
| Phase 17.3.1 | Completed / checklist | STM32 pre-power inspection and ST-LINK non-detection triage |
| Phase 17.3.2 | Procedure prepared / pending hardware | Controlled power rail validation |
| Phase 17.3.3 | Procedure prepared / pending hardware | ST-LINK attach / read-ID validation |
| Phase 17 hardware-independent firmware/software prep | Completed / host-tested | Portable common layer, STM32 scaffold stubs, and host protocol tests |
| Phase 17.4 | Completed / plan | UART Link Validation Plan |
| Phase 17.x | Planned | End-to-end Raspberry Pi to STM32 hardware demo |

TensorRT / Jetson deployment is optional future work and is not the current target path.

## Engineering Highlights

- End-to-end system boundary from perception to embedded controller behavior
- Deployment-grade model export and ONNX Runtime validation
- Practical PyTorch vs ONNX comparison instead of assuming export correctness
- Real preprocessing bug found and fixed through visual/runtime comparison
- FreeRTOS task ownership, queues, UART parsing, FSM execution, status reporting, and watchdog concepts
- Hardware integration moving toward an STM32F103C8T6 PCB with documented schematic blocks, pin mapping, bring-up planning, UART validation planning, and command-line toolchain inspection
- Phase-based documentation for reproducibility and recruiter review
