# AI Host PLAN Generation Interface

## Purpose

This folder contains the Phase 16.9 AI-host planning interface for the Adaptive Traffic Light Control project.

It converts vehicle counts or density estimates into a structured traffic-light PLAN message that a future UART integration layer can send to an MCU.

## Current Status

```text
PLAN generation interface prepared.
No UART hardware communication yet.
No MCU firmware integration yet.
No end-to-end hardware demo yet.
```

## How This Connects To ONNX Inference

The existing ONNX Runtime path produces vehicle detections. Later AI-host logic will assign detections to ROIs and turn them into direction-specific counts or density values.

Phase 16.9 starts after that perception step:

```text
vehicle counts / density input
        |
        v
traffic density estimate
        |
        v
green-time allocation
        |
        v
PLAN object
        |
        v
serializable PLAN message string
```

This phase does not require live YOLO inference. It can run with mock counts.

## How This Connects To Raspberry Pi Deployment

On Raspberry Pi, this package is intended to run after ONNX inference and ROI counting. It is standard-library only, so it should be portable to the Raspberry Pi environment prepared in Phase 16.7.

## How This Connects To STM32 / ESP32 UART Controller

The output of this phase is a UART-ready PLAN string.

Actual serial sending is intentionally deferred:

```text
Phase 16.9:
    Generate and validate PLAN message.

Phase 16.10:
    Prepare UART sending / integration scripts.

Phase 17:
    STM32 / ESP32 receives, validates, acknowledges, and executes PLAN.
```

## Message Format

Current Phase 16.9 PLAN format:

```text
PLAN,seq=1,mode=adaptive,ns_green=28,ew_green=17,yellow=3,all_red=1
```

Future controller responses are expected to follow simple text formats:

```text
ACK,seq=1
NACK,seq=1,reason=...
STATUS,seq=1,state=...,remaining=...
DIAG,seq=1,...
```

Older ESP32 Phase 15 materials used a positional format such as `PLAN,17,25,15,3,1`. Mapping this Phase 16.9 key-value format to final MCU firmware belongs to Phase 16.10 / Phase 17 integration work.

## Demo Command

```bash
.venv/bin/python deployment/ai_host/demo_plan_generation.py \
  --ns-count 12 \
  --ew-count 5 \
  --seq 1
```

Expected output includes:

```text
PLAN,seq=1,mode=adaptive,ns_green=28,ew_green=17,yellow=3,all_red=1
Validation:        OK
```

## Limitations

- No serial port is opened.
- No UART message is sent.
- No MCU ACK/NACK is read.
- No live camera or ONNX inference is required.
- The timing policy is a deterministic prototype, not final traffic optimization.

## Next Step

Phase 16.10 should prepare AI-to-MCU UART integration without modifying firmware behavior prematurely.
