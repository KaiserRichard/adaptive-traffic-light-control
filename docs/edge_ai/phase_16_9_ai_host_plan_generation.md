# Phase 16.9 AI Host PLAN Generation

## Goal

Create a hardware-independent AI-host interface that converts vehicle counts or density estimates into a controller-ready PLAN message.

This phase bridges perception output and future MCU communication without sending UART messages yet.

## Scope

Included:

- count validation
- density ratio calculation
- deterministic adaptive green-time allocation
- structured `TrafficPlan` object
- PLAN serialization and parsing
- hardware-free demo script

Not included:

- live YOLO or ONNX inference
- ROI counting implementation
- UART sending
- MCU ACK/NACK handling
- STM32 or ESP32 firmware changes
- end-to-end hardware demo

## Why This Phase Exists

Earlier Phase 16 work prepared the model runtime. Phase 16.9 prepares the control-intent boundary:

```text
vehicle counts / density
        |
        v
traffic timing plan
        |
        v
UART-ready PLAN message
```

Without this boundary, the AI host would jump from detections directly toward firmware integration, which would make testing harder and mix deployment concerns.

## Architecture

```text
Camera / Video Input
        |
        v
Raspberry Pi AI Host
        |
        v
YOLO / ONNX Runtime Inference
        |
        v
Vehicle Detection
        |
        v
ROI Counting / Density Estimation
        |
        v
AI Host PLAN Generation
        |
        v
UART-ready PLAN Message
        |
        v
ESP32 / STM32 Controller
```

Phase 16.9 implements the `AI Host PLAN Generation` block only.

## Traffic Density Concept

The current prototype accepts:

```text
north_south_count
east_west_count
```

These counts are converted into normalized density ratios:

```text
ns_ratio = north_south_count / total_count
ew_ratio = east_west_count / total_count
```

If both counts are zero, the system uses a neutral ratio:

```text
0.5 / 0.5
```

## Timing Policy

Default timing configuration:

```text
min_green = 10
max_green = 45
base_cycle_green = 45
yellow = 3
all_red = 1
```

Policy:

- the busier direction receives more green time
- equal or zero counts receive a safe split
- all green times stay inside configured bounds
- the algorithm is deterministic and testable

This is not a final traffic optimization algorithm. It is a deployment interface and control-intent prototype.

## PLAN Message Format

Current serialized format:

```text
PLAN,seq=1,mode=adaptive,ns_green=28,ew_green=17,yellow=3,all_red=1
```

Required fields:

| Field | Meaning |
| --- | --- |
| `PLAN` | Message type |
| `seq` | Host-generated sequence number |
| `mode` | `adaptive`, `fixed`, or `fallback` |
| `ns_green` | North-south green time in seconds |
| `ew_green` | East-west green time in seconds |
| `yellow` | Yellow time in seconds |
| `all_red` | All-red clearance time in seconds |

## Future ACK / NACK / STATUS Relationship

Future controller responses should be kept simple:

```text
ACK,seq=1
NACK,seq=1,reason=...
STATUS,seq=1,state=...,remaining=...
DIAG,seq=1,...
```

Phase 16.9 only defines PLAN generation and validation. Phase 16.10 should prepare host-side UART sending and response handling.

## How To Run Demo

```bash
.venv/bin/python deployment/ai_host/demo_plan_generation.py \
  --ns-count 12 \
  --ew-count 5 \
  --seq 1
```

Additional smoke cases:

```bash
.venv/bin/python deployment/ai_host/demo_plan_generation.py --ns-count 5 --ew-count 12 --seq 2
.venv/bin/python deployment/ai_host/demo_plan_generation.py --ns-count 0 --ew-count 0 --seq 3
.venv/bin/python deployment/ai_host/demo_plan_generation.py --ns-count 10 --ew-count 10 --seq 4
```

## Limitations

- No UART hardware communication.
- No MCU firmware integration.
- No end-to-end Raspberry Pi to MCU demo.
- No live ONNX inference required.
- No final traffic optimization claim.
- Existing ESP32 Phase 15 positional PLAN examples may need an adapter or firmware-side parser update later.

## Connection To Phase 16.10

Phase 16.10 should use the PLAN object/string from this phase and prepare host-side serial communication, timeout handling, and ACK/NACK parsing.

It should not change the planner algorithm unless new integration requirements are measured.

## Connection To Phase 17

Phase 17 should validate that STM32 / ESP32 firmware can receive, validate, acknowledge, and execute PLAN messages safely.

Phase 16.9 does not modify Phase 17 files and does not claim controller-side validation is complete.
