# Phase 16.9 PLAN Generation Results

## Status

```text
PLAN generation interface prepared.
No UART hardware communication yet.
No MCU firmware integration yet.
No end-to-end hardware demo yet.
```

## Commands Run

Interpreter used:

```text
.venv/bin/python
```

Commands:

```bash
.venv/bin/python deployment/ai_host/demo_plan_generation.py --ns-count 12 --ew-count 5 --seq 1
.venv/bin/python deployment/ai_host/demo_plan_generation.py --ns-count 5 --ew-count 12 --seq 2
.venv/bin/python deployment/ai_host/demo_plan_generation.py --ns-count 0 --ew-count 0 --seq 3
.venv/bin/python deployment/ai_host/demo_plan_generation.py --ns-count 10 --ew-count 10 --seq 4
```

## Example Inputs

| Case | NS Count | EW Count | Sequence |
| --- | ---: | ---: | ---: |
| NS heavier | `12` | `5` | `1` |
| EW heavier | `5` | `12` | `2` |
| Empty intersection | `0` | `0` | `3` |
| Equal counts | `10` | `10` | `4` |

## Example Outputs

### Case 1 - NS Heavier

```text
NS count:          12
EW count:          5
NS density ratio:  0.706
EW density ratio:  0.294
NS green:          28 s
EW green:          17 s
Generated plan:
PLAN,seq=1,mode=adaptive,ns_green=28,ew_green=17,yellow=3,all_red=1
Round-trip parse:  OK
Validation:        OK
```

### Case 2 - EW Heavier

```text
NS count:          5
EW count:          12
NS density ratio:  0.294
EW density ratio:  0.706
NS green:          17 s
EW green:          28 s
Generated plan:
PLAN,seq=2,mode=adaptive,ns_green=17,ew_green=28,yellow=3,all_red=1
Round-trip parse:  OK
Validation:        OK
```

### Case 3 - Empty Intersection

```text
NS count:          0
EW count:          0
NS density ratio:  0.500
EW density ratio:  0.500
NS green:          22 s
EW green:          23 s
Generated plan:
PLAN,seq=3,mode=adaptive,ns_green=22,ew_green=23,yellow=3,all_red=1
Round-trip parse:  OK
Validation:        OK
```

### Case 4 - Equal Counts

```text
NS count:          10
EW count:          10
NS density ratio:  0.500
EW density ratio:  0.500
NS green:          22 s
EW green:          23 s
Generated plan:
PLAN,seq=4,mode=adaptive,ns_green=22,ew_green=23,yellow=3,all_red=1
Round-trip parse:  OK
Validation:        OK
```

## Interpretation

The demo shows that:

- higher NS count increases NS green time
- higher EW count increases EW green time
- zero/equal counts use a safe split
- serialized PLAN messages can be parsed and validated without hardware

## Limitations

- No serial I/O was performed.
- No ACK/NACK response was received.
- No STM32 or ESP32 firmware was modified.
- No live ONNX detections were required.

## Next Phase

```text
Phase 16.10 - AI-to-MCU UART Integration Preparation
```
