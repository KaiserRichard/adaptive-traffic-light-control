# ATLC Phase 15 — ESP32 FreeRTOS Traffic Light Controller

## Phase 15.1 — FreeRTOS Queue Warm-Up with RawMessage

This firmware belongs to:

```text
Phase 15 — FreeRTOS-Based Embedded Controller and Edge Deployment Upgrade
```

## Goal

This Phase implements a minimal FreeRTOS queue demo:

```text
TaskSimulatedProducer
        ↓
rawMessageQueue
        ↓
TaskPlanParser
```

The producer sends a fake ATLC command:

```text
PLAN,17,25,15,3,1
```

The parser receives the message from the queue and prints it.

## Not Implemented Yet

```text
Real UART input
PLAN parsing
SignalPlan validation
Traffic light FSM
ACK/NACK protocol
STATUS messages
Software timers
Watchdog fallback
```

## Platform

```text
ESP32 DevKit
PlatformIO
Arduino framework
USB Serial
Baud rate: 115200
```

## Build

```bash
pio run
```

## Upload

```bash
pio run --target upload
```

## Monitor

```bash
pio device monitor -b 115200
```

## Expected Output

```text
[BOOT] ATLC Phase 15 FreeRTOS Controller
[BOOT] Phase 15.1 - Queue Warm-Up with RawMessage
[BOOT] rawMessageQueue created.
[BOOT] TaskSimulatedProducer created.
[BOOT] TaskPlanParser created.
[BOOT] Phase 15.1 system is running.
[PRODUCER] Sent raw message: PLAN,17,25,15,3,1
[PARSER] Received raw message: PLAN,17,25,15,3,1
```

## Git Commit

```bash
git add firmware/esp32_freertos_traffic_light
git commit -m "Phase 15: add FreeRTOS RawMessage queue warm-up"
```