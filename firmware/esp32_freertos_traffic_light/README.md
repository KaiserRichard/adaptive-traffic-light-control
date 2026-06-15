# ATLC Phase 15 — ESP32 FreeRTOS Traffic Light Controller

## Overview

Phase 15 upgrades the ATLC embedded controller from a simple Arduino loop-based design into a FreeRTOS-based architecture.

The long-term goal is to move traffic light control logic onto the ESP32 so the controller can operate independently from the host computer.

---

## Target Architecture

```text
Host / Raspberry Pi / MacBook
        ↓
USB Serial
        ↓
TaskUARTReceive
        ↓
rawMessageQueue
        ↓
TaskPlanParser
        ↓
planQueue
        ↓
TaskTrafficFSM
        ↓
Traffic Light Outputs
```

The host sends high-level signal plans.

The ESP32 validates plans and executes the traffic light FSM locally.

---

## Current Progress

* Phase 15.1 — FreeRTOS RawMessage Queue Warm-Up
* Phase 15.2 — Queue-Based Fake PLAN Parser
* Phase 15.3 — Validated SignalPlan Queue
* Phase 15.4 — Real USB Serial Receive Task
* Phase 15.5 — Traffic FSM Task
* Phase 15.6 — ACK/NACK Protocol
* Phase 15.7 — Safe Plan Apply at FSM Cycle Boundary
* Phase 15.8 — Software Timer STATUS Messages

---

## Current Status

### Implemented

* FreeRTOS task creation
* RawMessage queue
* Queue-based task-to-task communication
* Fake PLAN message generation
* PLAN command detection
* PLAN field extraction using `sscanf()`

### Planned

* Validated SignalPlan queue
* Real USB Serial receive task
* Traffic light FSM task
* ACK/NACK protocol
* STATUS messages
* Software timers
* Host timeout watchdog
* Runtime diagnostics

---

## Platform

```text
ESP32 DevKit
PlatformIO
Arduino Framework
FreeRTOS
USB Serial
115200 baud
```

---

## Build

```bash
pio run
```

---

## Upload

```bash
pio run --target upload
```

---

## Serial Monitor

```bash
pio device monitor -b 115200
```

---

## Project Structure

```text
firmware/
└── esp32_freertos_traffic_light/
    ├── platformio.ini
    ├── README.md
    └── src/
        └── main.cpp

docs/
└── embedded/
    ├── Phase_15_1_freertos_queue_warmup.md
    └── Phase_15_2_queue_based_fake_plan_parser.md
```

---

## Documentation

* Phase 15.1 — FreeRTOS RawMessage Queue Warm-Up
* Phase 15.2 — Queue-Based Fake PLAN Parser
* Phase 15.3 — Validated SignalPlan Queue
* Phase 15.4 — Real USB Serial Receive Task

Future documentation will be added incrementally as Phase 15 evolves.

---

## Git Workflow

```bash
git status
git add <files>
git commit -m "<message>"
git push
```

Example:

```bash
git commit -m "Phase 15: add queue-based fake PLAN parser"
```
