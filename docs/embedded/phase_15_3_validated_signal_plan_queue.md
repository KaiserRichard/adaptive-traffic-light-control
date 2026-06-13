# Phase 15.3 — Validated SignalPlan Queue

## Goal

Add a second FreeRTOS queue, `planQueue`, to transfer validated `SignalPlan` objects from the parser task to a placeholder FSM task.

## Architecture

```text
TaskSimulatedProducer
        ↓
rawMessageQueue
        ↓
TaskPlanParser
        ↓
planQueue
        ↓
TaskTrafficFSMPlaceholder
```

## FreeRTOS Book Mapping

This phase maps to Chapter 5 Queue Management.

The main idea is that different queues can carry different data types.

```text
rawMessageQueue stores RawMessage
planQueue stores SignalPlan
```

## Concept Summary

Phase 15.2 parsed raw PLAN text.

Phase 15.3 validates parsed values before sending them to the next task.

Valid plans are sent to `planQueue`.

Invalid plans are rejected.

## ATLC Mapping

In the final ATLC system, only validated timing plans should reach the traffic FSM.

This prevents unsafe plans such as:

```text
PLAN,19,2,15,3,1
```

from controlling the traffic lights.

## Implemented

- `SignalPlan` structure
- `planQueue`
- basic timing validation
- valid plan transfer to FSM placeholder task
- invalid plan rejection

## Not Implemented Yet

- real USB Serial input
- real traffic FSM
- ACK/NACK protocol
- STATUS messages
- watchdog fallback

## Build

```bash
cd firmware/esp32_freertos_traffic_light
pio run
```