# Phase 15.4 — Real USB Serial Receive Task

## Goal

Replace the fake simulated producer with a real USB Serial receive task.

The ESP32 now reads command lines from Serial Monitor or a future host computer and sends each completed line into `rawMessageQueue`.

## Architecture

```text
Host / Serial Monitor
        ↓ USB Serial
TaskUARTReceive
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

Relevant sections:

```
5.2.1 Data Storage
5.2.2 Access by Multiple Tasks
5.3.2 xQueueSendToBack()
5.3.3 xQueueReceive()
5.4   Receiving Data From Multiple Sources
```

## Concept Summary

The UART receive task is an I/O boundary task.

It only reads characters from USB Serial and builds one complete command line.

It does not parse the message.

The parser task still owns PLAN detection, parsing, and validation.

## ATLC Mapping

In the final ATLC system, the host computer or Raspberry Pi will send high-level signal plans to the ESP32 using USB Serial or UART.

Example command:

```
PLAN,17,25,15,3,1
```

The ESP32 receives this as text, validates it, and later the traffic FSM will apply it safely.

## Implemented

- `TaskUARTReceive`
- USB Serial line reading
- newline-based command framing
- RawMessage forwarding into `rawMessageQueue`
- existing parser and planQueue reused

## Not Implemented Yet

- real traffic light FSM
- ACK/NACK protocol
- STATUS messages
- watchdog fallback
- safe plan apply at FSM boundary

## Build

```bash
cd firmware/esp32_freertos_traffic_light
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

## Manual Test Commands

```
PLAN,17,25,15,3,1
PLAN,19,2,15,3,1
HELLO
```

## Expected Output

```
[UART] Received line: PLAN,17,25,15,3,1
[PARSER] Received raw message: PLAN,17,25,15,3,1
[PARSER] PLAN command detected.
[PARSER] plan_id=17 green_a=25 green_b=15 yellow=3 all_red=1
[PARSER] Valid SignalPlan sent to planQueue.
[FSM] Received validated SignalPlan from planQueue.
[PLAN] plan_id=17 green_a=25 green_b=15 yellow=3 all_red=1
```

## Common Mistakes

### Mistake 1 — Parsing inside TaskUARTReceive

Do not parse inside the UART receive task.

Correct:

```
TaskUARTReceive reads text only.
TaskPlanParser parses text.
```

### Mistake 2 — Forgetting newline

The UART task sends a message only after it receives `\n`.

In Serial Monitor, press Enter after typing a command.

### Mistake 3 — Removing planQueue

Do not remove `planQueue`.

Phase 15.4 builds on Phase 15.3.

### Mistake 4 — Reintroducing fake producer

Do not keep `TaskSimulatedProducer`.

Phase 15.4 replaces it with `TaskUARTReceive`.
