# Phase 15.13 — UART RX Event Notification and Final Firmware Refactor

## Goal

Finalize the Phase 15 FreeRTOS controller core before hardware circuit testing and full ATLC pipeline validation.

This phase combines two changes:

```text
1. Add UART RX event notification and deferred UART processing.
2. Refactor the firmware into clearer module folders.
```

## Implemented Runtime Architecture

```text
UART RX event callback
        ↓
notify TaskUARTReceive
        ↓
TaskUARTReceive wakes
        ↓
reads Serial bytes
        ↓
builds RawMessage
        ↓
rawMessageQueue
        ↓
TaskPlanParser
        ↓
planQueue
        ↓
TaskTrafficFSM
        ↓
traffic light outputs
```

The callback does not parse commands, update the FSM, or print logs.

It only wakes `TaskUARTReceive`.

## Folder Structure

```text
include/
├── app/
├── config/
├── core/
├── drivers/
├── messages/
├── protocol/
├── services/
└── tasks/

src/
├── app/
├── core/
├── drivers/
├── protocol/
├── services/
└── tasks/
```

## Module Ownership

```text
config/
    project constants

messages/
    shared message structures

core/
    queues and traffic FSM logic

drivers/
    UART RX event registration

tasks/
    FreeRTOS task entry functions

services/
    logging, STATUS, diagnostics

protocol/
    PLAN parsing, validation, ACK/NACK
```

## UART RX Driver

`Serial.onReceive()` is used as the Arduino-ESP32 UART receive event source.

This applies the deferred processing pattern:

```text
callback stays short
task performs the work
```

The callback uses:

```cpp
vTaskNotifyGiveFromISR()
```

`TaskUARTReceive` waits with:

```cpp
ulTaskNotifyTake()
```

## Queue vs Task Notification

```text
Task notification:
    wake TaskUARTReceive

Queue:
    transfer RawMessage and SignalPlan data
```

The notification is only an event.

The queues still carry data.

## Expected Manual Test

Build:

```bash
cd firmware/esp32_freertos_traffic_light
pio run
```

Upload:

```bash
pio run --target upload
```

Monitor:

```bash
pio device monitor -b 115200
```

Send:

```text
PLAN,17,25,15,3,1
```

Expected:

```text
ACK,17
STATUS,17,A_GREEN,...
DIAG,heap=...
```

Invalid input:

```text
PLAN,18,2,15,3,1
```

Expected:

```text
NACK,18,GREEN_A_OUT_OF_RANGE
```

## Completion Checklist

```text
[x] include/ folder refactored
[x] src/ folder refactored
[x] include paths updated
[x] uart_rx.h created
[x] uart_rx.cpp created
[x] UART receive callback registered
[x] callback only notifies TaskUARTReceive
[x] TaskUARTReceive waits on notification
[x] TaskUARTReceive still builds RawMessage
[x] rawMessageQueue still sends data to TaskPlanParser
[x] planQueue still sends SignalPlan to TaskTrafficFSM
[x] ACK output remains mutex-protected
[x] NACK output remains mutex-protected
[x] STATUS output remains task-notified
[x] DIAG output remains best-effort
[x] pio run succeeds
[ ] upload succeeds
[ ] serial monitor test succeeds
```

## Next Phase

Phase 16 should focus on hardware circuit testing and full pipeline validation:

```text
YOLO vehicle count -> signal plan -> UART -> ESP32 -> traffic lights
```
