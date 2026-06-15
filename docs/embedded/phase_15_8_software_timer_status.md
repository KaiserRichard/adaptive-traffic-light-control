# Phase 15.8 — Software Timer STATUS Messages

## Goal

Add periodic machine-readable status messages from the ESP32 controller to the host.

## STATUS Format

```text
STATUS,<plan_id>,<state>,<remaining_seconds>,<health>
```

Example:

```text
STATUS,17,A_GREEN,12,OK
```

## Architecture

```text
TaskTrafficFSM
        ↓ updates status snapshot
ControllerStatus
        ↓ read by software timer
StatusTimerCallback
        ↓
Serial STATUS output
```

## Implemented

- `ControllerStatus` snapshot
- `status_reporter.h`
- `status_reporter.cpp`
- FreeRTOS software timer
- periodic `STATUS` messages
- FSM status updates

## FreeRTOS Concept

This phase uses a FreeRTOS software timer:

```cpp
xTimerCreate()
xTimerStart()
```

The timer callback runs periodically and prints the current controller status.

## Important Design Note

The timer callback does not run the FSM.

The FSM still runs inside:

```text
TaskTrafficFSM
```

The timer callback only reports the latest status snapshot.

## STATUS Meaning

```text
plan_id
    current active plan

state
    current traffic state

remaining_seconds
    approximate time left in the current state

health
    current controller health
```

For this phase:

```text
health = OK
```

## Test Command

```text
PLAN,17,25,15,3,1
```

Expected protocol response:

```text
ACK,17
```

Expected status output:

```text
STATUS,17,A_GREEN,24,OK
```

## Not Implemented Yet

- watchdog fallback
- host timeout health state
- mutex-protected Serial logging
- dedicated logger task

## Completion Checklist

```text
[ ] ControllerStatus added
[ ] status_reporter.h added
[ ] status_reporter.cpp added
[ ] FreeRTOS software timer created
[ ] STATUS timer started
[ ] TaskTrafficFSM updates controller status
[ ] STATUS lines print every 1 second
[ ] valid PLAN still returns ACK
[ ] pending plan apply from Phase 15.7 still works
[ ] pio run succeeds
```