# Phase 15.6 — ACK/NACK Protocol

## Goal

Add machine-readable protocol responses from the ESP32 back to the host.

The ESP32 now replies with:

```text
ACK,<plan_id>
```

for accepted plans, and:

```text
NACK,<plan_id>,<reason>
```

for rejected plans.

## Protocol Examples

```text
ACK,17
NACK,19,GREEN_A_OUT_OF_RANGE
NACK,-1,MALFORMED_PLAN
NACK,-1,UNKNOWN_COMMAND
```

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
TaskTrafficFSM
```

`TaskPlanParser` now also sends ACK/NACK responses over Serial.

## Implemented

- `sendAck()`
- `sendNack()`
- validation reason strings
- `ACK,<plan_id>` response for accepted plans
- `NACK,<plan_id>,<reason>` response for rejected plans
- `PLAN_QUEUE_FULL` response if `planQueue` cannot accept a valid plan

## Not Implemented Yet

- STATUS messages
- watchdog fallback
- safe plan apply at FSM boundary
- dedicated Serial logger task
- mutex-protected Serial output

## Important Rule

ACK means:

```text
The plan was parsed, validated, and successfully sent to planQueue.
```

ACK does not mean:

```text
The traffic light has already applied the plan safely.
```

Safe application happens later in Phase 15.7.

## Manual Test Commands

```text
PLAN,17,25,15,3,1
PLAN,19,2,15,3,1
PLAN,20,25,200,3,1
PLAN,abc
HELLO
```

## Expected Protocol Responses

```text
ACK,17
NACK,19,GREEN_A_OUT_OF_RANGE
NACK,20,GREEN_B_OUT_OF_RANGE
NACK,-1,MALFORMED_PLAN
NACK,-1,UNKNOWN_COMMAND
```

## Completion Checklist

```text
[ ] Boot banner says Phase 15.6
[ ] sendAck() added to protocol.cpp
[ ] sendNack() added to protocol.cpp
[ ] protocol.h declares sendAck() and sendNack()
[ ] validateSignalPlan() returns reason strings
[ ] Valid PLAN returns ACK
[ ] Invalid PLAN returns NACK with specific reason
[ ] Malformed PLAN returns NACK,-1,MALFORMED_PLAN
[ ] Unknown command returns NACK,-1,UNKNOWN_COMMAND
[ ] ACK is sent only after xQueueSendToBack(planQueue) succeeds
[ ] pio run succeeds
```