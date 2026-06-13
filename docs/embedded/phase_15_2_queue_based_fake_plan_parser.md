# Phase 15 — FreeRTOS-Based Embedded Controller and Edge Deployment Upgrade

## Phase 15.2 — Queue-Based Fake PLAN Parser

## Goal

Upgrade the Phase 15.1 queue demo by making the parser detect fake `PLAN` commands.

Current architecture:

```text
TaskSimulatedProducer
        ↓
rawMessageQueue
        ↓
TaskPlanParser
        ↓
Detect PLAN / reject unknown command
```

This phase does not use real USB Serial input yet.

## FreeRTOS Book Mapping

Relevant Chapter 5 sections:

```text
5.2   Characteristics of a Queue
5.2.1 Data Storage
5.2.2 Access by Multiple Tasks
5.3.1 xQueueCreate()
5.3.2 xQueueSendToBack()
5.3.3 xQueueReceive()
```

## Concept Summary

A FreeRTOS queue transports data between tasks.

The queue does not parse the message.

The parser task decides what the message means.

In this phase:

```text
TaskSimulatedProducer sends raw text.
TaskPlanParser receives raw text.
TaskPlanParser detects whether the text is a PLAN command.
```

This is a good embedded design because the transport layer and the parser layer are separated.

## ATLC Mapping

The future host will send signal plans like:

```text
PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>
```

Example:

```text
PLAN,17,25,15,3,1
```

In this phase, the producer simulates host messages.

The parser checks whether a received message is a valid-looking `PLAN` command.

Later, valid plans will be sent into `planQueue`.

## Implementation Task

Modify `src/main.cpp` so that:

```text
TaskSimulatedProducer sends multiple fake messages.
TaskPlanParser receives messages from rawMessageQueue.
TaskPlanParser detects PLAN messages.
TaskPlanParser parses PLAN fields using sscanf().
TaskPlanParser rejects unknown or malformed messages.
```

Fake messages:

```text
PLAN,17,25,15,3,1
HELLO
BAD,17,25
PLAN,18,30,20,3,1
```

## Files to Create or Modify

```text
firmware/esp32_freertos_traffic_light/platformio.ini
firmware/esp32_freertos_traffic_light/src/main.cpp
firmware/esp32_freertos_traffic_light/README.md
docs/embedded/phase_15_2_queue_based_fake_plan_parser.md
```

## Build / Upload / Monitor Commands

```bash
cd firmware/esp32_freertos_traffic_light
pio run
pio run --target upload
pio device monitor -b 115200
```

## Expected Serial Output

```text
[BOOT] ATLC Phase 15 FreeRTOS Controller
[BOOT] Phase 15.2 - Queue-Based Fake PLAN Parser
[BOOT] rawMessageQueue created.
[BOOT] TaskSimulatedProducer created.
[BOOT] TaskPlanParser created.
[BOOT] Phase 15.2 system is running.
[PRODUCER] Sent raw message: PLAN,17,25,15,3,1
[PARSER] Received raw message: PLAN,17,25,15,3,1
[PARSER] PLAN command detected.
[PARSER] plan_id=17 green_a=25 green_b=15 yellow=3 all_red=1
[PRODUCER] Sent raw message: HELLO
[PARSER] Received raw message: HELLO
[PARSER] Unknown command format.
[PRODUCER] Sent raw message: BAD,17,25
[PARSER] Received raw message: BAD,17,25
[PARSER] Unknown command format.
[PRODUCER] Sent raw message: PLAN,18,30,20,3,1
[PARSER] Received raw message: PLAN,18,30,20,3,1
[PARSER] PLAN command detected.
[PARSER] plan_id=18 green_a=30 green_b=20 yellow=3 all_red=1
```

## Test Procedure

1. Open the firmware folder:

```bash
cd firmware/esp32_freertos_traffic_light
```

2. Build the firmware:

```bash
pio run
```

3. If ESP32 is available, upload:

```bash
pio run --target upload
```

4. Open Serial Monitor:

```bash
pio device monitor -b 115200
```

5. Confirm that fake messages cycle every 3 seconds.

6. Confirm that valid PLAN messages are detected.

7. Confirm that unknown messages are rejected.

## Git Commit

```bash
git add firmware/esp32_freertos_traffic_light/platformio.ini
git add firmware/esp32_freertos_traffic_light/src/main.cpp
git add firmware/esp32_freertos_traffic_light/README.md
git add docs/embedded/phase_15_2_queue_based_fake_plan_parser.md
git commit -m "Phase 15: add queue-based fake PLAN parser"
```

## Common Mistakes

### Mistake 1 — Parsing inside the producer task

Do not parse messages in `TaskSimulatedProducer`.

Wrong design:

```text
Producer creates message and parses it immediately.
```

Correct design:

```text
Producer only sends raw message.
Parser receives and interprets raw message.
```

This separation prepares the firmware for real USB Serial input later.

### Mistake 2 — Using real Serial input too early

Do not add this yet:

```cpp
Serial.read();
Serial.readStringUntil('\n');
```

Real USB Serial input belongs to:

```text
Phase 15.4 — Real USB Serial Receive Task
```

### Mistake 3 — Thinking sscanf() validates the values

This phase only checks format.

Example:

```text
PLAN,17,999,999,3,1
```

This may parse successfully, but the values are not valid for the traffic light system.

Range validation belongs to:

```text
Phase 15.3 — Validated SignalPlan Queue
```

### Mistake 4 — Wrong expected field count

The expected format has 5 integer fields:

```text
PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>
```

So this must return 5:

```cpp
sscanf(message->data, "PLAN,%d,%d,%d,%d,%d", ...);
```

### Mistake 5 — Forgetting that xQueueReceive() blocks only one task

This call blocks only `TaskPlanParser`:

```cpp
xQueueReceive(rawMessageQueue, &receivedMessage, portMAX_DELAY);
```

The ESP32 is still running other tasks.

## Completion Checklist

```text
Phase 15.2 documentation created
TaskSimulatedProducer sends multiple fake messages
rawMessageQueue still stores RawMessage by copy
TaskPlanParser receives messages
TaskPlanParser detects valid PLAN commands
TaskPlanParser rejects unknown commands
Parsed PLAN fields are printed
No real USB Serial input added
No planQueue added yet
No FSM added yet
No ACK/NACK added yet
No STATUS added yet
Firmware builds successfully
Git commit completed
```

## Next Phase

Phase 15.3 — Validated SignalPlan Queue

Next implementation:

```text
TaskSimulatedProducer
        ↓ rawMessageQueue
TaskPlanParser
        ↓ planQueue
TaskTrafficFSM placeholder
```

In Phase 15.3, we will add:

```text
SignalPlan struct
planQueue
value range validation
valid plans sent to planQueue
invalid plans rejected
```

Example valid plan:

```text
PLAN,17,25,15,3,1
```

Example invalid plan:

```text
PLAN,19,2,15,3,1
```

Reason:

```text
green_a is too short
```