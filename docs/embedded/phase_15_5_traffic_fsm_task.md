# Phase 15.5 — Traffic FSM Task and Firmware Refactor

## Goal

Replace the temporary FSM placeholder with a real FreeRTOS traffic light FSM task and continue the firmware refactor into a modular structure.

After this phase, the controller flow is:

```text
Host / Serial Monitor
        ↓ USB Serial
TaskUARTReceive
        ↓ rawMessageQueue
TaskPlanParser
        ↓ planQueue
TaskTrafficFSM
        ↓
ESP32 LED outputs
```

## What Changed

Phase 15.4 only received and printed validated `SignalPlan` objects.

Phase 15.5 adds:

```text
Real TaskTrafficFSM
Default active SignalPlan
TrafficState transitions
GPIO output control
Non-blocking planQueue check
Modular firmware structure
```

## Refactored Firmware Structure

```text
firmware/esp32_freertos_traffic_light/
├── platformio.ini
├── include/
│   ├── app_config.h
│   ├── messages.h
│   ├── protocol.h
│   ├── queues.h
│   ├── tasks.h
│   └── traffic_fsm.h
└── src/
    ├── main.cpp
    ├── protocol.cpp
    ├── queues.cpp
    ├── task_plan_parser.cpp
    ├── task_traffic_fsm.cpp
    ├── task_uart_receive.cpp
    └── traffic_fsm.cpp
```

## Module Responsibilities

### `main.cpp`

Responsible for:

```text
Serial initialization
Boot banner
Queue creation
GPIO initialization
Task creation
Minimal Arduino loop
```

### `app_config.h`

Responsible for:

```text
Serial baud rate
Queue sizes
Task stack sizes
Task priorities
GPIO pin mapping
Timing validation limits
FSM update period
```

### `messages.h`

Responsible for:

```text
RawMessage
ParsedPlanFields
SignalPlan
TrafficState
```

### `queues.h` / `queues.cpp`

Responsible for:

```text
rawMessageQueue
planQueue
createApplicationQueues()
```

### `protocol.h` / `protocol.cpp`

Responsible for:

```text
setRawMessage()
isPlanCommand()
parsePlanCommand()
makeSignalPlan()
validateSignalPlan()
printParsedPlan()
printSignalPlan()
```

### `traffic_fsm.h` / `traffic_fsm.cpp`

Responsible for:

```text
default SignalPlan
traffic light GPIO setup
state output control
state duration calculation
next-state calculation
state-to-string conversion
```

### `task_uart_receive.cpp`

Responsible for:

```text
Reading newline-terminated USB Serial commands
Creating RawMessage
Sending RawMessage into rawMessageQueue
```

### `task_plan_parser.cpp`

Responsible for:

```text
Receiving RawMessage from rawMessageQueue
Parsing PLAN commands
Validating timing values
Sending valid SignalPlan into planQueue
Rejecting invalid commands
```

### `task_traffic_fsm.cpp`

Responsible for:

```text
Running the local traffic light FSM
Receiving valid SignalPlan from planQueue
Updating activePlan
Controlling ESP32 LED outputs
Transitioning between traffic states
```

## FreeRTOS Concept

The key design decision is that `TaskTrafficFSM` must not block forever on `planQueue`.

Bad:

```cpp
xQueueReceive(planQueue, &receivedPlan, portMAX_DELAY);
```

Why this is bad:

```text
The FSM would wait forever for a new plan and stop updating traffic light states.
```

Correct:

```cpp
xQueueReceive(planQueue, &receivedPlan, 0);
```

Meaning:

```text
Check whether a new plan exists.
If yes, receive it.
If no, continue running the FSM.
```

This lets the FSM continue controlling traffic lights even when the host is not sending new plans.

## FSM States

```text
STATE_A_GREEN
STATE_A_YELLOW
STATE_ALL_RED_AFTER_A
STATE_B_GREEN
STATE_B_YELLOW
STATE_ALL_RED_AFTER_B
```

State meaning:

```text
STATE_A_GREEN          Road A green, Road B red
STATE_A_YELLOW         Road A yellow, Road B red
STATE_ALL_RED_AFTER_A  Both directions red for intersection clearance
STATE_B_GREEN          Road B green, Road A red
STATE_B_YELLOW         Road B yellow, Road A red
STATE_ALL_RED_AFTER_B  Both directions red for intersection clearance
```

## Default Active Plan

The FSM starts with a default timing plan before receiving any host command:

```text
plan_id = 0
green_a = 20
green_b = 20
yellow = 3
all_red = 1
```

This allows the controller to operate even if the host has not yet sent a valid `PLAN`.

## Important Implementation Detail

The FSM tracks when the current state started:

```cpp
TickType_t stateStartTick = xTaskGetTickCount();
```

Then it calculates elapsed time:

```cpp
uint32_t elapsedMs =
    static_cast<uint32_t>(now - stateStartTick) *
    portTICK_PERIOD_MS;
```

Meaning:

```text
elapsedMs = how long the current FSM state has been active
```

When:

```cpp
elapsedMs >= durationMs
```

the FSM transitions to the next state.

## Why `vTaskDelay()` Is Used

The FSM loop ends with:

```cpp
vTaskDelay(FSM_UPDATE_PERIOD_TICK);
```

This prevents the task from wasting CPU by spinning continuously.

For Phase 15.5, `vTaskDelay()` is acceptable because traffic light timings are in seconds, so a small timing drift is not critical.

Later, this can be upgraded to:

```cpp
vTaskDelayUntil()
```

for more precise periodic execution.

## Implemented Behavior

On boot:

```text
TaskTrafficFSM starts
Default active plan is loaded
Initial state is STATE_A_GREEN
GPIO outputs are applied
FSM transitions automatically based on timing
```

When a valid `SignalPlan` arrives:

```text
TaskTrafficFSM receives it from planQueue
activePlan is updated
Current state continues running
Future state durations use the new plan
```

## Current Limitation

In Phase 15.5, a new plan becomes active immediately:

```cpp
activePlan = receivedPlan;
```

However, the FSM state is not restarted.

This is acceptable for Phase 15.5.

Later, Phase 15.7 will improve this using:

```text
pendingPlan
hasPendingPlan
safe boundary apply
```

so that plans are applied only at a safe FSM cycle boundary.

## Expected Boot Output

```text
[BOOT] ATLC Phase 15 FreeRTOS Controller
[BOOT] Phase 15.5 - Traffic FSM Task
[BOOT] Traffic light GPIO pins initialized.
[BOOT] Application queues created.
[BOOT] TaskUARTReceive created.
[BOOT] TaskPlanParser created.
[BOOT] TaskTrafficFSM created.
[BOOT] Phase 15.5 system is running.
[FSM] Traffic FSM started.
[FSM] Initial state: A_GREEN
[FSM] Default active plan:
[PLAN] plan_id=0 green_a=20 green_b=20 yellow=3 all_red=1
```

## Expected FSM Output

```text
[FSM] Transition to A_YELLOW
[FSM] Transition to ALL_RED_AFTER_A
[FSM] Transition to B_GREEN
[FSM] Transition to B_YELLOW
[FSM] Transition to ALL_RED_AFTER_B
[FSM] Transition to A_GREEN
```

## Manual Test Command

Send through Serial Monitor:

```text
PLAN,17,25,15,3,1
```

Expected output:

```text
[UART] Received line: PLAN,17,25,15,3,1
[PARSER] Received raw message: PLAN,17,25,15,3,1
[PARSER] PLAN command detected.
[PARSER] plan_id=17 green_a=25 green_b=15 yellow=3 all_red=1
[PARSER] Valid SignalPlan sent to planQueue.
[FSM] New active SignalPlan received.
[PLAN] plan_id=17 green_a=25 green_b=15 yellow=3 all_red=1
```

## Build Command

From repo root:

```bash
cd firmware/esp32_freertos_traffic_light
pio run
```

## Upload Command

With ESP32 connected:

```bash
pio run --target upload
```

## Serial Monitor Command

```bash
pio device monitor -b 115200
```

## Test Procedure

1. Build the firmware.

```bash
pio run
```

2. If ESP32 is available, upload the firmware.

```bash
pio run --target upload
```

3. Open Serial Monitor.

```bash
pio device monitor -b 115200
```

4. Send a valid plan.

```text
PLAN,17,25,15,3,1
```

5. Confirm that:
   - UART task receives the line.
   - Parser task validates the plan.
   - FSM task receives the plan.
   - FSM continues transitioning between traffic states.

## Common Mistakes

### Mistake 1 — Blocking forever in FSM

Do not use:

```cpp
xQueueReceive(planQueue, &receivedPlan, portMAX_DELAY);
```

inside the FSM loop.

That would stop the FSM when no new plan is available.

Use:

```cpp
xQueueReceive(planQueue, &receivedPlan, 0);
```

### Mistake 2 — Keeping FSM runtime state in headers

Do not put runtime variables like these in header files:

```cpp
SignalPlan activePlan;
TrafficState currentState;
TickType_t stateStartTick;
```

They should live inside `task_traffic_fsm.cpp` or another `.cpp` file.

### Mistake 3 — Forgetting GPIO setup

`setupTrafficLightPins()` must be called in `setup()` before the FSM starts.

### Mistake 4 — Function name mismatch

Make sure the function name is consistent:

```cpp
getStateDurationMs()
```

not mixed with:

```cpp
getStateDurationsMs()
```

### Mistake 5 — Typo in time variable name

Prefer:

```cpp
elapsedMs
durationMs
```

instead of:

```cpp
elapseMs
durationMS
```

## Completion Checklist

```text
[ ] Firmware folder structure refactored
[ ] main.cpp simplified
[ ] traffic_fsm.h created or updated
[ ] traffic_fsm.cpp implemented
[ ] task_traffic_fsm.cpp implemented
[ ] TaskTrafficFSMPlaceholder replaced
[ ] tasks.h declares TaskTrafficFSM
[ ] main.cpp creates TaskTrafficFSM
[ ] GPIO pins initialized
[ ] Default SignalPlan works
[ ] FSM transitions automatically
[ ] Valid PLAN updates activePlan
[ ] pio run succeeds
[ ] Documentation file created
```

## Next Phase

```text
Phase 15.6 — ACK/NACK Protocol
```

Phase 15.6 will add machine-readable serial responses:

```text
ACK,<plan_id>
NACK,<plan_id>,<reason>
```