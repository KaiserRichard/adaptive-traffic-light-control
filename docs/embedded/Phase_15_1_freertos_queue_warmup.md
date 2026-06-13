# Phase 15 — FreeRTOS-Based Embedded Controller and Edge Deployment Upgrade

## Phase 15.1 — FreeRTOS Queue Warm-Up with RawMessage

## Goal

Create the first minimal FreeRTOS queue demo for the ATLC ESP32 controller.

Current architecture:

```text
TaskSimulatedProducer
        ↓
rawMessageQueue
        ↓
TaskPlanParser
```

This Phase only verifies task-to-task communication using a queue.

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

A FreeRTOS queue allows one task to send data to another task safely.

In this Phase, the queue stores:

```cpp
struct RawMessage
{
    char data[96];
};
```

The queue stores the full `RawMessage` by copy.

This is safe:

```cpp
RawMessage message;
xQueueSendToBack(rawMessageQueue, &message, portMAX_DELAY);
```

Even though `message` is a stack variable, FreeRTOS copies the contents into the queue.

## ATLC Mapping

The final Phase 15 controller will receive host messages like:

```text
PLAN,17,25,15,3,1
```

Later architecture:

```text
TaskUARTReceive
        ↓ rawMessageQueue
TaskPlanParser
        ↓ planQueue
TaskTrafficFSM
```

This Phase only simulates the host by using `TaskSimulatedProducer`.

## Implementation Task

Implement two tasks:

```text
TaskSimulatedProducer
TaskPlanParser
```

Create one queue:

```text
rawMessageQueue
```

Expected behavior:

```text
TaskSimulatedProducer sends a fake PLAN message every 3 seconds.
TaskPlanParser receives and prints the message.
```

## Files to Create or Modify

```text
firmware/esp32_freertos_traffic_light/platformio.ini
firmware/esp32_freertos_traffic_light/src/main.cpp
firmware/esp32_freertos_traffic_light/README.md
docs/embedded/Phase_15_1_freertos_queue_warmup.md
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
[BOOT] Phase 15.1 - Queue Warm-Up with RawMessage
[BOOT] rawMessageQueue created.
[BOOT] TaskSimulatedProducer created.
[BOOT] TaskPlanParser created.
[BOOT] Phase 15.1 system is running.
[PRODUCER] Sent raw message: PLAN,17,25,15,3,1
[PARSER] Received raw message: PLAN,17,25,15,3,1
```

## Test Procedure

1. Connect ESP32 DevKit to the computer using USB.
2. Open the project folder in VS Code.
3. Open PlatformIO terminal.
4. Build the firmware.
5. Upload the firmware.
6. Open Serial Monitor at 115200 baud.
7. Confirm that producer and parser messages appear repeatedly.

## Git Commit

```bash
git add firmware/esp32_freertos_traffic_light
git add docs/embedded/Phase_15_1_freertos_queue_warmup.md
git commit -m "Phase 15: add FreeRTOS RawMessage queue warm-up"
```

## Common Mistakes

### Mistake 1 — Creating tasks before creating the queue

Wrong:

```cpp
xTaskCreate(TaskPlanParser, ...);
rawMessageQueue = xQueueCreate(...);
```

Correct:

```cpp
rawMessageQueue = xQueueCreate(...);
xTaskCreate(TaskPlanParser, ...);
```

### Mistake 2 — Wrong queue item size

Wrong:

```cpp
xQueueCreate(5, sizeof(char *));
```

Correct:

```cpp
xQueueCreate(5, sizeof(RawMessage));
```

In this Phase, the queue stores full `RawMessage` objects, not pointers.

### Mistake 3 — Unsafe string copy

Avoid:

```cpp
strcpy(message.data, input);
```

Use:

```cpp
snprintf(message.data, sizeof(message.data), "%s", input);
```

### Mistake 4 — Expecting loop() to do the work

In Phase 15, FreeRTOS tasks should do the work.

The Arduino `loop()` stays minimal:

```cpp
void loop()
{
    vTaskDelay(pdMS_TO_TICKS(1000));
}
```

### Mistake 5 — Confusing blocking with freezing

`xQueueReceive()` with `portMAX_DELAY` does not freeze the whole ESP32.

It only blocks the parser task.

Other tasks can still run.

## Why xQueueReceive() Blocks Instead of Polling

Polling means checking again and again:

```cpp
if (xQueueReceive(rawMessageQueue, &message, 0) == pdPASS)
{
    Serial.println(message.data);
}
```

If there is no message, the task still keeps checking.

That wastes CPU time.

Blocking is better:

```cpp
xQueueReceive(rawMessageQueue, &message, portMAX_DELAY);
```

This tells FreeRTOS:

```text
Put this task into the Blocked state until a message arrives.
```

For ATLC, this is important because the plan parser should not waste CPU while waiting for host messages.

The parser should sleep until a message arrives.

## Completion Checklist

```text
ESP32 firmware folder created
platformio.ini created
src/main.cpp created
README.md created
Notion Phase note created
Firmware builds successfully
Firmware uploads successfully
Serial Monitor opens at 115200 baud
Producer sends fake PLAN message
Parser receives fake PLAN message
loop() remains minimal
No real UART implemented yet
No FSM implemented yet
No STATUS implemented yet
No watchdog implemented yet
```

## Next Phase

Phase 15.2 — Queue-Based Fake PLAN Parser

Next, we will keep using fake messages but upgrade the parser:

```text
Input:
PLAN,17,25,15,3,1

Parser behavior:
Detect whether the message starts with PLAN
Print PLAN detected
Reject unknown command format
```