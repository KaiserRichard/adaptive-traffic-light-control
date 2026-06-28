# Phase 15.12 — Task Notifications for STATUS Reporting

## Goal

Use FreeRTOS task notifications to move STATUS printing out of the software timer callback.

## Implemented Architecture

```text
StatusTimer
        ↓
StatusTimerCallback
        ↓ task notification
TaskStatusReporter
        ↓
STATUS output
```

## FreeRTOS Concept

This phase uses:

```cpp
xTaskNotifyGive()
ulTaskNotifyTake()
```

`xTaskNotifyGive()` sends a notification directly to a task.

`ulTaskNotifyTake()` lets a task block until a notification arrives.

## Why This Matters

Software timer callbacks run inside the FreeRTOS daemon task.

Therefore, callbacks should stay short.

The timer callback now only sends a notification.

The actual STATUS formatting and Serial output happen inside `TaskStatusReporter`.

## Queue vs Task Notification

```text
Queue:
    transfers data

Task notification:
    sends a direct event to one task
```

For ATLC:

```text
PLAN messages:
    queue

STATUS timer event:
    task notification
```

## Expected Output

```text
STATUS,17,A_GREEN,12,OK
```

## Important Design Rule

```text
Timer callback triggers work.
Application task performs work.
```

## Completion Checklist

```text
[x] STATUS_REPORTER_TASK_STACK_SIZE added
[x] STATUS_REPORTER_TASK_PRIORITY added
[x] TaskStatusReporter added
[x] StatusTimerCallback uses xTaskNotifyGive()
[x] TaskStatusReporter uses ulTaskNotifyTake()
[x] STATUS output still works
[x] Timer callback no longer prints directly
[ ] pio run succeeds
```
