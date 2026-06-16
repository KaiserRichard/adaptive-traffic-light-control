# Phase 15.10 — Runtime Diagnostics: Heap and Stack Monitoring

## Goal

Add runtime diagnostics to the ESP32 FreeRTOS controller.

The controller should periodically report:

```text
free heap memory
remaining task stack space
```

This helps detect memory-related issues before they become runtime failures.

---

## Why This Phase Matters

The controller now contains:

```text
TaskUARTReceive
TaskPlanParser
TaskTrafficFSM
StatusTimer
Host Timeout Watchdog
```

As the firmware grows, it becomes difficult to answer questions such as:

```text
How much RAM is still available?
How much stack is each task using?
Is any task close to stack overflow?
```

Phase 15.10 introduces a diagnostics subsystem that continuously monitors system health.

---

## FreeRTOS Concepts Used

### Heap

The FreeRTOS heap is used when creating:

```text
Tasks
Queues
Software Timers
Semaphores
Mutexes
```

The controller reports the remaining heap using:

```cpp
xPortGetFreeHeapSize()
```

---

### Task Stack

Each task owns its own stack.

Examples:

```text
TaskUARTReceive
TaskPlanParser
TaskTrafficFSM
```

A task stack stores:

```text
Local variables
Function calls
Temporary execution data
```

---

### Stack High-Water Mark

FreeRTOS provides:

```cpp
uxTaskGetStackHighWaterMark()
```

This reports the minimum amount of remaining stack since the task started.

It helps estimate:

```text
How close a task has come to stack overflow.
```

---

## Design Decision

Instead of creating another task:

```text
TaskDiagnostics
```

the project reuses the Software Timer architecture introduced in Phase 15.8.

A new timer:

```text
DiagnosticsTimer
```

runs periodically.

Architecture:

```text
DiagnosticsTimer
        ↓
DiagnosticsTimerCallback()
        ↓
printDiagnosticsLine()
        ↓
DIAG,heap=...,uart_stack=...,parser_stack=...,fsm_stack=...
```

This keeps diagnostics lightweight and avoids creating another application task.

---

## Task Handles

To inspect task stack usage, the diagnostics module needs task handles.

Example:

```cpp
TaskHandle_t trafficFsmTaskHandle;
```

A task handle is a FreeRTOS reference to an existing task.

The diagnostics module uses these handles to inspect stack usage:

```cpp
uxTaskGetStackHighWaterMark(
    trafficFsmTaskHandle
);
```

---

## Diagnostic Output Format

Example:

```text
DIAG,heap=245000,uart_stack=3200,parser_stack=2800,fsm_stack=3100
```

Field meaning:

```text
heap
    Remaining FreeRTOS heap memory.

uart_stack
    Remaining stack for TaskUARTReceive.

parser_stack
    Remaining stack for TaskPlanParser.

fsm_stack
    Remaining stack for TaskTrafficFSM.
```

---

## Files Added

```text
include/diagnostics.h
src/diagnostics.cpp
```

---

## Files Modified

```text
include/app_config.h
src/main.cpp
```

---

## Expected Boot Output

```text
[BOOT] ATLC Phase 15 FreeRTOS Controller
[BOOT] Phase 15.10 - Runtime Diagnostics
[BOOT] Traffic light GPIO pins initialized.
[BOOT] Status reporter initialized.
[BOOT] Application queues created.
[BOOT] TaskUARTReceive created.
[BOOT] TaskPlanParser created.
[BOOT] TaskTrafficFSM created.
[BOOT] Diagnostics reporter initialized.
[STATUS] StatusTimer started.
[DIAG] DiagnosticsTimer started.
[BOOT] Phase 15.10 system is running.
```

---

## Expected Runtime Output

```text
STATUS,17,A_GREEN,12,OK

DIAG,heap=245000,uart_stack=3200,parser_stack=2800,fsm_stack=3100

STATUS,17,A_GREEN,11,OK

DIAG,heap=244980,uart_stack=3200,parser_stack=2800,fsm_stack=3100
```

---

## Benefits

```text
Monitor free heap memory
Monitor task stack usage
Detect potential stack overflow risks
Improve firmware observability
Prepare for larger FreeRTOS applications
```

---

## Completion Checklist

```text
[ ] diagnostics.h created
[ ] diagnostics.cpp created
[ ] DiagnosticsTimer added
[ ] Task handles stored
[ ] Heap monitoring added
[ ] Stack monitoring added
[ ] DIAG line prints correctly
[ ] pio run succeeds
```

---

## Next Phase

```text
Phase 15.11 — Serial Logging Mutex
```

The next phase protects shared Serial output using a FreeRTOS mutex so STATUS, DIAG, FSM, UART and parser logs cannot interleave.
