# Phase 15.11 — Serial Logging Mutex

## Goal

Protect Serial output using a FreeRTOS mutex so machine-readable protocol lines cannot become corrupted when multiple tasks or timer callbacks print simultaneously.

---

## Why This Matters

Before this phase, multiple modules may print to Serial:

```text
TaskUARTReceive
TaskPlanParser
TaskTrafficFSM
StatusTimerCallback
DiagnosticsTimerCallback
protocol.cpp
status_reporter.cpp
diagnostics.cpp
```

Without protection, output may become interleaved.

Example:

```text
ACK,STATUS,17,A_GREEN,12,OK
17
```

or

```text
STATUS,17,A_GREEN,DIAG,heap=245000
12,OK
```

This is not only difficult to read.

It can break host-side protocol parsing because:

```text
ACK
NACK
STATUS
DIAG
```

are machine-readable messages.

---

## Design Principle

> Protect the interface, not just the print function.

Serial is now part of the controller communication interface.

Therefore:

```text
One complete protocol line
=
One protected operation
```

Good:

```text
lock mutex
print complete STATUS line
unlock mutex
```

Bad:

```text
lock mutex
print "STATUS,"
unlock mutex

lock mutex
print "17"
unlock mutex
```

The second approach still allows line corruption.

---

## Logging Reliability Classes

### Class 1 — Critical Protocol Output

Examples:

```text
ACK
NACK
```

Characteristics:

```text
Machine-readable
Protocol-critical
Host may depend on them
```

Policy:

```text
Atomic line
Bounded wait
Must not be corrupted
```

---

### Class 2 — Periodic Telemetry

Examples:

```text
STATUS
DIAG
```

Characteristics:

```text
Periodic monitoring
Useful but not individually critical
```

Policy:

```text
Non-blocking
Best-effort
Skip frame if Serial is busy
```

Reason:

```text
A later STATUS/DIAG frame will arrive anyway.
```

---

### Class 3 — Human Debug Logs

Examples:

```text
[FSM]
[UART]
[PARSER]
[BOOT]
```

Characteristics:

```text
Human-readable
Debug only
Not protocol-critical
```

Policy:

```text
Low priority
Best-effort
Should never disturb real-time control
```

---

## Architecture Before Phase 15.11

```text
TaskUARTReceive
        ↓
Serial

TaskPlanParser
        ↓
Serial

TaskTrafficFSM
        ↓
Serial

StatusTimerCallback
        ↓
Serial

DiagnosticsTimerCallback
        ↓
Serial
```

Problem:

```text
No shared output protection exists.
```

---

## Architecture After Phase 15.11

```text
Tasks / Callbacks
        ↓
Logging Module
        ↓
Serial Mutex
        ↓
Serial TX
```

Machine-readable outputs:

```text
ACK
NACK
STATUS
DIAG
```

must pass through the mutex.

---

## Why Use a Mutex?

Shared resource:

```text
Serial TX
```

Protection tool:

```text
FreeRTOS Mutex
```

Benefits:

```text
Mutual exclusion
Priority inheritance
Simple implementation
Short critical section
```

---

## Why Not a Logger Task Yet?

Alternative architecture:

```text
Tasks
        ↓
logQueue
        ↓
TaskLogger
        ↓
Serial
```

Advantages:

```text
Single Serial owner
Highly scalable
Cleaner architecture
```

Disadvantages:

```text
Extra task
Extra queue
More complexity
```

For Phase 15.11:

```text
Mutex solution is sufficient.
```

Future upgrade:

```text
TaskLogger + logQueue
```

---

## Priority Inheritance

FreeRTOS mutexes support priority inheritance.

Problem:

```text
Low-priority task owns Serial mutex.
High-priority task wants Serial mutex.
Medium-priority task becomes ready.
```

Without priority inheritance:

```text
Medium-priority task may delay both tasks.
```

With priority inheritance:

```text
Low-priority mutex owner temporarily inherits
the higher priority.
```

Result:

```text
Mutex released sooner.
Priority inversion becomes bounded.
```

Important:

```text
Priority inheritance does NOT remove
priority inversion.

It only reduces its impact.
```

---

## Timer Callback Rule

STATUS and DIAG are generated from software timer callbacks.

Software timer callbacks run inside:

```text
FreeRTOS Daemon Task
```

Therefore:

```text
Do not block.
```

Use:

```cpp
lockSerial(0)
```

or

```cpp
tryLogLine(...)
```

If Serial is busy:

```text
Skip the frame.
```

Do not use:

```cpp
lockSerial(portMAX_DELAY)
```

inside timer callbacks.

---

## Important Boundary

The Serial mutex protects only:

```text
Serial TX output
```

It does NOT protect:

```text
controllerStatus
activePlan
pendingPlan
traffic FSM state
watchdog timestamps
shared counters
```

Those require:

```text
ownership rules
queues
critical sections
other synchronization mechanisms
```

---

## Implemented

```text
logging.h
logging.cpp

initLogging()
lockSerial()
unlockSerial()
logLine()
tryLogLine()

Protected ACK output
Protected NACK output
Protected STATUS output
Protected DIAG output
```

---

## Expected Output

Before:

```text
ACK,STATUS,17,A_GREEN,12,OK
17
```

After:

```text
ACK,17
STATUS,17,A_GREEN,12,OK
DIAG,heap=245000,uart_stack=3200,parser_stack=2800,fsm_stack=3100
[FSM] Transition to B_GREEN
```

Each line remains complete.

---

## Professional Notes

> Machine-readable logs are part of the protocol, not decoration.

> A mutex is acceptable here because the protected section is short and the resource is truly shared.

> Timer callbacks should trigger work, not become work.

> Phase 15.11 intentionally chooses a mutex as a learning and implementation step before introducing a dedicated logger task.

---

## Completion Checklist

```text
[ ] logging.h created
[ ] logging.cpp created
[ ] Serial mutex created
[ ] initLogging() called
[ ] ACK output protected
[ ] NACK output protected
[ ] STATUS output protected
[ ] DIAG output protected
[ ] Timer callbacks use non-blocking logging
[ ] Serial output remains readable
[ ] pio run succeeds
```

## Next Phase

```text
Phase 15.12 — Task Notifications
```

Task notifications provide lightweight task-to-task signaling.

Compared to queues:

```text
Queue
    → transfers data

Task Notification
    → transfers a signal or small value
```

This becomes useful for efficient embedded event handling.
