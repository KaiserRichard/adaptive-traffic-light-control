# Phase 15.7 — Safe Plan Apply at FSM Cycle Boundary

## Goal

Store newly received `SignalPlan` objects as pending plans and apply them only at a safe FSM boundary.

## Architecture

```text
TaskPlanParser
        ↓
planQueue
        ↓
TaskTrafficFSM
        ↓
pendingPlan
        ↓ safe boundary
activePlan
        ↓
Traffic light outputs
```

## Why This Matters

Applying a new timing plan immediately can change the duration of the current state while it is already running.

Phase 15.7 avoids this by separating:

```text
plan reception
```

from:

```text
plan application
```

## Implemented

```text
pendingPlan
hasPendingPlan
safe boundary apply
```

The safe boundary is:

```text
STATE_A_GREEN
```

A pending plan is applied only when the FSM returns to `STATE_A_GREEN` after a full cycle.

## Key Behavior

When a valid plan arrives:

```text
The parser sends it to planQueue.
The FSM receives it.
The FSM stores it as pendingPlan.
The current traffic cycle continues.
The pending plan is applied when the FSM returns to STATE_A_GREEN.
```

## Important ACK Meaning

ACK means:

```text
The plan was parsed, validated, and accepted into planQueue.
```

ACK does not mean:

```text
The plan has already been applied to the traffic lights.
```

## Test Command

```text
PLAN,17,25,15,3,1
```

Expected first FSM response:

```text
[FSM] Pending SignalPlan received.
```

Expected later response:

```text
[FSM] Pending SignalPlan applied at safe boundary.
```

## Completion Checklist

```text
[ ] pendingPlan added
[ ] hasPendingPlan added
[ ] new plan no longer overwrites activePlan immediately
[ ] pending plan is applied only at STATE_A_GREEN
[ ] valid PLAN still produces ACK
[ ] FSM continues transitioning
[ ] pio run succeeds
```