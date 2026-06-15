# Phase 15.9 — Host Timeout Watchdog and Fallback Plan

## Goal

Detect when the host stops sending valid `PLAN` commands and switch the ESP32 controller into fallback behavior.

## Why This Matters

The ESP32 controller should remain safe even if:

```text
Python host crashes
YOLO pipeline freezes
USB serial connection fails
Raspberry Pi becomes overloaded
```

## Implemented Behavior

```text
Track last valid PLAN receive time
Detect host silence timeout
Set health to HOST_TIMEOUT
Switch to fallback SignalPlan
Continue FSM transitions
Continue STATUS reporting
Recover when valid PLAN returns
```

## Fallback Plan

The fallback plan uses the default safe timing plan:

```text
plan_id = 0
green_a = 20
green_b = 20
yellow = 3
all_red = 1
```

## Health States

```text
OK
HOST_TIMEOUT
```

## STATUS Example

Normal:

```text
STATUS,17,A_GREEN,12,OK
```

Timeout:

```text
STATUS,0,A_GREEN,12,HOST_TIMEOUT
```

## Important Design Decision

Normal valid plans are applied at a safe FSM boundary.

Host timeout fallback is applied immediately because it is a safety condition.

## Test Procedure

Temporarily set:

```cpp
static const uint32_t HOST_TIMEOUT_SECONDS = 10;
```

Build:

```bash
cd firmware/esp32_freertos_traffic_light
pio run
```

Upload:

```bash
pio run --target upload
```

Open monitor:

```bash
pio device monitor -b 115200
```

Wait without sending a `PLAN`.

Expected:

```text
[FSM] WARNING: Host timeout detected.
[FSM] Switching to fallback plan.
STATUS,0,A_GREEN,...,HOST_TIMEOUT
```

Then send:

```text
PLAN,17,25,15,3,1
```

Expected:

```text
ACK,17
[FSM] Host communication recovered.
[FSM] Pending SignalPlan received.
```

## Completion Checklist

```text
[ ] HOST_TIMEOUT_SECONDS added
[ ] HOST_TIMEOUT_MS added
[ ] controllerHealth added
[ ] hostTimedOut added
[ ] lastValidPlanTick added
[ ] timeout detection works
[ ] fallback plan applied on timeout
[ ] health becomes HOST_TIMEOUT
[ ] valid PLAN restores health to OK
[ ] STATUS reflects health state
[ ] pio run succeeds
```