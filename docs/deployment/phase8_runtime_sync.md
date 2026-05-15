# Phase 8 Runtime Synchronization Notes

## Purpose

This document records the runtime synchronization fixes added during the full pipeline UART integration phase.

The goal is to make the software-side traffic light overlay and the physical MCU traffic light circuit follow the same active signal plan and countdown.

## Problem Found

During testing, the scheduler output changed frequently from frame to frame.

This made debugging difficult because it was unclear which signal plan was actually being executed.

Observed issues:

```text
- signal_plan changed frequently
- physical MCU countdown could continue from an old state
- software overlay and hardware LEDs were difficult to compare
- 7-segment countdown needed an independent smoke test
Root Cause

The original pipeline treated each newly computed scheduler output as the current signal plan.

However, a real traffic light should not change timing every frame.

The system needs a stable runtime state.

Runtime Design

The system now separates:

raw_signal_plan
= latest scheduler recommendation from the current frame

pending_plan
= latest recommended plan waiting to be applied

active_plan
= plan currently being executed

runtime_state
= current traffic light state and remaining countdown
New Runtime Controller

A new host-side runtime controller was added:

pc_app/control/signal_runtime.py

Responsibilities:

- maintain current traffic state
- maintain countdown
- keep active_plan stable
- store pending_plan
- apply pending_plan only at safe cycle boundary
Video Overlay

A virtual traffic light panel was added to the video output.

It shows:

- current state
- remaining countdown
- active plan
- pending plan
- UART status
- ACK latency

This helps compare the software runtime state with the physical MCU LED circuit.

Startup Synchronization

A startup synchronization fix was added.

When pc_app.main starts and UART is enabled, Python immediately sends the current active plan to the MCU.

This resets the MCU FSM and countdown so that the physical countdown starts aligned with the software overlay.

Expected startup flow:

main.py starts
→ UART opens
→ startup PLAN is sent
→ MCU resets to A_GREEN
→ software overlay countdown and MCU countdown start together
7-Segment Smoke Test

A separate 7-segment smoke test firmware was added:

firmware/arduino_uno_7seg_smoke_test/

Purpose:

Verify the 7-segment wiring before integrating it with UART and the traffic light FSM.

Expected behavior:

The display counts from 00 to 99 repeatedly.
Conclusion

The main Phase 8 improvement is the separation between scheduler recommendation and runtime execution.

The system now uses a stable active plan for the overlay and MCU, making the full software-to-hardware loop easier to debug and validate.
