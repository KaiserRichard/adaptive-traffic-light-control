"""
signal_runtime.py

Purpose:
- Maintain a stable traffic light runtime state on the Python host side.
- Separate raw scheduler output from the plan currently being executed.
- Provide countdown state for video overlay and MCU synchronization debugging.

Why this exists:
- The scheduler may compute a slightly different signal_plan every frame.
- Traffic lights should not instantly change timing every frame.
- The system needs a stable active_plan and a pending_plan.

Concept:

    raw signal_plan
        = plan computed from current frame/density

    pending_plan
        = latest recommended plan waiting to be applied

    active_plan
        = plan currently being executed by the runtime controller and MCU

    runtime_state
        = current traffic light state and countdown

This makes video overlay and MCU behavior easier to compare.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


TRAFFIC_STATES = [
    "A_GREEN",
    "A_YELLOW",
    "ALL_RED_AFTER_A",
    "B_GREEN",
    "B_YELLOW",
    "ALL_RED_AFTER_B",
]


@dataclass
class RuntimeSnapshot:
    """
    Snapshot of the current traffic light runtime state.
    """

    state: str
    remaining_seconds: int
    active_plan: dict
    pending_plan: Optional[dict]
    cycle_count: int


class SignalRuntimeController:
    """
    Host-side traffic light runtime controller.

    This controller simulates the traffic light FSM on the Python side.

    Why:
    - The video overlay needs to show a stable current state and countdown.
    - The MCU should receive stable plans, not constantly changing raw plans.
    - Debugging is easier when the host and MCU can be compared visually.

    Important:
    This controller does not replace the MCU FSM.
    It provides a host-side reference for debugging and synchronization.
    """

    def __init__(self, initial_plan: dict) -> None:
        self.active_plan = self._copy_plan(initial_plan)
        self.pending_plan: Optional[dict] = None

        self.state_index = 0
        self.state = TRAFFIC_STATES[self.state_index]

        self.remaining_seconds = self._duration_for_state(self.state)
        self.accumulator_seconds = 0.0
        self.cycle_count = 0

    def _copy_plan(self, plan: dict) -> dict:
        """
        Keep only timing fields needed by runtime FSM.
        """

        return {
            "green_a": int(plan["green_a"]),
            "green_b": int(plan["green_b"]),
            "yellow": int(plan["yellow"]),
            "all_red": int(plan["all_red"]),
        }

    def update_pending_plan(self, plan: dict) -> None:
        """
        Store latest scheduler recommendation.

        This does not immediately change the active runtime plan.
        The pending plan is applied only at a safe boundary.
        """

        self.pending_plan = self._copy_plan(plan)

    def _duration_for_state(self, state: str) -> int:
        """
        Return duration in seconds for a traffic state.
        """

        if state == "A_GREEN":
            return int(self.active_plan["green_a"])

        if state == "A_YELLOW":
            return int(self.active_plan["yellow"])

        if state == "ALL_RED_AFTER_A":
            return int(self.active_plan["all_red"])

        if state == "B_GREEN":
            return int(self.active_plan["green_b"])

        if state == "B_YELLOW":
            return int(self.active_plan["yellow"])

        if state == "ALL_RED_AFTER_B":
            return int(self.active_plan["all_red"])

        return 1

    def _advance_state(self) -> bool:
        """
        Advance to next FSM state.

        Returns:
            True if a new cycle started at A_GREEN.
        """

        self.state_index = (self.state_index + 1) % len(TRAFFIC_STATES)
        self.state = TRAFFIC_STATES[self.state_index]

        new_cycle_started = self.state == "A_GREEN"

        if new_cycle_started:
            self.cycle_count += 1

            # Apply pending plan only at the start of a new full cycle.
            # This prevents mid-cycle timing jumps.
            if self.pending_plan is not None:
                self.active_plan = self.pending_plan
                self.pending_plan = None

        self.remaining_seconds = self._duration_for_state(self.state)

        return new_cycle_started

    def update(self, dt_seconds: float) -> bool:
        """
        Advance countdown using elapsed real time.

        Args:
            dt_seconds:
                Real elapsed time since last update.

        Returns:
            True if a new cycle started and pending plan may have been applied.
        """

        if dt_seconds <= 0:
            return False

        self.accumulator_seconds += dt_seconds

        new_cycle_started = False

        while self.accumulator_seconds >= 1.0:
            self.accumulator_seconds -= 1.0

            if self.remaining_seconds > 1:
                self.remaining_seconds -= 1
            else:
                started = self._advance_state()
                new_cycle_started = new_cycle_started or started

        return new_cycle_started

    def get_snapshot(self) -> RuntimeSnapshot:
        """
        Return current runtime state for overlay/debugging.
        """

        return RuntimeSnapshot(
            state=self.state,
            remaining_seconds=int(self.remaining_seconds),
            active_plan=self._copy_plan(self.active_plan),
            pending_plan=self._copy_plan(self.pending_plan)
            if self.pending_plan is not None
            else None,
            cycle_count=self.cycle_count,
        )

    def should_send_active_plan_to_mcu(self) -> bool:
        """
        Current simple strategy:
        Send active plan whenever a new full cycle starts.

        This keeps the MCU synchronized with the host runtime controller.
        """

        return True