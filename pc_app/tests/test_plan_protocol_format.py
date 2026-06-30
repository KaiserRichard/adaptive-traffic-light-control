#!/usr/bin/env python3
"""Host-only regression test for canonical ATLC PLAN formatting."""

from __future__ import annotations

import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


class _DummySerial:
    def __init__(self, *args, **kwargs) -> None:
        self.is_open = True
        self.in_waiting = 0

    def readline(self) -> bytes:
        return b""

    def write(self, data: bytes) -> int:
        return len(data)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.is_open = False


fake_serial = types.ModuleType("serial")
fake_serial.Serial = _DummySerial
sys.modules.setdefault("serial", fake_serial)

from pc_app.control.uart_sender import UartPlanSender, should_send_plan  # noqa: E402


def main() -> None:
    sender = object.__new__(UartPlanSender)
    sender.plan_id = 0

    message = sender.build_plan_message(
        {
            "green_a": 25,
            "green_b": 15,
            "yellow": 3,
            "all_red": 1,
        }
    )

    assert message == "PLAN,1,25,15,3,1"
    assert "PLAN,seq=" not in message
    assert "ns_green=" not in message
    assert "ew_green=" not in message

    second_message = sender.build_plan_message(
        {
            "green_a": 30,
            "green_b": 10,
            "yellow": 3,
            "all_red": 1,
        }
    )

    assert second_message == "PLAN,2,30,10,3,1"

    assert should_send_plan(None, {"green_a": 25, "green_b": 15, "yellow": 3, "all_red": 1})
    assert not should_send_plan(
        {"green_a": 25, "green_b": 15, "yellow": 3, "all_red": 1},
        {"green_a": 25, "green_b": 15, "yellow": 3, "all_red": 1},
    )

    print("test_plan_protocol_format.py: PASS")


if __name__ == "__main__":
    main()
