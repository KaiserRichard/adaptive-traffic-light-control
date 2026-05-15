"""
uart_sender.py

Purpose:
- Provide a reusable UART sender for MCU traffic light controllers.
- Convert scheduler signal_plan dictionaries into PLAN messages.
- Send PLAN messages through pyserial.
- Wait for ACK from Arduino Uno / ESP32 / STM32 firmware.
- Keep serial communication out of pc_app.main.

UART protocol:
    PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>

Expected MCU response:
    ACK,<plan_id>

The MCU may also print debug logs such as:
    [MCU/ACK] ACK,1

Therefore, ACK detection checks whether the expected ACK string appears
inside the received line.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import serial


@dataclass
class UartSendResult:
    """
    Result object returned after sending a PLAN message.
    """

    success: bool
    plan_id: int
    message: str
    ack_line: Optional[str]
    error: Optional[str]
    latency_ms: Optional[float]


class UartPlanSender:
    """
    UART sender for traffic light timing plans.

    This class owns the serial connection and exposes one main method:

        send_plan(signal_plan)

    The rest of the pipeline should not directly use serial.Serial.
    """

    def __init__(
        self,
        port: str,
        baud: int = 115200,
        ack_timeout: float = 2.0,
        boot_wait: float = 2.0,
    ) -> None:
        """
        Initialize UART sender.

        Parameters:
            port:
                Serial port, for example:
                /dev/cu.usbmodem141011
                /dev/cu.usbserial-XXXX
                /dev/ttyUSB0

            baud:
                Baud rate. Must match MCU firmware.

            ack_timeout:
                Maximum time to wait for ACK after sending PLAN.

            boot_wait:
                Time to wait after opening serial port.
                Some boards reset when serial opens.
        """

        if not port:
            raise ValueError("UART port is empty. Set UART_PORT in .env.")

        self.port = port
        self.baud = baud
        self.ack_timeout = ack_timeout
        self.plan_id = 0

        self.serial = serial.Serial(
            port=self.port,
            baudrate=self.baud,
            timeout=0.1,
        )

        # Many Arduino-compatible boards reset when serial is opened.
        # Wait briefly so boot messages do not interfere with ACK parsing.
        time.sleep(boot_wait)

        self._drain_old_lines()

    def _drain_old_lines(self) -> None:
        """
        Drain boot messages or old serial lines from the MCU.

        This prevents previous logs from being confused with the ACK for
        the next message.
        """

        while self.serial.in_waiting:
            try:
                self.serial.readline()
            except Exception:
                break

    def close(self) -> None:
        """
        Close serial port safely.
        """

        if self.serial and self.serial.is_open:
            self.serial.close()

    def build_plan_message(self, signal_plan: dict) -> str:
        """
        Convert a signal_plan dictionary into a UART PLAN message.

        Input example:
            {
                "green_a": 34,
                "green_b": 10,
                "yellow": 3,
                "all_red": 1,
            }

        Output example:
            PLAN,1,34,10,3,1
        """

        self.plan_id += 1

        green_a = int(signal_plan["green_a"])
        green_b = int(signal_plan["green_b"])
        yellow = int(signal_plan["yellow"])
        all_red = int(signal_plan["all_red"])

        return f"PLAN,{self.plan_id},{green_a},{green_b},{yellow},{all_red}"

    def send_plan(self, signal_plan: dict) -> UartSendResult:
        """
        Send signal_plan to MCU and wait for ACK.

        Returns:
            UartSendResult
        """

        message = self.build_plan_message(signal_plan)
        expected_ack = f"ACK,{self.plan_id}"

        start_time = time.perf_counter()

        try:
            self.serial.write((message + "\n").encode("utf-8"))
            self.serial.flush()
        except Exception as exc:
            return UartSendResult(
                success=False,
                plan_id=self.plan_id,
                message=message,
                ack_line=None,
                error=f"serial_write_failed: {exc}",
                latency_ms=None,
            )

        deadline = time.perf_counter() + self.ack_timeout
        last_line = None

        while time.perf_counter() < deadline:
            try:
                raw = self.serial.readline()
            except Exception as exc:
                return UartSendResult(
                    success=False,
                    plan_id=self.plan_id,
                    message=message,
                    ack_line=None,
                    error=f"serial_read_failed: {exc}",
                    latency_ms=None,
                )

            if not raw:
                continue

            line = raw.decode("utf-8", errors="replace").strip()

            if not line:
                continue

            last_line = line

            # Accept both:
            # ACK,1
            # [MCU/ACK] ACK,1
            if expected_ack in line:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                return UartSendResult(
                    success=True,
                    plan_id=self.plan_id,
                    message=message,
                    ack_line=line,
                    error=None,
                    latency_ms=latency_ms,
                )

            if "ERR" in line:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                return UartSendResult(
                    success=False,
                    plan_id=self.plan_id,
                    message=message,
                    ack_line=line,
                    error="mcu_returned_err",
                    latency_ms=latency_ms,
                )

        return UartSendResult(
            success=False,
            plan_id=self.plan_id,
            message=message,
            ack_line=last_line,
            error="ack_timeout",
            latency_ms=None,
        )


def should_send_plan(
    previous_plan: Optional[dict],
    current_plan: dict,
    min_green_change: int = 1,
) -> bool:
    """
    Decide whether the current plan should be sent to MCU.

    This prevents sending duplicate or nearly identical plans too often.
    """

    if previous_plan is None:
        return True

    keys = ["green_a", "green_b", "yellow", "all_red"]

    for key in keys:
        old_value = int(previous_plan[key])
        new_value = int(current_plan[key])

        if key in {"green_a", "green_b"}:
            if abs(new_value - old_value) >= min_green_change:
                return True
        else:
            if new_value != old_value:
                return True

    return False