"""
test_uart_sender.py

Purpose:
- Send a test traffic signal PLAN message from Python to ESP32.
- Read ACK response from ESP32.
- Verify PC/Raspberry Pi to ESP32 UART communication.

Example macOS:
    python -m experiments.test_uart_sender --port /dev/cu.usbserial-XXXX

Example Windows:
    python -m experiments.test_uart_sender --port COM5

Example with custom timing:
    python -m experiments.test_uart_sender --port /dev/cu.usbserial-XXXX --green-a 10 --green-b 5
"""

import argparse
import time

import serial


def build_plan_message(
    plan_id: int,
    green_a: int,
    green_b: int,
    yellow: int,
    all_red: int,
) -> str:
    """
    Build UART PLAN message.

    Format:
        PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>
    """

    return f"PLAN,{plan_id},{green_a},{green_b},{yellow},{all_red}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send traffic light PLAN message to ESP32 over UART."
    )

    parser.add_argument(
        "--port",
        required=True,
        help="Serial port, for example /dev/cu.usbserial-XXXX on macOS or COM5 on Windows.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Serial baud rate. Default: 115200.",
    )
    parser.add_argument(
        "--plan-id",
        type=int,
        default=1,
        help="Plan ID. Default: 1.",
    )
    parser.add_argument(
        "--green-a",
        type=int,
        default=10,
        help="Green time for direction A in seconds. Default: 10.",
    )
    parser.add_argument(
        "--green-b",
        type=int,
        default=5,
        help="Green time for direction B in seconds. Default: 5.",
    )
    parser.add_argument(
        "--yellow",
        type=int,
        default=3,
        help="Yellow time in seconds. Default: 3.",
    )
    parser.add_argument(
        "--all-red",
        type=int,
        default=1,
        help="All-red time in seconds. Default: 1.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Serial read timeout in seconds. Default: 3.0.",
    )

    args = parser.parse_args()

    message = build_plan_message(
        plan_id=args.plan_id,
        green_a=args.green_a,
        green_b=args.green_b,
        yellow=args.yellow,
        all_red=args.all_red,
    )

    print(f"Opening serial port: {args.port}")
    print(f"Baud rate: {args.baud}")

    with serial.Serial(args.port, args.baud, timeout=args.timeout) as ser:
        # ESP32 often resets when serial port opens.
        # Wait briefly so firmware can boot and print READY.
        time.sleep(2.0)

        # Clear old boot messages if any.
        while ser.in_waiting:
            old_line = ser.readline().decode(errors="replace").strip()
            if old_line:
                print(f"Boot/old message: {old_line}")

        payload = message + "\n"

        ser.write(payload.encode("utf-8"))
        ser.flush()

        print(f"Sent: {message}")

        start_time = time.time()
        ack_received = False

        while time.time() - start_time < args.timeout:
            line = ser.readline().decode(errors="replace").strip()

            if not line:
                continue

            print(f"Received: {line}")

            expected_ack = f"ACK,{args.plan_id}"

            if expected_ack in line:
                ack_received = True
                break

            if "ERR" in line:
                break

        if not ack_received:
            raise RuntimeError("ACK was not received from ESP32.")

        print("UART test passed.")


if __name__ == "__main__":
    main()