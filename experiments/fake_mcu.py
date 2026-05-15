"""
fake_mcu.py

Purpose:
- Simulate ESP32 UART behavior without real hardware.
- Useful for testing the PLAN/ACK protocol in software.

How to run:
    python -m experiments.fake_mcu

Then type:
    PLAN,1,10,5,3,1

Expected output:
    ACK,1
"""

from __future__ import annotations


def parse_plan(line: str) -> tuple[bool, str]:
    """
    Parse a PLAN message.

    Expected format:
        PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>
    """

    parts = [part.strip() for part in line.split(",")]

    if len(parts) != 6:
        return False, "ERR"

    if parts[0] != "PLAN":
        return False, "ERR"

    try:
        plan_id = int(parts[1])
        green_a = int(parts[2])
        green_b = int(parts[3])
        yellow = int(parts[4])
        all_red = int(parts[5])
    except ValueError:
        return False, "ERR"

    if plan_id <= 0:
        return False, "ERR"

    if green_a <= 0 or green_b <= 0 or yellow <= 0 or all_red <= 0:
        return False, "ERR"

    return True, f"ACK,{plan_id}"


def main() -> None:
    print("Fake MCU started.")
    print("Type PLAN messages, for example:")
    print("PLAN,1,10,5,3,1")

    while True:
        try:
            line = input("> ").strip()
        except KeyboardInterrupt:
            print("\nFake MCU stopped.")
            break

        if not line:
            continue

        ok, response = parse_plan(line)

        if ok:
            print(response)
        else:
            print("ERR")


if __name__ == "__main__":
    main()