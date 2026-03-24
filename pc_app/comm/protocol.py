'''
protocol.py

Purpose: 
    - Define a simple text protocol for host-to-MCU communication.
    - Keep message formatting and parsing seperate from transport code.
'''

from typing import Dict

''' Convert a signal plan dictionary into a protocol string.

    Example :
    PLAN,17,25,15,3,1
'''
def encode_signal_plan(plan_id: int, signal_plan: Dict[str, int]) -> str:
    green_a = int(signal_plan["green_a"])
    green_b = int(signal_plan["green_b"])
    yellow = int(signal_plan["yellow"])
    all_red = int(signal_plan["all_red"])

    # Host sends: PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>
    return f"PLAN,{plan_id},{green_a},{green_b},{yellow},{all_red}"

''' Parse an ACK message from MCU

    Example input:
        ACK, 17 -> len = 2

    Example output:
        {"plan_id: 17}    

'''
def parse_ack(message: str) -> Dict[str, int]:
    parts = message.strip().split(",")

    if len(parts) != 2 or parts[0] != "ACK":
        raise ValueError(f"Invalid ACK message: {message}")
    
    return{
        "plan_id": int(parts[1]),
    }