'''
test_protocol.py

Purpose:
    - Verify message encoding and ACK parsing for Phase 5 communication.
'''

from pc_app.comm.protocol import encode_signal_plan, parse_ack

def main():
    signal_plan = {
        "green_a": 25,
        "green_b": 15,
        "yellow": 3,
        "all_red": 1,
    }

    encoded = encode_signal_plan(plan_id=17, signal_plan=signal_plan)
    print("Encoded plan:", encoded)

    ack = parse_ack("ACK,17")
    print("Parsed ACK:",ack)

if __name__ == "__main__":
    main()