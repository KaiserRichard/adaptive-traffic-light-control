'''
uart_sender.py

Purpose:
    - Send signal timing plans from host to MCU / ESP32
    - Optionally wait for an ACK from the MCU
'''

from typing import Dict, Optional
import serial

from pc_app.comm.protocol import encode_signal_plan, parse_ack

# UART sender for traffic signal plans.
class UartPlanSender:
    def __init__(self, port: str, baud_rate: int, timeout: float ):
        self.serial_conn = serial.Serial(
            port=port,
            baudrate=baud_rate,
            timeout=timeout,
        )

    '''
    Send one signal plan to MCU.

    Returns:
    - ACK dictionary if wait_for_ack=True and ACK received
    - None otherwise
    '''
    def send_plan(
            self,
            plan_id: int,
            signal_plan: Dict[str, float|int],
            wait_for_act: bool = True,
    ) -> Optional[Dict[str, int]]:
        message = encode_signal_plan(plan_id, signal_plan)
        packet = (message + "\n").encode("utf-8") # \n : add delimiter
        self.serial_conn.write(packet)
        self.serial_conn.flush()

        if not wait_for_act:
            return None
        # readline(): read until newline (read everything until \n )
        # strip(): remove \n (newline)
        line = self.serial_conn.readline().decode("utf-8").strip() 
        if not line:
            raise TimeoutError("No ACK received from MCU.")

        return parse_ack(line)

    # Close serial connection 
    def close(self) -> None:
        if self.serial_conn.is_open:
            self.serial_conn.close()
    