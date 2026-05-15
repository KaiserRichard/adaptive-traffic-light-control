# UART ESP32 Integration Plan

## Purpose

This document describes Phase 7 of the adaptive traffic light control project.

The goal is to connect the Python host-side controller to an ESP32 traffic light controller using UART serial communication.

## Architecture

Development architecture:

```text
PC Python script
→ USB Serial
→ ESP32
→ 6 LED traffic light circuit

Final deployment architecture:

Raspberry Pi
→ Local YOLO detector
→ ROI + density + scheduler
→ UART Serial
→ ESP32
→ LED traffic light controller
UART Protocol

Python sends:

PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>

Example:

PLAN,1,10,5,3,1

ESP32 replies:

ACK,1

Invalid messages return:

ERR
ESP32 Pin Mapping
Direction A:
A_RED     = GPIO 25
A_YELLOW  = GPIO 26
A_GREEN   = GPIO 27

Direction B:
B_RED     = GPIO 14
B_YELLOW  = GPIO 12
B_GREEN   = GPIO 13

Each LED should be connected with a resistor:

ESP32 GPIO
→ resistor 220Ω or 330Ω
→ LED anode
→ LED cathode
→ GND
State Machine

The ESP32 firmware runs these states:

STATE_A_GREEN
STATE_A_YELLOW
STATE_ALL_RED_AFTER_A
STATE_B_GREEN
STATE_B_YELLOW
STATE_ALL_RED_AFTER_B

Example plan:

PLAN,1,10,5,3,1

Expected LED sequence:

A green 10s
A yellow 3s
All red 1s
B green 5s
B yellow 3s
All red 1s
Repeat
Testing
Manual serial monitor test

Upload firmware using PlatformIO.

Open serial monitor at:

115200 baud

Send:

PLAN,1,10,5,3,1

Expected:

ACK,1
Python UART test
python -m experiments.test_uart_sender --port <YOUR_PORT>

macOS example:

python -m experiments.test_uart_sender --port /dev/cu.usbserial-XXXX

Windows example:

python -m experiments.test_uart_sender --port COM5
Report Note

The ESP32 firmware is independent from the YOLO detector. It only receives timing commands through a defined UART protocol. This modular design separates perception and decision-making from real-time signal execution.


---
## Update
Arduino Uno was used as a temporary MCU validation target because ESP32 serial port was not detected on macOS.
The same UART protocol will later be migrated to ESP32.