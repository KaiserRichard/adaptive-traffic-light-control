"""
File:
    uart_framing.py

Phase:
    Phase 16.10 - AI-to-MCU UART Integration Preparation

Purpose:
    - Convert PLAN message strings into newline-terminated UART frames.
    - Validate and unframe UART bytes without opening a serial port.

Responsibilities:
    - Enforce ASCII-only messages.
    - Reject embedded newline characters.
    - Enforce bounded frame length.
    - Preserve a simple dry-run boundary for future UART sending.

This file should NOT:
    - Import pyserial.
    - Open serial ports.
    - Send real UART messages.
    - Modify MCU firmware.
"""

from __future__ import annotations

VALID_FRAME_TYPES = {"PLAN", "ACK", "NACK", "STATUS", "DIAG"}


def frame_uart_message(message: str, max_length: int = 128) -> bytes:
    """
    Convert a PLAN message string into newline-terminated UART bytes.

    Why:

        Newline-terminated ASCII is easy to inspect in logs and straightforward
        for MCU firmware to parse line-by-line. This helper prepares the frame
        without touching real serial hardware.
    """

    if not isinstance(message, str):
        raise ValueError("message must be a string.")
    if not message:
        raise ValueError("message must be non-empty.")
    if "\n" in message or "\r" in message:
        raise ValueError("message must not contain embedded newline characters.")

    try:
        frame = f"{message}\n".encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("message must contain ASCII characters only.") from exc

    validate_uart_frame(frame, max_length=max_length)
    return frame


def validate_uart_frame(frame: bytes, max_length: int = 128) -> None:
    """
    Validate one newline-terminated UART frame.

    Why:

        Phase 16.10 is a dry-run stage, but the frame should already satisfy
        the constraints a future UART sender and MCU line parser will depend on.
    """

    if isinstance(frame, bytearray):
        frame = bytes(frame)
    if not isinstance(frame, bytes):
        raise ValueError("frame must be bytes.")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 1:
        raise ValueError("max_length must be an integer greater than 1.")
    if not frame:
        raise ValueError("frame must be non-empty.")
    if len(frame) > max_length:
        raise ValueError(f"frame exceeds max_length={max_length}.")
    if not frame.endswith(b"\n"):
        raise ValueError("frame must end with one newline.")
    if frame.count(b"\n") != 1:
        raise ValueError("frame must contain exactly one newline.")
    if b"\r" in frame:
        raise ValueError("frame must not contain carriage returns.")

    try:
        decoded = frame.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("frame must be ASCII encoded.") from exc

    frame_type = decoded.rstrip("\n").split(",", 1)[0]
    if frame_type not in VALID_FRAME_TYPES:
        raise ValueError(f"invalid frame type: {frame_type}")


def unframe_uart_message(frame: bytes, max_length: int = 128) -> str:
    """
    Convert a validated UART frame back into a message string.

    Why:

        Round-trip tests prove that the dry-run sender is not hiding format
        drift before Phase 16.10 moves toward real serial transport.
    """

    validate_uart_frame(frame, max_length=max_length)
    return bytes(frame).decode("ascii").rstrip("\n")
