# failsafe.py

from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class FailsafeConfig:
    min_processing_fps: float = 3.0
    max_frame_age_sec: float = 5.0
    max_no_frame_sec: float = 5.0
    consecutive_error_limit: int = 3


class FailsafeMonitor:
    def __init__(self, config: Optional[FailsafeConfig] = None):
        self.config = config or FailsafeConfig()
        self.error_count = 0
        self.last_reason = "OK"

    def check(
        self,
        last_frame_time: Optional[float],
        processing_fps: Optional[float],
        now: Optional[float] = None,
    ) -> tuple[bool, str]:
        if now is None:
            now = time.time()

        reason = "OK"

        if last_frame_time is None:
            reason = "NO_FRAME_AVAILABLE"
        else:
            frame_age = now - last_frame_time

            if frame_age > self.config.max_no_frame_sec:
                reason = "CAMERA_OR_FRAME_TIMEOUT"
            elif frame_age > self.config.max_frame_age_sec:
                reason = "FRAME_TOO_OLD"
            elif processing_fps is not None and processing_fps < self.config.min_processing_fps:
                reason = "PROCESSING_TOO_SLOW"

        if reason == "OK":
            self.error_count = 0
            self.last_reason = "OK"
            return False, "OK"

        self.error_count += 1
        self.last_reason = reason

        if self.error_count >= self.config.consecutive_error_limit:
            return True, reason

        return False, reason