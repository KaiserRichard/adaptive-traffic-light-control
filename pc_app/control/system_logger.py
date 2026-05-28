# system_logger.py

"""
Purpose:
    - Persist runtime telemetry for the ATLC adaptive traffic-control system.
    - Store lightweight CSV and JSONL logs for:
        * debugging
        * scheduler inspection
        * benchmark evidence
        * report screenshots / plots
        * runtime replay analysis

Architecture Notes:
    - The current ATLC system is a 2-direction adaptive controller.
    - The runtime pipeline uses:
            direction_a
            direction_b
      instead of:
            east/west/north/south

    - This naming is intentionally aligned with:
            scheduler.py
            density.py
            ROI configuration
            runtime signal plans

    - The logger stores:
            counts
            occupancy proxies
            LOW/MEDIUM/HIGH levels
            processing FPS
            runtime mode
            active direction

    - JSONL is useful for:
            structured parsing
            later dashboard integration
            benchmark scripts

    - CSV is useful for:
            Excel
            plotting
            quick report inspection
"""

import csv
import json
import os
from datetime import datetime
from typing import Dict, Any


class SystemLogger:
    """
    Runtime telemetry logger for ATLC.
    """

import csv
import json
import os

from datetime import datetime
from typing import Dict, Any

from pc_app.config import RUNTIME_LOG_DIR


class SystemLogger:
    """
    Runtime telemetry logger for ATLC.
    """

    def __init__(
        self,
        log_dir: str | None = None,
    ):
        self.log_dir = (
            log_dir
            or RUNTIME_LOG_DIR
            or "./outputs/runtime_logs"
        )

        os.makedirs(
            self.log_dir,
            exist_ok=True,
        )

        self.csv_path = os.path.join(
            self.log_dir,
            "traffic_runtime_log.csv",
        )

        self.jsonl_path = os.path.join(
            self.log_dir,
            "traffic_runtime_log.jsonl",
        )

        self._initialize_csv()

    def _initialize_csv(self) -> None:  
        """
        Create CSV header only once.
        """

        if os.path.exists(self.csv_path):
            return

        with open(
            self.csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "timestamp",

                "frame_index",

                "mode",
                "active_direction",
                "phase",
                "green_time",

                "green_a",
                "green_b",

                "raw_green_a",
                "raw_green_b",

                "active_green_a",
                "active_green_b",

                "alert",

                "direction_a_count",
                "direction_b_count",

                "direction_a_occupancy",
                "direction_b_occupancy",

                "direction_a_level",
                "direction_b_level",

                "processing_fps",
            ])

    def write(self, state: Dict[str, Any]) -> None:
        """
        Append runtime state into:
            - CSV
            - JSONL
        """

        timestamp = datetime.now().isoformat(timespec="seconds")

        counts = state.get("counts", {})
        occupancies = state.get("occupancies", {})
        levels = state.get("levels", {})

        csv_row = [
            timestamp,

            state.get("frame_index", 0),

            state.get("mode", "AUTO"),
            state.get("active_direction", "NONE"),
            state.get("phase", "UNKNOWN"),
            state.get("green_time", 0),

            state.get("green_a", 0),
            state.get("green_b", 0),

            state.get("raw_green_a", 0),
            state.get("raw_green_b", 0),

            state.get("active_green_a", 0),
            state.get("active_green_b", 0),

            state.get("alert", "OK"),

            counts.get("direction_a", 0),
            counts.get("direction_b", 0),

            occupancies.get("direction_a", 0.0),
            occupancies.get("direction_b", 0.0),

            levels.get("direction_a", "LOW"),
            levels.get("direction_b", "LOW"),

            state.get("processing_fps", None),
        ]

        with open(
            self.csv_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)
            writer.writerow(csv_row)

        json_state = {
            "timestamp": timestamp,
            **state,
        }

        with open(
            self.jsonl_path,
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    json_state,
                    ensure_ascii=False
                ) + "\n"
            )