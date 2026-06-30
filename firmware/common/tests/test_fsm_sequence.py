#!/usr/bin/env python3
"""Host-side tests for the ATLC common FSM model."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
COMMON_DIR = REPO_ROOT / "firmware" / "common"


def compile_and_run(source: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        harness = tmp_path / "test_fsm_sequence.c"
        binary = tmp_path / "test_fsm_sequence"
        harness.write_text(source, encoding="utf-8")

        command = [
            "cc",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(COMMON_DIR / "include"),
            str(harness),
            str(COMMON_DIR / "src" / "atlc_fsm.c"),
            str(COMMON_DIR / "src" / "atlc_plan.c"),
            "-o",
            str(binary),
        ]

        subprocess.run(command, check=True)
        subprocess.run([str(binary)], check=True)


def main() -> None:
    compile_and_run(
        r'''
#include <stdio.h>
#include <string.h>

#include "atlc_fsm.h"

#define REQUIRE(condition, message) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "FAIL: %s\n", message); \
            return 1; \
        } \
    } while (0)

int main(void)
{
    atlc_fsm_t fsm;
    atlc_signal_plan_t initial = atlc_plan_default();
    atlc_signal_plan_t pending = atlc_plan_default();
    int applied = 0;

    pending.plan_id = 42;
    pending.green_a = 30;
    pending.green_b = 10;

    atlc_fsm_init(&fsm, &initial);

    REQUIRE(fsm.state == ATLC_STATE_A_GREEN, "FSM starts at A_GREEN");
    REQUIRE(fsm.active_plan.plan_id == 0, "default active plan starts with id 0");
    REQUIRE(
        strcmp(atlc_fsm_state_to_string(fsm.state), "A_GREEN") == 0,
        "state string A_GREEN"
    );
    REQUIRE(
        atlc_fsm_state_duration_seconds(ATLC_STATE_A_GREEN, &fsm.active_plan) == 20,
        "A_GREEN duration from active plan"
    );

    atlc_fsm_set_pending_plan(&fsm, &pending);
    REQUIRE(fsm.has_pending_plan, "pending plan stored");

    applied = atlc_fsm_advance_state(&fsm);
    REQUIRE(!applied, "pending not applied at A_YELLOW");
    REQUIRE(fsm.state == ATLC_STATE_A_YELLOW, "advanced to A_YELLOW");
    REQUIRE(fsm.active_plan.plan_id == 0, "active plan unchanged mid-cycle");

    REQUIRE(atlc_fsm_advance_state(&fsm) == false, "not applied at ALL_RED_AFTER_A");
    REQUIRE(fsm.state == ATLC_STATE_ALL_RED_AFTER_A, "advanced to all-red after A");
    REQUIRE(atlc_fsm_advance_state(&fsm) == false, "not applied at B_GREEN");
    REQUIRE(fsm.state == ATLC_STATE_B_GREEN, "advanced to B_GREEN");
    REQUIRE(atlc_fsm_advance_state(&fsm) == false, "not applied at B_YELLOW");
    REQUIRE(fsm.state == ATLC_STATE_B_YELLOW, "advanced to B_YELLOW");
    REQUIRE(atlc_fsm_advance_state(&fsm) == false, "not applied at ALL_RED_AFTER_B");
    REQUIRE(fsm.state == ATLC_STATE_ALL_RED_AFTER_B, "advanced to all-red after B");

    applied = atlc_fsm_advance_state(&fsm);
    REQUIRE(applied, "pending applied only when returning to A_GREEN");
    REQUIRE(fsm.state == ATLC_STATE_A_GREEN, "returned to A_GREEN");
    REQUIRE(fsm.active_plan.plan_id == 42, "pending became active");
    REQUIRE(!fsm.has_pending_plan, "pending flag cleared");
    REQUIRE(
        atlc_fsm_state_duration_seconds(ATLC_STATE_A_GREEN, &fsm.active_plan) == 30,
        "new active plan duration visible"
    );

    REQUIRE(
        atlc_fsm_next_state(ATLC_STATE_ALL_RED_AFTER_B) == ATLC_STATE_A_GREEN,
        "state sequence wraps to A_GREEN"
    );

    return 0;
}
'''
    )
    print("test_fsm_sequence.py: PASS")


if __name__ == "__main__":
    main()
