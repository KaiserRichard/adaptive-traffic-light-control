#!/usr/bin/env python3
"""Host-side tests for the ATLC common protocol parser."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
COMMON_DIR = REPO_ROOT / "firmware" / "common"


def compile_and_run(source: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        harness = tmp_path / "test_protocol_parser.c"
        binary = tmp_path / "test_protocol_parser"
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
            str(COMMON_DIR / "src" / "atlc_protocol.c"),
            str(COMMON_DIR / "src" / "atlc_plan.c"),
            str(COMMON_DIR / "src" / "atlc_fsm.c"),
            str(COMMON_DIR / "src" / "atlc_status.c"),
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

#include "atlc_plan.h"
#include "atlc_protocol.h"
#include "atlc_status.h"

#define REQUIRE(condition, message) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "FAIL: %s\n", message); \
            return 1; \
        } \
    } while (0)

int main(void)
{
    atlc_signal_plan_t plan;
    char response[ATLC_PROTOCOL_MAX_RESPONSE_LENGTH];
    atlc_controller_status_t status;

    REQUIRE(
        atlc_protocol_parse_plan("PLAN,17,25,15,3,1", &plan) == ATLC_PARSE_OK,
        "valid PLAN should parse"
    );
    REQUIRE(plan.plan_id == 17, "plan_id parsed");
    REQUIRE(plan.green_a == 25, "green_a parsed");
    REQUIRE(plan.green_b == 15, "green_b parsed");
    REQUIRE(plan.yellow == 3, "yellow parsed");
    REQUIRE(plan.all_red == 1, "all_red parsed");
    REQUIRE(atlc_plan_validate(&plan) == ATLC_PLAN_OK, "valid PLAN should validate");

    REQUIRE(
        atlc_protocol_parse_plan("PLAN,17,25,15,3", &plan) == ATLC_PARSE_MISSING_FIELD,
        "missing field rejected"
    );
    REQUIRE(
        atlc_protocol_parse_plan("PLAN,17,25,15,3,1,99", &plan) == ATLC_PARSE_EXTRA_FIELD,
        "extra field rejected"
    );
    REQUIRE(
        atlc_protocol_parse_plan("PLAN,17,abc,15,3,1", &plan) == ATLC_PARSE_NON_NUMERIC_FIELD,
        "non-numeric field rejected"
    );
    REQUIRE(
        atlc_protocol_parse_plan("PLAN,seq=17,ns_green=25,ew_green=15", &plan) ==
            ATLC_PARSE_OLD_FORMAT_REJECTED,
        "old seq/ns/ew format rejected"
    );
    REQUIRE(
        atlc_protocol_parse_plan("HELLO,17,25,15,3,1", &plan) == ATLC_PARSE_NOT_PLAN,
        "unknown command rejected"
    );

    REQUIRE(atlc_protocol_format_ack(17, response, sizeof(response)) > 0, "ACK formats");
    REQUIRE(strcmp(response, "ACK,17") == 0, "ACK format exact");

    REQUIRE(
        atlc_protocol_format_nack(19, "GREEN_A_OUT_OF_RANGE", response, sizeof(response)) > 0,
        "NACK formats"
    );
    REQUIRE(
        strcmp(response, "NACK,19,GREEN_A_OUT_OF_RANGE") == 0,
        "NACK format exact"
    );

    status.plan_id = 17;
    status.state = ATLC_STATE_A_GREEN;
    status.remaining_seconds = 12;
    status.health = ATLC_HEALTH_OK;

    REQUIRE(atlc_status_format(&status, response, sizeof(response)) > 0, "STATUS formats");
    REQUIRE(
        strcmp(response, "STATUS,17,A_GREEN,12,OK") == 0,
        "STATUS format exact"
    );

    return 0;
}
'''
    )
    print("test_protocol_parser.py: PASS")


if __name__ == "__main__":
    main()
