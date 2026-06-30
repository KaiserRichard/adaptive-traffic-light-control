#!/usr/bin/env python3
"""Host-side tests for the ATLC common plan validator."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
COMMON_DIR = REPO_ROOT / "firmware" / "common"


def compile_and_run(source: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        harness = tmp_path / "test_plan_validator.c"
        binary = tmp_path / "test_plan_validator"
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

#include "atlc_plan.h"

#define REQUIRE(condition, message) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "FAIL: %s\n", message); \
            return 1; \
        } \
    } while (0)

static atlc_signal_plan_t valid_plan(void)
{
    atlc_signal_plan_t plan;
    plan.plan_id = 1;
    plan.green_a = 25;
    plan.green_b = 15;
    plan.yellow = 3;
    plan.all_red = 1;
    return plan;
}

int main(void)
{
    atlc_signal_plan_t plan = valid_plan();

    REQUIRE(atlc_plan_validate(&plan) == ATLC_PLAN_OK, "valid plan accepted");
    REQUIRE(atlc_plan_validate(0) == ATLC_PLAN_NULL, "null plan rejected");

    plan = valid_plan();
    plan.plan_id = -1;
    REQUIRE(
        atlc_plan_validate(&plan) == ATLC_PLAN_ID_OUT_OF_RANGE,
        "negative plan_id rejected"
    );

    plan = valid_plan();
    plan.green_a = -1;
    REQUIRE(
        atlc_plan_validate(&plan) == ATLC_PLAN_GREEN_A_OUT_OF_RANGE,
        "negative green_a rejected"
    );

    plan = valid_plan();
    plan.green_a = 0;
    REQUIRE(
        atlc_plan_validate(&plan) == ATLC_PLAN_GREEN_A_OUT_OF_RANGE,
        "zero green_a rejected"
    );

    plan = valid_plan();
    plan.green_b = 46;
    REQUIRE(
        atlc_plan_validate(&plan) == ATLC_PLAN_GREEN_B_OUT_OF_RANGE,
        "too-large green_b rejected"
    );

    plan = valid_plan();
    plan.yellow = 0;
    REQUIRE(
        atlc_plan_validate(&plan) == ATLC_PLAN_YELLOW_OUT_OF_RANGE,
        "zero yellow rejected"
    );

    plan = valid_plan();
    plan.yellow = 4;
    REQUIRE(
        atlc_plan_validate(&plan) == ATLC_PLAN_YELLOW_OUT_OF_RANGE,
        "non-fixed yellow rejected"
    );

    plan = valid_plan();
    plan.all_red = 0;
    REQUIRE(
        atlc_plan_validate(&plan) == ATLC_PLAN_ALL_RED_OUT_OF_RANGE,
        "zero all_red rejected"
    );

    plan = valid_plan();
    plan.all_red = 2;
    REQUIRE(
        atlc_plan_validate(&plan) == ATLC_PLAN_ALL_RED_OUT_OF_RANGE,
        "non-fixed all_red rejected"
    );

    REQUIRE(
        strcmp(
            atlc_plan_result_to_string(ATLC_PLAN_GREEN_A_OUT_OF_RANGE),
            "GREEN_A_OUT_OF_RANGE"
        ) == 0,
        "reason string stable"
    );

    plan = atlc_plan_default();
    REQUIRE(plan.green_a == 20, "default green_a matches ESP32 reference");
    REQUIRE(plan.green_b == 20, "default green_b matches ESP32 reference");
    REQUIRE(atlc_plan_validate(&plan) == ATLC_PLAN_OK, "default plan valid");

    return 0;
}
'''
    )
    print("test_plan_validator.py: PASS")


if __name__ == "__main__":
    main()
