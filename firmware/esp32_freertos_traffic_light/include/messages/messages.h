#pragma once

#include <Arduino.h>

#include "config/app_config.h"

// RawMessage represents one raw command line.  
// Example: PLAN,17,25,15,3,1
struct RawMessage
{
    char data[RAW_MESSAGE_MAX_LENGTH];
};

// Temporary structure for parsed PLAN fields.
struct ParsedPlanFields
{
    int plan_id;
    int green_a;
    int green_b;
    int yellow;
    int all_red;
};

// SignalPlan represents a validated traffic signal timing plan.
struct SignalPlan
{
    int plan_id;
    int green_a;
    int green_b;
    int yellow;
    int all_red;
};

// Traffic state enum for the future FSM task.
enum TrafficState
{
    STATE_A_GREEN,
    STATE_A_YELLOW,
    STATE_ALL_RED_AFTER_A,
    STATE_B_GREEN,
    STATE_B_YELLOW,
    STATE_ALL_RED_AFTER_B
};

// Controller status structure
struct ControllerStatus
{
    int plan_id;
    TrafficState state;
    uint32_t remaining_seconds;
    const char *health;
};
