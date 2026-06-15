// status_reporter.cpp
#pragma once
#include <Arduino.h>
#include "messages.h"

void initStatusReporter();

void updateControllerStatus(
    const SignalPlan *activePlan,
    TrafficState currentState,
    TickType_t stateStartTick);

void startStatusTimer();
