// status_reporter.cpp
#pragma once
#include <Arduino.h>
#include "messages.h"

/*
 * Phase 15.12 - Task Notification STATUS Reporter
 * 
 * The STATUS software timer no longer print directly.
 * 
 * Instead:
 *
 *     StatusTimerCallback
 *             ↓
 *     task notification
 *             ↓
 *     TaskStatusReporter
 *             ↓
 *     STATUS line output
 *
 * This keeps the FreeRTOS daemon task lightweight.
 */

// Phase 15.12:  this function returns bool instead of void
// Because this phase creates both 
// StatusTimer and TaskStatusReporter.
// And initialization can fail.
bool initStatusReporter();

void updateControllerStatus(
    const SignalPlan *activePlan,
    TrafficState currentState,
    TickType_t stateStartTick,
    const char *health 
);

void startStatusTimer();


// Dedicated task that waits for STATUS notifications.
void TaskStatusReporter(void *pvParameters);
