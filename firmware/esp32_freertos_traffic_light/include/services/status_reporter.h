#pragma once

#include <Arduino.h>

#include "messages/messages.h"

/*
 * Phase 15.12 - Task Notification STATUS Reporter
 * 
 * The STATUS software timer no longer prints directly.
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

bool initStatusReporter();

void updateControllerStatus(
    const SignalPlan *activePlan,
    TrafficState currentState,
    TickType_t stateStartTick,
    const char *health
);

void startStatusTimer();

/*
 * Dedicated task that waits for STATUS notifications.
 *
 * This task blocks on ulTaskNotifyTake().
 * It wakes when StatusTimerCallback calls xTaskNotifyGive().
 */
void TaskStatusReporter(void *pvParameters);
