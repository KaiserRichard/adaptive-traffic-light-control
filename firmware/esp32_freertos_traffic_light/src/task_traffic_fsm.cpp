// task_traffic_fsm.cpp
/* 
 * TaskTrafficFSMPlaceholder:
 * Receives validated SignalPlan object from planQueue
 * 
 */
#include <Arduino.h>

#include "messages.h"
#include "queues.h"
#include "protocol.h"
#include "tasks.h"
#include "traffic_fsm.h"

uint32_t secondsToMs(int seconds)
{
    return static_cast<uint32_t>(seconds) * 1000UL;
}

void TaskTrafficFSMPlaceholder(void *pvParmeters)
{
    (void)pvParmeters;

    SignalPlan receivedPlan;

    for (;;)
    {
        BaseType_t receiveResult = xQueueReceive(
            planQueue,
            &receivedPlan,
            portMAX_DELAY
        );

        if (receiveResult == pdPASS)
        {
            Serial.println("[FSM] Received validated SignalPlan from planQueue.");
            printSignalPlan(&receivedPlan);
        }
        else
        {
            Serial.println("[FSM] ERROR: Failed to receive SignalPlan.");
        }
    }
}