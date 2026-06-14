// task_traffic_fsm.cpp
/* 
 * TaskTrafficFSMPlaceholder:
 * Receives validated SignalPlan object from planQueue
 * 
 */
#include <Arduino.h>

#include "app_config.h"
#include "messages.h"
#include "queues.h"
#include "protocol.h"
#include "tasks.h"
#include "traffic_fsm.h"

// Phase 15.5: We replace placeholder with real FSM task
// void TaskTrafficFSMPlaceholder(void *pvParmeters)
// {
//     (void)pvParmeters;

//     SignalPlan receivedPlan;

//     for (;;)
//     {
//         BaseType_t receiveResult = xQueueReceive(
//             planQueue,
//             &receivedPlan,
//             portMAX_DELAY
//         );

//         if (receiveResult == pdPASS)
//         {
//             Serial.println("[FSM] Received validated SignalPlan from planQueue.");
//             printSignalPlan(&receivedPlan);
//         }
//         else
//         {
//             Serial.println("[FSM] ERROR: Failed to receive SignalPlan.");
//         }
//     }
// }

/*
 * TaskTrafficFSM:
 * Runs the local traffic light state machine. 
 * Receives validated SignalPlan objects from planQueue.
 */
void TaskTrafficFSM(void *pvParameters)
{
    (void)pvParameters;

    SignalPlan activePlan = getDefaultSignalPlan();
    // Temporary buffer used when a new plan arrives from planQueue.
    SignalPlan receivedPlan;

    TrafficState currentState = STATE_A_GREEN;

    // Tick count when the current FSM state started.
    // Used to measure how long the current state has been active.
    TickType_t stateStartTick = xTaskGetTickCount();

    applyTrafficOutputs(currentState);

    Serial.println("[FSM] Traffic FSM started.");
    Serial.print("[FSM] Initial state: ");
    Serial.println(trafficStateToString(currentState));

    Serial.println("[FSM] Default active plan: ");
    printSignalPlan(&activePlan);

    for (;;)
    {
        /*
         * Non-blocking receive: 
         * If a new plan exists, receive it.
         * If no new plan exists, keep running the FSM.
         */
        BaseType_t receiveResult = xQueueReceive(
            planQueue,
            &receivedPlan,
            0                   // Non-blocking: check queue and continue immediately.
        );

        if (receiveResult == pdPASS)
        {
            // Replace the current timing plan with the new validated plan.
            activePlan = receivedPlan;

            Serial.println("[FSM] New active SignalPlan received.");
            printSignalPlan(&activePlan);
        }

        TickType_t now = xTaskGetTickCount();
        
        // How many milliseconds has the current FSM state been running?
        uint32_t elapsedMs =
            static_cast<uint32_t>(now - stateStartTick) *
            portTICK_PERIOD_MS;
        
        // Required duration for the current state.
        // Example:
        // A_GREEN  -> activePlan.green_a
        // A_YELLOW -> activePlan.yellow
        uint32_t durationMs = getStateDurationMs(
            currentState,
            &activePlan);
        
        if (elapsedMs >= durationMs)
        {
            // Current state has finished.
            // Move to the next traffic state.
            currentState = getNextTrafficState(currentState);
            stateStartTick = xTaskGetTickCount();

            applyTrafficOutputs(currentState);

            Serial.print("[FSM] Transition to ");
            Serial.println(trafficStateToString(currentState));
        }

        // Keep it simple first. 
        // Later use:  vTaskDelayUntil for more precise periodic execution
        vTaskDelay(FSM_UPDATE_PERIOD_TICK);
    }
}