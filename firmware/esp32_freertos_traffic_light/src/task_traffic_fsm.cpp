// task_traffic_fsm.cpp


/*
 * TaskTrafficFSM:
 * Runs the local traffic light state machine.
 * Receives validated SignalPlan objects from planQueue.
 *
 * Phase 15.7:
 * New plans are stored as pendingPlan first.
 * They are applied only at a safe FSM boundary.
 */

#include <Arduino.h>

#include "app_config.h"
#include "messages.h"
#include "queues.h"
#include "protocol.h"
#include "tasks.h"
#include "traffic_fsm.h"

void TaskTrafficFSM(void *pvParameters)
{
    (void)pvParameters;

    SignalPlan activePlan = getDefaultSignalPlan();
    // Temporary buffer used when a new plan arrives from planQueue.
    SignalPlan receivedPlan;

    // Latest valid plan waiting to be applied at a safe boundary
    SignalPlan pendingPlan;

    // True when pendingPlan contains a new valid plan
    bool hasPendingPlan = false;

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
            /*
             * Phase 15.7: Safe plan apply
             * Do not apply the plan immediately.
             * Store it as pending and wait for a safe boundary.
             */
            pendingPlan = receivedPlan;
            hasPendingPlan = true;

            Serial.println("[FSM] Pending SignalPlan received.");
            printSignalPlan(&pendingPlan);
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
            &activePlan
        );
        
        if (elapsedMs >= durationMs)
        {
            // Current state has finished.
            // Move to the next traffic state.
            currentState = getNextTrafficState(currentState);
            stateStartTick = xTaskGetTickCount();
            
            /*
             * Safe apply boundary: 
             * Apply the pending plan only when the FSM returns to A_GREEN.
             * Meaning one full cycle has completed
             */
            if (currentState == STATE_A_GREEN && hasPendingPlan)
            {
                activePlan = pendingPlan;
                hasPendingPlan = false;

                Serial.println("[FSM] Pending SignalPlan applied at safe boundary.");
                printSignalPlan(&activePlan); // pendingPlan now becomes activePlan
            }

            applyTrafficOutputs(currentState);

            Serial.print("[FSM] Transition to ");
            Serial.println(trafficStateToString(currentState));
        }

        // Keep it simple first. 
        // Later use:  vTaskDelayUntil for more precise periodic execution
        vTaskDelay(FSM_UPDATE_PERIOD_TICK);
    }
}