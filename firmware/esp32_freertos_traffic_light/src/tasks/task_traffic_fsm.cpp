// task_traffic_fsm.cpp


/*
 * TaskTrafficFSM:
 * Runs the local traffic light state machine.
 * Receives validated SignalPlan objects from planQueue.
 *
 * Phase 15.7:
 * New SignalPlans are first stored as pendingPlan.
 * The plan is only applied when the FSM reaches a safe boundary.
 *
 * Phase 15.9:
 * Detect host communication timeout.
 * Switch to a fallback plan if the host disappears.
 * Publish controller health through STATUS messages.
 */

#include <Arduino.h>

#include "config/app_config.h"
#include "core/queues.h"
#include "core/traffic_fsm.h"
#include "messages/messages.h"
#include "protocol/protocol.h"
#include "services/logging.h"
#include "services/status_reporter.h"
#include "tasks/task_traffic_fsm.h"

void TaskTrafficFSM(void *pvParameters)
{
    (void)pvParameters;

    // Current active timing plan used by the FSM.
    SignalPlan activePlan = getDefaultSignalPlan();

    // Safe fallback plan used when host stop sending valid PLAN messages.
    SignalPlan fallbackPlan = getDefaultSignalPlan();

    // Temporary buffer used when a new plan arrives from planQueue.
    SignalPlan receivedPlan;

    // Latest valid plan waiting to be applied at a safe boundary
    SignalPlan pendingPlan;

    // True when pendingPlan contains a new valid plan
    bool hasPendingPlan = false;

    /*
     * Phase 15.9: Host watchdog states
     * 
     */

    // Current controller health reported through STATUS messages.
    const char *controllerHealth = "OK";

    // True after a host timeout is detected.
    // Used to trigger fallback logic.
    bool isHostInTimeoutState = false;

    // Tick count when the last valid PLAN arrived.
    // Used by the host timeout watchdog.
    TickType_t lastValidPlanTick = xTaskGetTickCount();

    TrafficState currentState = STATE_A_GREEN;

    // Tick count when the current FSM state started.
    // Used to measure how long the current state has been active.
    TickType_t stateStartTick = xTaskGetTickCount();

    applyTrafficOutputs(currentState);

    /*
     * Publish initial controller state.
     *
     * STATUS timer cannot access local FSM variables.
     * Instead it reads the shared ControllerStatus snapshot.
     */
    updateControllerStatus(
        &activePlan,
        currentState,
        stateStartTick,
        controllerHealth
    );


    logLine("[FSM] Traffic FSM started.", DEBUG_LOG_WAIT_TICKS);

    char initialStateLine[80];
    snprintf(
        initialStateLine,
        sizeof(initialStateLine),
        "[FSM] Initial state: %s",
        trafficStateToString(currentState)
    );
    logLine(initialStateLine, DEBUG_LOG_WAIT_TICKS);

    logLine("[FSM] Default active plan:", DEBUG_LOG_WAIT_TICKS);
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
             * A valid PLAN means: 
             *      Host is alive.
             *      Refresh watchdog timer.
             *      Store plan for safe boundary apply.
             */
            pendingPlan = receivedPlan;
            hasPendingPlan = true;

            lastValidPlanTick = xTaskGetTickCount();
            
            // If the controller was previously in timeout mode,
            // This PLAN marks communication recovery.
            if(isHostInTimeoutState)
            {
                logLine("[FSM] Host communication recovered", DEBUG_LOG_WAIT_TICKS);
                isHostInTimeoutState = false;
            }
            controllerHealth = "OK";

            logLine("[FSM] Pending SignalPlan received.", DEBUG_LOG_WAIT_TICKS);
            printSignalPlan(&pendingPlan);
        }

        TickType_t now = xTaskGetTickCount();
        
        // How long has the host been silent ? 
        uint32_t hostSilentMs =
            static_cast<uint32_t>(now - lastValidPlanTick) *
            portTICK_PERIOD_MS;
        
        if(!isHostInTimeoutState && hostSilentMs >= HOST_TIMEOUT_MS)
        {
            // Host timeout detected.
            isHostInTimeoutState = true;
            controllerHealth = "HOST_TIMEOUT";

            activePlan = fallbackPlan; // Switch to fallback plan immediately.
            hasPendingPlan = false; // Clear pending plan to avoid applying old plans at safe boundary.

            logLine("[FSM] WARNING: Host communication timeout detected.", DEBUG_LOG_WAIT_TICKS);
            logLine("[FSM] Switching to fallback plan.", DEBUG_LOG_WAIT_TICKS);
            printSignalPlan(&activePlan);
        }

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

                logLine("[FSM] Pending SignalPlan applied at safe boundary.", DEBUG_LOG_WAIT_TICKS);
                printSignalPlan(&activePlan); // pendingPlan now becomes activePlan
            }

            applyTrafficOutputs(currentState);

            char transitionLine[80];
            snprintf(
                transitionLine,
                sizeof(transitionLine),
                "[FSM] Transition to %s",
                trafficStateToString(currentState)
            );
            logLine(transitionLine, DEBUG_LOG_WAIT_TICKS);
        }

        /*
         *Publish the latest FSM snapshot 
         *  
         * StatusTimerCallback never calculates FSM state
         * It only reads the latest published snapshot.
         * EX: STATUS,17,A_GREEN,12,OK
         */
        updateControllerStatus(
            &activePlan,
            currentState,
            stateStartTick,
            controllerHealth
        );

        vTaskDelay(FSM_TICK_PERIOD_TICKS);
    }
}
