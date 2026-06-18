// status_reporter.cpp
#include <Arduino.h>

#include "app_config.h"
#include "messages.h"
#include "status_reporter.h"
#include "traffic_fsm.h"
#include "logging.h"

/*
 * Shared status snapshot.
 *
 * TaskTrafficFSM updates this structure.
 * StatusTimerCallback reads this structure.
 *
 * Phase 15.8 goal:
 * Separate control logic from monitoring logic.
 */
static ControllerStatus controllerStatus;

// FreeRTOS software timer handle
static TimerHandle_t statusTimer = nullptr;

// Print one machine-readable STATUS line 
// ex :  STATUS,17,A_GREEN,12,OK

static void printStatusLine()
{
    /*
     * STATUS is periodic telemetry.
     * 
     * We only need to use tryLogLine(), which internally uses lockSerial(0)
     */
    char line[96];

    snprintf(
        line,
        sizeof(line),
        "STATUS,%d,%s,%lu,%s",
        controllerStatus.plan_id,
        trafficStateToString(controllerStatus.state),
        static_cast<unsigned long>(controllerStatus.remaining_seconds),
        controllerStatus.health
    );
    /*
     * Build complete line first.
     * Lock Serial briefly.
     * Print one complete line.
     * Unlock immediately.
     */

    tryLogLine(line);
}

// Software timer callback
// Executed by the FreeRTOS Daemon Task, not by TaskTrafficFSM
static void StatusTimerCallback(TimerHandle_t timerHandle)
{
    (void)timerHandle;
    // Keep call back short now.
    printStatusLine();
}

// Initialize status reporting subsystem.
void initStatusReporter()
{
    controllerStatus.plan_id = 0;
    controllerStatus.state = STATE_A_GREEN;
    controllerStatus.remaining_seconds = 0;
    controllerStatus.health = "OK";

    /*
     * Auto-reload timer: 
     * 
     * Every STATUS_TIMER_PERIOD_TICK
     * execute StatusTimerCallback().
     */
    statusTimer = xTimerCreate(
        "StatusTimer", 
        STATUS_TIMER_PERIOD_TICK,
        pdTRUE, // For Auto-reload mode
        nullptr,
        StatusTimerCallback
    );

    if (statusTimer == nullptr)
    {
        Serial.println("[STATUS] ERROR: Failed to create StatusTimer.");
    }
}


/*
 * Update the shared status snapshot
 * 
 * Called periodically by TaskTrafficFSM.
 * 
 * The timer callback never calculates FSM state.
 * It only reports the latest snapshot.
 */
void updateControllerStatus(
    const SignalPlan *activePlan,
    TrafficState currentState,
    TickType_t stateStartTick,
    const char *health
)
{
    if (activePlan == nullptr)
    {
        return;
    }

    // The count of ticks since vTaskStartScheduler was called.
    TickType_t now = xTaskGetTickCount();
    // How long has the current FSM state been active ?
    uint32_t elapsedMs =
        static_cast<uint32_t>(now - stateStartTick) *
        portTICK_PERIOD_MS; // Ticks -> Milliseconds

        /*
        * Duration of current state.
        *
        * Example:
        * A_GREEN -> green_a
        * B_GREEN -> green_b
        */
        uint32_t durationMs = getStateDurationMs(
        currentState,
        activePlan
        );

        uint32_t remainingMs = 0;
        if (durationMs > elapsedMs)
        {
            remainingMs = durationMs - elapsedMs;
        }

        /*
         * Update shared snapshot.
         */
        controllerStatus.state = currentState;
        controllerStatus.plan_id = activePlan->plan_id;
        controllerStatus.remaining_seconds = remainingMs / 1000;

        // Update Status Reporter Source
        // Health is supplied by TaskTrafficFSM
        if (health == nullptr)
        {
            controllerStatus.health = "UNKNOWN";
        }
        else
        {
            controllerStatus.health = health;
        }
}


/***
 * Start STATUS timer. 
 * 
 * Transition: Dormant State -> Running State
 * 
 * After start:
 * StatusTimerCallback() executes periodically
 */

void startStatusTimer()
{
    if(statusTimer == nullptr)
    {
        Serial.println("[STATUS] ERROR: StatusTimer is null.");
        return;
    }
    BaseType_t timerStarted = xTimerStart(
        statusTimer,
        0
    );

    if (timerStarted == pdPASS)
    {
        Serial.println("[STATUS] StatusTimer started.");
    }
    else
    {
        Serial.println("[STATUS] ERROR: Failed to start StatusTimer.");
    }
}