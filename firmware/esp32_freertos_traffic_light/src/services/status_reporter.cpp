#include <Arduino.h>

#include "config/app_config.h"
#include "core/traffic_fsm.h"
#include "messages/messages.h"
#include "services/logging.h"
#include "services/status_reporter.h"

/*
 * STATUS REPORTER
 *
 * "The traffic FSM owns the truth. The status reporter only tells the story."
 *
 * TaskTrafficFSM updates controllerStatus with the latest plan/state/health.
 * A FreeRTOS software timer decides when a STATUS line is due.
 * The timer callback wakes TaskStatusReporter, and the task prints the line.
 *
 * This keeps two jobs separate:
 *     - control logic: calculate and update traffic state
 *     - monitoring logic: publish the latest state as telemetry
 */
static ControllerStatus controllerStatus;

/*
 * ESP32 FreeRTOS critical sections require a portMUX_TYPE spinlock.
 *
 * In many FreeRTOS examples you will see:
 *
 *     taskENTER_CRITICAL();
 *     taskEXIT_CRITICAL();
 *
 * In ESP32 Arduino FreeRTOS we use:
 *
 *     taskENTER_CRITICAL(&controllerStatusMux);
 *     taskEXIT_CRITICAL(&controllerStatusMux);
 *
 * Reason:
 *     ESP32 is dual-core. The mux protects this shared snapshot from tasks
 *     running on either core.
 */
static portMUX_TYPE controllerStatusMux = portMUX_INITIALIZER_UNLOCKED;

// FreeRTOS software timer: decides WHEN a STATUS report is needed.
static TimerHandle_t statusTimer = nullptr;

// Dedicated reporter task: decides HOW the STATUS line is printed.
static TaskHandle_t statusReporterTaskHandle = nullptr;

/*
 * Take a safe local copy of the shared status.
 *
 * "Lock, copy, unlock" keeps the critical section very short.
 * After this function returns, formatting and Serial logging can happen without
 * holding the mux.
 */
static ControllerStatus copyControllerStatus()
{
    ControllerStatus snapshot;

    taskENTER_CRITICAL(&controllerStatusMux);
    snapshot = controllerStatus;
    taskEXIT_CRITICAL(&controllerStatusMux);

    return snapshot;
}

/*
 * Print one machine-readable STATUS line.
 *
 * Format:
 *     STATUS,<plan_id>,<state>,<remaining_seconds>,<health>
 *
 * Example:
 *     STATUS,17,A_GREEN,12,OK
 */
static void printStatusLine()
{
    ControllerStatus snapshot = copyControllerStatus();

    char line[96];

    snprintf(
        line,
        sizeof(line),
        "STATUS,%d,%s,%lu,%s",
        snapshot.plan_id,
        trafficStateToString(snapshot.state),
        static_cast<unsigned long>(snapshot.remaining_seconds),
        snapshot.health
    );

    /*
     * Build the complete line first, then print it with the Serial mutex.
     * tryLogLine() uses lockSerial(0), so telemetry never blocks the controller.
     */
    tryLogLine(line);
}

/*
 * Timer callback rule:
 *     Keep it tiny.
 *
 * FreeRTOS software timer callbacks run in the timer service task. If this
 * callback printed to Serial directly, it could delay other software timers.
 * Instead, it sends a notification to TaskStatusReporter.
 */
static void StatusTimerCallback(TimerHandle_t timerHandle)
{
    (void)timerHandle;

    if (statusReporterTaskHandle == nullptr)
    {
        return;
    }

    xTaskNotifyGive(statusReporterTaskHandle);
}

/*
 * Initialize the STATUS reporter subsystem.
 *
 * Creates:
 *     - initial controllerStatus snapshot
 *     - TaskStatusReporter
 *     - periodic StatusTimer
 *
 * Returns false if any required FreeRTOS object cannot be created.
 */
bool initStatusReporter()
{
    taskENTER_CRITICAL(&controllerStatusMux);
    controllerStatus.plan_id = 0;
    controllerStatus.state = STATE_A_GREEN;
    controllerStatus.remaining_seconds = 0;
    controllerStatus.health = "OK";
    taskEXIT_CRITICAL(&controllerStatusMux);

    BaseType_t taskCreated = xTaskCreate(
        TaskStatusReporter,
        "StatusReporter",
        STATUS_REPORTER_TASK_STACK_SIZE,
        nullptr,
        STATUS_REPORTER_TASK_PRIORITY,
        &statusReporterTaskHandle
    );

    if (taskCreated != pdPASS)
    {
        logLine("[STATUS] ERROR: Failed to create TaskStatusReporter.", pdMS_TO_TICKS(20));
        return false;
    }

    statusTimer = xTimerCreate(
        "StatusTimer",
        STATUS_TIMER_PERIOD_TICKS,
        pdTRUE,
        nullptr,
        StatusTimerCallback
    );

    if (statusTimer == nullptr)
    {
        logLine("[STATUS] ERROR: Failed to create StatusTimer.", pdMS_TO_TICKS(20));
        return false;
    }

    logLine("[STATUS] Status reporter initialized.", pdMS_TO_TICKS(20));
    return true;
}

/*
 * Update the shared STATUS snapshot.
 *
 * Called by TaskTrafficFSM after it calculates the current traffic state.
 * The timer callback never calculates FSM timing; it only asks for the latest
 * completed snapshot to be reported.
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

    TickType_t now = xTaskGetTickCount();

    /*
     * How long has this FSM state been active?
     *
     * stateStartTick was captured when the FSM entered currentState.
     * Tick difference gives elapsed scheduler ticks; portTICK_PERIOD_MS
     * converts that count into milliseconds.
     */
    uint32_t elapsedMs =
        static_cast<uint32_t>(now - stateStartTick) *
        portTICK_PERIOD_MS;

    /*
     * Duration expected for the current state.
     *
     * Examples:
     *     STATE_A_GREEN -> activePlan->green_a
     *     STATE_B_GREEN -> activePlan->green_b
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
     * Publish one coherent snapshot.
     *
     * All fields are updated inside the same critical section so the reporter
     * cannot read a half-old, half-new STATUS line.
     */
    taskENTER_CRITICAL(&controllerStatusMux);

    controllerStatus.plan_id = activePlan->plan_id;
    controllerStatus.state = currentState;
    controllerStatus.remaining_seconds = remainingMs / 1000;

    if (health == nullptr)
    {
        controllerStatus.health = "UNKNOWN";
    }
    else
    {
        controllerStatus.health = health;
    }

    taskEXIT_CRITICAL(&controllerStatusMux);
}

/*
 * Start the periodic STATUS timer.
 *
 * Transition:
 *     Dormant timer -> Running timer
 *
 * After this succeeds, StatusTimerCallback() runs once per
 * STATUS_TIMER_PERIOD_TICKS and wakes TaskStatusReporter.
 */
void startStatusTimer()
{
    if (statusTimer == nullptr)
    {
        logLine("[STATUS] ERROR: StatusTimer is null.", pdMS_TO_TICKS(20));
        return;
    }

    BaseType_t timerStarted = xTimerStart(
        statusTimer,
        0
    );

    if (timerStarted == pdPASS)
    {
        logLine("[STATUS] StatusTimer started.", pdMS_TO_TICKS(20));
    }
    else
    {
        logLine("[STATUS] ERROR: Failed to start StatusTimer.", pdMS_TO_TICKS(20));
    }
}

/*
 * Dedicated STATUS reporter task.
 *
 * It sleeps until the timer callback sends a task notification.
 * Then it prints exactly one latest STATUS snapshot.
 *
 * "The timer knocks. The task answers."
 */
void TaskStatusReporter(void *pvParameters)
{
    (void)pvParameters;

    logLine("[STATUS] TaskStatusReporter started.", pdMS_TO_TICKS(20));

    for (;;)
    {
        uint32_t notificationCount = ulTaskNotifyTake(
            pdTRUE,
            portMAX_DELAY
        );

        if (notificationCount > 0)
        {
            printStatusLine();
        }
    }
}
