#include <Arduino.h>

#include "config/app_config.h"
#include "services/diagnostics.h"
#include "services/logging.h"

// Task handles are stored here so the diagnostic timer can inspect task stack usage periodically.
static TaskHandle_t uartReceiveHandle = nullptr;
static TaskHandle_t planParserHandle = nullptr;
static TaskHandle_t trafficFsmHandle = nullptr;

// Software timer used for periodic diagnostics.
static TimerHandle_t diagnosticsTimer = nullptr;

/*
 * TaskHandle_t is already a pointer-like type.
 *
 * FreeRTOS task handles are used as references
 * to existing tasks. A nullptr means the task
 * handle has not been assigned yet.
 */
static UBaseType_t getStackRemainingBytes(TaskHandle_t taskHandle)
{
    if(taskHandle == nullptr)
    {
        return 0; // diagnostics can still run but that task will report stack value as 0.
    }

    // FreeRTOS returns stack high-water mark in words.
    // In this phase, we report the raw value for debugging.
    return uxTaskGetStackHighWaterMark(taskHandle);
}

// Print one complete diagnostic line.
static void printDiagnosticsLine()
{
    /*
     * DIAG is periodic telemetry.
     * 
     * It is useful, but not important enough to block the FreeRTOS daemond task
     * Therefore if Serial is busy, this DIAG frame can be skipped.
     */
    uint32_t freeHeapBytes = static_cast<uint32_t>(xPortGetFreeHeapSize());
    UBaseType_t uartStackBytes = getStackRemainingBytes(uartReceiveHandle);
    UBaseType_t parserStackBytes = getStackRemainingBytes(planParserHandle);
    UBaseType_t fsmStackBytes = getStackRemainingBytes(trafficFsmHandle);

    char line[160];
    snprintf(
        line,
        sizeof(line),
        "DIAG,heap=%lu,uart_stack=%lu,parser_stack=%lu,fsm_stack=%lu",
        static_cast<unsigned long>(freeHeapBytes),
        static_cast<unsigned long>(uartStackBytes),
        static_cast<unsigned long>(parserStackBytes),
        static_cast<unsigned long>(fsmStackBytes)
    );

    tryLogLine(line);
}

/*
 * FreeRTOS software timer callback.
 * 
 * Keep timer callbacks short.
 */

 static void DiagnosticsTimerCallback(TimerHandle_t timerHandle)
 {
     (void)timerHandle;

     printDiagnosticsLine();
}

void initDiagnosticsReporter(
    TaskHandle_t uartTaskHandle,
    TaskHandle_t parserTaskHandle,
    TaskHandle_t fsmTaskHandle
)
{
    uartReceiveHandle = uartTaskHandle;
    planParserHandle = parserTaskHandle;
    trafficFsmHandle = fsmTaskHandle;

    /*
     * Basic sanity check:
     * If any handle is null, diagnostics can still run but that task will report stack value as 0.
     */
    if (uartReceiveHandle == nullptr)
    {
        logLine("[DIAG] WARNING: UART task handle is null.", DEBUG_LOG_WAIT_TICKS);
    }

    if (planParserHandle == nullptr)
    {
        logLine("[DIAG] WARNING: Parser task handle is null.", DEBUG_LOG_WAIT_TICKS);
    }

    if (trafficFsmHandle == nullptr)
    {
        logLine("[DIAG] WARNING: FSM task handle is null.", DEBUG_LOG_WAIT_TICKS);
    }

    diagnosticsTimer = xTimerCreate(
        "DiagnosticsTimer",
        DIAGNOSTICS_TIMER_PERIOD_TICKS,
        pdTRUE,
        nullptr,
        DiagnosticsTimerCallback
    );
    if(diagnosticsTimer == nullptr)
    {
        logLine("[DIAG] ERROR: Failed to create DiagnosticsTimer.", DEBUG_LOG_WAIT_TICKS);
    }
}

void startDiagnosticsTimer()
{
    if (diagnosticsTimer == nullptr)
    {
        logLine("[DIAG] ERROR: DiagnosticsTimer is null.", DEBUG_LOG_WAIT_TICKS);
        return;
    }

    BaseType_t timerStarted = xTimerStart(
        diagnosticsTimer,
        0
    );

    if (timerStarted == pdPASS)
    {
        logLine("[DIAG] DiagnosticsTimer started.", DEBUG_LOG_WAIT_TICKS);
    }
    else
    {
        logLine("[DIAG] ERROR: Failed to start DiagnosticsTimer.", DEBUG_LOG_WAIT_TICKS);
    }
}
