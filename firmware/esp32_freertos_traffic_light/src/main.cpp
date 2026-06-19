#include <Arduino.h>

#include "app/app_startup.h"
#include "config/app_config.h"

#include "core/queues.h"
#include "core/traffic_fsm.h"

#include "drivers/uart_rx.h"

#include "services/diagnostics.h"
#include "services/logging.h"
#include "services/status_reporter.h"

#include "tasks/task_plan_parser.h"
#include "tasks/task_traffic_fsm.h"
#include "tasks/task_uart_receive.h"

void setup()
{
    Serial.begin(SERIAL_BAUD_RATE);
    delay(1000);

    printBootBanner();

    if (!initLogging())
    {
        haltSystem("Failed to initialize logging service.");
    }

    logLine("[BOOT] Logging service initialized.", DEBUG_LOG_WAIT_TICKS);

    if (!initQueues())
    {
        haltSystem("Failed to initialize FreeRTOS queues.");
    }

    logLine("[BOOT] FreeRTOS queues initialized.", DEBUG_LOG_WAIT_TICKS);

    initTrafficFsm();
    logLine("[BOOT] Traffic FSM initialized.", DEBUG_LOG_WAIT_TICKS);

    TaskHandle_t uartReceiveTaskHandle = nullptr;
    TaskHandle_t planParserTaskHandle = nullptr;
    TaskHandle_t trafficFsmTaskHandle = nullptr;

    BaseType_t uartTaskCreated = xTaskCreate(
        TaskUARTReceive,
        "UARTReceive",
        UART_RECEIVE_TASK_STACK_SIZE,
        nullptr,
        UART_RECEIVE_TASK_PRIORITY,
        &uartReceiveTaskHandle
    );

    if (uartTaskCreated != pdPASS)
    {
        haltSystem("Failed to create TaskUARTReceive.");
    }

    BaseType_t parserTaskCreated = xTaskCreate(
        TaskPlanParser,
        "PlanParser",
        PLAN_PARSER_TASK_STACK_SIZE,
        nullptr,
        PLAN_PARSER_TASK_PRIORITY,
        &planParserTaskHandle
    );

    if (parserTaskCreated != pdPASS)
    {
        haltSystem("Failed to create TaskPlanParser.");
    }

    BaseType_t fsmTaskCreated = xTaskCreate(
        TaskTrafficFSM,
        "TrafficFSM",
        TRAFFIC_FSM_TASK_STACK_SIZE,
        nullptr,
        TRAFFIC_FSM_TASK_PRIORITY,
        &trafficFsmTaskHandle
    );

    if (fsmTaskCreated != pdPASS)
    {
        haltSystem("Failed to create TaskTrafficFSM.");
    }

    logLine("[BOOT] FreeRTOS tasks created.", DEBUG_LOG_WAIT_TICKS);

    /*
     * Initialize UART RX driver after TaskUARTReceive exists.
     * The callback needs the task handle so it can notify the task.
     */
    initUartRxDriver(uartReceiveTaskHandle);
    logLine("[BOOT] UART RX event driver initialized.", DEBUG_LOG_WAIT_TICKS);

    if (!initStatusReporter())
    {
        haltSystem("Failed to initialize status reporter.");
    }

    startStatusTimer();
    logLine("[BOOT] Status reporter started.", DEBUG_LOG_WAIT_TICKS);

    initDiagnosticsReporter(
        uartReceiveTaskHandle,
        planParserTaskHandle,
        trafficFsmTaskHandle
    );

    startDiagnosticsTimer();
    logLine("[BOOT] Diagnostics reporter started.", DEBUG_LOG_WAIT_TICKS);

    logLine("[BOOT] Phase 15.13 system is running.", DEBUG_LOG_WAIT_TICKS);
}

void loop()
{
    /*
     * FreeRTOS tasks own the application behavior.
     * Arduino loop() remains intentionally minimal.
     */
    vTaskDelay(pdMS_TO_TICKS(1000));
}
