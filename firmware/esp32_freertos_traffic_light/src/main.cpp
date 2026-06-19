/*
 * ATLC Phase 15 — FreeRTOS-Based Embedded Controller and Edge Deployment Upgrade
 */

#include <Arduino.h>
#include "app_config.h"
#include "queues.h"
#include "tasks.h"
#include "traffic_fsm.h"
#include "status_reporter.h"
#include "diagnostics.h"
#include "logging.h"

static void printBootBanner()
{
    Serial.println();
    Serial.println("[BOOT] ATLC Phase 15 FreeRTOS Controller");
    Serial.println("[BOOT] Phase 15.12 - Task Notifications");
}

static void haltSystem()
{
    Serial.println("[BOOT] System halted.");

    while (true)
    {
        delay(1000);
    }
}

void setup()
{
    Serial.begin(SERIAL_BAUD_RATE);
    delay(1000);

    if (!initLogging())
    {
        haltSystem();
    }

    printBootBanner();

    setupTrafficLightPins();
    Serial.println("[BOOT] Traffic light GPIO pins initialized.");

    if (!initStatusReporter())
    {
        logLine("[BOOT] ERROR: Failed to initialize status reporter.", pdMS_TO_TICKS(20));
        haltSystem();
    }

    logLine("[BOOT] Status reporter initialized.", pdMS_TO_TICKS(20));

    if (!createApplicationQueues())
    {
        Serial.println("[BOOT] ERROR: Failed to create application queues.");
        haltSystem();
    }

    Serial.println("[BOOT] Application queues created.");

    /*
     * Phase 15.10:
     * Store task handles so diagnostics can inspect each task's stack high-water mark later.
     */
    TaskHandle_t uartReceiveTaskHandle = nullptr;
    TaskHandle_t planParserTaskHandle = nullptr;
    TaskHandle_t trafficFsmTaskHandle = nullptr;


    BaseType_t uartReceiveCreated = xTaskCreate(
        TaskUARTReceive,
        "UARTReceive",
        UART_RECEIVE_TASK_STACK_SIZE,
        nullptr,
        UART_RECEIVE_TASK_PRIORITY,
        &uartReceiveTaskHandle
    );

    if (uartReceiveCreated != pdPASS)
    {
        Serial.println("[BOOT] ERROR: Failed to create TaskUARTReceive.");
        haltSystem();
    }

    Serial.println("[BOOT] TaskUARTReceive created.");

    BaseType_t parserCreated = xTaskCreate(
        TaskPlanParser,
        "PlanParser",
        PARSER_TASK_STACK_SIZE,
        nullptr,
        PARSER_TASK_PRIORITY,
        &planParserTaskHandle
    );

    if (parserCreated != pdPASS)
    {
        Serial.println("[BOOT] ERROR: Failed to create TaskPlanParser.");
        haltSystem();
    }

    Serial.println("[BOOT] TaskPlanParser created.");

    BaseType_t fsmCreated = xTaskCreate(
        TaskTrafficFSM,
        "TrafficFSM",
        FSM_TASK_STACK_SIZE,
        nullptr,
        FSM_TASK_PRIORITY,
        &trafficFsmTaskHandle
    );

    if (fsmCreated != pdPASS)
    {
        Serial.println("[BOOT] ERROR: Failed to create TaskTrafficFSM.");
        haltSystem();
    }

    Serial.println("[BOOT] TaskTrafficFSM created.");

    /**
     * Start diagnostics after all tasks are created.
     * 
     * Diagnostics need valid task handles to measure task stack high-water marks.
     */

    initDiagnosticsReporter(
        uartReceiveTaskHandle,
        planParserTaskHandle,
        trafficFsmTaskHandle
    );
    
    Serial.println("[BOOT] Diagnostics reporter initialized.");

    startStatusTimer();
    startDiagnosticsTimer();
    logLine("[BOOT] Phase 15.12 system is running.", pdMS_TO_TICKS(20));
}

void loop()
{
    vTaskDelay(pdMS_TO_TICKS(1000));
}
