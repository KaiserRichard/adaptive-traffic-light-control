/*
 * ATLC Phase 15 — FreeRTOS-Based Embedded Controller and Edge Deployment Upgrade
 */

#include <Arduino.h>
#include "app_config.h"
#include "queues.h"
#include "tasks.h"
#include "traffic_fsm.h"
#include "status_reporter.h"



static void printBootBanner()
{
    Serial.println();
    Serial.println("[BOOT] ATLC Phase 15 FreeRTOS Controller");
    Serial.println("[BOOT] Phase 15.9 - Host Timeout Watchdog");
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

    printBootBanner();

    setupTrafficLightPins();
    Serial.println("[BOOT] Traffic light GPIO pins initialized.");

    initStatusReporter();
    Serial.println("[BOOT] Status reporter initialized.");

    if (!createApplicationQueues())
    {
        Serial.println("[BOOT] ERROR: Failed to create application queues.");
        haltSystem();
    }

    Serial.println("[BOOT] Application queues created.");

    BaseType_t uartReceiveCreated = xTaskCreate(
        TaskUARTReceive,
        "UARTReceive",
        UART_RECEIVE_TASK_STACK_SIZE,
        nullptr,
        UART_RECEIVE_TASK_PRIORITY,
        nullptr
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
        nullptr
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
        nullptr
    );

    if (fsmCreated != pdPASS)
    {
        Serial.println("[BOOT] ERROR: Failed to create TaskTrafficFSM.");
        haltSystem();
    }

    Serial.println("[BOOT] TaskTrafficFSM created.");

    startStatusTimer();
    Serial.println("[BOOT] Phase 15.9 system is running.");
}

void loop()
{
    vTaskDelay(pdMS_TO_TICKS(1000));
}