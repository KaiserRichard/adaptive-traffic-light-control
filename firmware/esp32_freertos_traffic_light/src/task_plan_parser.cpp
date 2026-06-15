// task_plan_parser.cpp
/*
 * TaskPlanParser:
 * Waits for RawMessage objects from rawMessageQueue.
 * Parser behavior: 
 * Phase 15.6: Add ACK and NACK into Valid and Invalid PLAN 
 * Valid PLAN
        → parse
        → validate
        → send to planQueue
        → ACK

    Invalid PLAN
        → parse
        → fail validation
        → reject
        → NACK 
    Unknown message
        → NACK 
 */
#include <Arduino.h>

#include "messages.h"
#include "queues.h"
#include "protocol.h"
#include "tasks.h"

void TaskPlanParser(void *pvParameters)
{
    (void)pvParameters;

    RawMessage receivedMessage;

    for (;;)
    {
        BaseType_t receiveResult = xQueueReceive(
            rawMessageQueue,
            &receivedMessage,
            portMAX_DELAY
        );

        if (receiveResult != pdPASS)
        {
            Serial.println("[PARSER] ERROR: Failed to receive raw message.");
            continue;
        }

        Serial.print("[PARSER] Received raw message: ");
        Serial.println(receivedMessage.data);

        if (!isPlanCommand(&receivedMessage))
        {
            Serial.println("[PARSER] Rejected: unknown command.");
            sendNack(-1, "UNKNOWN_COMMAND");
            continue;
        }

        ParsedPlanFields fields;

        if (!parsePlanCommand(&receivedMessage, &fields))
        {
            Serial.println("[PARSER] Rejected: malformed PLAN.");
            sendNack(-1, "MALFORMED_PLAN");
            continue;
        }

        Serial.println("[PARSER] PLAN command detected.");
        printParsedPlan(&fields);

        SignalPlan plan = makeSignalPlan(&fields);

        const char *validationReason = "OK";

        if (!validateSignalPlan(&plan, &validationReason))
        {
            Serial.print("[PARSER] Rejected SignalPlan: ");
            Serial.println(validationReason);

            sendNack(plan.plan_id, validationReason);
            continue;
        }

        BaseType_t sendResult = xQueueSendToBack(
            planQueue,
            &plan,
            pdMS_TO_TICKS(100)
        );

        if (sendResult == pdPASS)
        {
            Serial.println("[PARSER] Valid SignalPlan sent to planQueue.");
            sendAck(plan.plan_id);
        }
        else
        {
            Serial.println("[PARSER] ERROR: planQueue full.");
            sendNack(plan.plan_id, "PLAN_QUEUE_FULL");
        }
    }
}   