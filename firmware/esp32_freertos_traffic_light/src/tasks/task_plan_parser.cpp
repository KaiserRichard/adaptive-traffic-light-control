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

#include "config/app_config.h"
#include "core/queues.h"
#include "messages/messages.h"
#include "protocol/protocol.h"
#include "services/logging.h"
#include "tasks/task_plan_parser.h"

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
            logLine("[PARSER] ERROR: Failed to receive raw message.", DEBUG_LOG_WAIT_TICKS);
            continue;
        }

        char receivedLine[160];
        snprintf(
            receivedLine,
            sizeof(receivedLine),
            "[PARSER] Received raw message: %s",
            receivedMessage.data
        );
        logLine(receivedLine, DEBUG_LOG_WAIT_TICKS);

        if (!isPlanCommand(&receivedMessage))
        {
            logLine("[PARSER] Rejected: unknown command.", DEBUG_LOG_WAIT_TICKS);
            sendNack(-1, "UNKNOWN_COMMAND");
            continue;
        }

        ParsedPlanFields fields;

        if (!parsePlanCommand(&receivedMessage, &fields))
        {
            logLine("[PARSER] Rejected: malformed PLAN.", DEBUG_LOG_WAIT_TICKS);
            sendNack(-1, "MALFORMED_PLAN");
            continue;
        }

        logLine("[PARSER] PLAN command detected.", DEBUG_LOG_WAIT_TICKS);
        printParsedPlan(&fields);

        SignalPlan plan = makeSignalPlan(&fields);

        const char *validationReason = "OK";

        if (!validateSignalPlan(&plan, &validationReason))
        {
            char rejectLine[128];
            snprintf(
                rejectLine,
                sizeof(rejectLine),
                "[PARSER] Rejected SignalPlan: %s",
                validationReason
            );
            logLine(rejectLine, DEBUG_LOG_WAIT_TICKS);

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
            logLine("[PARSER] Valid SignalPlan sent to planQueue.", DEBUG_LOG_WAIT_TICKS);
            sendAck(plan.plan_id);
        }
        else
        {
            logLine("[PARSER] ERROR: planQueue full.", DEBUG_LOG_WAIT_TICKS);
            sendNack(plan.plan_id, "PLAN_QUEUE_FULL");
        }
    }
}
