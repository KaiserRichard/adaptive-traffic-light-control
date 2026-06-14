// task_plan_parser.cpp
/*
 * TaskPlanParser:
 * Waits for RawMessage objects from rawMessageQueue.
 * Parser behavior: 
 * Valid PLAN
        → parse
        → validate
        → send to planQueue

    Invalid PLAN
        → parse
        → fail validation
        → reject

    Unknown message
        → reject before parsing
 */
#include <Arduino.h>

#include "messages.h"
#include "queues.h"
#include "protocol.h"
#include "tasks.h"
void TaskPlanParser(void *pvParameters)
{
    // Prevent compiler warnings
    (void)pvParameters;

    // Local variable to store the message received from the queue
    RawMessage receivedMessage;

    for (;;)
    {
        BaseType_t receiveResult = xQueueReceive(
            rawMessageQueue,
            &receivedMessage,
            portMAX_DELAY
        );

        // If receive succeeded, print the raw message
        if (receiveResult == pdPASS)
        {
            Serial.print("[PARSER] Received raw message: ");
            Serial.println(receivedMessage.data);

            if (isPlanCommand(&receivedMessage))
            {
                ParsedPlanFields fields;

                if (parsePlanCommand(&receivedMessage, &fields))
                {
                    Serial.println("[PARSER] PLAN command detected.");
                    printParsedPlan(&fields);

                    SignalPlan plan = makeSignalPlan(&fields);

                    const char *validationReason = "OK";
                    if (validateSignalPlan(&plan, &validationReason))
                    {
                        BaseType_t sendResult = xQueueSendToBack(
                            planQueue,
                            &plan,
                            portMAX_DELAY);

                        if (sendResult == pdPASS)
                        {
                            Serial.println("[PARSER] Valid SignalPlan sent to planQueue.");
                        }
                        else
                        {
                            Serial.println("[PARSER] ERROR: Failed to send SignalPlan to planQueue.");
                        }
                    }
                    else
                    {
                        Serial.print("[PARSER] Rejected SignalPlan: ");
                        Serial.println(validationReason);
                    }
                }
                else
                {
                    Serial.println("[PARSER] ERROR: Malformed PLAN command.");
                }
            }
            else
            {
                Serial.println("[PARSER] Unknown command format.");
            }
        }
        else
        {
            Serial.println("[PARSER] ERROR: Failed to receive raw message.");
        }
    }
}