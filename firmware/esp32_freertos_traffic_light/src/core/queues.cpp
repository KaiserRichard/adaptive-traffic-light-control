#include <Arduino.h>

#include "config/app_config.h"
#include "core/queues.h"
#include "messages/messages.h"

QueueHandle_t rawMessageQueue = nullptr;
QueueHandle_t planQueue = nullptr;

bool initQueues()
{
    rawMessageQueue = xQueueCreate(
        RAW_MESSAGE_QUEUE_LENGTH,
        sizeof(RawMessage)
    );

    if (rawMessageQueue == nullptr)
    {
        return false;
    }

    planQueue = xQueueCreate(
        PLAN_QUEUE_LENGTH,
        sizeof(SignalPlan)
    );

    if (planQueue == nullptr)
    {
        return false;
    }

    return true;
}
