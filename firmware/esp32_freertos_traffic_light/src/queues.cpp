// queues.cpp 
#include "queues.h"
#include "app_config.h"

QueueHandle_t rawMessageQueue = nullptr;
QueueHandle_t planQueue = nullptr;

bool createApplicationQueues()
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