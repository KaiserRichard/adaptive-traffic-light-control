#pragma once

#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>

#include "messages/messages.h"

// Global queue handles.
// Defined once in queues.cpp.
extern QueueHandle_t rawMessageQueue;
extern QueueHandle_t planQueue;

bool initQueues();
