//queues.h
#pragma once

#include <Arduino.h>
#include "messages.h"

// Global queue handles.
// Defined once in queues.cpp.
extern QueueHandle_t rawMessageQueue;
extern QueueHandle_t planQueue;

bool createApplicationQueues();