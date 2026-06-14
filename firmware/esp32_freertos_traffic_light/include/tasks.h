// task.h
#pragma once

#include <Arduino.h>

void TaskUARTReceive(void *pvParameters);
void TaskPlanParser(void *pvParameters);
void TaskTrafficFSMPlaceholder(void *pvParameters);