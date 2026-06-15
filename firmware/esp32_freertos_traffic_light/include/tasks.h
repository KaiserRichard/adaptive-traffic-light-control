// task.h
#pragma once

#include <Arduino.h>

void TaskUARTReceive(void *pvParameters);
void TaskPlanParser(void *pvParameters);
void TaskTrafficFSM(void *pvParameters);