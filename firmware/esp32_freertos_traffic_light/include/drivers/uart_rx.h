#pragma once

#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

/*
 * UART RX Driver Layer
 *
 * This module owns UART receive event registration.
 *
 * The receive callback must stay small:
 *
 *     callback
 *         -> notify TaskUARTReceive
 *         -> return immediately
 *
 * The callback must not parse PLAN messages.
 * The callback must not update the FSM.
 * The callback must not print logs.
 */
void initUartRxDriver(TaskHandle_t uartReceiveTaskHandle);
