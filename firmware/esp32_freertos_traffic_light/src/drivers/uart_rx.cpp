#include <Arduino.h>

#include "drivers/uart_rx.h"

static TaskHandle_t uartRxTaskHandle = nullptr;

static void onSerialReceive()
{
    if (uartRxTaskHandle == nullptr)
    {
        return;
    }

    BaseType_t higherPriorityTaskWoken = pdFALSE;

    vTaskNotifyGiveFromISR(
        uartRxTaskHandle,
        &higherPriorityTaskWoken
    );

    if (higherPriorityTaskWoken == pdTRUE)
    {
        portYIELD_FROM_ISR();
    }
}

void initUartRxDriver(TaskHandle_t uartReceiveTaskHandle)
{
    uartRxTaskHandle = uartReceiveTaskHandle;

    /*
     * onlyOnTimeout = true calls the callback after the UART RX stream becomes
     * idle, which is a good fit for line-based commands.
     */
    Serial.onReceive(onSerialReceive, true);

    /*
     * Wake the task once after registration so it can drain any bytes that may
     * have arrived during startup.
     */
    xTaskNotifyGive(uartRxTaskHandle);
}
