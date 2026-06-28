#include <Arduino.h>

#include "config/app_config.h"
#include "core/queues.h"
#include "messages/messages.h"
#include "services/logging.h"
#include "tasks/task_uart_receive.h"

/*
 * TaskUARTReceive
 *
 * Owns Serial RX byte reading.
 *
 * UART RX callback:
 *     only wakes this task.
 *
 * This task:
 *     reads available Serial bytes,
 *     builds newline-terminated RawMessage objects,
 *     sends completed RawMessage objects to rawMessageQueue.
 */

static char rxBuffer[RAW_MESSAGE_MAX_LENGTH];
static size_t rxIndex = 0;

static void resetRxBuffer()
{
    rxIndex = 0;
    rxBuffer[0] = '\0';
}

static void submitRawMessage()
{
    if (rxIndex == 0)
    {
        return;
    }

    RawMessage message;
    snprintf(
        message.data,
        sizeof(message.data),
        "%s",
        rxBuffer
    );

    BaseType_t sendResult = xQueueSendToBack(
        rawMessageQueue,
        &message,
        PROTOCOL_LOG_WAIT_TICKS
    );

    if (sendResult == pdPASS)
    {
        char line[128];
        snprintf(
            line,
            sizeof(line),
            "[UART] Received line: %s",
            message.data
        );
        logLine(line, DEBUG_LOG_WAIT_TICKS);
    }
    else
    {
        logLine(
            "[UART] ERROR: rawMessageQueue full. Dropped RawMessage.",
            DEBUG_LOG_WAIT_TICKS
        );
    }

    resetRxBuffer();
}

static void processReceivedByte(char receivedChar)
{
    if (receivedChar == '\r')
    {
        return;
    }

    if (receivedChar == '\n')
    {
        submitRawMessage();
        return;
    }

    if (rxIndex >= (RAW_MESSAGE_MAX_LENGTH - 1))
    {
        logLine(
            "[UART] ERROR: RX line too long. Dropping line.",
            DEBUG_LOG_WAIT_TICKS
        );

        resetRxBuffer();
        return;
    }

    rxBuffer[rxIndex] = receivedChar;
    rxIndex++;
    rxBuffer[rxIndex] = '\0';
}

static void drainSerialInput()
{
    while (Serial.available() > 0)
    {
        char receivedChar = static_cast<char>(Serial.read());
        processReceivedByte(receivedChar);
    }
}

void TaskUARTReceive(void *pvParameters)
{
    (void)pvParameters;

    resetRxBuffer();

    logLine(
        "[UART] TaskUARTReceive started.",
        DEBUG_LOG_WAIT_TICKS
    );

    for (;;)
    {
        ulTaskNotifyTake(
            pdTRUE,
            UART_NOTIFY_WAIT_TICKS
        );

        drainSerialInput();

        /*
         * Safety net: if data arrives around callback registration or the
         * Arduino callback behavior changes, this prevents tight-loop churn.
         */
        vTaskDelay(UART_IDLE_DELAY_TICKS);
    }
}
