// task_uart_receive.cpp
/*
 * TaskUARTReceive: 
 * Reads command lines from USB Serial and sends them to rawMessageQueue.
 */
#include <Arduino.h>
#include "app_config.h"
#include "messages.h"
#include "queues.h"
#include "protocol.h"
#include "tasks.h"

void TaskUARTReceive(void *pvParameters)
{
    (void)pvParameters;

    char lineBuffer[SERIAL_LINE_BUFFER_SIZE];
    size_t lineIndex = 0;
    for (;;)
    {
        while(Serial.available() > 0) // Check if characters exist
        {
            char receivedChar = static_cast<char>(Serial.read());

            if (receivedChar == '\r') 
            {
                continue;
            }

            if (receivedChar == '\n') // New line means: User pressed Enter or Command complete.
            {
                // Check Empty line
                if (lineIndex > 0)
                {
                    lineBuffer[lineIndex] = '\0'; // Without \0, C does know where the string ends
                    RawMessage message;
                    setRawMessage(&message, lineBuffer);

                    // Send RawMessage to parser task
                    BaseType_t sendResult = xQueueSendToBack(
                        rawMessageQueue,
                        &message,
                        portMAX_DELAY
                    );

                        if (sendResult == pdPASS)
                        {
                            Serial.print("[UART] Received line: ");
                            Serial.println(message.data);
                        }
                        else
                        {
                            Serial.println("[UART] ERROR: Failed to send raw message.");
                        }

                        lineIndex = 0;
                }
            }
            else 
            {
                if (lineIndex < SERIAL_LINE_BUFFER_SIZE -1) 
                {
                    lineBuffer[lineIndex] = receivedChar;
                    lineIndex++;
                }
                else
                {
                    Serial.println("[UART] ERROR: Serial line too long. Dropping line.");
                    lineIndex = 0;
                }
            }
        }
        vTaskDelay(UART_RECEIVE_POLL_TICK);
    }
}