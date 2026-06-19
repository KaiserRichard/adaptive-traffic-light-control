#include <Arduino.h>

#include "services/logging.h"

// Serial mutex is private to this module.
static SemaphoreHandle_t serialMutex = nullptr;

// Create the FreeRTOS mutex used to protect Serial output.
bool initLogging()
{
    serialMutex = xSemaphoreCreateMutex();

    if (serialMutex == nullptr)
    {
        // Logging is not initialized yet, so direct Serial output is acceptable here.
        Serial.println("[LOGGING] ERROR: Failed to create Serial mutex.");
        return false;
    }

    // Serial mutex was created successfully
    Serial.println("[LOGGING] Serial mutex initialized.");
    return true;
}

// Attemp to obtain exclusive access to Serial TX.
bool lockSerial(TickType_t ticksToWait)
{
    if (serialMutex == nullptr)
    {
        // Logging was not initialized.
        return false;
    }

    BaseType_t result = xSemaphoreTake(
        serialMutex,
        ticksToWait
    );

    return result == pdTRUE;
}

// Release the Serial mutex after a protected print operation.
void unlockSerial()
{
    if(serialMutex == nullptr)
    {
        // Nothing to release.
        // This guard prevents accidental xSemaphoreGive(nullptr).
        return;
    }
    xSemaphoreGive(serialMutex);
}

// Print one complete line with mutex protection.
bool logLine(const char *message, TickType_t ticksToWait)
{
    if(message == nullptr)
    {
        // Nothing valid to print.
        return false;
    }
    // If we cannot lock Serial, abort this log attemp.
    bool success = lockSerial(ticksToWait);
    if (success == false)
    {
        return false;
    }

    // At this point, this task owns the Serial mutex.
    Serial.println(message);

    // Release Serial so other tasks can print.
    unlockSerial();
    return true;
}

// Best-effort one-line logging trying to print immediately.
//  If Serial is busy, skip the message.
bool tryLogLine(const char *message)
{
    return logLine(message, 0);
}
