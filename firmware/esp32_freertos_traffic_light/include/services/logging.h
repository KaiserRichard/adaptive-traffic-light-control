#pragma once

#include <Arduino.h>

// Init serial logging mutex.
bool initLogging();

// Lock Serial before printing a multi-part line.
bool lockSerial(TickType_t ticksToWait);

// Release Serial after printing a multi-part line.
void unlockSerial();

// Print one complete line with mutex protection.
bool logLine(const char *message, TickType_t ticksToWait);

/*
 * Best-effort one-line logging.
 *
 * Suitable for timer callbacks or low-importance debug output.
 * If Serial is busy, the line is skipped.
 */
bool tryLogLine(const char *message);
