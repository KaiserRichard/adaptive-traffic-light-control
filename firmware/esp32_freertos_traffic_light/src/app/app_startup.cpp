#include <Arduino.h>

#include "app/app_startup.h"

void printBootBanner()
{
    Serial.println();
    Serial.println("=================================================");
    Serial.println("[BOOT] ATLC Phase 15 FreeRTOS Controller");
    Serial.println("[BOOT] Phase 15.13 - UART RX Event + Final Refactor");
    Serial.println("=================================================");
}

void haltSystem(const char *reason)
{
    Serial.println("[BOOT] FATAL ERROR");

    if (reason != nullptr)
    {
        Serial.print("[BOOT] Reason: ");
        Serial.println(reason);
    }

    Serial.println("[BOOT] System halted.");

    while (true)
    {
        delay(1000);
    }
}
