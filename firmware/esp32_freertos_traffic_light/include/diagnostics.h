// diagnostic.h
#include <Arduino.h>
#pragma once

/*
 * Phase 15.10:
 * Runtime diagnostics module.
 *
 * This module periodically reports:
 * - free heap memory
 * - remaining stack space for important tasks
 *
 * Output format:
 * DIAG,heap=...,uart_stack=...,parser_stack=...,fsm_stack=...
 */

void initDiagnosticsReporter(
    TaskHandle_t uartTaskHandle,
    TaskHandle_t parserTaskHandle,
    TaskHandle_t fsmTaskHandle
);

void startDiagnosticsTimer();
