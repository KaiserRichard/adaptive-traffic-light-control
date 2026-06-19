#pragma once

#include <Arduino.h>
#include <freertos/FreeRTOS.h>

/*
 * Phase 15.13 - Final FreeRTOS Controller Configuration
 *
 * This file owns project-wide constants.
 * Do not put runtime state here.
 */

// -----------------------------------------------------------------------------
// Hardware pins
// -----------------------------------------------------------------------------

static const int A_RED_PIN = 25;
static const int A_YELLOW_PIN = 26;
static const int A_GREEN_PIN = 27;
static const int B_RED_PIN = 14;
static const int B_YELLOW_PIN = 12;
static const int B_GREEN_PIN = 13;

// -----------------------------------------------------------------------------
// Serial / protocol
// -----------------------------------------------------------------------------

static const uint32_t SERIAL_BAUD_RATE = 115200;
static const size_t RAW_MESSAGE_MAX_LENGTH = 96;

// -----------------------------------------------------------------------------
// Queues
// -----------------------------------------------------------------------------

static const UBaseType_t RAW_MESSAGE_QUEUE_LENGTH = 8;
static const UBaseType_t PLAN_QUEUE_LENGTH = 4;

// -----------------------------------------------------------------------------
// Task stack sizes
// -----------------------------------------------------------------------------

static const uint32_t UART_RECEIVE_TASK_STACK_SIZE = 4096;
static const uint32_t PLAN_PARSER_TASK_STACK_SIZE = 4096;
static const uint32_t TRAFFIC_FSM_TASK_STACK_SIZE = 4096;
static const uint32_t STATUS_REPORTER_TASK_STACK_SIZE = 3072;

// -----------------------------------------------------------------------------
// Task priorities
// -----------------------------------------------------------------------------

static const UBaseType_t UART_RECEIVE_TASK_PRIORITY = 3;
static const UBaseType_t PLAN_PARSER_TASK_PRIORITY = 2;
static const UBaseType_t TRAFFIC_FSM_TASK_PRIORITY = 2;
static const UBaseType_t STATUS_REPORTER_TASK_PRIORITY = 1;

// -----------------------------------------------------------------------------
// Timing
// -----------------------------------------------------------------------------

static const TickType_t UART_NOTIFY_WAIT_TICKS = portMAX_DELAY;
static const TickType_t UART_IDLE_DELAY_TICKS = pdMS_TO_TICKS(5);
static const TickType_t FSM_TICK_PERIOD_TICKS = pdMS_TO_TICKS(50);
static const TickType_t STATUS_TIMER_PERIOD_TICKS = pdMS_TO_TICKS(1000);
static const TickType_t DIAGNOSTICS_TIMER_PERIOD_TICKS = pdMS_TO_TICKS(5000);

// -----------------------------------------------------------------------------
// Logging
// -----------------------------------------------------------------------------

static const TickType_t PROTOCOL_LOG_WAIT_TICKS = pdMS_TO_TICKS(20);
static const TickType_t DEBUG_LOG_WAIT_TICKS = pdMS_TO_TICKS(20);

// -----------------------------------------------------------------------------
// Timing validation limits
// -----------------------------------------------------------------------------

static const int MIN_GREEN_SECONDS = 10;
static const int MAX_GREEN_SECONDS = 45;
static const int MIN_YELLOW_SECONDS = 3;
static const int MAX_YELLOW_SECONDS = 3;
static const int MIN_ALL_RED_SECONDS = 1;
static const int MAX_ALL_RED_SECONDS = 1;

// -----------------------------------------------------------------------------
// Host watchdog
// -----------------------------------------------------------------------------

static const uint32_t HOST_TIMEOUT_SECONDS = 30;
static const uint32_t HOST_TIMEOUT_MS = HOST_TIMEOUT_SECONDS * 1000;
