// app_config.h
#pragma once

#include <Arduino.h>

// Hardware Pin Mapping
static const int A_RED_PIN = 25;
static const int A_YELLOW_PIN = 26;
static const int A_GREEN_PIN = 27;

static const int B_RED_PIN = 14;
static const int B_YELLOW_PIN = 12;
static const int B_GREEN_PIN = 13;

// Serial Configuration
static const uint32_t SERIAL_BAUD_RATE = 115200;

// Queue Configuration
static const UBaseType_t RAW_MESSAGE_QUEUE_LENGTH = 5;
static const UBaseType_t PLAN_QUEUE_LENGTH = 3;

// Maximum serial command length, including null terminator
static const size_t SERIAL_LINE_BUFFER_SIZE = 96;

// Task Timing
static const TickType_t UART_RECEIVE_POLL_TICK = pdMS_TO_TICKS(20);

// Task Stack Sizes
static const uint32_t UART_RECEIVE_TASK_STACK_SIZE = 4096;
static const uint32_t PARSER_TASK_STACK_SIZE = 4096;
static const uint32_t FSM_TASK_STACK_SIZE = 4096;

// Task Priorities
static const UBaseType_t UART_RECEIVE_TASK_PRIORITY = 1;
static const UBaseType_t PARSER_TASK_PRIORITY = 1;
static const UBaseType_t FSM_TASK_PRIORITY = 1;

// Timing Validation Limits
static const int MIN_GREEN_SECONDS = 10;
static const int MAX_GREEN_SECONDS = 45;

static const int MIN_YELLOW_SECONDS = 3;
static const int MAX_YELLOW_SECONDS = 3;

static const int MIN_ALL_RED_SECONDS = 1;
static const int MAX_ALL_RED_SECONDS = 1;