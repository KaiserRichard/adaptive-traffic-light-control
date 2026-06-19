#pragma once

#include <Arduino.h>

#include "messages/messages.h"

void setRawMessage(RawMessage *message, const char *text);

bool isPlanCommand(const RawMessage *message);

bool parsePlanCommand(
    const RawMessage *message,
    ParsedPlanFields *fields
);

void printParsedPlan(const ParsedPlanFields *fields);

SignalPlan makeSignalPlan(const ParsedPlanFields *fields);

bool validateSignalPlan(
    const SignalPlan *plan,
    const char **reason
);

void printSignalPlan(const SignalPlan *plan);

// Phase 15.6: ACK/NACK help logic that the parser task should call
void sendAck(int planId);
void sendNack(
    int planId,
    const char *reason
);
