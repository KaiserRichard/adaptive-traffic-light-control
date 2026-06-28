#include <Arduino.h>

#include "config/app_config.h"
#include "messages/messages.h"
#include "protocol/protocol.h"
#include "services/logging.h"

void setRawMessage(RawMessage *message, const char *text)
{
    if (message == nullptr || text == nullptr)
    {
        return;
    }

    snprintf(
        message->data,
        sizeof(message->data),
        "%s",
        text
    );
}

bool isPlanCommand(const RawMessage *message)
{
    if (message == nullptr)
    {
        return false;
    }

    return strncmp(message->data, "PLAN,", 5) == 0;
}

bool parsePlanCommand(
    const RawMessage *message,
    ParsedPlanFields *fields
)
{
    if (message == nullptr || fields == nullptr)
    {
        return false;
    }

    int parsedCount = sscanf(
        message->data,
        "PLAN,%d,%d,%d,%d,%d",
        &fields->plan_id,
        &fields->green_a,
        &fields->green_b,
        &fields->yellow,
        &fields->all_red
    );

    return parsedCount == 5;
}

void printParsedPlan(const ParsedPlanFields *fields)
{
    if (fields == nullptr)
    {
        return;
    }

    char line[128];
    snprintf(
        line,
        sizeof(line),
        "[PARSER] plan_id=%d green_a=%d green_b=%d yellow=%d all_red=%d",
        fields->plan_id,
        fields->green_a,
        fields->green_b,
        fields->yellow,
        fields->all_red
    );

    logLine(line, DEBUG_LOG_WAIT_TICKS);
}

SignalPlan makeSignalPlan(const ParsedPlanFields *fields)
{
    SignalPlan plan;

    if (fields == nullptr)
    {
        plan.plan_id = -1;
        plan.green_a = 0;
        plan.green_b = 0;
        plan.yellow = 0;
        plan.all_red = 0;
        return plan;
    }

    plan.plan_id = fields->plan_id;
    plan.green_a = fields->green_a;
    plan.green_b = fields->green_b;
    plan.yellow = fields->yellow;
    plan.all_red = fields->all_red;

    return plan;
}

bool validateSignalPlan(
    const SignalPlan *plan,
    const char **reason
)
{
    if (reason != nullptr)
    {
        *reason = "OK";
    }

    if (plan == nullptr)
    {
        if (reason != nullptr)
        {
            *reason = "NULL_PLAN";
        }
        return false;
    }

    if(plan->plan_id < 0)
    {
        if (reason != nullptr)
        {
            *reason = "PLAN_ID_OUT_OF_RANGE";
        }
        return false;
    }

    if (plan->green_a < MIN_GREEN_SECONDS ||
        plan->green_a > MAX_GREEN_SECONDS)
    {
        if (reason != nullptr)
        {
            *reason = "GREEN_A_OUT_OF_RANGE";
        }
        return false;
    }

    if (plan->green_b < MIN_GREEN_SECONDS ||
        plan->green_b > MAX_GREEN_SECONDS)
    {
        if (reason != nullptr)
        {
            *reason = "GREEN_B_OUT_OF_RANGE";
        }
        return false;
    }

    if (plan->yellow < MIN_YELLOW_SECONDS ||
        plan->yellow > MAX_YELLOW_SECONDS)
    {
        if (reason != nullptr)
        {
            *reason = "YELLOW_OUT_OF_RANGE";
        }
        return false;
    }

    if (plan->all_red < MIN_ALL_RED_SECONDS ||
        plan->all_red > MAX_ALL_RED_SECONDS)
    {
        if (reason != nullptr)
        {
            *reason = "ALL_RED_OUT_OF_RANGE";
        }
        return false;
    }

    return true;
}

void printSignalPlan(const SignalPlan *plan)
{
    if (plan == nullptr)
    {
        return;
    }

    char line[128];
    snprintf(
        line,
        sizeof(line),
        "[PLAN] plan_id=%d green_a=%d green_b=%d yellow=%d all_red=%d",
        plan->plan_id,
        plan->green_a,
        plan->green_b,
        plan->yellow,
        plan->all_red
    );

    logLine(line, DEBUG_LOG_WAIT_TICKS);
}

void sendAck(int planId)
{
    // We use a bounded wait instead of portMAX_DELAY
    // So a Serial fault or programming error cannot block this task forever.
    if (lockSerial(pdMS_TO_TICKS(20)))
    {
        Serial.print("ACK,");
        Serial.println(planId);

        // ACK is protocol-critical
        // The whole line must be printed under one mutex lock.
        unlockSerial();
    }
}

void sendNack(int planId, const char *reason)
{
    if(lockSerial(pdMS_TO_TICKS(20)))
    {
        Serial.print("NACK,");
        Serial.print(planId);
        Serial.print(",");

        if (reason == nullptr)
        {
            Serial.println("UNKNOWN_REASON");
            unlockSerial();
            return;
        }

        Serial.println(reason);
        unlockSerial();
    }
}
