// protocol.cpp 
#include "protocol.h"
#include "app_config.h"

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

    Serial.print("[PARSER] plan_id=");
    Serial.print(fields->plan_id);
    Serial.print(" green_a=");
    Serial.print(fields->green_a);
    Serial.print(" green_b=");
    Serial.print(fields->green_b);
    Serial.print(" yellow=");
    Serial.print(fields->yellow);
    Serial.print(" all_red=");
    Serial.println(fields->all_red);
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

    Serial.print("[PLAN] plan_id=");
    Serial.print(plan->plan_id);
    Serial.print(" green_a=");
    Serial.print(plan->green_a);
    Serial.print(" green_b=");
    Serial.print(plan->green_b);
    Serial.print(" yellow=");
    Serial.print(plan->yellow);
    Serial.print(" all_red=");
    Serial.println(plan->all_red);
}

void sendAck(int planId)
{
    Serial.print("ACK,");
    Serial.println(planId);
}

void sendNack(int planId, const char *reason)
{
    Serial.print("NACK,");
    Serial.print(planId);
    Serial.print(",");

    if (reason == nullptr)
    {
        Serial.println("UNKNONW_REASON");
        return;
    }

    Serial.println(reason);
}

// 