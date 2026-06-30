#include "atlc_plan.h"

atlc_signal_plan_t atlc_plan_default(void)
{
    atlc_signal_plan_t plan;
    plan.plan_id = 0;
    plan.green_a = 20;
    plan.green_b = 20;
    plan.yellow = 3;
    plan.all_red = 1;
    return plan;
}

atlc_plan_result_t atlc_plan_validate(const atlc_signal_plan_t *plan)
{
    if (plan == 0)
    {
        return ATLC_PLAN_NULL;
    }

    if (plan->plan_id < 0)
    {
        return ATLC_PLAN_ID_OUT_OF_RANGE;
    }

    if (plan->green_a < ATLC_MIN_GREEN_SECONDS ||
        plan->green_a > ATLC_MAX_GREEN_SECONDS)
    {
        return ATLC_PLAN_GREEN_A_OUT_OF_RANGE;
    }

    if (plan->green_b < ATLC_MIN_GREEN_SECONDS ||
        plan->green_b > ATLC_MAX_GREEN_SECONDS)
    {
        return ATLC_PLAN_GREEN_B_OUT_OF_RANGE;
    }

    if (plan->yellow < ATLC_MIN_YELLOW_SECONDS ||
        plan->yellow > ATLC_MAX_YELLOW_SECONDS)
    {
        return ATLC_PLAN_YELLOW_OUT_OF_RANGE;
    }

    if (plan->all_red < ATLC_MIN_ALL_RED_SECONDS ||
        plan->all_red > ATLC_MAX_ALL_RED_SECONDS)
    {
        return ATLC_PLAN_ALL_RED_OUT_OF_RANGE;
    }

    return ATLC_PLAN_OK;
}

const char *atlc_plan_result_to_string(atlc_plan_result_t result)
{
    switch (result)
    {
        case ATLC_PLAN_OK:
            return "OK";
        case ATLC_PLAN_NULL:
            return "NULL_PLAN";
        case ATLC_PLAN_ID_OUT_OF_RANGE:
            return "PLAN_ID_OUT_OF_RANGE";
        case ATLC_PLAN_GREEN_A_OUT_OF_RANGE:
            return "GREEN_A_OUT_OF_RANGE";
        case ATLC_PLAN_GREEN_B_OUT_OF_RANGE:
            return "GREEN_B_OUT_OF_RANGE";
        case ATLC_PLAN_YELLOW_OUT_OF_RANGE:
            return "YELLOW_OUT_OF_RANGE";
        case ATLC_PLAN_ALL_RED_OUT_OF_RANGE:
            return "ALL_RED_OUT_OF_RANGE";
        default:
            return "UNKNOWN_PLAN_RESULT";
    }
}
