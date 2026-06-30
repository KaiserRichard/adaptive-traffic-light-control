#include "atlc_fsm.h"

void atlc_fsm_init(
    atlc_fsm_t *fsm,
    const atlc_signal_plan_t *initial_plan
)
{
    if (fsm == 0)
    {
        return;
    }

    if (initial_plan != 0)
    {
        fsm->active_plan = *initial_plan;
    }
    else
    {
        fsm->active_plan = atlc_plan_default();
    }

    fsm->pending_plan = fsm->active_plan;
    fsm->has_pending_plan = false;
    fsm->state = ATLC_STATE_A_GREEN;
}

const char *atlc_fsm_state_to_string(atlc_traffic_state_t state)
{
    switch (state)
    {
        case ATLC_STATE_A_GREEN:
            return "A_GREEN";
        case ATLC_STATE_A_YELLOW:
            return "A_YELLOW";
        case ATLC_STATE_ALL_RED_AFTER_A:
            return "ALL_RED_AFTER_A";
        case ATLC_STATE_B_GREEN:
            return "B_GREEN";
        case ATLC_STATE_B_YELLOW:
            return "B_YELLOW";
        case ATLC_STATE_ALL_RED_AFTER_B:
            return "ALL_RED_AFTER_B";
        default:
            return "UNKNOWN";
    }
}

atlc_traffic_state_t atlc_fsm_next_state(atlc_traffic_state_t state)
{
    switch (state)
    {
        case ATLC_STATE_A_GREEN:
            return ATLC_STATE_A_YELLOW;
        case ATLC_STATE_A_YELLOW:
            return ATLC_STATE_ALL_RED_AFTER_A;
        case ATLC_STATE_ALL_RED_AFTER_A:
            return ATLC_STATE_B_GREEN;
        case ATLC_STATE_B_GREEN:
            return ATLC_STATE_B_YELLOW;
        case ATLC_STATE_B_YELLOW:
            return ATLC_STATE_ALL_RED_AFTER_B;
        case ATLC_STATE_ALL_RED_AFTER_B:
            return ATLC_STATE_A_GREEN;
        default:
            return ATLC_STATE_A_GREEN;
    }
}

uint32_t atlc_fsm_state_duration_seconds(
    atlc_traffic_state_t state,
    const atlc_signal_plan_t *active_plan
)
{
    if (active_plan == 0)
    {
        return 1U;
    }

    switch (state)
    {
        case ATLC_STATE_A_GREEN:
            return (uint32_t)active_plan->green_a;
        case ATLC_STATE_A_YELLOW:
            return (uint32_t)active_plan->yellow;
        case ATLC_STATE_ALL_RED_AFTER_A:
            return (uint32_t)active_plan->all_red;
        case ATLC_STATE_B_GREEN:
            return (uint32_t)active_plan->green_b;
        case ATLC_STATE_B_YELLOW:
            return (uint32_t)active_plan->yellow;
        case ATLC_STATE_ALL_RED_AFTER_B:
            return (uint32_t)active_plan->all_red;
        default:
            return 1U;
    }
}

void atlc_fsm_set_pending_plan(
    atlc_fsm_t *fsm,
    const atlc_signal_plan_t *plan
)
{
    if (fsm == 0 || plan == 0)
    {
        return;
    }

    fsm->pending_plan = *plan;
    fsm->has_pending_plan = true;
}

bool atlc_fsm_advance_state(atlc_fsm_t *fsm)
{
    bool applied_pending_plan = false;

    if (fsm == 0)
    {
        return false;
    }

    fsm->state = atlc_fsm_next_state(fsm->state);

    if (fsm->state == ATLC_STATE_A_GREEN && fsm->has_pending_plan)
    {
        fsm->active_plan = fsm->pending_plan;
        fsm->has_pending_plan = false;
        applied_pending_plan = true;
    }

    return applied_pending_plan;
}
