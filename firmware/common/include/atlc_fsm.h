#ifndef ATLC_FSM_H
#define ATLC_FSM_H

#include <stdbool.h>
#include <stdint.h>

#include "atlc_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    ATLC_STATE_A_GREEN = 0,
    ATLC_STATE_A_YELLOW,
    ATLC_STATE_ALL_RED_AFTER_A,
    ATLC_STATE_B_GREEN,
    ATLC_STATE_B_YELLOW,
    ATLC_STATE_ALL_RED_AFTER_B
} atlc_traffic_state_t;

typedef struct
{
    atlc_signal_plan_t active_plan;
    atlc_signal_plan_t pending_plan;
    bool has_pending_plan;
    atlc_traffic_state_t state;
} atlc_fsm_t;

void atlc_fsm_init(
    atlc_fsm_t *fsm,
    const atlc_signal_plan_t *initial_plan
);

const char *atlc_fsm_state_to_string(atlc_traffic_state_t state);

atlc_traffic_state_t atlc_fsm_next_state(atlc_traffic_state_t state);

uint32_t atlc_fsm_state_duration_seconds(
    atlc_traffic_state_t state,
    const atlc_signal_plan_t *active_plan
);

void atlc_fsm_set_pending_plan(
    atlc_fsm_t *fsm,
    const atlc_signal_plan_t *plan
);

bool atlc_fsm_advance_state(atlc_fsm_t *fsm);

#ifdef __cplusplus
}
#endif

#endif
