#include "atlc_status.h"

#include <stdio.h>

const char *atlc_health_to_string(atlc_health_t health)
{
    switch (health)
    {
        case ATLC_HEALTH_OK:
            return "OK";
        case ATLC_HEALTH_HOST_TIMEOUT:
            return "HOST_TIMEOUT";
        case ATLC_HEALTH_FAULT:
            return "FAULT";
        default:
            return "UNKNOWN_HEALTH";
    }
}

int atlc_status_format(
    const atlc_controller_status_t *status,
    char *out,
    size_t out_size
)
{
    if (status == 0 || out == 0 || out_size == 0)
    {
        return -1;
    }

    return snprintf(
        out,
        out_size,
        "STATUS,%ld,%s,%lu,%s",
        (long)status->plan_id,
        atlc_fsm_state_to_string(status->state),
        (unsigned long)status->remaining_seconds,
        atlc_health_to_string(status->health)
    );
}
