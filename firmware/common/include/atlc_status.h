#ifndef ATLC_STATUS_H
#define ATLC_STATUS_H

#include <stddef.h>
#include <stdint.h>

#include "atlc_fsm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    ATLC_HEALTH_OK = 0,
    ATLC_HEALTH_HOST_TIMEOUT,
    ATLC_HEALTH_FAULT
} atlc_health_t;

typedef struct
{
    int32_t plan_id;
    atlc_traffic_state_t state;
    uint32_t remaining_seconds;
    atlc_health_t health;
} atlc_controller_status_t;

const char *atlc_health_to_string(atlc_health_t health);

int atlc_status_format(
    const atlc_controller_status_t *status,
    char *out,
    size_t out_size
);

#ifdef __cplusplus
}
#endif

#endif
