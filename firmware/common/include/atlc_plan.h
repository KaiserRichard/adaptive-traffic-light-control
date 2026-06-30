#ifndef ATLC_PLAN_H
#define ATLC_PLAN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ATLC_MIN_GREEN_SECONDS 10
#define ATLC_MAX_GREEN_SECONDS 45
#define ATLC_MIN_YELLOW_SECONDS 3
#define ATLC_MAX_YELLOW_SECONDS 3
#define ATLC_MIN_ALL_RED_SECONDS 1
#define ATLC_MAX_ALL_RED_SECONDS 1

typedef struct
{
    int32_t plan_id;
    int32_t green_a;
    int32_t green_b;
    int32_t yellow;
    int32_t all_red;
} atlc_signal_plan_t;

typedef enum
{
    ATLC_PLAN_OK = 0,
    ATLC_PLAN_NULL,
    ATLC_PLAN_ID_OUT_OF_RANGE,
    ATLC_PLAN_GREEN_A_OUT_OF_RANGE,
    ATLC_PLAN_GREEN_B_OUT_OF_RANGE,
    ATLC_PLAN_YELLOW_OUT_OF_RANGE,
    ATLC_PLAN_ALL_RED_OUT_OF_RANGE
} atlc_plan_result_t;

atlc_signal_plan_t atlc_plan_default(void);

atlc_plan_result_t atlc_plan_validate(const atlc_signal_plan_t *plan);

const char *atlc_plan_result_to_string(atlc_plan_result_t result);

#ifdef __cplusplus
}
#endif

#endif
