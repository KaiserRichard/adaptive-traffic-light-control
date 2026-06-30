#ifndef ATLC_PROTOCOL_H
#define ATLC_PROTOCOL_H

#include <stddef.h>

#include "atlc_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ATLC_PROTOCOL_MAX_LINE_LENGTH 96
#define ATLC_PROTOCOL_MAX_RESPONSE_LENGTH 96

typedef enum
{
    ATLC_PARSE_OK = 0,
    ATLC_PARSE_NULL,
    ATLC_PARSE_NOT_PLAN,
    ATLC_PARSE_OLD_FORMAT_REJECTED,
    ATLC_PARSE_MISSING_FIELD,
    ATLC_PARSE_EXTRA_FIELD,
    ATLC_PARSE_NON_NUMERIC_FIELD,
    ATLC_PARSE_LINE_TOO_LONG
} atlc_parse_result_t;

atlc_parse_result_t atlc_protocol_parse_plan(
    const char *line,
    atlc_signal_plan_t *out_plan
);

const char *atlc_parse_result_to_string(atlc_parse_result_t result);

int atlc_protocol_format_ack(
    int plan_id,
    char *out,
    size_t out_size
);

int atlc_protocol_format_nack(
    int plan_id,
    const char *reason,
    char *out,
    size_t out_size
);

#ifdef __cplusplus
}
#endif

#endif
