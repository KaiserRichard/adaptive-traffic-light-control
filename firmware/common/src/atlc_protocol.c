#include "atlc_protocol.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int starts_with(const char *text, const char *prefix)
{
    return strncmp(text, prefix, strlen(prefix)) == 0;
}

static void copy_trimmed_line(
    const char *line,
    char *out,
    size_t out_size
)
{
    size_t length = strlen(line);

    while (length > 0 &&
           (line[length - 1] == '\n' ||
            line[length - 1] == '\r' ||
            line[length - 1] == ' ' ||
            line[length - 1] == '\t'))
    {
        length--;
    }

    if (length >= out_size)
    {
        length = out_size - 1;
    }

    memcpy(out, line, length);
    out[length] = '\0';
}

static atlc_parse_result_t parse_int_field(
    const char **cursor,
    int is_last,
    int *out_value
)
{
    char *end = 0;
    long value = 0;

    if (cursor == 0 || *cursor == 0 || out_value == 0)
    {
        return ATLC_PARSE_NULL;
    }

    if (**cursor == '\0' || **cursor == ',')
    {
        return ATLC_PARSE_MISSING_FIELD;
    }

    errno = 0;
    value = strtol(*cursor, &end, 10);

    if (end == *cursor || errno != 0)
    {
        return ATLC_PARSE_NON_NUMERIC_FIELD;
    }

    if (!is_last)
    {
        if (*end == '\0')
        {
            return ATLC_PARSE_MISSING_FIELD;
        }

        if (*end != ',')
        {
            return ATLC_PARSE_NON_NUMERIC_FIELD;
        }

        *cursor = end + 1;
    }
    else
    {
        if (*end == ',')
        {
            return ATLC_PARSE_EXTRA_FIELD;
        }

        if (*end != '\0')
        {
            return ATLC_PARSE_NON_NUMERIC_FIELD;
        }

        *cursor = end;
    }

    *out_value = (int)value;
    return ATLC_PARSE_OK;
}

atlc_parse_result_t atlc_protocol_parse_plan(
    const char *line,
    atlc_signal_plan_t *out_plan
)
{
    char buffer[ATLC_PROTOCOL_MAX_LINE_LENGTH];
    const char *cursor = 0;
    int values[5] = {0};
    size_t line_length = 0;

    if (line == 0 || out_plan == 0)
    {
        return ATLC_PARSE_NULL;
    }

    line_length = strlen(line);
    if (line_length >= ATLC_PROTOCOL_MAX_LINE_LENGTH)
    {
        return ATLC_PARSE_LINE_TOO_LONG;
    }

    copy_trimmed_line(line, buffer, sizeof(buffer));

    if (strstr(buffer, "seq=") != 0 ||
        strstr(buffer, "ns_green=") != 0 ||
        strstr(buffer, "ew_green=") != 0)
    {
        return ATLC_PARSE_OLD_FORMAT_REJECTED;
    }

    if (!starts_with(buffer, "PLAN,"))
    {
        return ATLC_PARSE_NOT_PLAN;
    }

    cursor = buffer + strlen("PLAN,");

    for (int i = 0; i < 5; i++)
    {
        atlc_parse_result_t result =
            parse_int_field(&cursor, i == 4, &values[i]);

        if (result != ATLC_PARSE_OK)
        {
            return result;
        }
    }

    out_plan->plan_id = values[0];
    out_plan->green_a = values[1];
    out_plan->green_b = values[2];
    out_plan->yellow = values[3];
    out_plan->all_red = values[4];

    return ATLC_PARSE_OK;
}

const char *atlc_parse_result_to_string(atlc_parse_result_t result)
{
    switch (result)
    {
        case ATLC_PARSE_OK:
            return "OK";
        case ATLC_PARSE_NULL:
            return "NULL_ARGUMENT";
        case ATLC_PARSE_NOT_PLAN:
            return "UNKNOWN_COMMAND";
        case ATLC_PARSE_OLD_FORMAT_REJECTED:
            return "OLD_PLAN_FORMAT_REJECTED";
        case ATLC_PARSE_MISSING_FIELD:
            return "MALFORMED_PLAN";
        case ATLC_PARSE_EXTRA_FIELD:
            return "MALFORMED_PLAN";
        case ATLC_PARSE_NON_NUMERIC_FIELD:
            return "MALFORMED_PLAN";
        case ATLC_PARSE_LINE_TOO_LONG:
            return "LINE_TOO_LONG";
        default:
            return "UNKNOWN_PARSE_RESULT";
    }
}

int atlc_protocol_format_ack(
    int plan_id,
    char *out,
    size_t out_size
)
{
    if (out == 0 || out_size == 0)
    {
        return -1;
    }

    return snprintf(out, out_size, "ACK,%d", plan_id);
}

int atlc_protocol_format_nack(
    int plan_id,
    const char *reason,
    char *out,
    size_t out_size
)
{
    const char *safe_reason = reason;

    if (out == 0 || out_size == 0)
    {
        return -1;
    }

    if (safe_reason == 0)
    {
        safe_reason = "UNKNOWN_REASON";
    }

    return snprintf(out, out_size, "NACK,%d,%s", plan_id, safe_reason);
}
