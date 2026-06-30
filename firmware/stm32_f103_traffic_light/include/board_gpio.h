#ifndef ATLC_STM32_BOARD_GPIO_H
#define ATLC_STM32_BOARD_GPIO_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    ATLC_BOARD_GPIO_STATUS_LED = 0,
    ATLC_BOARD_GPIO_TRAFFIC_LED_0,
    ATLC_BOARD_GPIO_TRAFFIC_LED_1,
    ATLC_BOARD_GPIO_TRAFFIC_LED_2,
    ATLC_BOARD_GPIO_TRAFFIC_LED_3,
    ATLC_BOARD_GPIO_TRAFFIC_LED_4,
    ATLC_BOARD_GPIO_TRAFFIC_LED_5
} atlc_board_gpio_output_t;

int atlc_board_gpio_init(void);

int atlc_board_gpio_set_output(
    atlc_board_gpio_output_t output,
    bool enabled
);

const char *atlc_board_gpio_status(void);

#ifdef __cplusplus
}
#endif

#endif
