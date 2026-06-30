#ifndef ATLC_STM32_PORT_STATUS_H
#define ATLC_STM32_PORT_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    const char *target_mcu;
    const char *target_core;
    int power_validated;
    int stlink_validated;
    int gpio_validated;
    int uart_validated;
    int seven_segment_validated;
    int buildable_firmware_created;
    int flashable_binary_created;
} atlc_stm32_port_status_t;

atlc_stm32_port_status_t atlc_stm32_get_port_status(void);

const char *atlc_stm32_port_status_summary(void);

#ifdef __cplusplus
}
#endif

#endif
