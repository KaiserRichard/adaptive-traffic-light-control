#ifdef ATLC_STM32_HARDWARE_BUILD
#error "STM32 hardware port status must be reviewed before a real hardware build."
#endif

#include "board_config.h"
#include "stm32_port_status.h"

/*
 * Not validated on STM32 hardware yet.
 * This file exposes honest compile-time status for the scaffold.
 */

atlc_stm32_port_status_t atlc_stm32_get_port_status(void)
{
    atlc_stm32_port_status_t status;

    status.target_mcu = ATLC_STM32_TARGET_MCU;
    status.target_core = ATLC_STM32_TARGET_CORE;
    status.power_validated = ATLC_STM32_POWER_VALIDATED;
    status.stlink_validated = ATLC_STM32_STLINK_VALIDATED;
    status.gpio_validated = ATLC_STM32_GPIO_VALIDATED;
    status.uart_validated = ATLC_STM32_UART_VALIDATED;
    status.seven_segment_validated = ATLC_STM32_SEVEN_SEGMENT_VALIDATED;
    status.buildable_firmware_created = 0;
    status.flashable_binary_created = 0;

    return status;
}

const char *atlc_stm32_port_status_summary(void)
{
    return "STM32 scaffold only: no buildable firmware, flashing, or hardware validation";
}
