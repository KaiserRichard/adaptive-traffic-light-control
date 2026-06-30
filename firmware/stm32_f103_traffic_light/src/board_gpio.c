#ifdef ATLC_STM32_HARDWARE_BUILD
#error "GPIO hardware implementation is pending CMSIS/HAL/LL and PCB validation."
#endif

#include "board_gpio.h"

/*
 * Not validated on STM32 hardware yet.
 * Requires future CMSIS/HAL/LL selection, startup file, linker script, and
 * toolchain before this can drive real GPIO.
 */

int atlc_board_gpio_init(void)
{
    return -1;
}

int atlc_board_gpio_set_output(
    atlc_board_gpio_output_t output,
    bool enabled
)
{
    (void)output;
    (void)enabled;
    return -1;
}

const char *atlc_board_gpio_status(void)
{
    return "STM32 GPIO not implemented; hardware validation pending";
}
