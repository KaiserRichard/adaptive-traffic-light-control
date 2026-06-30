#ifdef ATLC_STM32_HARDWARE_BUILD
#error "STM32 hardware build is not implemented yet. Select CMSIS/HAL/LL, startup, linker, and toolchain first."
#endif

#include "stm32_port_status.h"

/*
 * Host-safe STM32 scaffold entry point.
 *
 * Not validated on STM32 hardware yet.
 * Requires future CMSIS/HAL/LL selection, startup file, linker script, and
 * toolchain before this can become real firmware.
 */
int main(void)
{
    atlc_stm32_port_status_t status = atlc_stm32_get_port_status();

    (void)status;

    return 0;
}
