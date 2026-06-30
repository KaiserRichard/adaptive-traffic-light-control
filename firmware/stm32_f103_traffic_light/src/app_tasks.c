#ifdef ATLC_STM32_HARDWARE_BUILD
#error "STM32 FreeRTOS task wiring is pending toolchain and hardware validation."
#endif

#include "app_tasks.h"

/*
 * Not validated on STM32 hardware yet.
 * Requires future FreeRTOS port configuration, CMSIS/HAL/LL decision, startup
 * file, linker script, and toolchain before scheduler startup can exist.
 */

int atlc_app_tasks_init(void)
{
    return -1;
}

int atlc_app_tasks_start_scheduler(void)
{
    return -1;
}

const char *atlc_app_tasks_status(void)
{
    return "STM32 FreeRTOS task wiring not implemented; hardware validation pending";
}
