#ifdef ATLC_STM32_HARDWARE_BUILD
#error "UART hardware implementation is pending CMSIS/HAL/LL and PCB validation."
#endif

#include "board_uart.h"

/*
 * Not validated on STM32 hardware yet.
 * Requires future CMSIS/HAL/LL selection, startup file, linker script, and
 * toolchain before this can configure USART1.
 */

int atlc_board_uart_init(uint32_t baud_rate)
{
    (void)baud_rate;
    return -1;
}

int atlc_board_uart_send_line(const char *line)
{
    (void)line;
    return -1;
}

int atlc_board_uart_poll_line(
    char *buffer,
    size_t buffer_size
)
{
    (void)buffer;
    (void)buffer_size;
    return -1;
}

const char *atlc_board_uart_status(void)
{
    return "STM32 UART not implemented; hardware validation pending";
}
