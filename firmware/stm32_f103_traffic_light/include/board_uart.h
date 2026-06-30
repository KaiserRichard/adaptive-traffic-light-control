#ifndef ATLC_STM32_BOARD_UART_H
#define ATLC_STM32_BOARD_UART_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int atlc_board_uart_init(uint32_t baud_rate);

int atlc_board_uart_send_line(const char *line);

int atlc_board_uart_poll_line(
    char *buffer,
    size_t buffer_size
);

const char *atlc_board_uart_status(void);

#ifdef __cplusplus
}
#endif

#endif
