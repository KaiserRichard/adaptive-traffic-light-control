#ifndef ATLC_STM32_BOARD_CONFIG_H
#define ATLC_STM32_BOARD_CONFIG_H

/*
 * STM32F103C8T6 board configuration scaffold.
 *
 * Not validated on STM32 hardware yet.
 * Requires future CMSIS/HAL/LL selection, startup file, linker script, and
 * toolchain before this can become a real hardware build.
 */

#define ATLC_STM32_TARGET_MCU "STM32F103C8T6"
#define ATLC_STM32_TARGET_CORE "ARM Cortex-M3"
#define ATLC_STM32_UART_BAUD_RATE 115200

#define ATLC_STM32_POWER_VALIDATED 0
#define ATLC_STM32_STLINK_VALIDATED 0
#define ATLC_STM32_GPIO_VALIDATED 0
#define ATLC_STM32_UART_VALIDATED 0
#define ATLC_STM32_SEVEN_SEGMENT_VALIDATED 0

/*
 * Schematic-observed pins only. These are not hardware-validated.
 * Do not use as final firmware ownership until the PCB is measured.
 */
#define ATLC_STM32_STATUS_LED_PIN_TEXT "PC13"
#define ATLC_STM32_USART1_TX_PIN_TEXT "PA9"
#define ATLC_STM32_USART1_RX_PIN_TEXT "PA10"
#define ATLC_STM32_SWDIO_PIN_TEXT "PA13"
#define ATLC_STM32_SWCLK_PIN_TEXT "PA14"

#define ATLC_STM32_TRAFFIC_LED_0_PIN_TEXT "PB9"
#define ATLC_STM32_TRAFFIC_LED_1_PIN_TEXT "PB8"
#define ATLC_STM32_TRAFFIC_LED_2_PIN_TEXT "PB7"
#define ATLC_STM32_TRAFFIC_LED_3_PIN_TEXT "PB6"
#define ATLC_STM32_TRAFFIC_LED_4_PIN_TEXT "PB5"
#define ATLC_STM32_TRAFFIC_LED_5_PIN_TEXT "PB3"

#define ATLC_STM32_SEG_A_PIN_TEXT "PA8"
#define ATLC_STM32_SEG_B_PIN_TEXT "PB15"
#define ATLC_STM32_SEG_C_PIN_TEXT "PB14"
#define ATLC_STM32_SEG_D_PIN_TEXT "PB13"
#define ATLC_STM32_SEG_E_PIN_TEXT "PB12"
#define ATLC_STM32_SEG_F_PIN_TEXT "PA12"
#define ATLC_STM32_SEG_G_PIN_TEXT "PA11"
#define ATLC_STM32_DIGIT_0_PIN_TEXT "PA3"
#define ATLC_STM32_DIGIT_1_PIN_TEXT "PA4"

#endif
