#ifndef ATLC_STM32_APP_TASKS_H
#define ATLC_STM32_APP_TASKS_H

#ifdef __cplusplus
extern "C" {
#endif

int atlc_app_tasks_init(void);

int atlc_app_tasks_start_scheduler(void);

const char *atlc_app_tasks_status(void);

#ifdef __cplusplus
}
#endif

#endif
