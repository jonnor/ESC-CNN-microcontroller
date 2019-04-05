/**
  ******************************************************************************
  * @file    aiSystemPerformance.h
  * @author  MCD Vertical Application Team
  * @brief   Entry points for AI system performance application
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) YYYY STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0044, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0044
  *
  ******************************************************************************
  */

#ifndef __AI_SYSTEM_PERFORMANCE_H__
#define __AI_SYSTEM_PERFORMANCE_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int aiSystemPerformanceInit(void);
int aiSystemPerformanceProcess(void);
void aiSystemPerformanceDeInit(void);

#ifdef __cplusplus
}
#endif

#endif /* __AI_SYSTEM_PERFORMANCE_H__ */
