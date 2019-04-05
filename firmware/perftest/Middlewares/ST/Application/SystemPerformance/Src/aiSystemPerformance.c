/**
  ******************************************************************************
  * @file    aiSystemPerformance.c
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
 
 /*
 * Description:
 *
 * - Simple STM32 application to measure and report the system performance of
 *   a generated NN
 * - STM32CubeMX.AI tools version: Client API "1.0" / AI Platform 3.2.x
 * - Only STM32F7, STM32F4 or STM32L4 MCU-base family are supported
 * - Random input values are injected in the NN to measure the inference time
 *   and to monitor the usage of the stack and/or the heap. Output value are
 *   skipped.
 * - After N iterations (_APP_ITER_ C-define), results are reported through a
 *   re-target printf
 * - aiSystemPerformanceInit()/aiSystemPerformanceProcess() functions should
 *   be called from the main application code.
 * - Only UART (to re-target the printf) & CORE clock setting are expected
 *   by the initial run-time (main function).
 *   CRC IP should be also enabled for AI Platform >= 3.0.0
 *
 * Atollic/AC6 IDE (GCC-base toolchain)
 *  - Linker options "-Wl,--wrap=malloc -Wl,--wrap=free" should be used
 *    to support the HEAP monitoring
 *
 * TODO:
 *  - complete the returned HEAP data
 *  - add HEAP monitoring for IAR tool-chain
 *  - add HEAP/STACK monitoring MDK-ARM Keil tool-chain
 *
 * History:
 *  - v1.0 - Initial version
 *  - v1.1 - Complete minimal interactive console
 *  - v1.2 - Adding STM32H7 MCU support
 *  - v1.3 - Adding STM32F3 MCU support
 *  - v1.4 - Adding Profiling mode
 *  - v2.0 - Adding Multiple Network support
 *  - v2.1 - Adding F3 str description
 */

/* System headers */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

#if defined(__GNUC__)
#include <errno.h>
#include <sys/unistd.h> /* STDOUT_FILENO, STDERR_FILENO */
#elif defined (__ICCARM__)
#if (__IAR_SYSTEMS_ICC__ <= 8) 
/* Temporary workaround - LowLevelIOInterface.h seems not available
   with IAR 7.80.4 */
#define _LLIO_STDIN  0
#define _LLIO_STDOUT 1
#define _LLIO_STDERR 2
#define _LLIO_ERROR ((size_t)-1) /* For __read and __write. */
#else
#include <LowLevelIOInterface.h> /* _LLIO_STDOUT, _LLIO_STDERR */
#endif

#elif defined (__CC_ARM)

#endif

#include <aiSystemPerformance.h>
#include <bsp_ai.h>

/* APP configuration 0: disabled 1: enabled */
#define _APP_DEBUG_         0
#if defined(__GNUC__)
#define _APP_STACK_MONITOR_ 1
#define _APP_HEAP_MONITOR_  1
#elif defined (__ICCARM__)
#define _APP_STACK_MONITOR_ 1
#define _APP_HEAP_MONITOR_  0   /* not yet supported */
#else
#define _APP_STACK_MONITOR_ 0   /* not yet supported */
#define _APP_HEAP_MONITOR_  0   /* not yet supported */
#endif


extern UART_HandleTypeDef UartHandle;


/* AI header files */
#include "ai_platform.h"

#if !defined(STM32F7) && !defined(STM32L4) && !defined(STM32F4) && !defined(STM32H7) && !defined(STM32F3)
#error Only STM32H7, STM32F7, STM32F4, STM32L4 or STM32F3 device are supported
#endif

#define _APP_VERSION_MAJOR_     (0x02)
#define _APP_VERSION_MINOR_     (0x01)
#define _APP_VERSION_   ((_APP_VERSION_MAJOR_ << 8) | _APP_VERSION_MINOR_)

#define _APP_NAME_      "AI system performance measurement"

#define _APP_ITER_       16  /* number of iteration for perf. test */

struct dwtTime {
    uint32_t fcpu;
    int s;
    int ms;
    int us;
};

/* -----------------------------------------------------------------------------
 * Device-related functions
 * -----------------------------------------------------------------------------
 */
 
__STATIC_INLINE void crcIpInit(void)
{
#if defined(STM32H7)
    /* By default the CRC IP clock is enabled */
    __HAL_RCC_CRC_CLK_ENABLE();  
#else
    if (!__HAL_RCC_CRC_IS_CLK_ENABLED())
        printf("W: CRC IP clock is NOT enabled\r\n");

    /* By default the CRC IP clock is enabled */
    __HAL_RCC_CRC_CLK_ENABLE();
#endif
}

__STATIC_INLINE void dwtIpInit(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;

#ifdef STM32F7
    DWT->LAR = 0xC5ACCE55;
#endif

    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk | DWT_CTRL_CPIEVTENA_Msk;
}

__STATIC_INLINE void dwtReset(void)
{
    DWT->CYCCNT = 0; /* Clear DWT cycle counter */
}

__STATIC_INLINE uint32_t dwtGetCycles(void)
{
    return DWT->CYCCNT;
}

__STATIC_INLINE uint32_t systemCoreClock(void)
{
#if !defined(STM32H7) 
    return HAL_RCC_GetHCLKFreq();
#else
    return HAL_RCC_GetSysClockFreq();
#endif
}

__STATIC_INLINE int dwtCyclesToTime(uint64_t clks, struct dwtTime *t)
{
    if (!t)
        return -1;
    uint32_t fcpu = systemCoreClock();
    uint64_t s  = clks / fcpu;
    uint64_t ms = (clks * 1000) / fcpu;
    uint64_t us = (clks * 1000 * 1000) / fcpu;
    ms -= (s * 1000);
    us -= (ms * 1000 + s * 1000000);
    t->fcpu = fcpu;
    t->s = s;
    t->ms = ms;
    t->us = us;
    return 0;
}

__STATIC_INLINE const char *devIdToStr(uint16_t dev_id)
{
    const char *str;
    switch (dev_id) {
    case 0x422: str = "STM32F303xB/C"; break;
    case 0x438: str = "STM32F303x6/8"; break;
    case 0x446: str = "STM32F303xD/E"; break;
    case 0x431: str = "STM32F411xC/E"; break;
    case 0x435: str = "STM32L43xxx"; break;
    case 0x462: str = "STM32L45xxx"; break;
    case 0x415: str = "STM32L4x6xx"; break;
    case 0x470: str = "STM32L4Rxxx"; break;
    case 0x449: str = "STM32F74xxx"; break;
    case 0x450: str = "STM32H743/753 and STM32H750"; break;
    default:    str = "UNKNOWN";
    }
    return str;
}

#if !defined(STM32F3)
__STATIC_INLINE const char* bitToStr(uint32_t val)
{
    if (val)
        return "True";
    else
        return "False";
}
#endif

__STATIC_INLINE void logDeviceConf(void)
{
#if !defined(STM32F3)
    uint32_t acr = FLASH->ACR ;
#endif
    uint32_t val;

    printf("STM32 Runtime configuration...\r\n");

    printf(" Device       : DevID:0x%08x (%s) RevID:0x%08x\r\n",
            (int)HAL_GetDEVID(),
            devIdToStr(HAL_GetDEVID()),
            (int)HAL_GetREVID()
    );

    printf(" Core Arch.   : M%d - %s %s\r\n",
            __CORTEX_M,
#if (__FPU_PRESENT == 1)
            "FPU PRESENT",
            __FPU_USED ? "and used" : "and not used!"
#else
            "!FPU NOT PRESENT",
            ""
#endif
    );

    printf(" HAL version  : 0x%08x\r\n", (int)HAL_GetHalVersion());

    val = systemCoreClock()/1000000;

#if !defined(STM32H7)
    printf(" system clock : %u MHz\r\n", (int)val);
#else
    printf(" SYSCLK clock : %u MHz\r\n", (int)val);
    printf(" HCLK clock   : %u MHz\r\n", (int)HAL_RCC_GetHCLKFreq()/1000000);    
#endif

#if defined(STM32F7) || defined(STM32H7)
    val = SCB->CCR;
#if !defined(STM32H7)
    printf(" FLASH conf.  : ACR=0x%08x - Prefetch=%s ART=%s latency=%d\r\n",
            (int)acr,
            bitToStr((acr & FLASH_ACR_PRFTEN_Msk) >> FLASH_ACR_PRFTEN_Pos),
            bitToStr((acr & FLASH_ACR_ARTEN_Msk) >> FLASH_ACR_ARTEN_Pos),
            (int)((acr & FLASH_ACR_LATENCY_Msk) >> FLASH_ACR_LATENCY_Pos));
#else
    printf(" FLASH conf.  : ACR=0x%08x - latency=%d\r\n",
            (int)acr,
            (int)((acr & FLASH_ACR_LATENCY_Msk) >> FLASH_ACR_LATENCY_Pos));
#endif
    printf(" CACHE conf.  : $I/$D=(%s,%s)\r\n",
            bitToStr(val & SCB_CCR_IC_Msk),
            bitToStr(val & SCB_CCR_DC_Msk));
#else
#if !defined(STM32F3)
    printf(" FLASH conf.  : ACR=0x%08x - Prefetch=%s $I/$D=(%s,%s) latency=%d\r\n",
            (int)acr,
            bitToStr((acr & FLASH_ACR_PRFTEN_Msk) >> FLASH_ACR_PRFTEN_Pos),
            bitToStr((acr & FLASH_ACR_ICEN_Msk) >> FLASH_ACR_ICEN_Pos),
            bitToStr((acr & FLASH_ACR_DCEN_Msk) >> FLASH_ACR_DCEN_Pos),
            (int)((acr & FLASH_ACR_LATENCY_Msk) >> FLASH_ACR_LATENCY_Pos));
#endif
#endif
}

__STATIC_INLINE uint32_t disableInts(void)
{
    uint32_t state;

    state = __get_PRIMASK();
    __disable_irq();

    return state;
}

__STATIC_INLINE void restoreInts(uint32_t state)
{
   __set_PRIMASK(state);
}


/* -----------------------------------------------------------------------------
 * low-level I/O functions
 * -----------------------------------------------------------------------------
 */

static struct ia_malloc {
    uint32_t cfg;
    uint32_t alloc;
    uint32_t free;
    uint32_t alloc_req;
    uint32_t free_req;
} ia_malloc;

#define MAGIC_MALLOC_NUMBER 0xefdcba98


static int ioGetUint8(uint8_t *buff, int count, uint32_t timeout)
{
    HAL_StatusTypeDef status;

    if ((!buff) || (count <= 0))
        return -1;

    status = HAL_UART_Receive(&UartHandle, (uint8_t *)buff, count,
            timeout);

    if (status == HAL_TIMEOUT)
        return -1;

    return (status == HAL_OK ? count : 0);
}


#if defined(__GNUC__)

int _write(int fd, const void *buff, int count);

int _write(int fd, const void *buff, int count)
{
    HAL_StatusTypeDef status;

    if ((count < 0) && (fd != STDOUT_FILENO) && (fd != STDERR_FILENO)) {
        errno = EBADF;
        return -1;
    }

    status = HAL_UART_Transmit(&UartHandle, (uint8_t *)buff, count,
            HAL_MAX_DELAY);

    return (status == HAL_OK ? count : 0);
}

void* __real_malloc(size_t bytes);
void __real_free(void *ptr);

void* __wrap_malloc(size_t bytes)
{
    uint8_t *ptr;

    ia_malloc.cfg |= 1 << 1;

    /* ensure alignment for magic number */
    bytes = (bytes + 3) & ~3;

    /* add 2x32-bit for size and magic  number */
    ptr = (uint8_t*)__real_malloc(bytes + 8);

    /* remember size */
    if (ptr) {
        *((uint32_t*)ptr) = bytes;
        *((uint32_t*)(ptr + 4 + bytes)) = MAGIC_MALLOC_NUMBER;
    }

    if ((ptr) && (ia_malloc.cfg & 1UL)) {
        ia_malloc.alloc_req++;
        ia_malloc.alloc += bytes;
    }
    return ptr?(ptr + 4):NULL;
}

void __wrap_free(void *ptr)
{
    uint8_t* p;
    uint32_t bytes;

    ia_malloc.cfg |= 1 << 2;

    if (!ptr)
        return;

    p = (uint8_t*)ptr - 4;
    bytes = *((uint32_t*)p);

    if (*((uint32_t*)(p + 4 + bytes)) == MAGIC_MALLOC_NUMBER) {
        *((uint32_t*)(p + 4 + bytes)) = 0;
    }

    if (ia_malloc.cfg & 1UL) {
        ia_malloc.free_req++;
        ia_malloc.free += bytes;
    }
    __real_free(p);
}


#elif defined (__ICCARM__)

__ATTRIBUTES  size_t __write(int handle, const unsigned char *buffer,
                             size_t size);

__ATTRIBUTES  size_t __write(int handle, const unsigned char *buffer,
                             size_t size)
{
    HAL_StatusTypeDef status;

    /*
     * This means that we should flush internal buffers.  Since we
     * don't we just return.  (Remember, "handle" == -1 means that all
     * handles should be flushed.)
     */
    if (buffer == 0)
      return 0;

    /* This template only writes to "standard out" and "standard err",
     * for all other file handles it returns failure.
     */
    if ((handle != _LLIO_STDOUT) && (handle != _LLIO_STDERR))
        return _LLIO_ERROR;

    status = HAL_UART_Transmit(&UartHandle, (uint8_t *)buffer, size,
            HAL_MAX_DELAY);

    return (status == HAL_OK ? size : _LLIO_ERROR);
}

#if _APP_HEAP_MONITOR_ == 1
#undef _APP_HEAP_MONITOR_
#define _APP_HEAP_MONITOR_ 0
#warning HEAP monitor is not YET supported
#endif

#elif defined (__CC_ARM)

int fputc(int ch, FILE *f)
{
      HAL_UART_Transmit(&UartHandle, (uint8_t *)&ch, 1,
            HAL_MAX_DELAY);
    return ch;
}

#if _APP_STACK_MONITOR_ == 1
#undef _APP_STACK_MONITOR_
#define _APP_STACK_MONITOR_ 0
#warning STACK monitor is not YET supported
#endif

#if _APP_HEAP_MONITOR_ == 1
#undef _APP_HEAP_MONITOR_
#define _APP_HEAP_MONITOR_ 0
#warning HEAP monitor is not YET supported
#endif

#else
#error ARM MCU tool-chain is not supported.
#endif


/* -----------------------------------------------------------------------------
 * AI-related functions
 * -----------------------------------------------------------------------------
 */

struct network_exec_ctx {
    ai_handle handle;
    ai_network_report report;
} net_ctx[AI_MNETWORK_NUMBER] = {0};

#define AI_BUFFER_NULL(ptr_)  \
  AI_BUFFER_OBJ_INIT( \
    AI_BUFFER_FORMAT_NONE|AI_BUFFER_FMT_FLAG_CONST, \
    0, 0, 0, 0, \
    AI_HANDLE_PTR(ptr_))

AI_ALIGNED(4)
static ai_u8 activations[AI_MNETWORK_DATA_ACTIVATIONS_SIZE];

static ai_float in_data[AI_MNETWORK_IN_1_SIZE];
static ai_float out_data[AI_MNETWORK_OUT_1_SIZE];

__STATIC_INLINE void aiLogErr(const ai_error err, const char *fct)
{
    if (fct)
        printf("E: AI error (%s) - type=%d code=%d\r\n", fct,
                err.type, err.code);
    else
        printf("E: AI error - type=%d code=%d\r\n", err.type, err.code);
}

__STATIC_INLINE const char* aiBufferFormatToStr(uint32_t val)
{
    if (val == AI_BUFFER_FORMAT_NONE)
        return "AI_BUFFER_FORMAT_NONE";
    else if (val == AI_BUFFER_FORMAT_FLOAT)
        return "AI_BUFFER_FORMAT_FLOAT";
    else if (val == AI_BUFFER_FORMAT_U8)
        return "AI_BUFFER_FORMAT_U8";
    else if (val == AI_BUFFER_FORMAT_Q15)
        return "AI_BUFFER_FORMAT_Q15";
    else if (val == AI_BUFFER_FORMAT_Q7)
        return "AI_BUFFER_FORMAT_Q7";
    else
        return "UNKNOWN";
}

__STATIC_INLINE ai_u32 aiBufferSize(const ai_buffer* buffer)
{
    return buffer->height * buffer->width * buffer->channels;
}

__STATIC_INLINE void aiPrintLayoutBuffer(const char *msg,
        const ai_buffer* buffer)
{
    printf("%s HWC layout:%d,%d,%ld (s:%ld f:%s)\r\n",
      msg, buffer->height, buffer->width, buffer->channels,
      aiBufferSize(buffer),
      aiBufferFormatToStr(buffer->format));
}

__STATIC_INLINE void aiPrintNetworkInfo(const ai_network_report* report)
{
  printf("Network configuration...\r\n");
  printf(" Model name         : %s\r\n", report->model_name);
  printf(" Model signature    : %s\r\n", report->model_signature);
  printf(" Model datetime     : %s\r\n", report->model_datetime);
  printf(" Compile datetime   : %s\r\n", report->compile_datetime);
  printf(" Runtime revision   : %s (%d.%d.%d)\r\n", report->runtime_revision,
    report->runtime_version.major,
    report->runtime_version.minor,
    report->runtime_version.micro);
  printf(" Tool revision      : %s (%d.%d.%d)\r\n", report->tool_revision,
    report->tool_version.major,
    report->tool_version.minor,
    report->tool_version.micro);
  printf("Network info...\r\n");
  printf("  signature         : 0x%lx\r\n", report->signature);
  printf("  nodes             : %ld\r\n", report->n_nodes);
  printf("  complexity        : %ld MACC\r\n", report->n_macc);
  printf("  activation        : %ld bytes\r\n", aiBufferSize(&report->activations));
  printf("  weights           : %ld bytes\r\n", aiBufferSize(&report->weights));
  printf("  inputs/outputs    : %u/%u\r\n", report->n_inputs, report->n_outputs);
  aiPrintLayoutBuffer("  IN tensor format  :", &report->inputs);
  aiPrintLayoutBuffer("  OUT tensor format :", &report->outputs);
}

static int aiBootstrap(const char *nn_name, const int idx)
{
    ai_error err;

    /* Creating the network */
    printf("Creating the network \"%s\"..\r\n", nn_name);
    err = ai_mnetwork_create(nn_name, &net_ctx[idx].handle, NULL);
    if (err.type) {
        aiLogErr(err, "ai_mnetwork_create");
        return -1;
    }

    /* Query the created network to get relevant info from it */
    if (ai_mnetwork_get_info(net_ctx[idx].handle, &net_ctx[idx].report)) {
        aiPrintNetworkInfo(&net_ctx[idx].report);
    } else {
        err = ai_mnetwork_get_error(net_ctx[idx].handle);
        aiLogErr(err, "ai_mnetwork_get_info");
        ai_mnetwork_destroy(net_ctx[idx].handle);
        net_ctx[idx].handle = AI_HANDLE_NULL;
        return -2;
    }

    /* Initialize the instance */
    printf("Initializing the network\r\n");
    /* build params structure to provide the reference of the
     * activation and weight buffers */
    const ai_network_params params = {
            AI_BUFFER_NULL(NULL),
            AI_BUFFER_NULL(activations) };

    if (!ai_mnetwork_init(net_ctx[idx].handle, &params)) {
        err = ai_mnetwork_get_error(net_ctx[idx].handle);
        aiLogErr(err, "ai_mnetwork_init");
        ai_mnetwork_destroy(net_ctx[idx].handle);
        net_ctx[idx].handle = AI_HANDLE_NULL;
        return -4;
    }
    return 0;
}

static int aiInit(void)
{
    const char *nn_name;
    int idx;

    printf("\r\nAI Network (AI platform API %d.%d.%d)...\r\n",
            AI_PLATFORM_API_MAJOR,
            AI_PLATFORM_API_MINOR,
            AI_PLATFORM_API_MICRO);

    /* Discover and init the embedded network */
    idx = 0;
    do {
    	nn_name = ai_mnetwork_find(NULL, idx);
    	if (nn_name) {
    		printf("\r\nFound network \"%s\"\r\n", nn_name);
    		if (aiBootstrap(nn_name, idx))
    		    return -1;
    	}
    	idx++;
    } while (nn_name);

    return 0;
}

static void aiDeInit(void)
{
    ai_error err;
    int idx;

    printf("Releasing the network(s)...\r\n");

    for (idx=0; idx<AI_MNETWORK_NUMBER; idx++) {
        if (net_ctx[idx].handle) {
            if (ai_mnetwork_destroy(net_ctx[idx].handle) != AI_HANDLE_NULL) {
                err = ai_mnetwork_get_error(net_ctx[idx].handle);
                aiLogErr(err, "ai_mnetwork_destroy");
            }
            net_ctx[idx].handle = NULL;
        }
    }
}

/* -----------------------------------------------------------------------------
 * Specific APP/test functions
 * -----------------------------------------------------------------------------
 */

#if defined(__GNUC__)
extern uint32_t _estack[];
#elif defined (__ICCARM__)
extern int CSTACK$$Limit;
extern int CSTACK$$Base;
#elif defined (__CC_ARM)
#if _APP_STACK_MONITOR_ == 1
#error STACK monitoring is not yet supported
#endif
#else
#error ARM MCU tool-chain is not supported.
#endif

static bool profiling_mode = false;
static int  profiling_factor = 5;

#if defined(__GNUC__)
// #pragma GCC push_options
// #pragma GCC optimize ("O0")
#elif defined (__ICCARM__)
// #pragma optimize=none
#endif

static int aiTestPerformance(int idx)
{
    int iter;
    uint32_t irqs;
    ai_i32 batch;
    int niter;

    struct dwtTime t;
    uint64_t tcumul;
    uint32_t tstart, tend;
    uint32_t tmin;
    uint32_t tmax;
    uint32_t cmacc;

    ai_buffer ai_input[1];
    ai_buffer ai_output[1];


#if _APP_STACK_MONITOR_ == 1
    uint32_t ctrl;
    bool stack_mon;
    uint32_t susage;

    uint32_t ustack_size; /* used stack before test */
    uint32_t estack;      /* end of stack @ */
    uint32_t mstack_size; /* minimal master stack size */
    uint32_t cstack;      /* current stack @ */
    uint32_t bstack;      /* base stack @ */
#endif

    if (net_ctx[idx].handle == AI_HANDLE_NULL) {
        printf("E: network handle is NULL\r\n");
        return -1;
    }

#if _APP_STACK_MONITOR_ == 1
    /* Reading ARM Core registers */
    ctrl = __get_CONTROL();
    cstack = __get_MSP();

#if defined(__GNUC__)
    estack = (uint32_t)_estack;
    bstack = estack - MIN_STACK_SIZE;
    mstack_size = MIN_STACK_SIZE;
#elif defined (__ICCARM__)
    estack = (uint32_t)&CSTACK$$Limit;
    bstack = (uint32_t)&CSTACK$$Base;
    mstack_size = (uint32_t)&CSTACK$$Limit - (uint32_t)&CSTACK$$Base;
#endif

#endif

    if (profiling_mode)
        niter = _APP_ITER_ * profiling_factor;
    else
        niter = _APP_ITER_;

    printf("\r\nRunning PerfTest on \"%s\" with random inputs (%d iterations)...\r\n",
            net_ctx[idx].report.model_name, niter);

 #if _APP_STACK_MONITOR_ == 1
   /* Check that MSP is the active stack */
    if (ctrl & CONTROL_SPSEL_Msk) {
        printf("E: MSP is not the active stack (stack monitoring is disabled)\r\n");
        stack_mon = false;
    } else
      stack_mon = true;

    /* Calculating used stack before test */
    ustack_size = estack - cstack;

    if ((stack_mon) && (ustack_size > mstack_size)) {
        printf("E: !stack overflow detected %ld > %ld\r\n", ustack_size,
                mstack_size);
        stack_mon = false;
    }
#endif

#if _APP_DEBUG_ == 1
    printf("D: stack before test (0x%08lx-0x%08lx %ld/%ld ctrl=0x%08lx\n",
            estack, cstack, ustack_size, mstack_size, ctrl);
#endif

    irqs = disableInts();

#if _APP_STACK_MONITOR_ == 1
    /* Fill the remaining part of the stack */
    if (stack_mon) {
      uint32_t *pw =  (uint32_t*)((bstack + 3) & (~3));

#if _APP_DEBUG_ == 1
      printf("D: fill stack 0x%08lx -> 0x%08lx (%ld)\n", pw, cstack,
             cstack - (uint32_t)pw);
#endif
      while ((uint32_t)pw < cstack) {
        *pw = 0xDEDEDEDE;
        pw++;
      }
    }
#endif

    /* reset/init cpu clock counters */
    tcumul = 0ULL;
    tmin = UINT32_MAX;
    tmax = 0UL;

    memset(&ia_malloc,0,sizeof(struct ia_malloc));

    ai_input[0] = net_ctx[idx].report.inputs;
    ai_output[0] = net_ctx[idx].report.outputs;

    ai_input[0].n_batches  = 1;
    ai_input[0].data = AI_HANDLE_PTR(in_data);
    ai_output[0].n_batches = 1;
    ai_output[0].data = AI_HANDLE_PTR(out_data);

    if (profiling_mode) {
        printf("Profiling mode (%d)...\r\n", profiling_factor);
        fflush(stdout);
    }

    for (iter = 0; iter < niter; iter++) {
      /* Fill input vector */
        for (ai_size i = 0; i < aiBufferSize(&ai_input[0]); ++i) {
            /* uniform distribution between -1.0 and 1.0 */
            in_data[i] = 2.0f * (ai_float) rand() / (ai_float) RAND_MAX - 1.0f;
        }

#if _APP_HEAP_MONITOR_ == 1
        ia_malloc.cfg |= 1UL;
#endif
        dwtReset();
        tstart = dwtGetCycles();

        batch = ai_mnetwork_run(net_ctx[idx].handle, &ai_input[0], &ai_output[0]);
        if (batch != 1) {
            aiLogErr(ai_mnetwork_get_error(net_ctx[idx].handle),
                    "ai_mnetwork_run");
            break;
        }

        tend = dwtGetCycles() - tstart;

#if _APP_HEAP_MONITOR_ == 1
        ia_malloc.cfg &= ~1UL;
#endif

        if (tend < tmin)
            tmin = tend;

        if (tend > tmax)
            tmax = tend;

        tcumul += (uint64_t)tend;

#if _APP_DEBUG_ == 1
        dwtCyclesToTime(tend, &t);
        printf(" #%02d %8d.%03dms (%lu cycles)\r\n", iter,
                t.ms, t.us, tend);
#else
        if (!profiling_mode) {
            printf(".");
            fflush(stdout);
        }
#endif
    }

#if _APP_DEBUG_ != 1
    printf("\r\n");
#endif

#if _APP_STACK_MONITOR_ == 1
    if (__get_MSP() != cstack) {
       printf("E: !current stack address is not coherent 0x%08lx instead 0x%08lx\r\n",
              __get_MSP(), cstack);
    }
#endif

#if _APP_STACK_MONITOR_ == 1
   /* Calculating the used stack */
   susage = 0UL;
   if (stack_mon) {
        uint32_t rstack = mstack_size - ustack_size;
        uint32_t *pr =  (uint32_t*)((bstack + 3) & (~3));
        bool overflow = false;

        /* check potential stack overflow with 8 last words*/
        for (int i = 0; i < 8; i++) {
          if (*pr != 0xDEDEDEDE)
            overflow = true;
          pr++;
        }

        if (!overflow) {
          susage = 8*4;
          while ((*pr == 0xDEDEDEDE) && ((uint32_t)pr < cstack)) {
            pr++;
            susage += 4;
          }
          susage = rstack - susage;
        } else {
            printf("E: !stack overflow detected > %ld\r\n", rstack);
        }
    }
#endif

    restoreInts(irqs);

    printf("\r\n");

    tcumul /= (uint64_t)iter;

    dwtCyclesToTime(tcumul, &t);

    printf("Results for \"%s\", %d inferences @%ldMHz/%ldMHz (complexity: %lu MACC)\r\n",
            net_ctx[idx].report.model_name, iter,
            HAL_RCC_GetSysClockFreq() / 1000000,
            HAL_RCC_GetHCLKFreq() / 1000000,
            net_ctx[idx].report.n_macc);

    printf(" duration     : %d.%03d ms (average)\r\n", t.s * 1000 + t.ms, t.us);
    printf(" CPU cycles   : %lu -%lu/+%lu (average,-/+)\r\n",
            (uint32_t)(tcumul), (uint32_t)(tcumul - tmin),
            (uint32_t)(tmax - tcumul));
    printf(" CPU Workload : %d%c\r\n", (int)((tcumul * 100) / t.fcpu), '%');
    cmacc = (uint32_t)((tcumul * 100)/ net_ctx[idx].report.n_macc);
    printf(" cycles/MACC  : %lu.%02lu (average for all layers)\r\n",
            cmacc / 100, cmacc - ((cmacc / 100) * 100));
#if _APP_STACK_MONITOR_ == 1
    if (stack_mon)
      printf(" used stack   : %ld bytes\r\n", susage);
    else
      printf(" used stack   : NOT CALCULATED\r\n");
#else
    printf(" used stack   : DISABLED\r\n");
#endif
#if _APP_HEAP_MONITOR_ == 1
    printf(" used heap    : %ld:%ld %ld:%ld (req:allocated,req:released) cfg=%ld\r\n",
            ia_malloc.alloc_req, ia_malloc.alloc,
            ia_malloc.free_req, ia_malloc.free,
            (ia_malloc.cfg & (3 << 1)) >> 1);
#else
    printf(" used heap    : DISABLED or NOT YET SUPPORTED\r\n");
#endif

    return 0;
}

#if defined(__GNUC__)
// #pragma GCC pop_options
#endif

#define CONS_EVT_TIMEOUT    (0)
#define CONS_EVT_QUIT       (1)
#define CONS_EVT_RESTART    (2)
#define CONS_EVT_HELP       (3)
#define CONS_EVT_PAUSE      (4)
#define CONS_EVT_PROF       (5)

#define CONS_EVT_UNDEFINED  (100)

static int aiTestConsole(void)
{
    uint8_t c = 0;

    if (ioGetUint8(&c, 1, 5000) == -1) /* Timeout */
        return CONS_EVT_TIMEOUT;

    if ((c == 'q') || (c == 'Q'))
        return CONS_EVT_QUIT;

    if ((c == 'r') || (c == 'R'))
        return CONS_EVT_RESTART;

    if ((c == 'h') || (c == 'H') || (c == '?'))
        return CONS_EVT_HELP;

    if ((c == 'p') || (c == 'P'))
        return CONS_EVT_PAUSE;

    if ((c == 'x') || (c == 'X'))
        return CONS_EVT_PROF;

    return CONS_EVT_UNDEFINED;
}


/* -----------------------------------------------------------------------------
 * Exported/Public functions
 * -----------------------------------------------------------------------------
 */

int aiSystemPerformanceInit(void)
{
    printf("\r\n#\r\n");
    printf("# %s %d.%d\r\n", _APP_NAME_ , _APP_VERSION_MAJOR_,
            _APP_VERSION_MINOR_ );
    printf("#\r\n");

#if defined(__GNUC__)
    printf("Compiled with GCC %d.%d.%d\r\n", __GNUC__, __GNUC_MINOR__,
            __GNUC_PATCHLEVEL__);
#elif defined(__ICCARM__)
    printf("Compiled with IAR %d (build %d)\r\n", __IAR_SYSTEMS_ICC__,
            __BUILD_NUMBER__
    );
#elif defined (__CC_ARM)
    printf("Compiled with MDK-ARM Keil %d\r\n", __ARMCC_VERSION);
#endif

    dwtIpInit();
    crcIpInit();
    logDeviceConf();

    aiInit();

    srand(3); /* deterministic outcome */

    dwtReset();
    return 0;
}

int aiSystemPerformanceProcess(void)
{
    int r;
    int idx = 0;

    do {
        r = aiTestPerformance(idx);
        idx = (idx+1) % AI_MNETWORK_NUMBER;

        if (!r) {
            r = aiTestConsole();

            if (r == CONS_EVT_UNDEFINED) {
                r = 0;
            } else if (r == CONS_EVT_HELP) {
                printf("\r\n");
                printf("Possible key for the interactive console:\r\n");
                printf("  [q,Q]      quit the application\r\n");
                printf("  [r,R]      re-start (NN de-init and re-init)\r\n");
                printf("  [p,P]      pause\r\n");
                printf("  [h,H,?]    this information\r\n");
                printf("   xx        continue immediately\r\n");
                printf("\r\n");
                printf("Press any key to continue..\r\n");

                while ((r = aiTestConsole()) == CONS_EVT_TIMEOUT) {
                    HAL_Delay(1000);
                }
                if (r == CONS_EVT_UNDEFINED)
                    r = 0;
            }
            if (r == CONS_EVT_PROF) {
                profiling_mode = true;
                profiling_factor *= 2;
                r = 0;
            }

            if (r == CONS_EVT_RESTART) {
                profiling_mode = false;
                profiling_factor = 5;
                printf("\r\n");
                aiDeInit();
                aiSystemPerformanceInit();
                r = 0;
            }
            if (r == CONS_EVT_QUIT) {
                profiling_mode = false;
                printf("\r\n");
                disableInts();
                aiDeInit();
                printf("\r\n");
                printf("Board should be reseted...\r\n");
                while (1) {
                    HAL_Delay(1000);
                }
            }
            if (r == CONS_EVT_PAUSE) {
                printf("\r\n");
                printf("Press any key to continue..\r\n");
                while ((r = aiTestConsole()) == CONS_EVT_TIMEOUT) {
                    HAL_Delay(1000);
                }
                r = 0;
            }
        }
    } while (r==0);

    return r;
}

void aiSystemPerformanceDeInit(void)
{
    printf("\r\n");
    aiDeInit();
    printf("bye bye ...\r\n");
}

