/**
  ******************************************************************************
  * @file    core_utils.h
  * @author  AST Embedded Analytics Research Platform
  * @date    16-Aug-2018
  * @brief   header file of core utils routines
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2018 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0044, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0044
  *
  ******************************************************************************
  */

#ifndef __CORE_UTILS_H_
#define __CORE_UTILS_H_
#pragma once

#include "ai_platform.h"
#include "ai_platform_interface.h"

#include "core_common.h"

AI_API_DECLARE_BEGIN

/*!
 * @defgroup core_utils Core Utils Routines
 * @brief Implementation of core utils such has checksums algorithms, etc. 
 */


/*!
 * @brief Computes 32bit checksum from a buffer array of bytes
 * @ingroup core_utils
 * @param[in] buffer in an opaque handler to the buffer we want to compute crc code
 * @param[in] byte_size the size in byte of the buffer provided
 */
AI_INTERNAL_API
ai_u32    core_utils_compute_crc32(
  const ai_handle buffer, const ai_size byte_size);

/*!
 * @brief Computes network signature given a network context
 * @ingroup core_utils
 * @param[in] net_ctx a pointer to the network context to be signed
 * @param[out] signature a pointer to the checksum signature computed 
 * from the network context
 * @return false if failed to compute the signature, true otherwise
 */
AI_INTERNAL_API
ai_bool    core_utils_generate_network_signature(
  const ai_network* net_ctx, ai_signature* signature);

AI_API_DECLARE_END

#endif    /*__CORE_UTILS_H_*/
