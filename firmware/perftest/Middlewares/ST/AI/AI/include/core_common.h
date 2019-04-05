/**
  ******************************************************************************
  * @file    core_common.h
  * @author  AST Embedded Analytics Research Platform
  * @date    20-Lug-2018
  * @brief   header file of common core datatypes
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

#ifndef __CORE_COMMON_H_
#define __CORE_COMMON_H_
#pragma once

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_datatypes_internal.h"

#include "core_datatypes.h"
#include "core_log.h"

/*!
 * @defgroup core_common Common Core Library Routines
 * @brief Common macros, datatypes and routines of core common module
 * @details This module contains the definitons and handling of the @ref ai_node
 * datastructures. An ai_node is a generic abstraction for a network node that 
 * could be either a fixed function layer or an operator. Ideally the platform 
 * interface defined in api module should handle an process generic nodes in the
 * network, not relying on the fact that they are layers or operators datastructs
 * Specific implementative details should be kept inside layers and operators
 * modules. The core module implements additionally common routines used in the
 * layers and operators modules.
 */

/******************************************************************************/
#ifdef HAVING_AI_ASSERT
  #define ASSERT_ARRAY_SANITY(a_) \
    AI_ASSERT((a_) && (a_)->size>0)

  #define ASSERT_ARRAY_DATA_SANITY(a_) \
    ASSERT_ARRAY_SANITY(a_) \
    AI_ASSERT((a_)->data && (a_)->data_start)

  #define ASSERT_TENSOR_SANITY(t_) \
    AI_ASSERT((t_) && (t_)->data) \
    AI_ASSERT(ai_shape_get_size(&(t_)->shape)>0) \
    ASSERT_ARRAY_SANITY((t_)->data)

  #define ASSERT_TENSOR_DATA_SANITY(t_) \
    ASSERT_TENSOR_SANITY(t_) \
    ASSERT_ARRAY_DATA_SANITY((t_)->data)

  #define ASSERT_NODE_SANITY(node_) \
    do { \
      AI_ASSERT(AI_NODE_OBJ(node_)->tensors && AI_NODE_OBJ(node_)->tensors->chain) \
      ASSERT_TENSOR_SANITY(GET_TENSOR_IN(AI_NODE_OBJ(node_)->tensors, 0)) \
      ASSERT_TENSOR_SANITY(GET_TENSOR_OUT(AI_NODE_OBJ(node_)->tensors, 0)) \
    } while (0);
#else
  #define ASSERT_ARRAY_SANITY(a_)         AI_NOP
  #define ASSERT_ARRAY_DATA_SANITY(a_)    AI_NOP
  #define ASSERT_TENSOR_SANITY(t_)        AI_NOP
  #define ASSERT_TENSOR_DATA_SANITY(t_)   AI_NOP
  #define ASSERT_NODE_SANITY(node_)       AI_NOP
#endif  /*HAVING_AI_ASSERT*/


/******************************************************************************/
#define AI_NODE_TYPE(type_)  \
  ( (ai_node_type)((ai_u32)(type_)&0xFFFF) )

#define AI_NODE_OBJ(obj_) \
  ((ai_node*)(obj_))

#define AI_NODE_FORWARD_FUNC(func_) \
  ((node_forward_func)(func_))
          
#define AI_NODE_IS_FIRST(node) \
  (AI_NODE_OBJ(node)==AI_NODE_OBJ(AI_NODE_OBJ(node)->network->input_node))

#define AI_NODE_IS_LAST(node_) \
  ((AI_NODE_OBJ(node_)==AI_NODE_OBJ(node_)->next) || \
   (AI_NODE_OBJ(node_)->next==NULL))

#define AI_NODE_COMMON_FIELDS_DECLARE \
  ai_node_type type;              /*!< node type id (see @ref ai_node_type) */ \
  ai_id_obj id;                   /*!< node object instance id (see @ref ai_id_obj) */ \
  struct ai_network_* network;    /*!< handle to global network context */ \
  struct ai_node_* next;          /*!< the next node object in the sequence */ \
  node_forward_func forward;      /*!< forward function for the node */ \
  ai_klass_obj klass;             /*!< opaque handler to specific layer implementations */ \
  AI_CONST ai_tensor_chain* tensors; /*!< pointer to node tensor chain */

#define AI_NODE_COMMON_INIT(type_, id_, forward_, next_, network_, klass_) \
  .type = AI_NODE_TYPE(type_), \
  .id   = AI_ID_OBJ(id_), \
  .network = AI_NETWORK_OBJ(network_), \
  .next = AI_NODE_OBJ(next_), \
  .forward = AI_NODE_FORWARD_FUNC(forward_), \
  .klass = AI_KLASS_OBJ(klass_)

#define AI_FOR_EACH_NODE_DO(node_, nodes_) \
  for ( ai_node* node_ = AI_NODE_OBJ(nodes_); (node_); \
        node_ = ((AI_NODE_IS_LAST(node_)) ? NULL : (node_)->next) )


/**  TENSOR CHAINS LOOP MACROS & GETTERS  *************************************/
#define AI_FOR_EACH_TENSOR_CHAIN_DO(tlist_ptr_, tensors_) \
  for ( ai_tensor_list* tlist_ptr_=(tensors_)->chain; \
        tlist_ptr_<(((tensors_)->chain)+AI_TENSOR_CHAIN_SIZE); \
        tlist_ptr_++ )

#define AI_FOR_EACH_TENSOR_LIST_DO(tensor_, tlist_ptr_) \
    ai_tensor* tensor_= ((tlist_ptr_)->tensor) ? (tlist_ptr_)->tensor[0] : NULL; \
    for ( ai_u32 idx_=0; tensor_; tensor_=(tlist_ptr_)->tensor[++idx_] )

#define GET_TENSOR_LIST(tensors_, type_) \
  ( &(tensors_)->chain[AI_CONCAT(AI_TENSOR_CHAIN_, type_)] )

#define GET_TENSOR_IN(tensors_, pos_) \
  ( (tensors_)->chain[AI_TENSOR_CHAIN_INPUT].tensor[pos_] )

#define GET_TENSOR_OUT(tensors_, pos_) \
  ( (tensors_)->chain[AI_TENSOR_CHAIN_OUTPUT].tensor[pos_] )

#define GET_TENSOR_WEIGHTS(tensors_, pos_) \
  ( (tensors_)->chain[AI_TENSOR_CHAIN_WEIGHTS].tensor[pos_] )

#define GET_TENSOR_SCRATCH(tensors_, pos_) \
  ( (tensors_)->chain[AI_TENSOR_CHAIN_SCRATCH].tensor[pos_] )

#define AI_NODE_IO_GET(node_, in_, out_) \
  ASSERT_NODE_SANITY(node_) \
  ai_tensor* in_  = GET_TENSOR_IN((node_)->tensors, 0); \
  ai_tensor* out_ = GET_TENSOR_OUT((node_)->tensors, 0); \
  ASSERT_TENSOR_SANITY(in_) \
  ASSERT_TENSOR_SANITY(out_)

/******************************************************************************/
#if 1
  #define SECTION_SERIAL(expr)    expr
  #define SECTION_PARALLEL(expr)
#else
  #define SECTION_SERIAL(expr)
  #define SECTION_PARALLEL(expr)  expr
#endif

AI_API_DECLARE_BEGIN

/*!
 * @struct ai_node_type
 * @ingroup core_common
 * @brief generic network node numeric type ID
 *
 */
typedef uint16_t ai_node_type;

/*!
 * @typedef void (*node_forward_func)(struct ai_node_* node)
 * @ingroup core_common
 * @brief Callback signatures for all forward functions
 */
typedef void (*node_forward_func)(struct ai_node_* node);

/*!
 * @typedef ai_float (*func_nl_el)(const ai_float x)
 * @ingroup core_common
 * @brief Fuction pointer for generic elementwise transforms
 *
 * This function pointer abstracts a generic nonlinear function applied to a
 * single element. See @ref ai_math_sqrt in @ref math_helpers as examples.
 */
typedef ai_float (*func_nl_el)(const ai_float x);

/*!
 * @struct ai_node
 * @ingroup core_common
 * @brief Structure encoding a generic node of the network
 *
 * The node struct includes information about the network it belong to, the
 * next node in a sequential network and the forward function. The forward
 * functions are implemented in the @ref layers module.
 */
typedef AI_ALIGNED_TYPE(struct, 4) ai_node_ {
  AI_NODE_COMMON_FIELDS_DECLARE
} ai_node;

/*!
 * @brief initialize core module
 * @ingroup core_common
 * @return false if initialization fails, false otherwise
 */
AI_INTERNAL_API
ai_bool core_init(void);

/*!
 * @brief get 1st error raised during processing 
 * @ingroup core_common
 * @param[out] error the @ref ai_error recorded during processing
 * @return the 1st error generated during processing. If no errors AI_ERROR_NONE
 */
AI_INTERNAL_API
ai_error core_get_error(ai_error* error);

/*!
 * @brief set error recorded during processing
 * @ingroup core_common
 * @param[out] error the @ref ai_error to set
 * @param[in] type the specific error type to set
 * @param[in] code the specific error code to set
 * @return true if the error is set, false in case a precedent error was already 
 * set
 */
AI_INTERNAL_API
ai_bool core_set_error(
  ai_error* error, const ai_error_type type, const ai_error_code code);

AI_API_DECLARE_END

#endif    /*__CORE_COMMON_H_*/
