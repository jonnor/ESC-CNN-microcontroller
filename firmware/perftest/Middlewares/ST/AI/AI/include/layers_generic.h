/**
  ******************************************************************************
  * @file    layers_generic.h
  * @author  AST Embedded Analytics Research Platform
  * @date    18-Apr-2018
  * @brief   header file of AI platform generic layers datatypes
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
#ifndef __LAYERS_GENERIC_H_
#define __LAYERS_GENERIC_H_
#pragma once

#include "layers_common.h"

/*!
 * @defgroup layers_generic Generic Layers Definitions
 * @brief definition 
 *
 */

AI_API_DECLARE_BEGIN

/*!
 * @struct ai_layer_time_delay
 * @ingroup layers_generic
 * @brief TimeDelay layer with sparse kernel
 */
typedef AI_ALIGNED_TYPE(struct, 4) ai_layer_time_delay_ {
  AI_LAYER_COMMON_FIELDS_DECLARE
  // TODO Remove this and use an ai_tensor inside the chain instead?
  ai_array* mask;      /*!< sparse filter mask */
} ai_layer_time_delay;

/*!
 * @struct ai_layer_split
 * @ingroup layers_generic
 * @brief Split layer definition
 *
 * This layer defines the params of a splitting layer. It is intended to be used
 * by his associated forward function @ref forward_split
 */
typedef AI_ALIGNED_TYPE(struct, 4) ai_layer_split_ {
  AI_LAYER_COMMON_FIELDS_DECLARE
  ai_u16             out_layers_count; /*!< number of output layers to split*/
  ai_u16             out_layer_curr;   /*!< current layer to split  */
  ai_layer**         out_layers;  /*!< output layers list */
  ai_tensor**        out_tensors; /*!< output tensors list */
  ai_tensor*         in_tensor;   /*!< input tensor */
  func_copy_tensor   copy_to_out_tensor; /*!< pointer to copy tensor func
                                         (NULL = no copy) */ 
} ai_layer_split;

/*!
 * @struct ai_layer_slice
 * @ingroup layers_generic
 * @brief Slice layer definition
 *
 * This layer defines the params of a slicing layer. It is intended to be used
 * by his associated forward function @ref forward_slice
 */
typedef AI_ALIGNED_TYPE(struct, 4) ai_layer_slice_ {
  AI_LAYER_COMMON_FIELDS_DECLARE
  ai_array* axes;    /*!< Axes that 'starts' and 'ends' apply to. It's optional*/
  ai_array* starts;  /*!< Starting indices of corrisponding axis in axes*/
  ai_array* ends;    /*!< Ending indices (exclusive) of corrisponding axis in axes*/
} ai_layer_slice;

/*!
 * @struct ai_layer_add
 * @ingroup layers_generic
 * @brief Add layer definition
 *
 * This layer defines the params of an add layer. 
 */
typedef AI_ALIGNED_TYPE(struct, 4) ai_layer_add_ {
  AI_LAYER_COMMON_FIELDS_DECLARE
  ai_u16             in_layers_count; /*!< number of input layers to concat */
  ai_u16             in_layer_curr;   /*!< current layer to concat  */
  ai_tensor**        in_tensors;  /*!< input tensors list (if NULL==no copy) */
  ai_tensor*         out_tensor;  /*!< output tensor (if NULL==no copy) */
  func_copy_tensor   copy_to_out_tensor; /*!< pointer to copy tensor func
                                         (NULL = no copy) */ 
  ai_layer*  split_layer; /*!< pointer to associated split layer */
  ai_layer*  next_layer;  /*!< pointer to next layer to process */
} ai_layer_add;

/*!
 * @struct ai_layer_permute
 * @ingroup layers_generic
 * @brief Permute layer datastruct declaration. This defines the params of a
 * permute layer. It is intended to be used by his associated forward function
 * @ref forward_permute
 */
typedef AI_ALIGNED_TYPE(struct, 4) ai_layer_permute_ {
  AI_LAYER_COMMON_FIELDS_DECLARE
  ai_shape out_mapping;       /*!< permute output mapping order. I.e. tt is a
                                   permutation of the input tensor shape */
} ai_layer_permute;


#define AI_TIME_DISTRIBUTED_AXIS    (AI_SHAPE_HEIGHT)

/*!
 * @struct ai_layer_time_distributed
 * @ingroup layers_generic
 * @brief Time distributed layer datastruct declaration. This defines the params
 * of a time distributed layer. It is intended to be used by his associated
 * forward function @ref forward_time_distributed
 */
typedef AI_ALIGNED_TYPE(struct, 4) ai_layer_time_distributed_ {
  AI_LAYER_COMMON_FIELDS_DECLARE
  ai_layer*  inner_layer;       /*!< inner layer to process */
} ai_layer_time_distributed;

/*!
 * @struct ai_layer_concat
 * @ingroup layers_generic
 * @brief Concatenation layer
 *
 * Concat Layer.
 * It is a sequential layer. see @ref ai_layer_sequential
 */
typedef AI_ALIGNED_TYPE(struct, 4) ai_layer_concat_ {
  AI_LAYER_COMMON_FIELDS_DECLARE
  ai_shape_dimension axis;       /*!< which axis to concatenate on */
} ai_layer_concat;


/******************************************************************************/
/* Forward Functions Section                                                  */
/******************************************************************************/

/*!
 * @brief Computes the activations of a TimeDelay layer.
 * @ingroup layers_generic
 * @param layer the time delay layer
 */
AI_INTERNAL_API
void forward_time_delay(ai_layer* layer);

/*!
 * @brief Split network computation in N parallel branches.
 * @ingroup layers_generic
 * @param layer the split layer
 */
AI_INTERNAL_API
void forward_split(ai_layer* layer);

/*!
 * @brief Add network computation from N parallel branches.
 * @ingroup layers_generic
 * @param layer the add layer
 */
AI_INTERNAL_API
void forward_add(ai_layer* layer);

/*!
 * @brief Permute a tensor along a pivot and save permuted values into an output
 * tensor
 * @ingroup layers_generic
 * @param layer the permute layer
 */
AI_INTERNAL_API
void forward_permute(ai_layer* layer);

/*!
 * @brief TimeDistrubuted forward layer function. This forward function
 * implements the timedistributed layer. 
 * @ingroup layers_generic
 * @param layer the time distributed layer
 */
AI_INTERNAL_API
void forward_time_distributed(ai_layer* layer);


/*!
 * @brief Concatenates a list of tensors into a single tensor.
 * @ingroup layers_generic
 * @param layer the concatenation layer
 */
AI_INTERNAL_API
void forward_concat(ai_layer* layer);

/*!
 * @brief Slice an input tensors
 * @ingroup layers_generic
 * @param layer the sliced layer
 */
AI_INTERNAL_API
void forward_slice(ai_layer* layer);


AI_API_DECLARE_END

#endif    /*__LAYERS_GENERIC_H_*/
