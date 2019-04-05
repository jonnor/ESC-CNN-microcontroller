/**
  ******************************************************************************
  * @file    core_conv2d_kernels_float.h
  * @author  AST Embedded Analytics Research Platform
  * @date    30-Oct-2018
  * @brief   implementation file of float conv2D kernels routines
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
#ifndef __CORE_CONV2D_KERNELS_FLOAT_H_
#define __CORE_CONV2D_KERNELS_FLOAT_H_
#pragma once

#include "ai_datatypes_defines.h"
#include "ai_datatypes_internal.h"
#include "ai_memory_manager.h"

#include "core_common.h"

/*!
 * @defgroup core_conv2d_kernels_float
 * @brief Conv2D Float Kernels Optimized Code
 * @details Conv2d kernel with weights stored following this schema (OHWI):
 * [ch_out, y, x, ch_in]
 */

/*!
  * @brief ai_conv2d_kernel_simple_f32
  * @details element wise conv2d internal kernel
  * @param node
  * @param out_data
  * @param in_data
  * @param weights_data
  * @param bias_data
  * @param beta multiplier of C
  * @param n_channel_in
  * @param n_channel_out
  * @param width_in
  * @param filt_width
  * @param filt_height
  * @param y_start
  * @param y_size
  * @param x_start
  * @param x_size
  */
AI_DECLARE_STATIC
void ai_conv2d_kernel_simple_f32(ai_node* node,
                                 ai_handle out_data,
                                 const ai_handle in_data,
                                 const ai_handle weights_data,
                                 const ai_handle bias,
                                 const ai_size n_channel_in,
                                 const ai_size n_channel_out,
                                 const ai_size width_in,
                                 const ai_size filt_width,
                                 const ai_size filt_height,
                                 ai_i32 y_start, const ai_i32 y_size,
                                 ai_i32 x_start, const ai_i32 x_size)
{
  /* Offsets pre-calculations */
  const ai_offset ch_y_offset =  (width_in - x_size) * n_channel_in;
  const ai_offset ch_weights_offset = n_channel_in * (filt_width - x_size);
  const ai_offset ch_weights_offset_2 = n_channel_in * filt_width *
                                              (filt_height - y_size);
  const ai_float *ch_weights = ((const ai_float*)weights_data) +
                (y_start * filt_width + x_start) * n_channel_in;
  const ai_offset inner_loop_size = n_channel_in * x_size;

  AI_ASSERT(in_data && out_data && weights_data && bias)

  /* Filtering */
  const ai_float bias_factor = (bias==out_data) ? 0.0f : 1.0f;

#if defined(HAS_X86) || defined(__CC_ARM)      /* X86 OR ARM Keil Compiler */
  register ai_vec4_float ch_in_f, weights_in_f;
#endif
  const ai_float* in_ptr = ((const ai_float*)in_data) +
                            y_start * width_in * n_channel_in;

  for (ai_size out = 0; out < n_channel_out; out++)
  {
    ai_float conv_out = ((const ai_float*)bias)[out] * bias_factor;
    const ai_float *ch_in = in_ptr;
    for (ai_i32 y_filt = 0; y_filt < y_size; y_filt++) {
      ai_i32 x_filt = 0;
#if defined(HAS_X86) || defined(__CC_ARM)      /* X86 OR ARM Keil Compiler */
      for ( ; x_filt<(inner_loop_size&(~0x3)); x_filt+=4 )
      {
        ch_in_f = AI_VEC4_FLOAT(ch_in);
        weights_in_f = AI_VEC4_FLOAT(ch_weights);
        conv_out += weights_in_f.a1 * ch_in_f.a1;
        conv_out += weights_in_f.a2 * ch_in_f.a2;
        conv_out += weights_in_f.a3 * ch_in_f.a3;
        conv_out += weights_in_f.a4 * ch_in_f.a4;
        ch_in    += 4;
        ch_weights += 4;
      }
#endif
      for ( ; x_filt<inner_loop_size; x_filt++) {
        conv_out += (*ch_weights++) * (*ch_in++);
      }
      ch_in      += ch_y_offset;
      ch_weights += ch_weights_offset;
    }
    ch_weights += ch_weights_offset_2;
    AI_PUSH_CONV_OUT(node, &conv_out, ((ai_float*)out_data)+out, ai_float*)
  }
}

/*!
  * @brief ai_conv2d_kernel_depthwise_f32
  * @details 
  * @param node
  * @param out_data
  * @param in_data
  * @param weights_data
  * @param bias_data
  * @param beta multiplier of C
  * @param n_channel_in
  * @param n_channel_out
  * @param width_in
  * @param filt_width
  * @param filt_height
  * @param y_start
  * @param y_size
  * @param x_start
  * @param x_size
  */
AI_DECLARE_STATIC
void ai_conv2d_kernel_depthwise_f32(ai_node* node,
                                    ai_handle out_data,
                                    const ai_handle in_data,
                                    const ai_handle weights_data,
                                    const ai_handle bias,
                                    const ai_size n_channel_in,
                                    const ai_size n_channel_out,
                                    const ai_size width_in,
                                    const ai_size filt_width,
                                    const ai_size filt_height,
                                    ai_i32 y_start, const ai_i32 y_size,
                                    ai_i32 x_start, const ai_i32 x_size)
{
  /* Pre-calculate offsets */
  const ai_size n_channel_out_for_group = n_channel_out / n_channel_in;
  const ai_offset ch_y_offset = (width_in - x_size) * n_channel_in;

  const ai_offset ch_weights_offset = (filt_width - x_size);
  const ai_offset ch_weights_offset_2 = filt_width * (filt_height - y_size);
  const ai_float *ch_weights = ((const ai_float*)weights_data) + 
                               (y_start * filt_width + x_start);
  
  AI_ASSERT(in_data && out_data && weights_data && bias)

  /* Filtering */
  const ai_float bias_factor = (bias==out_data) ? 0.0f : 1.0f;
  const ai_float* in_ptr = ((const ai_float*)in_data) + 
                           y_start*width_in*n_channel_in;
  ai_size out  = 0;
  //for(ai_size group = 0, out = 0; group < n_channel_in  ; group++) {
  for (const ai_float* in_curr=in_ptr; in_curr<in_ptr+n_channel_in; in_curr++)
  {
    //ai_ptr_offset in_base = y_start*width_in*n_channel_in + group;
    for (ai_size i = 0; i < n_channel_out_for_group; i++) {
      ai_float conv_out = ((const ai_float*)bias)[out] * bias_factor;
      const ai_float *ch_in = in_curr; //((const ai_float*)in_data) + in_base;     
      for (ai_i32 y_filt = 0; y_filt < y_size; y_filt++) {
        ai_i32 x_filt = 0;
#if defined(HAS_X86) || defined(__CC_ARM)      /* X86 OR ARM Keil Compiler */
        for ( ; x_filt<(x_size&(~0x3)); x_filt+=4 )
        {
          register ai_vec4_float weights_in_f = AI_VEC4_FLOAT(ch_weights);
          conv_out += weights_in_f.a1 * (*ch_in); ch_in += n_channel_in;
          conv_out += weights_in_f.a2 * (*ch_in); ch_in += n_channel_in;
          conv_out += weights_in_f.a3 * (*ch_in); ch_in += n_channel_in;
          conv_out += weights_in_f.a4 * (*ch_in); ch_in += n_channel_in;
          ch_weights += 4;
        }
#endif
        for( ; x_filt < x_size; x_filt++) {
          conv_out += (*ch_weights++) * (*ch_in);
          ch_in    += n_channel_in;
        }
        ch_in      += ch_y_offset;
        ch_weights += ch_weights_offset;
      }
      ch_weights += ch_weights_offset_2;
      AI_PUSH_CONV_OUT(node, &conv_out, ((ai_float*)out_data+out), ai_float*)
      out++;
    }
  }
}


/*!
  * @brief ai_conv2d_kernel_group_f32
  * @details conv2d group conv2d kernel
  * @param node
  * @param out_data
  * @param in_data
  * @param weights_data
  * @param bias_data
  * @param beta multiplier of C
  * @param n_channel_in
  * @param n_channel_out
  * @param width_in
  * @param filt_width
  * @param filt_height
  * @param y_start
  * @param y_size
  * @param x_start
  * @param x_size
  */
AI_DECLARE_STATIC
void ai_conv2d_kernel_group_f32(ai_node* node,
                                ai_handle out_data,
                                const ai_handle in_data,
                                const ai_handle weights_data,
                                const ai_handle bias,
                                const ai_size n_channel_in,
                                const ai_size n_channel_out,
                                const ai_size width_in,
                                const ai_size filt_width,
                                const ai_size filt_height,
                                const ai_size n_groups,
                                ai_i32 y_start, const ai_i32 y_size,
                                ai_i32 x_start, const ai_i32 x_size)
{
  /* Pre-calculate offsets */
  const ai_size n_channel_in_for_group = n_channel_in / n_groups;
  const ai_size n_channel_out_for_group = n_channel_out / n_groups;
  const ai_offset ch_y_offset = (width_in - x_size) * n_channel_in;
  const ai_offset ch_in_grp_offset = n_channel_in-n_channel_in_for_group;

  const ai_offset ch_weights_offset =
        n_channel_in_for_group * (filt_width - x_size);
  const ai_offset ch_weights_offset_2 =
        n_channel_in_for_group * filt_width * (filt_height - y_size);
  const ai_float *ch_weights = ((const ai_float*)weights_data) +
                      (y_start * filt_width + x_start) * n_channel_in_for_group;

  AI_ASSERT(in_data && out_data && weights_data && bias)
  
  /* Filtering */
  const ai_float bias_factor = (bias==out_data) ? 0.0f : 1.0f;
  const ai_float* in_ptr = ((const ai_float*)in_data) + 
                           y_start*width_in*n_channel_in;
  ai_size out = 0;
  for(ai_size group = 0; group < n_groups; group++)
  {
    //const ai_i32 in_base = y_start*width_in*n_channel_in + n_channel_in_for_group*group;
    for (ai_size i = 0; i < n_channel_out_for_group; i++) {
      ai_float conv_out = ((const ai_float*)bias)[out] * bias_factor;
      const ai_float *ch_in = in_ptr; //((const ai_float*)in_data) + in_base;     
      for (ai_i32 y_filt = 0; y_filt < y_size; y_filt++) {
        for(ai_i32 x_filt = 0; x_filt < x_size; x_filt++) {
          const ai_float* ch_in_end = ch_in + n_channel_in_for_group;
          for ( ; ch_in<ch_in_end; ) {
          //for (ai_size j = 0; j < n_channel_in_for_group; j++ ) {
            conv_out += (*ch_weights++) * (*ch_in++);
          }
          ch_in += ch_in_grp_offset;
        }
        ch_in      += ch_y_offset;
        ch_weights += ch_weights_offset;
      }
      ch_weights += ch_weights_offset_2;
      AI_PUSH_CONV_OUT(node, &conv_out, ((ai_float*)out_data)+out, ai_float*)
      out++;
    }
    in_ptr += n_channel_in_for_group;
  }
}

/*!
  * @brief ai_conv2d_kernel_f32
  * @details main conv2d kernel routine
  * @param node
  * @param out_data
  * @param in_data
  * @param weights_data
  * @param bias_data
  * @param beta multiplier of C
  * @param n_channel_in
  * @param n_channel_out
  * @param width_in
  * @param filt_width
  * @param filt_height
  * @param y_start
  * @param y_end
  * @param x_start
  * @param x_end
  */
AI_DECLARE_STATIC
void ai_conv2d_kernel_f32(ai_node* node,
                          ai_handle out_data,
                          const ai_handle in_data,
                          const ai_handle weights_data,
                          const ai_handle bias_data,
                          const ai_size n_channel_in,
                          const ai_size n_channel_out,
                          const ai_size width_in,
                          const ai_size filt_width,
                          const ai_size filt_height,
                          const ai_size n_groups,
                          ai_i32 y_start, const ai_i32 y_end,
                          ai_i32 x_start, const ai_i32 x_end)
{
  /* avoid null pointer exceptions in called routines */
  const ai_handle bias = (bias_data) ? bias_data : out_data;
  const ai_i32 y_size  = (y_end - y_start);
  const ai_i32 x_size  = (x_end - x_start);
  AI_ASSERT(bias)
  AI_ASSERT(x_size>=0 && y_size>=0)

  if (n_groups == 1) {
    /* ordinary convolutional layer */
    ai_conv2d_kernel_simple_f32(node, out_data, in_data, weights_data, bias,
                                n_channel_in, n_channel_out, width_in,
                                filt_width, filt_height,
                                y_start, y_size, x_start, x_size);
  } else if (n_groups == n_channel_in) {
    /* depthwise separable convolutional layer */
    ai_conv2d_kernel_depthwise_f32(node, out_data, in_data, weights_data, bias,
                                   n_channel_in, n_channel_out, width_in,
                                   filt_width, filt_height,
                                   y_start, y_size, x_start, x_size);
  } else {
    /* convolutional layer with groups (general case) */
    ai_conv2d_kernel_group_f32(node, out_data, in_data, weights_data, bias,
                               n_channel_in, n_channel_out, width_in,
                               filt_width, filt_height,n_groups,
                               y_start, y_size, x_start, x_size);
  }
}

#undef AI_PUSH_CONV_OUT

#endif    /*__CORE_CONV2D_KERNELS_FLOAT_H_*/
