/* ============================================================================
# [2017] - Robik AI Ltd - Paul Balanca
# All Rights Reserved.

# NOTICE: All information contained herein is, and remains
# the property of Robik AI Ltd, and its suppliers
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Robik AI Ltd
# and its suppliers and may be covered by U.S., European and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from Robik AI Ltd.
# =========================================================================== */
#ifndef TFRT_CUDA_SEGMENTATION_H
#define TFRT_CUDA_SEGMENTATION_H

#include "cudaUtility.h"

/** Overlay a segmentation result with an original image.
 * The original image is supposed to uint8 RGB(X) format.
 * The mask is supposed to be an uint8 monochrome image (or equivalent).
 * The masks is upscaled to the dimension of the input image.
 */
cudaError_t cuda_seg_overlay(
    uint8_t* d_img, uint8_t* d_mask, 
    uint32_t img_width, uint32_t img_height,
    uint32_t img_stride_x, uint32_t img_stride_y,
    uint32_t mask_width, uint32_t mask_height,
    uint32_t mask_stride_x, uint32_t mask_stride_y);

/** Convert a RGBX image to CHW format. Both are supposed to be stored on
 * device/CUDA space and have same size. The input image is uint8 RGBA format.
 */
// cudaError_t cuda_rgba_to_chw(uint8_t* d_input, float* d_output, 
//     uint32_t width, uint32_t height, uint32_t stride_x, uint32_t stride_y);

#endif
