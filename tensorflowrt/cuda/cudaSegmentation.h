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

/** Post-processing of RAW segmentation probabilities: find classes
 * and scores. If necessary, also apply transformation (for instance, stabilization).
 */
cudaError_t cuda_seg_post_process(
    float* d_raw_prob, uint8_t* d_classes, float* d_scores, float* d_tr_matrix,
    uint32_t seg_width, uint32_t seg_height, uint32_t num_classes,
    bool empty_class, float threshold);

#endif
