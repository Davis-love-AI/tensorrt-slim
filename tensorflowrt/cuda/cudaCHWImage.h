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
#ifndef TFRT_CUDA_CHW_IMAGE_H
#define TFRT_CUDA_CHW_IMAGE_H

#include "cudaUtility.h"

/** Convert a RGBX image to CHW format. Both are supposed to be stored on
 * device/CUDA space. Note: input and output are supposed to have equal size.
 */
cudaError_t cuda_rgba_to_chw(float4* d_input, float* d_output, uint32_t height, uint32_t width);

#endif
