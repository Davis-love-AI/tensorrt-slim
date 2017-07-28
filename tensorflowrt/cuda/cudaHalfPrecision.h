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

#include "cudaUtility.h"

#ifndef TFRT_CUDA_HALFPRECISION_H
#define TFRT_CUDA_HALFPRECISION_H

/** Convert an array of float into an array of half precision floats.
 * Take as input host memory allocated arrays, and size.
 * Extremely not well optimised!!! Allocation of memory on device at every call.
 * Note: use type uint16_t for half precision float storage.
 */
cudaError_t cuda_float2half_array(float* host_input, uint16_t* host_output, uint32_t size);

/** Convert an array of float into an array of half precision floats.
 * Take as input host memory allocated arrays, and size.
 * Extremely not well optimised!!! Allocation of memory on device at every call.
 * Note: use type uint16_t for half precision float storage.
 */
cudaError_t cuda_half2float_array(uint16_t* host_input, float* host_output, uint32_t size);

#endif
