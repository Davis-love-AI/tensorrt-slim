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

__global__ void kernel_rgbx_to_chw(uint8_t* input, float* output, int height, int width)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = width * height;
    if( x >= width || y >= height ) {
        return;
    }
    // Use stride to compute the index?
    const int idx = y * width * 4 + x * 4;
    // Simple re-ordering. Nothing fancy!
    const float3 rgb = make_float3(input[idx], input[idx+1], input[idx+2]);
    output[n * 0 + y * width + x] = rgb.x;
    output[n * 1 + y * width + x] = rgb.y;
    output[n * 2 + y * width + x] = rgb.z;
}
cudaError_t cuda_rgba_to_chw(uint8_t* d_input, float* d_output, uint32_t height, uint32_t width)
{
    if( !d_input || !d_output ) {
        return cudaErrorInvalidDevicePointer;
    }
    if( height == 0 || width == 0) {
        return cudaErrorInvalidValue;
    }
    // Launch convertion kernel.
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(width, blockDim.x), iDivUp(height, blockDim.y));
    kernel_rgbx_to_chw<<<gridDim, blockDim>>>(d_input, d_output, height, width);
    return CUDA(cudaGetLastError());
}
