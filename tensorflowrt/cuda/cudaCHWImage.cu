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

// ========================================================================== //
// RGBX <=> CHW.
// ========================================================================== //
__global__ void kernel_rgbx_to_chw(uint8_t* input, float* output, 
    int width, int height, uint32_t stride_x, uint32_t stride_y)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = width * height;
    if( x >= width || y >= height ) {
        return;
    }
    // Use stride to compute the index.
    const int idx = y * stride_y + x * stride_x;
    // Simple re-ordering. Nothing fancy!
    const float3 rgb = make_float3(input[idx+0], input[idx+1], input[idx+2]);
    output[n * 0 + y * width + x] = rgb.x;
    output[n * 1 + y * width + x] = rgb.y;
    output[n * 2 + y * width + x] = rgb.z;
}
cudaError_t cuda_rgba_to_chw(uint8_t* d_input, float* d_output, 
    uint32_t width, uint32_t height, uint32_t stride_x, uint32_t stride_y)
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
    kernel_rgbx_to_chw<<<gridDim, blockDim>>>(d_input, d_output, width, height, stride_x, stride_y);
    return CUDA(cudaGetLastError());
}


__global__ void kernel_chw_to_rgbx(float* input, uint8_t* output, 
    int width, int height, uint32_t stride_x, uint32_t stride_y)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = width * height;
    if( x >= width || y >= height ) {
        return;
    }
    // Get the pixel values.
    const uchar4 rgb = make_uchar4(
        input[n * 0 + y * width + x],
        input[n * 1 + y * width + x],
        input[n * 2 + y * width + x], 255);
    // Use stride to compute the image index.
    const int idx_out = y * stride_y + x * stride_x;
    output[idx_out+0] = rgb.x;
    output[idx_out+1] = rgb.y;
    output[idx_out+2] = rgb.z;
    output[idx_out+3] = rgb.w;
}
cudaError_t cuda_chw_to_rgba(float* d_input, uint8_t* d_output, 
    uint32_t width, uint32_t height, uint32_t stride_x, uint32_t stride_y)
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
    kernel_chw_to_rgbx<<<gridDim, blockDim>>>(d_input, d_output, width, height, stride_x, stride_y);
    return CUDA(cudaGetLastError());
}

// ========================================================================== //
// RGBX <=> CHW resize.
// ========================================================================== //
__global__ void kernel_rgbx_to_chw_resize(uint8_t* input, float* output, 
    uint32_t inwidth, uint32_t inheight, uint32_t instride_x, uint32_t instride_y,
    uint32_t outwidth, uint32_t outheight)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = outwidth * outheight;
    if( x >= outwidth || y >= outheight ) {
        return;
    }
    // Input coordinates.
    const int in_x = round(float(x) / float(outwidth) * float(inwidth)); 
    const int in_y = round(float(y) / float(outheight) * float(inheight)); 
    // Use stride to compute the index.
    const int idx = in_y * instride_y + in_x * instride_x;
    // Simple re-ordering. Nothing fancy!
    const float3 rgb = make_float3(input[idx+0], input[idx+1], input[idx+2]);
    output[n * 0 + y * outwidth + x] = rgb.x;
    output[n * 1 + y * outwidth + x] = rgb.y;
    output[n * 2 + y * outwidth + x] = rgb.z;
}
cudaError_t cuda_rgba_to_chw_resize(uint8_t* d_input, float* d_output, 
    uint32_t inwidth, uint32_t inheight, uint32_t instride_x, uint32_t instride_y,
    uint32_t outwidth, uint32_t outheight)
{
    if( !d_input || !d_output ) {
        return cudaErrorInvalidDevicePointer;
    }
    if( inwidth == 0 || inheight == 0 || outwidth == 0 || outheight == 0) {
        return cudaErrorInvalidValue;
    }
    // Launch convertion kernel.
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(outwidth, blockDim.x), iDivUp(outheight, blockDim.y));
    kernel_rgbx_to_chw_resize<<<gridDim, blockDim>>>(d_input, d_output, 
        inwidth, inheight, instride_x, instride_y, outwidth, outheight);
    return CUDA(cudaGetLastError());
}
