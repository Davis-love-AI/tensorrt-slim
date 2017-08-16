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
#include <vector>
#include <array>
#include "cudaUtility.h"

#define ALPHA_COLOR 128

__global__ void kernel_seg_overlay(
    uint8_t* d_img, uint8_t* d_mask, float2 scale,
    uint32_t img_width, uint32_t img_height,
    uint32_t img_stride_x, uint32_t img_stride_y,
    uint32_t mask_width, uint32_t mask_height,
    uint32_t mask_stride_x, uint32_t mask_stride_y)
{
    // UGLY! I know!
    static uchar4 colors[] = {
        {0, 0, 0, ALPHA_COLOR},
        {0, 0, 142, ALPHA_COLOR},
        {0, 110, 100, ALPHA_COLOR},
        {30, 30, 70, ALPHA_COLOR},
        {0, 60, 100, ALPHA_COLOR},
        {119, 11, 32, ALPHA_COLOR},
        {50, 0, 230, ALPHA_COLOR},
        {220, 20, 60, ALPHA_COLOR},
        {128, 64, 128, ALPHA_COLOR},
        {244, 35, 232, ALPHA_COLOR},
        {152, 251, 152, ALPHA_COLOR},
        {220, 220, 0, ALPHA_COLOR},
        {250, 170, 30, ALPHA_COLOR},
        {107, 142, 35, ALPHA_COLOR},
        {70, 70, 70, ALPHA_COLOR},
        {70, 130, 180, ALPHA_COLOR},
        {190, 153, 153, ALPHA_COLOR},
        {153, 53, 153, ALPHA_COLOR},
        {250, 170, 160, ALPHA_COLOR}
    };

    // Image coordinates.
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if( x >= img_width || y >= img_height ) {
        return;
    }
    const int idx = y * img_stride_y + x * img_stride_x;
    // Convertion to mask coordinates.
    const int mx = max(min(int((float)x * scale.x), mask_width-1), 0);
    const int my = max(min(int((float)y * scale.y), mask_height-1), 0);

    const int mval = d_mask[my * mask_stride_y + mx * mask_stride_x];
    // const uchar3 color = make_uchar3(colors[mval].x, colors[mval].y, colors[mval].z);
    // const float alpha = float(colors[mval].w) / 255.f;
    // const int mval = 1;
    const uchar3 color = make_uchar3(colors[mval].x, colors[mval].y, colors[mval].z);
    const float alpha = float(colors[mval].w) / 255.f;
    // const uchar3 color = make_uchar3(255, 0, 0); 
    // const float alpha = 0.3;

    // Mask overlay.
    const uchar3 rgb = make_uchar3(
        d_img[idx + 0] * (1. - alpha) + color.x * alpha, 
        d_img[idx + 1] * (1. - alpha) + color.y * alpha, 
        d_img[idx + 2] * (1. - alpha) + color.z * alpha);
    d_img[idx + 0] = rgb.x;
    d_img[idx + 1] = rgb.y;
    d_img[idx + 2] = rgb.z;
}
cudaError_t cuda_seg_overlay(
    uint8_t* d_img, uint8_t* d_mask,
    uint32_t img_width, uint32_t img_height,
    uint32_t img_stride_x, uint32_t img_stride_y,
    uint32_t mask_width, uint32_t mask_height,
    uint32_t mask_stride_x, uint32_t mask_stride_y)
{
    if( !d_img || !d_mask ) {
        return cudaErrorInvalidDevicePointer;
    }
    if( img_width == 0 || img_height == 0 || img_stride_x == 0 || img_stride_y == 0 ) {
        return cudaErrorInvalidValue;
    }
    if( mask_width == 0 || mask_height == 0 || mask_stride_x == 0 || mask_stride_y == 0 ) {
        return cudaErrorInvalidValue;
    }
    // Scale between image and mask.
    const float2 scale = make_float2( float(mask_width) / float(img_width),
                                      float(mask_height) / float(img_height) );
    // Launch kernel!
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(img_width,blockDim.x), iDivUp(img_height,blockDim.y));
    kernel_seg_overlay<<<gridDim, blockDim>>>(d_img, d_mask, scale, 
        img_width, img_height, img_stride_x, img_stride_y, 
        mask_width, mask_height, mask_stride_x, mask_stride_y);
    return cudaGetLastError();   
}