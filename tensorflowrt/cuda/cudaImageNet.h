/*
 * inference-101
 */

#ifndef __CUDA_IMAGENET_H__
#define __CUDA_IMAGENET_H__


#include "cudaUtility.h"

/** Resize and normalize an RGBA input tensor, and convert RGB.
 */
cudaError_t cudaPreImageNetMean(
    float4* input, size_t inputWidth, size_t inputHeight,
    float* output, size_t outputWidth, size_t outputHeight,
    const float3& mean_value);

/** Resize an RGBA input tensor, and convert RGB.
 */
cudaError_t cudaPreImageNet(
    float4* input, size_t inputWidth, size_t inputHeight,
	float* output, size_t outputWidth, size_t outputHeight);

#endif
