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
#include <cstdint>
#include <cuda_fp16.h>
#include "cudaUtility.h"

#define nTPB 256

/* ============================================================================
 * float2half and half2float conversions.
 * ========================================================================== */
__global__ void float2half_array(float* din, half* dout, uint32_t dsize)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx < dsize){
        dout[idx] = __float2half(din[idx]);
    }
}
cudaError_t cuda_float2half_array(float* host_input, uint16_t* host_output, uint32_t size)
{
    // Allocate memory on device.
    float* d_in;
    half* d_out;
    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(half));
    // Copy input, call convertion kernel and copy back.
    cudaMemcpy(d_in, host_input, size*sizeof(float), cudaMemcpyHostToDevice);
    float2half_array<<<(iDivUp(size, nTPB)),nTPB>>>(d_in, d_out, size);
    cudaMemcpy(host_output, d_out, size*sizeof(half), cudaMemcpyDeviceToHost);
    // Free everything and run!
    cudaFree(d_in);
    cudaFree(d_out);
    return CUDA(cudaGetLastError());
}

__global__ void half2float_array(half* din, float* dout, uint32_t dsize)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx < dsize){
        dout[idx] = __half2float(din[idx]);
    }
}
cudaError_t cuda_half2float_array(uint16_t* host_input, float* host_output, uint32_t size)
{
    // Allocate memory on device.
    half* d_in;
    float* d_out;
    cudaMalloc(&d_in, size * sizeof(half));
    cudaMalloc(&d_out, size * sizeof(float));
    // Copy input, call convertion kernel and copy back.
    cudaMemcpy(d_in, host_input, size*sizeof(half), cudaMemcpyHostToDevice);
    half2float_array<<<(iDivUp(size, nTPB)),nTPB>>>(d_in, d_out, size);
    cudaMemcpy(host_output, d_out, size*sizeof(float), cudaMemcpyDeviceToHost);
    // Free everything and run!
    cudaFree(d_in);
    cudaFree(d_out);
    return CUDA(cudaGetLastError());
}

    // return CUDA(cudaGetLastError());