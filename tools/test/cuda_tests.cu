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

#define DSIZE 4
#define SCF 0.5f
#define nTPB 256


/**
 * iDivUp
 * @ingroup util
 */
inline __device__ __host__ int iDivUp( int a, int b )  	{ return (a % b != 0) ? (a / b + 1) : (a / b); }

__global__ void half_scale_kernel(float *din, float *dout, int dsize){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  if (idx < dsize){
    half scf = __float2half(SCF);
    half kin = __float2half(din[idx]);
    half kout;
#if __CUDA_ARCH__ >= 530
    kout = __hmul(kin, scf);
#else
    kout = __float2half(__half2float(kin)*__half2float(scf));
#endif
    dout[idx] = __half2float(kout);
    }
}

/* ============================================================================
 * float2half and half2float
 * ========================================================================== */
__global__ void float2half_array(float* din, half* dout, uint32_t dsize)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx < dsize){
        dout[idx] = __float2half(din[idx]);
    }
}
void cuda_float2half_array(float* host_input, uint16_t* host_output, uint32_t size)
{
    // Allocate memory on device.
    float* d_in;
    half* d_out;
    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(half));
    // Copy input.
    cudaMemcpy(d_in, host_input, size*sizeof(float), cudaMemcpyHostToDevice);
    // Call the convertion kernel...
    float2half_array<<<(iDivUp(size, nTPB)),nTPB>>>(d_in, d_out, size);
    // Copy back.
    cudaMemcpy(host_output, d_out, size*sizeof(half), cudaMemcpyDeviceToHost);
}

__global__ void half2float_array(half* din, float* dout, uint32_t dsize)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx < dsize){
        dout[idx] = __half2float(din[idx]);
    }
}
void cuda_half2float_array(uint16_t* host_input, float* host_output, uint32_t size)
{
    // Allocate memory on device.
    half* d_in;
    float* d_out;
    cudaMalloc(&d_in, size * sizeof(half));
    cudaMalloc(&d_out, size * sizeof(float));
    // Copy input.
    cudaMemcpy(d_in, host_input, size*sizeof(half), cudaMemcpyHostToDevice);
    // Call the convertion kernel...
    half2float_array<<<(iDivUp(size, nTPB)),nTPB>>>(d_in, d_out, size);
    // Copy back.
    cudaMemcpy(host_output, d_out, size*sizeof(float), cudaMemcpyDeviceToHost);
}
// half_scale_kernel<<<(DSIZE+nTPB-1)/nTPB,nTPB>>>(din, dout, DSIZE);
