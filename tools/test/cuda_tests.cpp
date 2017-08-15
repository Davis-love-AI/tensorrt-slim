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
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sys/stat.h>
#include <time.h>
#include <chrono>

#include <half.hpp>

#include <VX/vx.h>
#include <VX/vxu.h>
#include <NVX/nvx.h>
#include <OVX/UtilityOVX.hpp>

#include <tensorflowrt.h>
#include <tensorflowrt_nets.h>
#include <tensorflowrt_models.h>

#include <cuda/cudaCHWImage.h>

DEFINE_double(value, 1.0, "Float value.");

// CUDA methods...
void cuda_float2half_array(float* host_input, uint16_t* host_output, uint32_t size);
void cuda_half2float_array(uint16_t* host_input, float* host_output, uint32_t size);


void print_half(uint16_t* half)
{
    uint8_t* ptr = (uint8_t*) half;
    LOG(INFO) << "HALF: 0x" << std::hex << int(ptr[0]) << int(ptr[1]) << std::endl;
}

int main(int argc, char **argv)
{
    // google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Half precision tests.
    LOG(INFO) << "Half size: " << sizeof(half_float::half);
    // Convert a vector of float to half.
    std::vector<float> vec_f = {FLAGS_value};
    std::vector<float> vec_f2 = {0.0};
    std::vector<uint16_t> vec_h = {0};

    LOG(INFO) << "Original data: " << std::setprecision(6) << vec_f[0] << " | " << vec_f2[0];
    cuda_float2half_array(vec_f.data(), vec_h.data(), vec_f.size());
    cuda_half2float_array(vec_h.data(), vec_f2.data(), vec_f.size());
    LOG(INFO) << "Half data: " << std::setprecision(6) << vec_f[0] << " | " << vec_f2[0];
    print_half(vec_h.data());

    // VX image testing...
    ovxio::ContextGuard context;
    vxRegisterLogCallback(context, &ovxio::stdoutLogCallback, vx_false_e);
    vxDirective(context, VX_DIRECTIVE_ENABLE_PERFORMANCE);

    vx_pixel_value_t initVal;
    initVal.RGBX[0] = 255;
    initVal.RGBX[1] = 128;
    initVal.RGBX[2] = 64;
    initVal.RGBX[3] = 32;
    vx_image frame = vxCreateUniformImage(context, 100, 100, VX_DF_IMAGE_RGBX, &initVal);

    vx_rectangle_t rect;
    vxGetValidRegionImage(frame, &rect);

    vx_map_id src_map_id;
    vx_uint8* src_ptr;
    vx_imagepatch_addressing_t src_addr;
    NVXIO_SAFE_CALL(vxMapImagePatch(frame, nullptr, 0, &src_map_id, &src_addr, (void **)&src_ptr, VX_READ_ONLY, NVX_MEMORY_TYPE_CUDA, 0));
    LOG(INFO) << "CUDA ptr: " << (void*)src_ptr << ", ID: " << src_map_id;
    LOG(INFO) << "CUDA addr: " << src_addr.stride_x << " | " << src_addr.stride_y;
    LOG(INFO) << "CUDA addr: " << src_addr.dim_x << " | " << src_addr.dim_y;
    LOG(INFO) << "CUDA addr: " << src_addr.scale_x << " | " << src_addr.scale_y;
    
    // Interaction with cuda tensor?
    tfrt::cuda_tensor ctensor{"test", {1, 3, 100, 100}};
    ctensor.allocate();
    LOG(INFO) << "CUDA tensor: " << tfrt::dims_str(ctensor.shape);
    auto r = cuda_rgba_to_chw(src_ptr, ctensor.cuda, 100, 100);
    
    
    CUDA(cudaDeviceSynchronize());
    int start = 10000;
    for (int i = start ; i < start+200 ; ++i) {
        LOG(INFO) << "CUDA tensor: " << i << " | " << ctensor.cpu[i] << " | " << ctensor.cpu[i];
    }
    

    
    return 0;
}
