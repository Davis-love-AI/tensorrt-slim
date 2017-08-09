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

#include <tensorflowrt.h>
#include <tensorflowrt_nets.h>
#include <tensorflowrt_models.h>

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
    google::InitGoogleLogging(argv[0]);
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
    return 0;
}
