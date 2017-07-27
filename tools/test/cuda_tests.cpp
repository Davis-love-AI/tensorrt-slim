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
#include <sys/stat.h>
#include <time.h>
#include <chrono>

#include <half.hpp>

#include <tensorflowrt.h>
#include <tensorflowrt_nets.h>
#include <tensorflowrt_ssd_models.h>

DEFINE_double(value, 1.0, "Float value.");

// CUDA methods...
void cuda_half2float_array(float* input, uint16_t* output, uint32_t size);

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Half precision tests.
    LOG(INFO) << "Half size: " << sizeof(half_float::half);
    // Convert a vector of float to half.
    std::vector<float> vec_f = {1.0};
    std::vector<uint16_t> vec_h = {0};

    cuda_half2float_array(vec_f.data(), vec_h.data(), vec_f.size());



    return 0;
}
