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

#include <VX/vx.h>
#include <VX/vxu.h>
#include <NVX/nvx.h>
#include <OVX/UtilityOVX.hpp>

#include <tensorflowrt.h>
#include <tensorflowrt_nets.h>
#include <tensorflowrt_models.h>

#include <cuda/cudaCHWImage.h>

DEFINE_string(network, "seg_inception2_v1", "SegNet to test.");
DEFINE_string(network_pb, "../data/networks/seg_inception2_v1.25.tfrt32",
    "Network protobuf parameter file.");
DEFINE_int32(batch_size, 2, "Batch size.");
DEFINE_int32(height, 224, "Input height.");
DEFINE_int32(width, 384, "Input width.");

/* ============================================================================
 * Static collection of nets.
 * ========================================================================== */
std::unique_ptr<tfrt::network>&& networks_map(const std::string& key)
{
    static std::map<std::string, std::unique_ptr<tfrt::network> > nets;
    // Fill the map at first call!
    if(nets.empty()) {
        nets["ssd_inception2_v0"] = std::make_unique<ssd_inception2_v0::net>();
        nets["seg_inception2_v1"] = std::make_unique<seg_inception2_v1::net>();
    }
    return std::move(nets.at(key));
}

void fill_input_tensor(tfrt::nchw<uint8_t>& t)
{

    CUDA(cudaDeviceSynchronize());
}

int main(int argc, char **argv)
{
    // google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Create network.
    LOG(INFO) << "Loading network: " << FLAGS_network;
    auto segnet = networks_map(FLAGS_network);
    segnet->load(FLAGS_network_pb);
    // Input CUDA buffer, uint8.
    tfrt::cuda_tensor_u8  in_tensor{"input_u8", {1, 3, FLAGS_height, FLAGS_width}};
    in_tensor.allocate();

    return 0;
}
