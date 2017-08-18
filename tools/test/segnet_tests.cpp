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
DEFINE_int32(height, 225, "Input height.");
DEFINE_int32(width, 385, "Input width.");

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

template <class T>
void print_tensor_hw(const typename tfrt::nchw<T>::tensor& t, int _i, int _j, int size, bool nhwc=false)
{
    int c = 0;
    // LOG(INFO) << "Sub-tensor of size: " << size;
    for (int i = _i ; i < _i+size ; ++i) {
        for (int j = _j ; j < _j+size ; ++j) {
            if (nhwc) {
                std::cout << float(t(0, i, j, c)) << " ";                 
            }
            else {
                std::cout << float(t(0, c, i, j)) << " "; 
            }
        }
        std::cout << std::endl;
    }
}
void fill_input_tensor(tfrt::nchw<uint8_t>::tensor_map&& t)
{
    // Fill with pseudo-random values. NHWC organisation!
    // uint8_t c = 0;
    for (long n = 0 ; n < t.dimension(0) ; ++n) {
        for (long i = 0 ; i < t.dimension(1) ; ++i) {
            for (long j = 0 ; j < t.dimension(2) ; ++j) {
                // Channel loop.
                for (long k = 0 ; k < t.dimension(3) ; ++k) {
                    t(n, i, j, k) = (i+1)*(j+2)*(k+3) % 256;
                    // c = (i+1)*(j+2)*(k+3) % 256;
                }
            }
        }
    }
    // CUDA(cudaDeviceSynchronize());
}

int main(int argc, char **argv)
{
    // google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Create network.
    LOG(INFO) << "Loading network: " << FLAGS_network;
    auto segnet = networks_map(FLAGS_network);
    segnet->load(FLAGS_network_pb);
    auto inshape = segnet->input_shape();

    // Input CUDA buffer, uint8.
    tfrt::cuda_tensor_u8  in_tensor{"input_u8", {1, inshape.h(), inshape.w(), 4}};
    in_tensor.allocate();
    fill_input_tensor(in_tensor.tensor());
    LOG(INFO) << "Sub-u8 tensor... " << tfrt::dims_str(in_tensor.shape);
    print_tensor_hw<uint8_t>(in_tensor.tensor(), 0, 0, 7, true);

    // Copy input to network input buffer and execute network.
    LOG(INFO) << "Inference on network: " << FLAGS_network;
    cuda_rgba_to_chw(in_tensor.cuda, segnet->m_cuda_input.cuda, 
        inshape.w(), inshape.h(), 4, inshape.w()*4);
    CUDA(cudaDeviceSynchronize());
    segnet->m_nv_context->execute(FLAGS_batch_size, (void**)segnet->m_cached_bindings.data());
    CUDA(cudaDeviceSynchronize());
    
    // Printing information...
    LOG(INFO) << "Sub-input tensor... " << tfrt::dims_str(segnet->m_cuda_input.shape);
    print_tensor_hw<float>(segnet->m_cuda_input.tensor(), 0, 0, 7, false);

    for(size_t i = 0 ; i < segnet->m_cuda_outputs.size() ; ++i) {
        LOG(INFO) << "Sub-output tensor: " << segnet->m_cuda_outputs[i].name << " | " << tfrt::dims_str(segnet->m_cuda_outputs[i].shape);
        print_tensor_hw<float>(segnet->m_cuda_outputs[i].tensor(), 0, 0, 7, false);
    }

    return 0;
}
