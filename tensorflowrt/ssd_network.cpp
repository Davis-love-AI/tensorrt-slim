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
#include <glog/logging.h>

#include "utils.h"
#include "ssd_network.h"
#include "cuda/cudaImageNet.h"

namespace tfrt
{

/* ============================================================================
 * tfrt::ssd_anchor2d methods.
 * ========================================================================== */
ssd_anchor2d::ssd_anchor2d(const tfrt_pb::ssd_anchor2d& anchor) :
    size{anchor.size()},
    scales{anchor.scales().begin(), anchor.scales().end()}
{}

/* ============================================================================
 * tfrt::ssd_feature methods.
 * ========================================================================== */
ssd_feature::ssd_feature(const tfrt_pb::ssd_feature& feature) :
    name{feature.name()}, fullname{feature.fullname()},
    shape{dims_pb(feature.shape())},
    anchors2d{feature.anchors2d().begin(), feature.anchors2d().end()},
    outputs{nullptr, nullptr, nullptr, nullptr}
{}

/* ============================================================================
 * tfrt::ssd_network methods.
 * ========================================================================== */
ssd_network::~ssd_network()
{
}
size_t ssd_network::nb_features() const
{
    return m_pb_ssd_network->features_size();
}
const std::vector<ssd_feature>& ssd_network::features() const
{
    if(m_cached_features.size() == 0) {
        // Copy from protobuf objects.
        std::vector<ssd_feature> features{
            m_pb_ssd_network->features().begin(), m_pb_ssd_network->features().end()};
        // A bit hacky here...
        for(size_t i = 0 ; i < features.size() ; ++i) {
            auto& fout = features[i].outputs;
            auto& pb_fout = m_pb_ssd_network->features(i).outputs();
            // Find CUDA tensors.
            fout.predictions2d = this->find_cuda_output(pb_fout.predictions2d());
            fout.boxes2d = this->find_cuda_output(pb_fout.boxes2d());
            fout.predictions3d = this->find_cuda_output(pb_fout.predictions3d());
            fout.boxes3d = this->find_cuda_output(pb_fout.boxes3d());
        }
        m_cached_features = features;
    }
    return m_cached_features;
}

bool ssd_network::load_weights(const std::string& filename)
{
    // Free everything!
    m_cached_features.empty();

    LOG(INFO) << "Loading SSD network parameters and weights from: " << filename;
    bool r = parse_protobuf(filename, m_pb_ssd_network.get());
    // Hacky swaping!
    m_pb_network.reset(m_pb_ssd_network->release_network());
    return r;
}

tfrt::boxes2d::bboxes2d ssd_network::raw_detect2d(
        float* rgba, uint32_t height, uint32_t width, float threshold, size_t max_detections)
{
    CHECK(rgba) << "Invalid image buffer.";
    CHECK(height) << "Invalid image height.";
    CHECK(width) << "Invalid image width.";
    // Downsample and convert to RGB.
    cudaError_t r = cudaPreImageNet((float4*)rgba, width, height,
        m_cuda_input.cuda, m_cuda_input.shape.w(), m_cuda_input.shape.h());
    CHECK_EQ(r, cudaSuccess) << "Failed to resize image to ImageNet network input shape."
        << "CUDA error: " << r;

    // Execute TensorRT network (batch size = 1) TODO.
    m_nv_context->execute(1, (void**)m_cached_bindings.data());
    // Simple post-processing of outputs of every feature layer.
    const auto& features = this->features();


    tfrt::boxes2d::bboxes2d bboxes2d{max_detections};
    return bboxes2d;
}

}
