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
std::vector<ssd_feature> ssd_network::features() const
{
    // Copy from protobuf objects.
    std::vector<ssd_feature> features{
        m_pb_ssd_network->features().begin(), m_pb_ssd_network->features().end()};
    for(size_t i = 0 ; i < features.size() ; ++i) {
        auto& fout = features[i].outputs;
        auto& pb_fout = m_pb_ssd_network->features(i).outputs();
        // Find CUDA tensors.
        fout.predictions2d = this->find_cuda_output(pb_fout.predictions2d());
        fout.boxes2d = this->find_cuda_output(pb_fout.boxes2d());
        fout.predictions3d = this->find_cuda_output(pb_fout.predictions3d());
        fout.boxes3d = this->find_cuda_output(pb_fout.boxes3d());
    }
    return features;
}

bool ssd_network::load_weights(const std::string& filename)
{
    LOG(INFO) << "Loading SSD network parameters and weights from: " << filename;
    bool r = parse_protobuf(filename, m_pb_ssd_network.get());
    // Hacky swaping!
    m_pb_network.reset(m_pb_ssd_network->release_network());
    return r;
}

tfrt::boxes2d::bboxes2d ssd_network::raw_detect2d(
        float* rgba, uint32_t height, uint32_t width, float threshold, size_t max_detections)
{

}

}
