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
 * tfrt::ssd_network methods.
 * ========================================================================== */
ssd_anchor2d::ssd_anchor2d(const tfrt_pb::ssd_anchor2d& anchor) :
    size{anchor.size()},
    scales{anchor.scales().begin(), anchor.scales().end()}
{}
ssd_feature::ssd_feature(const tfrt_pb::ssd_feature& feature) :
    name{feature.name()}, fullname{feature.fullname()},
    shape{dims_pb(feature.shape())},
    anchors2d{feature.anchors2d().begin(), feature.anchors2d().end()}
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
    return std::vector<ssd_feature>{
        m_pb_ssd_network->features().begin(), m_pb_ssd_network->features().end()};
}

bool ssd_network::load_weights(const std::string& filename)
{
    LOG(INFO) << "Loading SSD network parameters and weights from: " << filename;
    bool r = parse_protobuf(filename, m_pb_ssd_network.get());
    // Hacky swaping!
    m_pb_network.reset(m_pb_ssd_network->release_network());
    return r;
}

}
