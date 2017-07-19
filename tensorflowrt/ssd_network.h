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
#ifndef TFRT_SSD_NETWORK_H
#define TFRT_SSD_NETWORK_H

// #include <cmath>
// #include <memory>
#include <NvInfer.h>

#include "network.h"
#include "ssd_network.pb.h"

namespace tfrt
{

/** Structure representing 2D SSD anchors.
 */
struct ssd_anchor2d
{
    // Anchor reference size, in pixels.
    float size;
    // Anchor scales: height / width.
    std::vector<float> scales;

public:
    /** Construct from equivalent protobuf object. */
    ssd_anchor2d(const tfrt_pb::ssd_anchor2d& anchor);
};
/** Structure representing a feature of an SSD network.
 */
struct ssd_feature
{
    // Parameters: basic name + full name + shape.
    std::string name;
    std::string fullname;
    nvinfer1::DimsCHW shape;
    // List of 2D anchors associated.
    std::vector<ssd_anchor2d> anchors2d;

public:
    /** Construct from an equivalent protobuf object. */
    ssd_feature(const tfrt_pb::ssd_feature& feature);
};

class ssd_network : public tfrt::network
{
public:
    /** Create SSD network, specifying the name.
     */
    ssd_network(std::string name) :
        tfrt::network(name),
        m_pb_ssd_network(std::make_unique<tfrt_pb::ssd_network>()) {
    }
    virtual ~ssd_network();

public:
    /** Get the number of features. */
    size_t nb_features() const;
    /** Get the list of features. */
    std::vector<ssd_feature> features() const;
    // Number of classes...
    int num_classes_2d() const {  return m_pb_ssd_network->num_classes_2d();  }
    int num_classes_3d() const {  return m_pb_ssd_network->num_classes_3d();  }
    int num_classes_seg() const {  return m_pb_ssd_network->num_classes_seg();  }

public:
    /** Load weights and configuration from .tfrt file. */
    virtual bool load_weights(const std::string& filename);
    /** Clear out the collections of network weights, to save memory. */
    // virtual void clear_weights();

protected:
    // Protobuf network object.
    std::unique_ptr<tfrt_pb::ssd_network>  m_pb_ssd_network;
};

}

#endif
