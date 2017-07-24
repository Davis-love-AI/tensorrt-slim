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

#include "types.h"
#include "network.h"
#include "ssd_network.pb.h"
#include "boxes2d/boxes2d.h"

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

    /** Output CUDA tensors. */
    struct outputs_cuda {
        // 2D predictions + boxes.
        tfrt::cuda_tensor*  predictions2d;
        tfrt::cuda_tensor*  boxes2d;
        // 3D predictions + boxes.
        tfrt::cuda_tensor*  predictions3d;
        tfrt::cuda_tensor*  boxes3d;
    };
    // CUDA outputs.
    outputs_cuda outputs;

public:
    /** Construct from an equivalent protobuf object. */
    ssd_feature(const tfrt_pb::ssd_feature& feature);
    /** Number of anchors. */
    size_t num_anchors2d() const {
        return anchors2d.size();
    }
    /** Total number of anchors, with different scales. */
    size_t num_anchors2d_total() const {
        size_t nanchors{0};
        for(auto&& a : anchors2d) {
            nanchors += a.scales.size();
        }
        return nanchors;
    }
public:
    /** Get the 2D predictions as a NACHW tensor. */
    tfrt::nachw<float>::tensor predictions2d() const;
    /** Get the 2D boxes as a NACHW tensor. */
    tfrt::nachw<float>::tensor boxes2d() const;
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
    const std::vector<ssd_feature>& features() const;
    // Number of classes...
    int num_classes_2d() const {  return m_pb_ssd_network->num_classes_2d();  }
    int num_classes_3d() const {  return m_pb_ssd_network->num_classes_3d();  }
    int num_classes_seg() const {  return m_pb_ssd_network->num_classes_seg();  }

public:
    /** Load weights and configuration from .tfrt file. */
    virtual bool load_weights(const std::string& filename);
    /** Clear out the collections of network weights, to save memory. */
    // virtual void clear_weights();

    /** Raw detection of 2Dobject on one image.
     */
    tfrt::boxes2d::bboxes2d raw_detect2d(
        float* rgba, uint32_t height, uint32_t width,
        float threshold, size_t max_detections);

protected:
    /** Fill 2D bounding boxes collection from raw output tensors.
     */
    void fill_bboxes2d(
        const tfrt::nachw<float>::tensor& predictions2d,
        const tfrt::nachw<float>::tensor& boxesd2d,
        float threshold, size_t max_detections, size_t batch,
        size_t& bboxes2d_idx, tfrt::boxes2d::bboxes2d& bboxes2d) const;

protected:
    // Protobuf network object.
    std::unique_ptr<tfrt_pb::ssd_network>  m_pb_ssd_network;
    // Cached features.
    mutable std::vector<ssd_feature>  m_cached_features;
};

}

#endif
