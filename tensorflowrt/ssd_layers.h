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
#ifndef TFRT_SSD_LAYERS_H
#define TFRT_SSD_LAYERS_H

#include <tuple>
#include "layers.h"

namespace tfrt
{
typedef std::pair<nvinfer1::ITensor*, nvinfer1::ITensor*> tensors_pair;
typedef std::tuple<nvinfer1::ITensor*, nvinfer1::ITensor*> tensors_tuple2;
typedef std::tuple<nvinfer1::ITensor*, nvinfer1::ITensor*, nvinfer1::ITensor*>
    tensors_tuple3;
typedef std::tuple<nvinfer1::ITensor*, nvinfer1::ITensor*, nvinfer1::ITensor*, nvinfer1::ITensor*>
    tensors_tuple4;

/* ============================================================================
 * SSD layers.
 * ========================================================================== */
class ssd_boxes2d_decode : public tfrt::layer
{
public:
    ssd_boxes2d_decode(const tfrt::scope& sc, const std::string& lname="decode") :
        layer(sc, lname) {}

    /** Add the decoding layer to network graph. Perform two scaling operations,
     * a channelwise and then an elementwise.
     */
    virtual nvinfer1::ITensor* operator()(nvinfer1::ITensor* net) {
        LOG(INFO) << "LAYER SSD boxes2d decode '" << this->m_scope.name() << "'. "
            << "Input shape: " << dims_str(net->getDimensions());
        // Channel and elementwise scalings.
        net = tfrt::scale(m_scope, "scale_channel").mode(nvinfer1::ScaleMode::kCHANNEL)(net);
        net = tfrt::scale(m_scope, "scale_elementwise").mode(nvinfer1::ScaleMode::kELEMENTWISE)(net);
        return this->mark_output(net);
    }
};

/* ============================================================================
 * SSD blocks.
 * ========================================================================== */
/* 2D bounding block: classification + boxes regression.
 * Return a pair of tensors <classification, boxes2d>
 */
inline tensors_pair ssd_boxes2d_block(
    nvinfer1::ITensor* net, tfrt::scope sc,
    int num_anchors, int num_classes,
    bool mark_outputs=true, bool decode_boxes=true)
{
    typedef tfrt::convolution2d<tfrt::ActivationType::NONE, tfrt::PaddingType::SAME, false>  conv2d;
    // Classification + boxes regression convolutions.
    auto net_cls = conv2d(sc, "conv_cls")
        .noutputs(num_anchors * num_classes).ksize({3, 3})(net);
    auto net_loc = conv2d(sc, "conv_loc")
        .noutputs(num_anchors * 4).ksize({3, 3})(net);
    // Decode boxes.
    if(decode_boxes) {
        net_loc = ssd_boxes2d_decode(sc, "decoce")(net_loc);
    }
    return std::make_pair(net_cls, net_loc);
}


}
#endif
