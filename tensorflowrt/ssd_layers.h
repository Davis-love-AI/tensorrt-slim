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
/* 2D bounding block: predictions + boxes regression.
 * Configured as an output layer by default.
 * Return a pair of tensors <predictions, boxes2d> as outputs.
 */
class ssd_boxes2d_block : public tfrt::layer
{
public:
    ssd_boxes2d_block(const tfrt::scope& sc, const std::string& lname) :
        layer(sc, lname), m_num_classes{0}, m_num_anchors{0}, m_decode_boxes{true}
    {
        this->is_output(true);
    }

    /** Named parameter: number of classes. */
    ssd_boxes2d_block& num_classes(int num_classes) {
        m_num_classes = num_classes;
        return *this;
    }
    /** Named parameter: number of anchors. */
    ssd_boxes2d_block& num_anchors(int num_anchors) {
        m_num_anchors = num_anchors;
        return *this;
    }
    /** Named parameter: decode 2d boxes. */
    ssd_boxes2d_block& decode_boxes(bool decode_boxes) {
        m_decode_boxes = decode_boxes;
        return *this;
    }
    // Getters...
    int num_classes() const {  return m_num_classes;  }
    int num_anchors() const {  return m_num_anchors;  }
    bool decode_boxes() const {  return m_decode_boxes;  }

public:
    /** Add the decoding layer to network graph. Perform two scaling operations,
     * a channelwise and then an elementwise.
     * Return a nullptr. Needs to use outputs() to get all outputs.
     */
    virtual nvinfer1::ITensor* operator()(nvinfer1::ITensor* net) {
        typedef tfrt::convolution2d<tfrt::ActivationType::NONE, tfrt::PaddingType::SAME, false>  conv2d;
        auto& sc = this->m_scope;
        LOG(INFO) << "LAYER SSD boxes2d block '" << sc.name() << "'. "
                << "Input shape: " << dims_str(net->getDimensions());
        // Classification + boxes regression convolutions.
        auto net_cls = conv2d(sc, "conv_cls")
            .noutputs(m_num_anchors * m_num_classes).ksize({3, 3})(net);
        auto net_loc = conv2d(sc, "conv_loc")
            .noutputs(m_num_anchors * 4).ksize({3, 3})(net);
        // Decode boxes.
        if(m_decode_boxes) {
            net_loc = ssd_boxes2d_decode(sc, "decode")(net_loc);
        }
        // Mark outputs.
        this->mark_output(net_cls, "predictions");
        this->mark_output(net_loc, "boxes");
        // return this->mark_output(net);
        // return std::make_pair(net_cls, net_loc);
        return nullptr;
    }

private:
    // Number of classes.
    int m_num_classes;
    // Number of anchors in the block.
    int m_num_anchors;
    // Decode 2D boxes.
    bool m_decode_boxes;
};



// inline tensors_pair ssd_boxes2d_block(
//     nvinfer1::ITensor* net, tfrt::scope sc,
//     int num_anchors, int num_classes,
//     bool mark_outputs=true, bool decode_boxes=true)
// {
//     typedef tfrt::convolution2d<tfrt::ActivationType::NONE, tfrt::PaddingType::SAME, false>  conv2d;
//     LOG(INFO) << "BLOCK SSD boxes2d '" << sc.name() << "'. "
//             << "Input shape: " << dims_str(net->getDimensions());
//     // Classification + boxes regression convolutions.
//     auto net_cls = conv2d(sc, "conv_cls")
//         .noutputs(num_anchors * num_classes).ksize({3, 3})(net);
//     auto net_loc = conv2d(sc, "conv_loc")
//         .noutputs(num_anchors * 4).ksize({3, 3})(net);
//     // Decode boxes.
//     if(decode_boxes) {
//         net_loc = ssd_boxes2d_decode(sc, "decode")(net_loc);
//     }
//     if(mark_outputs) {
//         sc.network()->markOutput(*net_cls);
//         sc.network()->markOutput(*net_loc);
//     }
//     return std::make_pair(net_cls, net_loc);
// }

}
#endif
