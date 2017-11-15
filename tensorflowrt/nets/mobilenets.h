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

#ifndef TFRT_MOBILENETS
#define TFRT_MOBILENETS

#include <map>
#include <fmt/format.h>

#include <NvInfer.h>
#include "../tensorflowrt.h"

namespace mobilenets
{
/** Arg scope for Inception v2: SAME padding + batch normalization + ReLU.
 */
typedef tfrt::separable_convolution2d<
    tfrt::ActivationType::RELU, tfrt::PaddingType::SAME, false> separable_conv2d;

typedef tfrt::convolution2d<
tfrt::ActivationType::RELU, tfrt::PaddingType::SAME, true>  conv2d;
typedef tfrt::convolution2d<
    tfrt::ActivationType::NONE, tfrt::PaddingType::SAME, true>  conv2d_none;
// typedef tfrt::convolution2d<tfrt::ActivationType::NONE, tfrt::PaddingType::SAME, true>  conv2d;
typedef tfrt::convolution2d<
    tfrt::ActivationType::NONE, tfrt::PaddingType::VALID, true>  conv2d_valid;

typedef tfrt::max_pooling2d<tfrt::PaddingType::SAME>    max_pool2d;
typedef tfrt::avg_pooling2d<tfrt::PaddingType::SAME>    avg_pool2d;
typedef tfrt::concat_channels                           concat_channels;

// ========================================================================== //
// Mobilenets main block.
// ========================================================================== //
/** Mobilenets block. */
inline nvinfer1::ITensor* block(nvinfer1::ITensor* net, tfrt::scope sc,
    size_t noutputs, size_t stride)
{
    net = separable_conv2d(sc, "sep_conv2d").noutputs(noutputs).stride(stride).ksize(3)(net);
    return net;
}
inline nvinfer1::ITensor* base(nvinfer1::ITensor* input, tfrt::scope sc)
{
    nvinfer1::ITensor* net{input};
    // First convolution.
    net = conv2d_valid(sc, "conv1").noutputs(32).ksize(3).stride(2)(net);

    // First series of blocks.
    net = block(net, sc.sub("block2"), 64, 1);
    net = block(net, sc.sub("block3"), 128, 2);
    net = block(net, sc.sub("block4"), 128, 1);
    net = block(net, sc.sub("block5"), 256, 2);
    net = block(net, sc.sub("block6"), 256, 1);
    net = block(net, sc.sub("block7"), 512, 2);
    // 5 Intermediate blocks.
    for (size_t i = 0 ; i < 5 ; ++i) {
        std::string name = fmt::format("block{}", i+8);
        net = block(net, sc.sub(name), 512, 1);
    }
    // Final parts.
    net = block(net, sc.sub("block13"), 1024, 2);
    net = block(net, sc.sub("block14"), 1024, 1);
    return net;
}

class net : public tfrt::imagenet_network
{
public:
    net() : tfrt::imagenet_network("mobilenets", 1000, true) {
    }
    virtual nvinfer1::ITensor* build(tfrt::scope sc) {
        auto net = tfrt::input(sc)();
        // Mobilenets base.
        net = base(net, sc);
        // Classification.
        net = conv2d(sc, "logits").noutputs(1000).ksize({1, 1})(net);
        net = tfrt::softmax(sc, "Softmax")(net);
        return net;
    }
};
}

#endif
