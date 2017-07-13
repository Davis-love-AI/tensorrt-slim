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
#ifndef TFRT_INCEPTION1_H
#define TFRT_INCEPTION1_H

#include <NvInfer.h>
#include "../tensorflowrt.h"

namespace inception1
{
/** Arg scope for Inception v2: SAME padding + batch normalization + ReLU.
 */
typedef tfrt::convolution2d<tfrt::ActivationType::RELU, tfrt::PaddingType::SAME, true>  conv2d;
typedef tfrt::max_pooling2d<tfrt::PaddingType::SAME>    max_pool2d;
typedef tfrt::avg_pooling2d<tfrt::PaddingType::SAME>    avg_pool2d;
typedef tfrt::concat_channels                           concat_channels;

/* ============================================================================
 * Inception1 main mixed blocks.
 * ========================================================================== */
/** Major mixed block used in Inception v2 (4 branches).
 * Max pooling version.
 */
template <int B0, int B10, int B11, int B20, int B21, int B3>
inline nvinfer1::ITensor* block_mixed_max(nvinfer1::ITensor* input, tfrt::scope sc)
{
    nvinfer1::ITensor* net{input};
    // Branch 0.
    auto ssc = sc.sub("Branch_0");
    auto branch0 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(B0).ksize({1, 1})(net);
    // Branch 1.
    ssc = sc.sub("Branch_1");
    auto branch1 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(B10).ksize({1, 1})(net);
    branch1 = conv2d(ssc, "Conv2d_0b_3x3").noutputs(B11).ksize({3, 3})(branch1);
    // Branch 2.
    ssc = sc.sub("Branch_2");
    auto branch2 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(B20).ksize({1, 1})(net);
    branch2 = conv2d(ssc, "Conv2d_0b_3x3").noutputs(B21).ksize({3, 3})(branch2);
    // Branch 2.
    ssc = sc.sub("Branch_3");
    auto branch3 = max_pool2d(ssc, "MaxPool_0a_3x3").ksize({3, 3})(net);
    branch3 = conv2d(ssc, "Conv2d_0b_1x1").noutputs(B3).ksize({1, 1})(branch3);
    // Concat everything!
    net = concat_channels(sc)({branch0, branch1, branch2, branch3});
    return net;
}

/* ============================================================================
 * Inception1: blocks 1 to 5.
 * ========================================================================== */
inline nvinfer1::ITensor* block1(nvinfer1::ITensor* net, tfrt::scope sc)
{
    // 7x7 convolution.
    net = conv2d(sc, "Conv2d_1a_7x7").noutputs(64).ksize({7, 7}).stride({2, 2})(net);
    return net;
}
inline nvinfer1::ITensor* block2(nvinfer1::ITensor* net, tfrt::scope sc)
{
    net = max_pool2d(sc, "MaxPool_2a_3x3").ksize({3, 3}).stride({2, 2})(net);
    net = conv2d(sc, "Conv2d_2b_1x1").noutputs(64).ksize({1, 1})(net);
    net = conv2d(sc, "Conv2d_2c_3x3").noutputs(192).ksize({3, 3})(net);
    return net;
}
inline nvinfer1::ITensor* block3(nvinfer1::ITensor* net, tfrt::scope sc)
{
    // Mixed block 3b and 3c.
    net = max_pool2d(sc, "MaxPool_3a_3x3").ksize({3, 3}).stride({2, 2})(net);
    net = block_mixed_max<64, 96, 128, 16, 32, 32>(net, sc.sub("Mixed_3b"));
    net = block_mixed_max<128, 128, 192, 32, 96, 64>(net, sc.sub("Mixed_3c"));
    return net;
}
inline nvinfer1::ITensor* block4(nvinfer1::ITensor* net, tfrt::scope sc)
{
    // Mixed blocks 4a to 4e.
    net = max_pool2d(sc, "MaxPool_4a_3x3").ksize({3, 3}).stride({2, 2})(net);
    net = block_mixed_max<192, 96, 208, 16, 48, 64>(net, sc.sub("Mixed_4b"));
    net = block_mixed_max<160, 112, 224, 24, 64, 64>(net, sc.sub("Mixed_4c"));
    net = block_mixed_max<128, 128, 256, 24, 64, 64>(net, sc.sub("Mixed_4d"));
    net = block_mixed_max<112, 144, 288, 32, 64, 64>(net, sc.sub("Mixed_4e"));
    net = block_mixed_max<256, 160, 320, 32, 128, 128>(net, sc.sub("Mixed_4f"));
    return net;
}
inline nvinfer1::ITensor* block5(nvinfer1::ITensor* net, tfrt::scope sc)
{
    typedef tfrt::max_pooling2d<tfrt::PaddingType::VALID>    max_pool2d;
    // Mixed blocks 5a to 5c.
    net = max_pool2d(sc, "MaxPool_5a_2x2").ksize({2, 2}).stride({2, 2})(net);
    net = block_mixed_max<256, 160, 320, 32, 128, 128>(net, sc.sub("Mixed_5b"));
    net = block_mixed_max<384, 192, 384, 48, 128, 128>(net, sc.sub("Mixed_5c"));
    return net;
}

/* ============================================================================
 * Inception1 network: base + full network
 * ========================================================================== */
inline nvinfer1::ITensor* base(nvinfer1::ITensor* input, tfrt::scope sc)
{
    nvinfer1::ITensor* net{input};
    // Main blocks 1 to 5.
    net = block1(input, sc);
    net = block2(net, sc);
    net = block3(net, sc);
    net = block4(net, sc);
    net = block5(net, sc);
    return net;
}
inline nvinfer1::ITensor* inception1(nvinfer1::ITensor* input,
                                     tfrt::scope sc,
                                     int num_classes=1001)
{
    nvinfer1::ITensor* net;
    // Construct backbone network.
    net = base(input, sc);
    // Logits end block.
    {
        typedef tfrt::avg_pooling2d<tfrt::PaddingType::VALID>  avg_pool2d;
        typedef tfrt::convolution2d<tfrt::ActivationType::NONE, tfrt::PaddingType::SAME, false>  conv2d;
        auto ssc = sc.sub("Logits");
        net = avg_pool2d(ssc, "AvgPool_1a_7x7").ksize({7, 7})(net);
        net = conv2d(ssc, "Conv2d_0c_1x1").noutputs(num_classes).ksize({1, 1})(net);
    }
    net = tfrt::softmax(sc, "Softmax")(net);
    return net;
}

/* ============================================================================
 * Inception1 class: as imagenet network.
 * ========================================================================== */
class net : public tfrt::imagenet_network
{
public:
    /** Constructor with default name.  */
    net() : tfrt::imagenet_network("InceptionV1", 1000, true) {}

    /** Inception2 building method. Take a network scope and do the work!
     */
    virtual nvinfer1::ITensor* build(tfrt::scope sc) {
        auto net = tfrt::input(sc)();
        net = inception1(net, sc, 1001);
        return net;
    }
};

}

#endif
