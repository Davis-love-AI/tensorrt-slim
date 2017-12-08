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

#ifndef TFRT_INCEPTION_V2B_H
#define TFRT_INCEPTION_V2B_H

#include <map>

#include <NvInfer.h>
#include "../tensorflowrt.h"

namespace inception_v2b
{
/** Arg scope for Inception v2: SAME padding + batch normalization + ReLU.
 */
typedef tfrt::separable_convolution2d<tfrt::ActivationType::RELU, tfrt::PaddingType::SAME, false> separable_conv2d;
typedef tfrt::convolution2d<tfrt::ActivationType::RELU, tfrt::PaddingType::SAME, true>  conv2d;
// typedef tfrt::convolution2d_grouped<tfrt::ActivationType::RELU, tfrt::PaddingType::SAME, true>  conv2d;
typedef tfrt::depthwise_convolution2d<
    tfrt::ActivationType::RELU, tfrt::PaddingType::SAME, true> depthwise_conv2d;

typedef tfrt::max_pooling2d<tfrt::PaddingType::SAME>    max_pool2d;
typedef tfrt::avg_pooling2d<tfrt::PaddingType::SAME>    avg_pool2d;
typedef tfrt::concat_channels                           concat_channels;

typedef std::pair<nvinfer1::ITensor*, nvinfer1::ITensor*>  tensor_pair;

/* ============================================================================
 * Inception2 main mixed blocks.
 * ========================================================================== */
/** Major mixed block used in Inception v2 (4 branches).
 * Average pooling version.
 */
template <int B0, int B10, int B11, int B20, int B21, int B3>
inline tensor_pair block_mixed_avg(tensor_pair inputs, tfrt::scope sc,
                                   tfrt::map_tensor* end_points=nullptr)
{
    tensor_pair outputs;
    std::vector<nvinfer1::ITensor*>  block1, block2;
    {
        nvinfer1::ITensor* net{inputs.first};
        // Branch 0.
        auto ssc = sc.sub("Branch_0l");
        auto branch0 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(B0/2).ksize({1, 1})(net);
        // Branch 1.
        ssc = sc.sub("Branch_1l");
        auto branch1 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(B10/2).ksize({1, 1})(net);
        branch1 = conv2d(ssc, "Conv2d_0b_3x3").noutputs(B11/2).ksize({3, 3})(branch1);
        // Branch 2.
        ssc = sc.sub("Branch_2l");
        auto branch2 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(B20/2).ksize({1, 1})(net);
        branch2 = conv2d(ssc, "Conv2d_0b_3x3").noutputs(B21/2).ksize({3, 3})(branch2);
        branch2 = conv2d(ssc, "Conv2d_0c_3x3").noutputs(B21/2).ksize({3, 3})(branch2);
        // Branch 2.
        ssc = sc.sub("Branch_3l");
        auto branch3 = avg_pool2d(ssc, "AvgPool_0a_3x3").ksize({3, 3})(net);
        branch3 = conv2d(ssc, "Conv2d_0b_1x1").noutputs(B3/2).ksize({1, 1})(branch3);

        outputs.first = concat_channels(sc.sub("left"))({branch0, branch1, branch2, branch3});
    }
    {
        nvinfer1::ITensor* net{inputs.second};
        // Branch 0.
        auto ssc = sc.sub("Branch_0r");
        auto branch0 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(B0/2).ksize({1, 1})(net);
        // Branch 1.
        ssc = sc.sub("Branch_1r");
        auto branch1 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(B10/2).ksize({1, 1})(net);
        branch1 = conv2d(ssc, "Conv2d_0b_3x3").noutputs(B11/2).ksize({3, 3})(branch1);
        // Branch 2.
        ssc = sc.sub("Branch_2r");
        auto branch2 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(B20/2).ksize({1, 1})(net);
        branch2 = conv2d(ssc, "Conv2d_0b_3x3").noutputs(B21/2).ksize({3, 3})(branch2);
        branch2 = conv2d(ssc, "Conv2d_0c_3x3").noutputs(B21/2).ksize({3, 3})(branch2);
        // Branch 2.
        ssc = sc.sub("Branch_3r");
        auto branch3 = avg_pool2d(ssc, "AvgPool_0a_3x3").ksize({3, 3})(net);
        branch3 = conv2d(ssc, "Conv2d_0b_1x1").noutputs(B3/2).ksize({1, 1})(branch3);

        outputs.second = concat_channels(sc.sub("right"))({branch0, branch1, branch2, branch3});
    }
    return outputs;
}

/** Major mixed block used in Inception v2 (4 branches).
 * Max pooling version.
 */
template <int B0, int B10, int B11, int B20, int B21, int B3>
inline tensor_pair block_mixed_max(tensor_pair inputs, tfrt::scope sc,
                                   tfrt::map_tensor* end_points=nullptr)
{
    tensor_pair outputs;
    {
        nvinfer1::ITensor* net{inputs.first};
        // Branch 0.
        auto ssc = sc.sub("Branch_0l");
        auto branch0 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(B0/2).ksize({1, 1})(net);
        // Branch 1.
        ssc = sc.sub("Branch_1l");
        auto branch1 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(B10/2).ksize({1, 1})(net);
        branch1 = conv2d(ssc, "Conv2d_0b_3x3").noutputs(B11/2).ksize({3, 3})(branch1);
        // Branch 2.
        ssc = sc.sub("Branch_2l");
        auto branch2 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(B20/2).ksize({1, 1})(net);
        branch2 = conv2d(ssc, "Conv2d_0b_3x3").noutputs(B21/2).ksize({3, 3})(branch2);
        branch2 = conv2d(ssc, "Conv2d_0c_3x3").noutputs(B21/2).ksize({3, 3})(branch2);
        // Branch 2.
        ssc = sc.sub("Branch_3l");
        auto branch3 = max_pool2d(ssc, "MaxPool_0a_3x3").ksize({3, 3})(net);
        branch3 = conv2d(ssc, "Conv2d_0b_1x1").noutputs(B3/2).ksize({1, 1})(branch3);
        // Concat everything!
        outputs.first = concat_channels(sc.sub("left"))({branch0, branch1, branch2, branch3});
    }
    {
        nvinfer1::ITensor* net{inputs.second};
        // Branch 0.
        auto ssc = sc.sub("Branch_0r");
        auto branch0 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(B0/2).ksize({1, 1})(net);
        // Branch 1.
        ssc = sc.sub("Branch_1r");
        auto branch1 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(B10/2).ksize({1, 1})(net);
        branch1 = conv2d(ssc, "Conv2d_0b_3x3").noutputs(B11/2).ksize({3, 3})(branch1);
        // Branch 2.
        ssc = sc.sub("Branch_2r");
        auto branch2 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(B20/2).ksize({1, 1})(net);
        branch2 = conv2d(ssc, "Conv2d_0b_3x3").noutputs(B21/2).ksize({3, 3})(branch2);
        branch2 = conv2d(ssc, "Conv2d_0c_3x3").noutputs(B21/2).ksize({3, 3})(branch2);
        // Branch 2.
        ssc = sc.sub("Branch_3r");
        auto branch3 = max_pool2d(ssc, "MaxPool_0a_3x3").ksize({3, 3})(net);
        branch3 = conv2d(ssc, "Conv2d_0b_1x1").noutputs(B3/2).ksize({1, 1})(branch3);
        // Concat everything!
        outputs.second = concat_channels(sc.sub("right"))({branch0, branch1, branch2, branch3});
    }
    return outputs;
}
/** Specific mixed block with stride 2 used in Inception v2.
 */
template <int B00, int B01, int B10, int B11>
inline tensor_pair block_mixed_s2(tensor_pair inputs, tfrt::scope sc,
                                  tfrt::map_tensor* end_points=nullptr)
{
    tensor_pair outputs;
    {
        nvinfer1::ITensor* net{inputs.first};
        // Branch 0.
        auto ssc = sc.sub("Branch_0l");
        auto branch0 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(B00/2).ksize({1, 1})(net);
        branch0 = conv2d(ssc, "Conv2d_1a_3x3").noutputs(B01/2).ksize({3, 3}).stride({2, 2})(branch0);
        // Branch 2.
        ssc = sc.sub("Branch_1l");
        auto branch1 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(B10/2).ksize({1, 1})(net);
        branch1 = conv2d(ssc, "Conv2d_0b_3x3").noutputs(B11/2).ksize({3, 3})(branch1);
        branch1 = conv2d(ssc, "Conv2d_1a_3x3").noutputs(B11/2).ksize({3, 3}).stride({2, 2})(branch1);
        // Branch 2.
        ssc = sc.sub("Branch_2l");
        auto branch2 = max_pool2d(ssc, "MaxPool_1a_3x3").ksize({3, 3}).stride({2, 2})(net);
        // Concat everything!
        outputs.first = concat_channels(sc.sub("left"))({branch0, branch1, branch2});
    }
    {
        nvinfer1::ITensor* net{inputs.second};
        // Branch 0.
        auto ssc = sc.sub("Branch_0r");
        auto branch0 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(B00/2).ksize({1, 1})(net);
        branch0 = conv2d(ssc, "Conv2d_1a_3x3").noutputs(B01/2).ksize({3, 3}).stride({2, 2})(branch0);
        // Branch 2.
        ssc = sc.sub("Branch_1r");
        auto branch1 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(B10/2).ksize({1, 1})(net);
        branch1 = conv2d(ssc, "Conv2d_0b_3x3").noutputs(B11/2).ksize({3, 3})(branch1);
        branch1 = conv2d(ssc, "Conv2d_1a_3x3").noutputs(B11/2).ksize({3, 3}).stride({2, 2})(branch1);
        // Branch 2.
        ssc = sc.sub("Branch_2r");
        auto branch2 = max_pool2d(ssc, "MaxPool_1a_3x3").ksize({3, 3}).stride({2, 2})(net);
        // Concat everything!
        outputs.second = concat_channels(sc.sub("right"))({branch0, branch1, branch2});
    }
    return outputs;
}

/* ============================================================================
 * Inception2 blocks 1 to 5.
 * ========================================================================== */
inline nvinfer1::ITensor* block1(nvinfer1::ITensor* net, tfrt::scope sc,
                                 tfrt::map_tensor* end_points=nullptr)
{
    int depthwise_multiplier = std::min(int(64 / 3), 8);
    // // 7x7 depthwise convolution.
    net = separable_conv2d(sc, "Conv2d_1a_7x7")
        .dw_group_size(1024)
        .depthmul(depthwise_multiplier)
        .noutputs(64).ksize({7, 7}).stride({2, 2})(net);
    // net = max_pool2d(sc, "MaxPool_1a_3x3").ksize({3, 3}).stride({2, 2})(net);
    return net;
}
inline tensor_pair block2(nvinfer1::ITensor* net, tfrt::scope sc,
                          tfrt::map_tensor* end_points=nullptr)
{
    tensor_pair outputs;
    net = max_pool2d(sc, "MaxPool_2a_3x3").ksize({3, 3}).stride({2, 2})(net);
    net = conv2d(sc, "Conv2d_2b_1x1").noutputs(64).ksize({1, 1})(net);
    outputs.first = conv2d(sc, "Conv2d_2c_3x3l").noutputs(192/2).ksize({3, 3})(net);
    outputs.second = conv2d(sc, "Conv2d_2c_3x3r").noutputs(192/2).ksize({3, 3})(net);
    return outputs;
}
inline tensor_pair block3(tensor_pair net, tfrt::scope sc,
                          tfrt::map_tensor* end_points=nullptr)
{
    // Mixed block 3b and 3c.
    // net = max_pool2d(sc, "MaxPool_3a_3x3").ksize({3, 3}).stride({2, 2})(net);
    net.first = max_pool2d(sc, "MaxPool_3a_3x3l").ksize({3, 3}).stride({2, 2})(net.first);
    net.second = max_pool2d(sc, "MaxPool_3a_3x3r").ksize({3, 3}).stride({2, 2})(net.second);

    net = block_mixed_avg<64, 64, 64, 64, 96, 32>(net, sc.sub("Mixed_3b"), end_points);
    net = block_mixed_avg<64, 64, 96, 64, 96, 64>(net, sc.sub("Mixed_3c"), end_points);
    return net;
}
inline tensor_pair block4(tensor_pair net, tfrt::scope sc,
                          tfrt::map_tensor* end_points=nullptr)
{
    // Mixed blocks 4a to 4e.
    net = block_mixed_s2<128, 160, 64, 96>(net, sc.sub("Mixed_4a"));
    net = block_mixed_avg<224, 64, 96, 96, 128, 128>(net, sc.sub("Mixed_4b"), end_points);
    net = block_mixed_avg<192, 96, 128, 96, 128, 128>(net, sc.sub("Mixed_4c"), end_points);
    net = block_mixed_avg<160, 128, 160, 128, 160, 96>(net, sc.sub("Mixed_4d"), end_points);
    net = block_mixed_avg<96, 128, 192, 160, 192, 96>(net, sc.sub("Mixed_4e"), end_points);
    return net;
}
inline tensor_pair block5(tensor_pair net, tfrt::scope sc,
                          tfrt::map_tensor* end_points=nullptr)
{
    // Mixed blocks 5a to 5c.
    net = block_mixed_s2<128, 192, 192, 256>(net, sc.sub("Mixed_5a"));
    net = block_mixed_avg<352, 192, 320, 160, 224, 128>(net, sc.sub("Mixed_5b"), end_points);
    net = block_mixed_max<352, 192, 320, 192, 224, 128>(net, sc.sub("Mixed_5c"), end_points);
    return net;
}

/* ============================================================================
 * Inception2 network: functional declaration.
 * ========================================================================== */
inline nvinfer1::ITensor* base(nvinfer1::ITensor* input, tfrt::scope sc,
                               tfrt::map_tensor* end_points=nullptr)
{
    nvinfer1::ITensor* net{input};
    tensor_pair nets;
    // Main blocks 1 to 5.
    net = block1(net, sc, end_points);
    nets = block2(net, sc, end_points);
    nets = block3(nets, sc, end_points);
    nets = block4(nets, sc, end_points);
    nets = block5(nets, sc, end_points);
    net = concat_channels(sc.sub("concat_final"))({nets.first, nets.second});
    return net;
}
inline nvinfer1::ITensor* inception_v2b(nvinfer1::ITensor* input,
                                        tfrt::scope sc,
                                        int num_classes=1001)
{
    nvinfer1::ITensor* net;
    // Construct backbone network.
    net = base(input, sc.sub("b"));
    // Logits end block.
    {
        typedef tfrt::avg_pooling2d<tfrt::PaddingType::VALID>  avg_pool2d;
        typedef tfrt::convolution2d<tfrt::ActivationType::NONE, tfrt::PaddingType::SAME, false>  conv2d;
        auto ssc = sc.sub("Logits");
        net = avg_pool2d(ssc, "AvgPool_1a_7x7").ksize({7, 7})(net);
        net = conv2d(ssc, "Conv2d_1c_1x1").noutputs(num_classes).ksize({1, 1})(net);
    }
    net = tfrt::softmax(sc, "Softmax")(net);
    return net;
}

/* ============================================================================
 * Inception2 class: as imagenet network.
 * ========================================================================== */
class net : public tfrt::imagenet_network
{
public:
    /** Constructor with default name.  */
    net() : tfrt::imagenet_network("InceptionV2B", 1000, true) {}

    /** Inception2 building method. Take a network scope and do the work!
     */
    virtual nvinfer1::ITensor* build(tfrt::scope sc) {
        auto net = tfrt::input(sc)();
        // auto net = tfrt::input(sc).shape({64, 112, 112})();
        net = inception_v2b(net, sc, 1001);
        return net;
    }
};

}

#endif
