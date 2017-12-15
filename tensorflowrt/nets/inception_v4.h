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

#ifndef TFRT_INCEPTION4_H
#define TFRT_INCEPTION4_H

#include <map>
#include <fmt/format.h>

#include <NvInfer.h>
#include "../tensorflowrt.h"

namespace inception4
{
/** Arg scope for Inception v4: SAME padding + batch normalization + ReLU.
 */
typedef tfrt::separable_convolution2d<tfrt::ActivationType::RELU, tfrt::PaddingType::SAME, false> separable_conv2d;
typedef tfrt::convolution2d<tfrt::ActivationType::RELU, tfrt::PaddingType::SAME, true>  conv2d;
typedef tfrt::max_pooling2d<tfrt::PaddingType::SAME>    max_pool2d;
typedef tfrt::avg_pooling2d<tfrt::PaddingType::SAME>    avg_pool2d;
typedef tfrt::concat_channels                           concat_channels;

typedef tfrt::convolution2d<tfrt::ActivationType::RELU, tfrt::PaddingType::VALID, true>  conv2d_valid;
typedef tfrt::max_pooling2d<tfrt::PaddingType::VALID>    max_pool2d_valid;

/* ============================================================================
 * Inception4 main mixed blocks.
 * ========================================================================== */
inline nvinfer1::ITensor* block_a(nvinfer1::ITensor* input, tfrt::scope sc,
                                  tfrt::map_tensor* end_points=nullptr)
{
    nvinfer1::ITensor* net{input};
    // Branch 0.
    auto ssc = sc.sub("Branch_0");
    auto branch0 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(96).ksize({1, 1})(net);
    // Branch 1.
    ssc = sc.sub("Branch_1");
    auto branch1 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(64).ksize({1, 1})(net);
    branch1 = conv2d(ssc, "Conv2d_0b_3x3").noutputs(96).ksize({3, 3})(branch1);
    // Branch 2.
    ssc = sc.sub("Branch_2");
    auto branch2 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(64).ksize({1, 1})(net);
    branch2 = conv2d(ssc, "Conv2d_0b_3x3").noutputs(96).ksize({3, 3})(branch2);
    branch2 = conv2d(ssc, "Conv2d_0c_3x3").noutputs(96).ksize({3, 3})(branch2);
    // Branch 2.
    ssc = sc.sub("Branch_3");
    auto branch3 = avg_pool2d(ssc, "AvgPool_0a_3x3").ksize({3, 3})(net);
    branch3 = conv2d(ssc, "Conv2d_0b_1x1").noutputs(96).ksize({1, 1})(branch3);
    // Concat everything!
    net = concat_channels(sc)({branch0, branch1, branch2, branch3});
    return tfrt::add_end_point(end_points, sc.name(), net);
}
inline nvinfer1::ITensor* block_reduc_a(
    nvinfer1::ITensor* input, tfrt::scope sc, tfrt::map_tensor* end_points=nullptr)
{
    nvinfer1::ITensor* net{input};
    // Branch 0.
    auto ssc = sc.sub("Branch_0");
    auto branch0 = conv2d_valid(ssc, "Conv2d_1a_3x3").noutputs(384).ksize({3, 3}).stride(2)(net);
    // Branch 1.
    ssc = sc.sub("Branch_1");
    auto branch1 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(192).ksize({1, 1})(net);
    branch1 = conv2d(ssc, "Conv2d_0b_3x3").noutputs(224).ksize({3, 3})(branch1);
    branch1 = conv2d_valid(ssc, "Conv2d_1a_3x3").noutputs(256).ksize({3, 3}).stride(2)(branch1);
    // Branch 2.
    ssc = sc.sub("Branch_2");
    auto branch2 = max_pool2d_valid(ssc, "MaxPool_1a_3x3").ksize({3, 3}).stride(2)(net);
    // Concat everything!
    net = concat_channels(sc)({branch0, branch1, branch2});
    return tfrt::add_end_point(end_points, sc.name(), net);
}

inline nvinfer1::ITensor* block_b(nvinfer1::ITensor* input, tfrt::scope sc,
                                  tfrt::map_tensor* end_points=nullptr)
{
    nvinfer1::ITensor* net{input};
    // Branch 0.
    auto ssc = sc.sub("Branch_0");
    auto branch0 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(384).ksize({1, 1})(net);
    // Branch 1.
    ssc = sc.sub("Branch_1");
    auto branch1 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(192).ksize({1, 1})(net);
    branch1 = conv2d(ssc, "Conv2d_0b_1x7").noutputs(224).ksize({1, 7})(branch1);
    branch1 = conv2d(ssc, "Conv2d_0c_7x1").noutputs(256).ksize({7, 1})(branch1);
    // Branch 2.
    ssc = sc.sub("Branch_2");
    auto branch2 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(192).ksize({1, 1})(net);
    branch2 = conv2d(ssc, "Conv2d_0b_7x1").noutputs(192).ksize({1, 7})(branch2);
    branch2 = conv2d(ssc, "Conv2d_0c_1x7").noutputs(224).ksize({7, 1})(branch2);
    branch2 = conv2d(ssc, "Conv2d_0d_7x1").noutputs(224).ksize({1, 7})(branch2);
    branch2 = conv2d(ssc, "Conv2d_0e_1x7").noutputs(256).ksize({7, 1})(branch2);
    // Branch 2.
    ssc = sc.sub("Branch_3");
    auto branch3 = avg_pool2d(ssc, "AvgPool_0a_3x3").ksize({3, 3})(net);
    branch3 = conv2d(ssc, "Conv2d_0b_1x1").noutputs(128).ksize({1, 1})(branch3);
    // Concat everything!
    net = concat_channels(sc)({branch0, branch1, branch2, branch3});
    return tfrt::add_end_point(end_points, sc.name(), net);
}
inline nvinfer1::ITensor* block_reduc_b(nvinfer1::ITensor* input, tfrt::scope sc,
                                  tfrt::map_tensor* end_points=nullptr)
{
    nvinfer1::ITensor* net{input};
    // Branch 0.
    auto ssc = sc.sub("Branch_0");
    auto branch0 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(192).ksize({1, 1})(net);
    branch0 = conv2d_valid(ssc, "Conv2d_1a_3x3").noutputs(192).ksize({3, 3}).stride(2)(branch0);
    // Branch 1.
    ssc = sc.sub("Branch_1");
    auto branch1 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(256).ksize({1, 1})(net);
    branch1 = conv2d(ssc, "Conv2d_0b_1x7").noutputs(256).ksize({1, 7})(branch1);
    branch1 = conv2d(ssc, "Conv2d_0c_7x1").noutputs(320).ksize({7, 1})(branch1);
    branch1 = conv2d_valid(ssc, "Conv2d_1a_3x3").noutputs(320).ksize({3, 3}).stride(2)(branch1);
    // Branch 2.
    ssc = sc.sub("Branch_2");
    auto branch2 = max_pool2d_valid(ssc, "MaxPool_1a_3x3").ksize({3, 3}).stride(2)(net);
    // Concat everything!
    net = concat_channels(sc)({branch0, branch1, branch2});
    return tfrt::add_end_point(end_points, sc.name(), net);
}

inline nvinfer1::ITensor* block_c(nvinfer1::ITensor* input, tfrt::scope sc,
                                  tfrt::map_tensor* end_points=nullptr)
{
    nvinfer1::ITensor* net{input};
    // Branch 0.
    auto ssc = sc.sub("Branch_0");
    auto branch0 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(256).ksize({1, 1})(net);
    // Branch 1.
    ssc = sc.sub("Branch_1");
    auto branch1 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(384).ksize({1, 1})(net);
    auto branch1l = conv2d(ssc, "Conv2d_0b_1x3").noutputs(256).ksize({1, 3})(branch1);
    auto branch1r = conv2d(ssc, "Conv2d_0c_3x1").noutputs(256).ksize({3, 1})(branch1);
    // Branch 2.
    ssc = sc.sub("Branch_2");
    auto branch2 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(384).ksize({1, 1})(net);
    branch2 = conv2d(ssc, "Conv2d_0b_3x1").noutputs(448).ksize({1, 3})(branch2);
    branch2 = conv2d(ssc, "Conv2d_0c_1x3").noutputs(512).ksize({3, 1})(branch2);
    auto branch2l = conv2d(ssc, "Conv2d_0d_1x3").noutputs(256).ksize({1, 3})(branch2);
    auto branch2r = conv2d(ssc, "Conv2d_0e_3x1").noutputs(256).ksize({3, 1})(branch2);
    // Branch 2.
    ssc = sc.sub("Branch_3");
    auto branch3 = avg_pool2d(ssc, "AvgPool_0a_3x3").ksize({3, 3})(net);
    branch3 = conv2d(ssc, "Conv2d_0b_1x1").noutputs(256).ksize({1, 1})(branch3);
    // Concat everything!
    net = concat_channels(sc)({branch0, branch1l, branch1r, branch2l, branch2r, branch3});
    return tfrt::add_end_point(end_points, sc.name(), net);
}

inline nvinfer1::ITensor* block_stem(nvinfer1::ITensor* input, tfrt::scope sc,
                                     tfrt::map_tensor* end_points=nullptr)
{
    nvinfer1::ITensor* net{input};
    // 299 x 299 x 3
    net = conv2d_valid(sc, "Conv2d_1a_3x3").noutputs(32).ksize({3, 3}).stride(2)(net);
    // 149 x 149 x 32
    net = conv2d_valid(sc, "Conv2d_2a_3x3").noutputs(32).ksize({3, 3})(net);
    // 147 x 147 x 32
    net = conv2d(sc, "Conv2d_2b_3x3").noutputs(64).ksize({3, 3})(net);
    // 147 x 147 x 64
    {
        auto ssc = sc.sub("Mixed_3a").sub("Branch_0");
        auto branch0 = max_pool2d_valid(ssc, "MaxPool_0a_3x3").ksize({3, 3}).stride(2)(net);
        ssc = sc.sub("Mixed_3a").sub("Branch_1");
        auto branch1 = conv2d_valid(ssc, "Conv2d_0a_3x3").noutputs(96).ksize({3, 3}).stride(2)(net);
        net = concat_channels(sc.sub("Mixed_3a"))({branch0, branch1});
    }
    // 73 x 73 x 160
    {
        auto ssc = sc.sub("Mixed_4a").sub("Branch_0");
        auto branch0 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(64).ksize({1, 1})(net);
        branch0 = conv2d_valid(ssc, "Conv2d_1a_3x3").noutputs(96).ksize({3, 3})(branch0);

        ssc = sc.sub("Mixed_4a").sub("Branch_1");
        auto branch1 = conv2d(ssc, "Conv2d_0a_1x1").noutputs(64).ksize({1, 1})(net);
        branch1 = conv2d(ssc, "Conv2d_0b_1x7").noutputs(64).ksize({1, 7})(branch1);
        branch1 = conv2d(ssc, "Conv2d_0c_7x1").noutputs(64).ksize({7, 1})(branch1);
        branch1 = conv2d_valid(ssc, "Conv2d_1a_3x3").noutputs(96).ksize({3, 3})(branch1);

        net = concat_channels(sc.sub("Mixed_4a"))({branch0, branch1});
    }
    // 71 x 71 x 192
    {
        auto ssc = sc.sub("Mixed_5a").sub("Branch_0");
        auto branch0 = conv2d_valid(ssc, "Conv2d_1a_3x3").noutputs(96).ksize({3, 3}).stride(2)(net);
        ssc = sc.sub("Mixed_5a").sub("Branch_1");
        auto branch1 = max_pool2d_valid(ssc, "MaxPool_1a_3x3").ksize({3, 3}).stride(2)(net);
        net = concat_channels(sc.sub("Mixed_5a"))({branch0, branch1});
    }
    // 35 x 35 x 384
    return tfrt::add_end_point(end_points, sc.name(), net);
}

inline nvinfer1::ITensor* base(nvinfer1::ITensor* input, tfrt::scope sc,
                               tfrt::map_tensor* end_points=nullptr)
{
    nvinfer1::ITensor* net{input};
    // Stem block.
    net = block_stem(net, sc, end_points);
    // 35 x 35 x 384. 4 blocks A.
    for (auto i = 0 ; i < 4 ; ++i) {
        auto ssc = sc.sub( fmt::format("Mixed_5_{}", i+1) );
        net = block_a(net, ssc, end_points);
    }
    // 35 x 35 x 384. Reduction block A
    net = block_reduc_a(net, sc.sub("Mixed_6a"), end_points);
    // 17 x 17 x 1024. 7 blocks B.
    for (auto i = 0 ; i < 7 ; ++i) {
        auto ssc = sc.sub( fmt::format("Mixed_6_{}", i+1) );
        net = block_b(net, ssc, end_points);
    }
    // 17 x 17 x 1024. Reduction block B
    net = block_reduc_b(net, sc.sub("Mixed_7a"), end_points);
    // 8 x 8 x 1536. 3 blocks C
    for (auto i = 0 ; i < 3 ; ++i) {
        auto ssc = sc.sub( fmt::format("Mixed_7_{}", i+1) );
        net = block_c(net, ssc, end_points);
    }
    return net;
}

/* ============================================================================
 * Inception4 network: functional declaration.
 * ========================================================================== */
inline nvinfer1::ITensor* inception4(nvinfer1::ITensor* input,
                                     tfrt::scope sc,
                                     int num_classes=1001)
{
    nvinfer1::ITensor* net;
    // sc = sc.sub("v4");
    // Construct backbone network.
    net = base(input, sc.sub("v4"));
    // Logits end block.
    {
        typedef tfrt::avg_pooling2d<tfrt::PaddingType::VALID>  avg_pool2d;
        typedef tfrt::convolution2d<tfrt::ActivationType::NONE, tfrt::PaddingType::SAME, false>  conv2d;
        auto ssc = sc.sub("v4").sub("Logits");
        net = avg_pool2d(ssc, "AvgPool_1a").ksize({5, 5})(net);
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
    net() : tfrt::imagenet_network("InceptionV4", 1000, true) {}

    /** Inception2 building method. Take a network scope and do the work!
     */
    virtual nvinfer1::ITensor* build(tfrt::scope sc) {
        auto net = tfrt::input(sc)();
        // auto net = tfrt::input(sc).shape({64, 112, 112})();
        net = inception4(net, sc, 1001);
        return net;
    }
};

}

#endif
