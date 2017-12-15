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
#ifndef TFRT_SSD_INCEPTION2_V0
#define TFRT_SSD_INCEPTION2_V0

#include <NvInfer.h>

#include "../tensorflowrt.h"
#include "../nets/inception_v2.h"
#include "robik_classes.h"

namespace ssd_inception2_v0
{
/* ============================================================================
 * SSD Inception2 V0 network: functional declaration.
 * ========================================================================== */
typedef tfrt::convolution2d<tfrt::ActivationType::RELU, tfrt::PaddingType::SAME, true>  conv2d;
typedef tfrt::separable_convolution2d<tfrt::ActivationType::RELU, tfrt::PaddingType::SAME, true> separable_conv2d;

/** Additional feature layer.
 */
inline nvinfer1::ITensor* inception2_extra_feature(nvinfer1::ITensor* net, tfrt::scope sc,
                                                   int num_outputs, tfrt::map_tensor* end_points=nullptr)
{
    LOG(INFO) << "BLOCK SSD inception2 extra-features '" << sc.name() << "'. "
            << "Input shape: " << tfrt::dims_str(net->getDimensions());
    // 1x1 compression convolution.
    net = conv2d(sc, "conv1x1").noutputs(num_outputs / 2).ksize({1, 1})(net);
    // 3x3 convolution with stride=2.
    net = conv2d(sc, "conv3x3").noutputs(num_outputs).ksize({3, 3}).stride({2, 2})(net);
    return tfrt::add_end_point(end_points, sc.name(), net);
}
/** Inception2 base network.
 */
inline nvinfer1::ITensor* block1(nvinfer1::ITensor* net, tfrt::scope sc,
                                 tfrt::map_tensor* end_points=nullptr)
{
    int depthwise_multiplier = std::min(int(64 / 3), 8);
    // // 7x7 depthwise convolution.
    net = separable_conv2d(sc, "Conv2d_1a_7x7")
        .depthmul(depthwise_multiplier)
        .noutputs(64).ksize({7, 7}).stride({2, 2})(net);
    return net;
}
inline nvinfer1::ITensor* inception2_base(nvinfer1::ITensor* input, tfrt::scope sc,
                                          tfrt::map_tensor* end_points=nullptr)
{
    nvinfer1::ITensor* net{input};
    // Main blocks 1 to 5.
    // net = inception2::block1(net, sc, end_points);
    net = block1(net, sc, end_points);
    net = inception2::block2(net, sc, end_points);
    net = inception2::block3(net, sc, end_points);
    net = inception2::block4(net, sc, end_points);
    net = inception2::block5(net, sc, end_points);
    return net;
}

/* ============================================================================
 * SSD Inception2 class: as ssd_network.
 * ========================================================================== */
class net : public tfrt::ssd_network
{
public:
    /** Constructor with default name.  */
    net() : tfrt::ssd_network("ssd_inception2") {}

    /** SSD Inception2 building method. Take a network scope and do the work!
     */
    virtual nvinfer1::ITensor* build(tfrt::scope sc) {
        auto net = tfrt::input(sc)();
        tfrt::map_tensor end_points;

        // Build the Inception2 base.
        net = inception2_base(net, sc.sub("inception2_base"), &end_points);
        // Add extra-features.
        auto features = this->features();
        auto ssc = sc.sub("feat_layers_extra");
        for(auto&& f : features) {
            int num_outputs = f.shape.c();
            if(num_outputs > 0) {
                net = inception2_extra_feature(net, ssc.sub(f.name), num_outputs, &end_points);
            }
        }
        // Add SSD boxed2d blocks.
        ssc = sc.sub("ssd_boxes2d_blocks");
        for(auto&& f : features) {
            net = tfrt::find_end_point(&end_points, f.name);
            // SSD boxes2d layer.
            tfrt::ssd_boxes2d_block(ssc, f.name + "_boxes")
                .num_anchors(f.num_anchors2d_total())
                .num_classes(this->num_classes_2d())
                .decode_boxes(true)(net);
            // tfrt::ssd_boxes2d_block(net, ssc.sub(f.name + "_boxes"),
            //     f.num_anchors2d_total(), this->num_classes_2d(),
            //     true, true);
        }
        // Clear any cached stuff...
        this->clear_cache();
        return net;
    }
};

}

#endif
