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
#ifndef TFRT_SEG_INCEPTION2_v1
#define TFRT_SEG_INCEPTION2_v1

#include <iostream>
#include <NvInfer.h>

#include "robik_classes.h"
#include "../tensorflowrt.h"
#include "../nets/inception2.h"

namespace seg_inception2_v1
{
/* ============================================================================
 * SEG Inception2 V1 network: functional declaration.
 * ========================================================================== */
typedef tfrt::convolution2d<tfrt::ActivationType::RELU, tfrt::PaddingType::SAME, true>  conv2d;
typedef tfrt::separable_convolution2d<tfrt::ActivationType::RELU, tfrt::PaddingType::SAME, false> separable_conv2d;
typedef tfrt::convolution2d_transpose<tfrt::ActivationType::RELU, tfrt::PaddingType::CUSTOM, false>  conv2d_transpose;

/** Additional feature layer.
 */
inline nvinfer1::ITensor* seg_inception2_extra_feature(
    nvinfer1::ITensor* net, nvinfer1::ITensor* net_side, tfrt::scope sc,
    int num_outputs, tfrt::map_tensor* end_points=nullptr)
{
    LOG(INFO) << "BLOCK SEG inception2 extra-features '" << sc.name() << "'. "
            << "Input shape: " << tfrt::dims_str(net->getDimensions());
    // 2x2 transpose convolution with stride=2.
    net = conv2d_transpose(sc, "tconv3x3")
        .noutputs(num_outputs).ksize({2, 2}).stride({2, 2}).padding({0, 0})(net);
    // 3x3 convolution to smooth out the result...
    net = conv2d(sc, "conv3x3").noutputs(num_outputs).ksize({3, 3})(net);
    // Additional side feature to add.
    if(net_side != nullptr) {
        LOG(INFO) << "Additional link shape: " << tfrt::dims_str(net_side->getDimensions());
        // 1x1 compression convolution and sum with rest...
        net_side = conv2d(sc, "conv1x1").noutputs(num_outputs).ksize({1, 1})(net_side);
        net = tfrt::add(sc, "sum")(net, net_side);
    }
    return tfrt::add_end_point(end_points, sc.name(), net);
}
inline nvinfer1::ITensor* seg_inception2_last_layer(
    nvinfer1::ITensor* net, tfrt::scope sc, int num_outputs, tfrt::map_tensor* end_points=nullptr)
{
    typedef tfrt::convolution2d<tfrt::ActivationType::NONE, tfrt::PaddingType::SAME, true>  conv2d;

    LOG(INFO) << "BLOCK SEG inception2 last layer '" << sc.name() << "'. "
            << "Input shape: " << tfrt::dims_str(net->getDimensions());
    // 3x3 transpose convolution with stride=2.
    net = conv2d(sc, "conv3x3").noutputs(num_outputs).ksize({3, 3})(net);
    return tfrt::add_end_point(end_points, sc.name(), net);
}

/** Inception2 base network.
 */
inline nvinfer1::ITensor* block1(
    nvinfer1::ITensor* net, tfrt::scope sc, tfrt::map_tensor* end_points=nullptr)
{
    int depthwise_multiplier = std::min(int(64 / 3), 8);
    // // 7x7 depthwise convolution.
    net = separable_conv2d(sc, "Conv2d_1a_7x7")
        .depthmul(depthwise_multiplier)
        .noutputs(64).ksize({7, 7}).stride({2, 2})(net);
    return tfrt::add_end_point(end_points, sc.sub("Conv2d_1a_7x7").name(), net);;
}
inline nvinfer1::ITensor* inception2_base(
    nvinfer1::ITensor* input, tfrt::scope sc, tfrt::map_tensor* end_points=nullptr)
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
 * SEG Inception2 class: as seg_network.
 * ========================================================================== */
class net : public tfrt::seg_network
{
public:
    /** Constructor with default name.  */
    net() : tfrt::seg_network("seg_inception2", robik::seg_descriptions().size(), false)
    {
        // Set up the descriptions of the classes.
        this->m_desc_classes = robik::seg_descriptions();
    }
    /** SEG Inception2 building method. Take a network scope and do the work!
     */
    virtual nvinfer1::ITensor* build(tfrt::scope sc)
    {
        auto inshape = this->input_shape();
        auto net = tfrt::input(sc)();
        tfrt::map_tensor end_points;

        // Set convolution formulas...
        m_deconv2d_formula = tfrt::tf_conv2d_transpose_formula{1};
        sc.network()->setDeconvolutionOutputDimensionsFormula(&m_deconv2d_formula);
        // sc.network()->setConvolutionOutputDimensionsFormula(&tf_out_formula);
        // sc.network()->setPoolingOutputDimensionsFormula(&tf_out_formula);

        // Build the Inception2 base.
        net = inception2_base(net, sc.sub("inception2_base"), &end_points);
        // Add segmentation extra-features.
        std::vector<std::string> feat_names = {"block6", "block7", "block8", "block9"};
        std::vector<std::string> feat_names_side =
            {"Mixed_4e", "Mixed_3c", "Conv2d_2c_3x3", "Conv2d_1a_7x7"};
        std::vector<std::size_t> feat_size = {384, 192, 96, 48};
        auto ssc = sc.sub("feat_layers_extra");
        for (size_t i = 0 ; i < feat_names.size() ; ++i) {
            // auto net1 = tfrt::find_end_point(&end_points, feat_names[i]);
            auto net_in = tfrt::find_end_point(&end_points, feat_names_side[i]);
            net = seg_inception2_extra_feature(net, net_in, ssc.sub(feat_names[i]), feat_size[i]);
        }
        // Last convolution layer and softmax.
        int num_classes = this->num_classes() - int(!m_empty_class);
        net = seg_inception2_last_layer(net, ssc.sub("block10"), num_classes);
        net = tfrt::softmax(sc, "Softmax")(net);

        // Clear any cached stuff...
        // this->clear_cache();
        return net;
    }

private:
    // TF output formulas.
    tfrt::tf_conv2d_formula  m_conv2d_formula;
    tfrt::tf_conv2d_transpose_formula  m_deconv2d_formula;
};

}

#endif
