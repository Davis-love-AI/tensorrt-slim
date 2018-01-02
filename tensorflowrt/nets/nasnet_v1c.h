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
#ifndef TFRT_NASNET_V1C_H
#define TFRT_NASNET_V1C_H

#include <map>
#include <fmt/format.h>

#include <NvInfer.h>
#include "../tensorflowrt.h"

namespace nasnet_v1c
{
/** Arg scope for NASNet: SAME padding + batch normalization. ReLU before.
 */
// typedef tfrt::separable_convolution2d_test<
//     tfrt::ActivationType::NONE, tfrt::PaddingType::SAME, true> separable_conv2d;
typedef tfrt::separable_convolution2d<
    tfrt::ActivationType::NONE, tfrt::PaddingType::SAME, true, false> separable_conv2d;
typedef tfrt::depthwise_convolution2d<
    tfrt::ActivationType::NONE, tfrt::PaddingType::SAME, true> depthwise_conv2d;
typedef tfrt::convolution2d<
    tfrt::ActivationType::NONE, tfrt::PaddingType::SAME, true>  conv2d;
typedef tfrt::convolution2d<
    tfrt::ActivationType::NONE, tfrt::PaddingType::VALID, true>  conv2d_valid;

typedef tfrt::max_pooling2d<tfrt::PaddingType::SAME>    max_pool2d;
typedef tfrt::avg_pooling2d<tfrt::PaddingType::SAME>    avg_pool2d;
typedef tfrt::concat_channels                           concat_channels;

/* ============================================================================
 * NASNet Abstract cell
 * ========================================================================== */
class base_cell
{
public:
    base_cell(tfrt::scope sc, size_t num_filters, float filter_scaling) :
        m_scope{sc}, m_num_conv_filters{num_filters}, m_filter_scaling{filter_scaling}
    {
    }
    /** Base of every cell: 1x1 convolution. */
    nvinfer1::ITensor* base_cell_n(nvinfer1::ITensor* net, tfrt::scope sc) const
    {
        size_t fsize = this->filter_size();
        // ReLU + conv.
        // net = tfrt::relu(sc, "relu")(net);
        net = conv2d(sc, "1x1").noutputs(fsize).ksize(1)(net);
        return net;
    }
    /** Reduce n-1 layer to correct shape. */
    nvinfer1::ITensor* reduce_n_1(
        nvinfer1::ITensor* net_n, nvinfer1::ITensor* net_n_1, tfrt::scope sc) const
    {
        nvinfer1::ITensor* net = net_n_1;
        // No previous layer?
        if (net_n_1 == nullptr) {
            return net_n;
        }
        size_t fsize = this->filter_size();
        auto shape_n = tfrt::dims_chw(net_n);
        auto shape_n_1 = tfrt::dims_chw(net_n_1);

        // Approximation of the original implementation. TODO: FIX stride=2
        // First case: different HW shape.
        if (shape_n.h() != shape_n_1.h() || shape_n.w() != shape_n_1.w()) {
            net = avg_pool2d(sc).ksize(3).stride(2)(net);
        }
        // Number of channels different?
        if (int(fsize) != shape_n_1.c()) {
            net = conv2d(sc, "1x1").noutputs(fsize).ksize(1)(net);
        }
        return net;
    }
    /** Identity block: reduce to number of outputs, or stride > 1. */
    nvinfer1::ITensor* identity(nvinfer1::ITensor* net, tfrt::scope sc,
        size_t stride, size_t num_outputs, bool relu=false) const
    {
        auto ssc = sc.sub("identity");
        auto shape = tfrt::dims_chw(net);
        if (stride > 1 || shape.c() != int(num_outputs)) {
            if (relu) {
                net = tfrt::relu(ssc, "relu")(net);
            }
            net = conv2d(ssc, "1x1").noutputs(num_outputs).ksize(1).stride(1)(net);
        }
        return net;
    }
    /** Stack of separable convolutions. */
    nvinfer1::ITensor* sep_conv2d_stacked(nvinfer1::ITensor* net, tfrt::scope sc,
        size_t ksize, size_t dilation, size_t stride, size_t ngroups_pw,
        size_t num_outputs, size_t num_layers) const
    {
        auto ssc = sc.sub("stack");
        std::string name;
        for (size_t i = 0 ; i < num_layers-1 ; ++i) {
            // Separable conv.
            name = fmt::format("separable_{0}x{0}_{1}", ksize, i+1);
            net = separable_conv2d(ssc, name)
                .dw_group_size(1).pw_ngroups(ngroups_pw)
                .dilation(dilation)
                .noutputs(num_outputs).stride(stride).ksize(ksize)(net);
            name = fmt::format("relu_{}", i+1);
            net = tfrt::relu(ssc, name)(net);
            stride = 1;
        }
        name = fmt::format("separable_{0}x{0}_{1}", ksize, num_layers);
        net = depthwise_conv2d(ssc, name)
            .group_size(1).dilation(dilation).stride(stride).ksize(ksize)(net);
        return net;
    }
public:
    size_t filter_size() const {
        return size_t(m_num_conv_filters * m_filter_scaling);
    }

protected:
    /** General scope of the cell. */
    tfrt::scope  m_scope;
    /** Number of convolution filters, normalized. */
    size_t  m_num_conv_filters;
    /** Filter scaling, used to compute the final number of filters. */
    float  m_filter_scaling;
};

/* ============================================================================
 * NASNet Normal cell
 * ========================================================================== */
class normal_cell : public base_cell
{
public:
    normal_cell(tfrt::scope sc, size_t num_filters, float filter_scaling) :
        base_cell(sc, num_filters, filter_scaling)
    {
    }
    /** Operator: taking layers n and n-1. */
    nvinfer1::ITensor* operator()(nvinfer1::ITensor* net_n, nvinfer1::ITensor* net_n_1)
    {
        nvinfer1::ITensor *net, *net_3x3;
        std::vector<nvinfer1::ITensor*> blocks;
        tfrt::scope sc{m_scope};

        // Basic cell + reduce n-1.
        net_n = this->base_cell_n(net_n, m_scope.sub("base_cell_n"));
        net_n_1 = this->reduce_n_1(net_n, net_n_1, m_scope.sub("reduce_n_1"));
        size_t fsize = this->filter_size();

        blocks.push_back( net_n );
        auto net_c1 = tfrt::concat_channels(sc.sub("c1"))({net_n, net_n_1});

        // Average pooling...
        net = avg_pool2d(sc.sub("avg_pool2d")).ksize(3)(net_c1);
        blocks.push_back( net );
        // First 3x3 sep. conv2d.
        net_3x3 = this->sep_conv2d_stacked(
            net_c1, sc.sub("sep_3x3_1"), 3, 2, 1, 2, fsize*2, 2);
        blocks.push_back( net_3x3 );
        // Second 3x3 sep. conv2d.
        net_3x3 = this->sep_conv2d_stacked(
            net_c1, sc.sub("sep_3x3_2"), 3, 1, 1, 2, fsize*3, 2);
        blocks.push_back( net_3x3 );

        // Concat this big mess!
        net = tfrt::concat_channels(sc)(blocks);
        return net;
    }
};

/* ============================================================================
 * NASNet Reduction cell
 * ========================================================================== */
class reduction_cell : public base_cell
{
public:
    reduction_cell(tfrt::scope sc, size_t num_filters, float filter_scaling) :
        base_cell(sc, num_filters, filter_scaling)
    {
    }
    /** Operator: taking layers n and n-1. */
    nvinfer1::ITensor* operator()(nvinfer1::ITensor* net_n, nvinfer1::ITensor* net_n_1)
    {
        nvinfer1::ITensor *net, *net_3x3;
        std::vector<nvinfer1::ITensor*> blocks;
        tfrt::scope sc{m_scope};

        // Basic cell + reduce n-1.
        net_n = this->base_cell_n(net_n, m_scope.sub("base_cell_n"));
        net_n_1 = this->reduce_n_1(net_n, net_n_1, m_scope.sub("reduce_n_1"));
        size_t fsize = this->filter_size();

        // blocks.push_back( net_n );
        auto net_c1 = tfrt::concat_channels(sc.sub("c1"))({net_n, net_n_1});
        // Average pooling...
        net = max_pool2d(sc.sub("max_pool2d")).stride(2).ksize(3)(net_c1);
        blocks.push_back( net );
        net = avg_pool2d(sc.sub("avg_pool2d")).stride(2).ksize(3)(net_c1);

        // First 3x3 sep. conv2d.
        net_3x3 = this->sep_conv2d_stacked(
            net_c1, sc.sub("sep_3x3_1"), 3, 2, 2, 2, fsize*2, 2);
        auto net_c2 = tfrt::concat_channels(sc.sub("c2"))({net_3x3, net});
        // Second 3x3 sep. conv2d.
        net_3x3 = this->sep_conv2d_stacked(
            net_c2, sc.sub("sep_3x3_2"), 3, 1, 1, 4, fsize*4, 2);
        blocks.push_back( net_3x3 );

        // Concat this big mess!
        net = tfrt::concat_channels(sc)(blocks);
        return net;
    }
};

/* ============================================================================
 * NASNet network.
 * ========================================================================== */
class net : public tfrt::imagenet_network
{
public:
    net(std::string name,
        float stem_multiplier, float filter_scaling_rate,
        size_t num_cells, size_t num_reduction_layers, size_t num_conv_filters,
        bool skip_reduction_layer_input) :
            tfrt::imagenet_network(name, 1000, true),
            m_stem_multiplier{stem_multiplier},
            m_filter_scaling_rate{filter_scaling_rate},
            m_num_cells{num_cells},
            m_num_reduction_layers{num_reduction_layers},
            m_num_conv_filters{num_conv_filters},
            m_skip_reduction_layer_input{skip_reduction_layer_input}
    {
    }
    /** Build the network: core + softmax. Whohohoooo! */
    virtual nvinfer1::ITensor* build(tfrt::scope sc) {
        // Input + core.
        auto net = tfrt::input(sc)();
        net = this->core(net, sc);

        // Logits?
        // auto ssc = sc.sub("Logits");
        // net = avg_pool2d(ssc, "AvgPool_1a_7x7").ksize({7, 7})(net);
        // net = conv2d(ssc, "Conv2d_1c_1x1").noutputs(num_classes).ksize({1, 1})(net);
        net = tfrt::softmax(sc, "Softmax")(net);
        return net;
    }
    /** Core of the network. */
    nvinfer1::ITensor* core(nvinfer1::ITensor* input, tfrt::scope sc) {
        nvinfer1::ITensor *net, *net_n_1{nullptr};
        // ImageNet stem cells.
        auto cell_outputs = imagenet_stem(input, sc);
        net = cell_outputs[cell_outputs.size()-1];
        LOG(INFO) << "ImageNet stem done!";

        // Core of the network.
        auto reduction_cells = this->reduction_layers();
        double filter_scaling = 1.0;
        size_t true_cell_num = 2;
        for (size_t i = 0 ; i < m_num_cells ; ++i) {
            LOG(INFO) << "CELL building: " << (i+1);
            // Skip reduction cell input?
            if (m_skip_reduction_layer_input) {
                net_n_1 = cell_outputs[cell_outputs.size()-2];
            }
            // Reduction cell?
            if (reduction_cells[i]) {
                LOG(INFO) << "REDUCTION CELL building: " << (i+1);
                filter_scaling *= m_filter_scaling_rate;
                // Scope name + cell construction...
                auto ssc = sc.sub( fmt::format("reduction_cell_{}", i) );
                auto cell = reduction_cell(ssc, m_num_conv_filters, filter_scaling);

                // Apply!
                net = cell(net, cell_outputs[cell_outputs.size()-2]);
                cell_outputs.push_back(net);
                true_cell_num++;
            }
            // Skip reduction cell input?
            if (!m_skip_reduction_layer_input) {
                net_n_1 = cell_outputs[cell_outputs.size()-2];
            }

            LOG(INFO) << "NORMAL CELL building: " << (i+1);
            // Scope name + normal cell construction...
            auto ssc = sc.sub( fmt::format("cell_{}", i) );
            auto cell = normal_cell(ssc, m_num_conv_filters, filter_scaling);
            // Apply!
            net = cell(net, net_n_1);
            cell_outputs.push_back(net);
            true_cell_num++;
        }
        return net;
    }

protected:
    /** ImageNet stem: 3x3 conv + reduction cells. */
    std::vector<nvinfer1::ITensor*> imagenet_stem(nvinfer1::ITensor* net, tfrt::scope sc) const {
        // First 3x3 convolution.
        size_t num_stem_filters = int(32 * m_stem_multiplier);
        // VALID padding?
        net = conv2d_valid(sc, "conv0").ksize(3).stride(2).noutputs(num_stem_filters)(net);

        // Reduction cells.
        size_t num_stem_cells = 2;
        std::vector<nvinfer1::ITensor*> cell_outputs;
        cell_outputs.push_back(nullptr);
        cell_outputs.push_back(net);

        // Filter scaling?
        double filter_scaling = 1.0 / pow(double(m_filter_scaling_rate), double(num_stem_cells));
        for (size_t i = 0 ; i < num_stem_cells ; ++i) {
            // Scope name + cell construction...
            auto ssc = sc.sub( fmt::format("cell_stem_{}", i) );
            auto cell = reduction_cell(ssc, m_num_conv_filters, filter_scaling);

            // Apply!
            int n = cell_outputs.size();
            net = cell(cell_outputs[n-1], cell_outputs[n-2]);
            cell_outputs.push_back(net);
            // Update filter scaling.
            filter_scaling *= m_filter_scaling_rate;
        }
        return cell_outputs;
    }
    /** Reduction layers: bool vector */
    std::vector<bool> reduction_layers() const {
        std::vector<bool> r_layers(m_num_cells, false);
        for (size_t i = 1 ; i <= m_num_reduction_layers ; ++i) {
            float l_idx = (float(i) / float(m_num_reduction_layers + 1)) * m_num_cells;
            r_layers[int(l_idx)] = true;
        }
        return r_layers;
    }

protected:
    /** Stem multiplier. */
    float  m_stem_multiplier;
    /** Filter scaling. */
    float  m_filter_scaling_rate;
    /** Number of cells. */
    size_t  m_num_cells;
    /** Number of reduction layers. */
    size_t  m_num_reduction_layers;
    /** Number of convolutional filters. */
    size_t  m_num_conv_filters;
    /** Skip reduction layer input (for n-1 layer). */
    bool  m_skip_reduction_layer_input;
};



}

/* ============================================================================
 * NASNet mobile.
 * ========================================================================== */
namespace nasnet_v1c_mobile
{
class net : public nasnet_v1c::net
{
public:
    /** NASNet mobile definition. */
    net() : nasnet_v1c::net("nasnet_v1c_mobile", 1.0, 2.0, 12, 2, 44, false)
    {}

    // stem_multiplier=1.0,
    // dense_dropout_keep_prob=0.5,
    // num_cells=12,
    // filter_scaling_rate=2.0,
    // drop_path_keep_prob=1.0,
    // num_conv_filters=44,
    // use_aux_head=1,
    // num_reduction_layers=2,
    // data_format='NHWC',
    // skip_reduction_layer_input=0,
    // total_training_steps=250000,
};
}
/* ============================================================================
 * NASNet large.
 * ========================================================================== */
namespace nasnet_v1c_large
{
class net : public nasnet_v1c::net
{
public:
    /** NASNet mobile definition. */
    net() : nasnet_v1c::net("nasnet_v1c_large", 3.0, 2.0, 18, 2, 168, true)
    {}
    // stem_multiplier=3.0,
    // dense_dropout_keep_prob=0.5,
    // num_cells=18,
    // filter_scaling_rate=2.0,
    // num_conv_filters=168,
    // drop_path_keep_prob=drop_path_keep_prob,
    // use_aux_head=1,
    // num_reduction_layers=2,
    // data_format='NHWC',
    // skip_reduction_layer_input=1,
    // total_training_steps=250000,
};
}
#endif
