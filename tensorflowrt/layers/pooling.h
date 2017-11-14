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
#ifndef TFRT_LAYERS_POOLING_H
#define TFRT_LAYERS_POOLING_H

#include "abstract.h"

namespace tfrt
{


/** Generic pooling layer.
 */
template <PaddingType PAD, nvinfer1::PoolingType POOL>
class pooling2d : public operation2d<ActivationType::NONE, PAD, false>
{
public:
    /** Constructor: declare the layer.
     */
    pooling2d(const tfrt::scope& sc, const std::string& lname) :
        operation2d<ActivationType::NONE, PAD, false>(sc, lname) {
    }
    /** Add the layer to network graph, using operator(root).
     */
    virtual nvinfer1::ITensor* operator()(nvinfer1::ITensor* net) {
        LOG(INFO) << "LAYER 2D contrib pooling '" << this->m_scope.name() << "'. "
            << "Input shape: " << dims_str(net->getDimensions());
        net = this->pooling(net);
        return this->mark_output(net);
    }

private:
    /** Set up the convolution operation.
     */
    nvinfer1::ITensor* pooling(nvinfer1::ITensor* input) {
        auto inshape = static_cast<nvinfer1::DimsCHW&&>(input->getDimensions());
        nvinfer1::IPoolingLayer* poollayer = nullptr;
        poollayer = this->m_scope.network()->addPooling(
            *input, POOL, DIMRT(this->ksize()));
        CHECK_NOTNULL(poollayer);
        // Set name, padding and stride.
        poollayer->setName(this->m_scope.cname());
        poollayer->setPadding(DIMRT(this->padding(inshape)));
        poollayer->setStride(DIMRT(this->stride()));
        return poollayer->getOutput(0);
    }
    using operation2d<ActivationType::NONE, PAD, false>::noutputs;
};

/** 2D max pooling layer.
 */
template <PaddingType PAD>
class max_pooling2d : public pooling2d<PAD, nvinfer1::PoolingType::kMAX>
{
public:
    max_pooling2d(const tfrt::scope& sc, const std::string& lname="MaxPool2D") :
        pooling2d<PAD, nvinfer1::PoolingType::kMAX>(sc, lname) {
    }
};
/** 2D average pooling layer.
 */
template <PaddingType PAD>
class avg_pooling2d : public pooling2d<PAD, nvinfer1::PoolingType::kAVERAGE>
{
public:
    avg_pooling2d(const tfrt::scope& sc, const std::string& lname="AvgPool2D") :
        pooling2d<PAD, nvinfer1::PoolingType::kAVERAGE>(sc, lname) {
    }
};

/* ============================================================================
 * Non-standard layers!
 * ========================================================================== */
/** Bilinear 2d interpolation. Input is supposed to be "checkerboard" tensor,
 * with zero entries between known values. Using a 3x3 filter by default.
 * This implementation is based on average pooling.
 */
class bilinear2d_pool : public layer
{
public:
    /** Constructor: declare the layer. */
    bilinear2d_pool(const tfrt::scope& sc, const std::string& lname) :
        layer(sc, lname) {
    }
    /** Add the layer to network graph, using operator(root).*/
    virtual nvinfer1::ITensor* operator()(nvinfer1::ITensor* net) {
        LOG(INFO) << "LAYER 2D bilinear-pool interpolation '"
            << this->m_scope.name() << "'. "
            << "Input shape: " << dims_str(net->getDimensions());
        net = this->interpolation(net);
        return this->mark_output(net);
    }

private:
    /** Bilinear interpolation. */
    nvinfer1::ITensor* interpolation(nvinfer1::ITensor* input) {
        auto tf_net = this->m_scope.tfrt_network();
        auto dt = tf_net->datatype();
        auto inshape = static_cast<nvinfer1::DimsCHW&&>(input->getDimensions());

        // Average pooling as a first step.
        auto avglayer = this->m_scope.network()->addPooling(
            *input, nvinfer1::PoolingType::kAVERAGE, {3, 3});
        CHECK_NOTNULL(avglayer);
        // Set name, padding and stride.
        avglayer->setName(this->m_scope.sub("avg_pool2d").cname());
        avglayer->setPadding({1, 1});
        avglayer->setStride({1, 1});
        auto net = avglayer->getOutput(0);

        // Compute scaling weights.
        auto wname = m_scope.sub("weights_scale").name();
        auto& wtensor = tf_net->create_tensor(wname, this->weights_scale(inshape), dt);
        nvinfer1::Weights wscale = tf_net->tensor_to_weights(wtensor);
        nvinfer1::Weights wzero{dt, nullptr, 0};
        // Second step: rescaling to adjust weights.
        auto slayer = this->m_scope.network()->addScale(
            *net, nvinfer1::ScaleMode::kELEMENTWISE,  wzero, wscale, wzero);
        CHECK_NOTNULL(slayer);
        slayer->setName(this->m_scope.sub("scale").cname());
        net = slayer->getOutput(0);
        return net;
    }
    /** Generate scaling weights.  */
    tfrt::chw<float>::tensor weights_scale(const nvinfer1::DimsCHW& inshape)
    {
        tfrt::chw<float>::tensor w{inshape.c(), inshape.h(), inshape.w()};
        // Generate scaling weights. A bit of a dirty hack for now...
        for (long i = 0 ; i < w.dimension(1) ; ++i) {
            for (long j = 0 ; j < w.dimension(2) ; ++j) {
                // Rescaling value.
                float val = 1.;
                if (i % 2 == 0 && j % 2 == 0) {
                    val = 9.;
                }
                else if (i % 2 == 1 && j % 2 == 0) {
                    val = 9. / 2.;
                }
                else if (i % 2 == 0 && j % 2 == 1) {
                    val = 9. / 2.;
                }
                else if (i % 2 == 1 && j % 2 == 1) {
                    val = 9. / 4.;
                }
                for (long k = 0 ; k < w.dimension(0) ; ++k) {
                    w(k, i, j) = val;
                }
            }
        }
        return w;
    }
};


/** Bilinear 2d interpolation. Input is supposed to be "checkerboard" tensor,
 * with zero entries between known values. Using a 3x3 filter by default.
 * This implementation is based 3x3 convolution, which surprisingly seems
 * to perform better.
 */
class bilinear2d_conv : public layer
{
public:
    /** Constructor: declare the layer. */
    bilinear2d_conv(const tfrt::scope& sc, const std::string& lname) :
        layer(sc, lname) {
    }
    /** Add the layer to network graph, using operator(root). */
    virtual nvinfer1::ITensor* operator()(nvinfer1::ITensor* net) {
        LOG(INFO) << "LAYER 2D bilinear-conv interpolation '"
            << this->m_scope.name() << "'. "
            << "Input shape: " << dims_str(net->getDimensions());
        net = this->interpolation(net);
        return this->mark_output(net);
    }

private:
    /** Bilinear interpolation. */
    nvinfer1::ITensor* interpolation(nvinfer1::ITensor* input) {
        auto tf_net = this->m_scope.tfrt_network();
        auto dt = tf_net->datatype();
        auto inshape = static_cast<nvinfer1::DimsCHW&&>(input->getDimensions());
        // Interpolation weights.
        auto wname = m_scope.sub("weights_interpolation").name();
        auto& wtensor = tf_net->create_tensor(wname,
            this->weights_interpolation(inshape), dt);
        nvinfer1::Weights weights = tf_net->tensor_to_weights(wtensor);
        nvinfer1::Weights biases{dt, nullptr, 0};
        // Convolution layer.
        nvinfer1::IConvolutionLayer* convlayer = nullptr;
        convlayer = this->m_scope.network()->addConvolution(
            *input, inshape.c(), {3, 3}, weights, biases);
        CHECK_NOTNULL(convlayer);
        // Set name, padding, stride and nb groups.
        convlayer->setName(this->m_scope.sub("conv2d_interpolation").cname());
        convlayer->setPadding({1, 1});
        convlayer->setStride({1, 1});
        convlayer->setNbGroups(1);
        return convlayer->getOutput(0);
    }
    /** Generate convolution weights. */
    tfrt::nchw<float>::tensor weights_interpolation(const nvinfer1::DimsCHW& inshape)
    {
        // Try group convolution?
        // GKCRS order: G == nb groups, K: nb outputs, C: nb inputs.
        tfrt::nchw<float>::tensor w{inshape.c(), inshape.c(), 3, 3};
        w.setConstant(0.0);
        for (long i = 0 ; i < inshape.c() ; ++i) {
            // Interpolation grid.
            w(i, i, 0, 0) = 0.25;
            w(i, i, 2, 2) = 0.25;
            w(i, i, 0, 2) = 0.25;
            w(i, i, 2, 0) = 0.25;

            w(i, i, 1, 0) = 0.5;
            w(i, i, 0, 1) = 0.5;
            w(i, i, 2, 1) = 0.5;
            w(i, i, 1, 2) = 0.5;

            w(i, i, 1, 1) = 1.;
        }
        return w;
    }
};

/* ============================================================================
 * DEFAULT layers name.
 * ========================================================================== */
typedef max_pooling2d<PaddingType::SAME>  max_pool2d;
typedef avg_pooling2d<PaddingType::SAME>  avg_pool2d;
// typedef bilinear2d_conv  bilinear2d;
typedef bilinear2d_pool  bilinear2d;

}

#endif

