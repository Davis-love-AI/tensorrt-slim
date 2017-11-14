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
#ifndef TFRT_LAYERS_UNARY_H
#define TFRT_LAYERS_UNARY_H

#include "abstract.h"

namespace tfrt
{


/* ============================================================================
 * UNARY layers definition.
 * ========================================================================== */
/** Scaling layer: output = (input * scale + shift)^power
 */
class scale : public layer
{
public:
    /** Standard constructor, with scope and layer sub-name.
     * Initialize mode to UNIFORM.
     */
    scale(const tfrt::scope& sc, const std::string& lname) :
        layer(sc, lname), m_mode{nvinfer1::ScaleMode::kUNIFORM} {
    }
    /** Named parameter: scaling mode. */
    scale& mode(nvinfer1::ScaleMode mode) {
        m_mode = mode;
        return *this;
    }
    nvinfer1::ScaleMode mode() const {
        return m_mode;
    }
    /** Add the scaling layer to network graph, using operator(root).
     */
    virtual nvinfer1::ITensor* operator()(nvinfer1::ITensor* net) {
        LOG(INFO) << "LAYER scale '" << this->m_scope.name() << "'. "
            << "Input shape: " << dims_str(net->getDimensions());
        net = this->scale_op(net);
        return this->mark_output(net);
    }

protected:
    /** Set up a scaling operation.
     */
    nvinfer1::ITensor* scale_op(nvinfer1::ITensor* net) {
        auto inshape = static_cast<nvinfer1::DimsCHW&&>(net->getDimensions());
        auto wshape = this->weights_shape(inshape);
        LOG(INFO) << "OP scaling. Input shape: " << dims_str(inshape);
        // Get weights and add scale layer.
        auto drift = m_scope.weights("drift", wshape);
        auto scale = m_scope.weights("scale", wshape);
        auto power = m_scope.weights("power", wshape);
        auto slayer = this->m_scope.network()->addScale(*net, m_mode, drift, scale, power);
        CHECK_NOTNULL(slayer);
        slayer->setName(m_scope.cname());
        return slayer->getOutput(0);
    }
    /** Get weights shape. */
    nvinfer1::Dims weights_shape(const nvinfer1::DimsCHW& inshape)
    {
        if (m_mode == nvinfer1::ScaleMode::kUNIFORM) {
            return nvinfer1::DimsC{1};
        }
        else if (m_mode == nvinfer1::ScaleMode::kCHANNEL) {
            return nvinfer1::DimsC{inshape.c()};
        }
        else {
            return nvinfer1::DimsCHW{inshape.c(), inshape.h(), inshape.w()};
        }
    }

protected:
    // Scaling mode
    nvinfer1::ScaleMode m_mode;
};



/** Simple batch norm layer. Nothing else! */
class batch_norm : public operation2d<tfrt::ActivationType::NONE, tfrt::PaddingType::SAME, true>
{
public:
    /** Constructor: declare the layer. */
    batch_norm(const tfrt::scope& sc, const std::string& lname="batch_norm") :
         operation2d<tfrt::ActivationType::NONE, tfrt::PaddingType::SAME, true>(sc, lname) {
    }
    /** Add the layer to network graph, using operator(root). */
    virtual nvinfer1::ITensor* operator()(nvinfer1::ITensor* net) {
        LOG(INFO) << "LAYER 2D contrib batch norm '" << this->m_scope.name() << "'. "
            << "Input shape: " << dims_str(net->getDimensions());
        net = this->operation2d<ActivationType::NONE, PaddingType::SAME, true>::batch_norm(net);
        return this->mark_output(net);
    }
private:
    // Deactivate a few methods.
    using operation2d<ActivationType::NONE, PaddingType::SAME, true>::noutputs;
    using operation2d<ActivationType::NONE, PaddingType::SAME, true>::ksize;
    using operation2d<ActivationType::NONE, PaddingType::SAME, true>::stride;
    using operation2d<ActivationType::NONE, PaddingType::SAME, true>::padding;
};
/** Simple activation layer. Nothing else! */
template <ActivationType ACT>
class activation : public operation2d<ACT, PaddingType::SAME, false>
{
public:
    /** Constructor: declare the layer.
     */
    activation(const tfrt::scope& sc, const std::string& lname="Activation2d") :
        operation2d<ACT, PaddingType::SAME, false>(sc, lname) {
    }
    /** Add the layer to network graph.
     */
    virtual nvinfer1::ITensor* operator()(nvinfer1::ITensor* net) {
        LOG(INFO) << "LAYER 2D activation '" << this->m_scope.name() << "'. "
            << "Input shape: " << dims_str(net->getDimensions());
        net = this->operation2d<ACT, PaddingType::SAME, false>::activation(net);
        return this->mark_output(net);
    }

private:
    // Deactivate a few methods.
    using operation2d<ACT, PaddingType::SAME, false>::noutputs;
    using operation2d<ACT, PaddingType::SAME, false>::ksize;
    using operation2d<ACT, PaddingType::SAME, false>::stride;
    using operation2d<ACT, PaddingType::SAME, false>::padding;
};

/* ============================================================================
 * DEFAULT layers name.
 * ========================================================================== */
typedef activation<ActivationType::RELU>    relu;
typedef activation<ActivationType::SIGMOID> sigmoid;
typedef activation<ActivationType::TANH>    tanh;
typedef activation<ActivationType::SOFTMAX> softmax;


}

#endif

