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
#ifndef TFRT_LAYERS_OPERATIONS_H
#define TFRT_LAYERS_OPERATIONS_H

#include "abstract.h"

namespace tfrt
{
/* ============================================================================
 * BINARY operations.
 * ========================================================================== */
/** Elementwise max.
 */
class max : public layer
{
public:
    /** Standard constructor, with scope and layer sub-name.
     */
    max(const tfrt::scope& sc, const std::string& lname) :
        layer(sc, lname) {
    }
    virtual nvinfer1::ITensor* operator()(nvinfer1::ITensor* net) {
        LOG(ERROR) << "LAYER operation not implemented.";
        return nullptr;
    }
    nvinfer1::ITensor* operator()(nvinfer1::ITensor* net1, nvinfer1::ITensor* net2) {
        LOG(INFO) << "LAYER maximum '" << this->m_scope.name() << "'. "
            << "Input shape: " << dims_str(net1->getDimensions())
            << " and " << dims_str(net2->getDimensions());
        auto layer = this->m_scope.network()->addElementWise(
            *net1, *net2, nvinfer1::ElementWiseOperation::kMAX);
        CHECK_NOTNULL(layer);
        layer->setName(m_scope.cname());
        return this->mark_output(layer->getOutput(0));
    }
};
/** Element wise sum. TODO: refactor into a single class sum, multiply and max.
 */
class add : public layer
{
public:
    /** Standard constructor, with scope and layer sub-name.
     */
    add(const tfrt::scope& sc, const std::string& lname) :
        layer(sc, lname) {
    }
    virtual nvinfer1::ITensor* operator()(nvinfer1::ITensor* net) {
        LOG(ERROR) << "LAYER operation not implemented.";
        return nullptr;
    }
    nvinfer1::ITensor* operator()(nvinfer1::ITensor* net1, nvinfer1::ITensor* net2) {
        LOG(INFO) << "LAYER add '" << this->m_scope.name() << "'. "
            << "Inputs shape: " << dims_str(net1->getDimensions())
            << " and " << dims_str(net2->getDimensions());
        auto layer = this->m_scope.network()->addElementWise(
            *net1, *net2, nvinfer1::ElementWiseOperation::kSUM);
        CHECK_NOTNULL(layer);
        layer->setName(m_scope.cname());
        return this->mark_output(layer->getOutput(0));
    }
};
/** Element wise multiply.
 */
class multiply : public layer
{
public:
    /** Standard constructor, with scope and layer sub-name.
     */
    multiply(const tfrt::scope& sc, const std::string& lname) :
        layer(sc, lname) {
    }
    virtual nvinfer1::ITensor* operator()(nvinfer1::ITensor* net) {
        LOG(ERROR) << "LAYER operation not implemented.";
        return nullptr;
    }
    nvinfer1::ITensor* operator()(nvinfer1::ITensor* net1, nvinfer1::ITensor* net2) {
        LOG(INFO) << "LAYER multiply '" << this->m_scope.name() << "'. "
            << "Input shape: " << dims_str(net1->getDimensions())
            << " and " << dims_str(net2->getDimensions());
        auto layer = this->m_scope.network()->addElementWise(
            *net1, *net2, nvinfer1::ElementWiseOperation::kPROD);
        CHECK_NOTNULL(layer);
        layer->setName(m_scope.cname());
        return this->mark_output(layer->getOutput(0));
    }
};


/** Concat Tensors along channel dimension.
 */
class concat_channels : public layer
{
public:
    /**Standard constructor, with scope and layer sub-name.
     */
    concat_channels(const tfrt::scope& sc, const std::string& lname="Concat") :
        layer(sc, lname) {
    }
    /** Add the layer to network graph, using operator(root).
     */
    nvinfer1::ITensor* operator()(const std::vector<nvinfer1::ITensor*>& inputs) {
        LOG(INFO) << "LAYER concat '" << this->m_scope.name() << "'.";
        auto clayer = this->m_scope.network()->addConcatenation(&inputs[0], inputs.size());
        CHECK_NOTNULL(clayer);
        clayer->setName(this->m_scope.name().c_str());
        auto net = clayer->getOutput(0);
        return this->mark_output(net);
    }

protected:
    // Should not be used!
    nvinfer1::ITensor* operator()(nvinfer1::ITensor*) { return nullptr; }
};

/* ============================================================================
 * DEFAULT layers name.
 * ========================================================================== */
typedef add sum;
typedef multiply mul;

}

#endif

