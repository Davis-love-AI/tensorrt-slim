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
    add(const tfrt::scope& sc, const std::string& lname="Add") :
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
    multiply(const tfrt::scope& sc, const std::string& lname="Mul") :
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
 * MISC operations.
 * ========================================================================== */
/** Shuffle layer. Similarly to TensorRT, 3 steps operations:
 * a first transpose operation, a reshape operation and a second transpose operation.
 */
class shuffle : public layer
{
public:
    /** Standard constructor, with scope and layer sub-name.
     */
    shuffle(const tfrt::scope& sc, const std::string& lname="Shuffle") :
        layer(sc, lname),
        m_use_default(3, true),
        m_first_perm{{0, 1, 2, 3, 4, 5, 6, 7}},
        m_second_perm{{0, 1, 2, 3, 4, 5, 6, 7}},
        m_reshape{nvinfer1::DimsCHW{0, 0, 0}}
    {
    }
    /** Add the layer to network graph, using operator(root).
     */
    nvinfer1::ITensor* operator()(nvinfer1::ITensor* net) {
        LOG(INFO) << "LAYER shuffle '" << this->m_scope.name() << "'.";
        auto layer = this->m_scope.network()->addShuffle(*net);
        CHECK_NOTNULL(layer);
        // Transpose + reshape.
        if (!m_use_default[0]) {
            layer->setFirstTranspose(m_first_perm);
        }
        if (!m_use_default[1]) {
            layer->setReshapeDimensions(m_reshape);
        }
        if (!m_use_default[2]) {
            layer->setSecondTranspose(m_second_perm);
        }
        // Layer options.
        layer->setName(this->m_scope.name().c_str());
        net = layer->getOutput(0);
        return this->mark_output(net);
    }

public:
    /** Named parameter: permutation and reshape.
     */
    shuffle& first(nvinfer1::Permutation first_perm) {
        m_use_default[0] = false;
        m_first_perm = first_perm;
        return *this;
    }
    nvinfer1::Permutation first() const {
        return m_first_perm;
    }
    shuffle& reshape(nvinfer1::Dims reshape) {
        m_use_default[1] = false;
        m_reshape = reshape;
        return *this;
    }
    nvinfer1::Dims reshape() const {
        return m_reshape;
    }
    shuffle& second(nvinfer1::Permutation second_perm) {
        m_use_default[2] = false;
        m_second_perm = second_perm;
        return *this;
    }
    nvinfer1::Permutation second() const {
        return m_second_perm;
    }

protected:
    /** Use default parameters. */
    std::vector<bool>  m_use_default;
    /** First permutation. */
    nvinfer1::Permutation  m_first_perm;
    /** Second permutation. */
    nvinfer1::Permutation  m_second_perm;
    /** Reshape. */
    nvinfer1::Dims  m_reshape;
};

/* ============================================================================
 * DEFAULT layers name.
 * ========================================================================== */
typedef add sum;
typedef multiply mul;

}

#endif

