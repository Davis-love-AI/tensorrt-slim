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
#ifndef TFRT_LAYERS_ABSTRACT_H
#define TFRT_LAYERS_ABSTRACT_H

#include <vector>
#include <algorithm>

#include "../utils.h"
#include "../scope.h"

namespace tfrt
{
/* ============================================================================
 * GENERAL enums...
 * ========================================================================== */
/** TensorFlow padding types: SAME or VALID.
 */
enum class PaddingType : int
{
    SAME = 0,           //!< TF SAME padding.
    VALID = 1,          //!< TF VALID padding.
    CUSTOM = 2          //!< Give custom values.
};
/** Type of activation for a layer: None, ReLU, Sigmoid and Tanh.
 * Should be compatible with TensorRT enum.
 */
enum class ActivationType : int
{
	NONE = -1,			//!< no activation
	RELU = 0,			//!< rectified linear activation
	SIGMOID = 1,		//!< sigmoid activation
	TANH = 2,			//!< TanH activation
    SOFTMAX = 3         //!< Sotfmax activation
};
inline std::string ActivationName(ActivationType type) {
    if(type == ActivationType::NONE) {  return "None";  }
    // Common names for the activation layers...
    static const std::vector<std::string> names = {
        "ReLU", "Sigmoid", "Tanh", "Softmax"
    };
    return names[int(type)];
}

/* ============================================================================
 * ABSTRACT layers definitions.
 * ========================================================================== */
/** Generic layer class, with a scope associated.
 */
class layer
{
public:
    layer(const tfrt::scope& sc, const std::string& lname="Layer") :
            m_scope(sc.sub(lname)), m_is_output{false} {
        // TODO: ugly initialization...
        m_is_output = this->net_is_output();
    }
    tfrt::scope scope() const {
        return m_scope;
    }
    /** Layer construction on some input vector.
     */
    virtual nvinfer1::ITensor* operator()(nvinfer1::ITensor*) = 0;

public:
    /** Named parameter: is it output layer?
     */
    layer& is_output(bool is_output) {
        m_is_output = is_output;
        return *this;
    }
    bool is_output() const {
        return m_is_output;
    }

protected:
    /** Check from network parameters if it is an output.
     * Return: whether the layer is an output layer or not.
     */
    bool net_is_output() {
        // TODO: a bit ugly this stuff. Move to scope class?
        auto onames = m_scope.tfrt_network()->outputs_name(true, false);
        auto lname = m_scope.name();
        bool r = (std::find(std::begin(onames), std::end(onames), lname) != std::end(onames));
        return r;
    }
    /** Mark a tensor as an output if the layer is an output layer.
     */
    nvinfer1::ITensor* mark_output(nvinfer1::ITensor* tensor, std::string suffix="output") {
        tensor->setName(this->m_scope.sub(suffix).cname());
        if(m_is_output) {
            LOG(INFO) << "MARK output (layer) on tensor: " << tensor->getName();
            m_scope.network()->markOutput(*tensor);
        }
        return tensor;
    }

protected:
    // Variable scope used by the layer.
    tfrt::scope  m_scope;
    // Is it an output?
    bool  m_is_output;
};


/** Generic 2D operation, with the following common structure:
 * op2d (with padding) -> batch norm/bias -> activation.
 */
template <ActivationType ACT, PaddingType PAD, bool BN>
class operation2d : public layer
{
public:
    /**Standard constructor, with scope and layer sub-name.
     * Initialize kernel size and stride to {1, 1} values.
     */
    operation2d(const tfrt::scope& sc, const std::string& lname) :
        layer(sc, lname), m_noutputs{0},
        m_ksize{1, 1},
        m_stride{1, 1},
        m_padding{0, 0},
        m_dilation{1, 1}
    {
    }
    // Implementation of th named-parameter idiom.
    /** Named parameter: number of outputs.
     */
    operation2d& noutputs(int noutputs) {
        m_noutputs = noutputs;
        return *this;
    }
    /** Named parameter: kernel size.
     */
    operation2d& ksize(nvinfer1::DimsHW ksize) {
        m_ksize = ksize;
        return *this;
    }
    operation2d& ksize(int ksize) {
        m_ksize = nvinfer1::DimsHW({ksize, ksize});
        return *this;
    }
    /** Named parameter: stride.
     */
    operation2d& stride(nvinfer1::DimsHW stride) {
        m_stride = stride;
        return *this;
    }
    operation2d& stride(int stride) {
        m_stride = nvinfer1::DimsHW({stride, stride});
        return *this;
    }
    /** Named parameter: custom padding.
     */
    operation2d& padding(nvinfer1::DimsHW padding) {
        m_padding = padding;
        return *this;
    }
    operation2d& padding(int padding) {
        m_padding = nvinfer1::DimsHW({padding, padding});
        return *this;
    }
    /** Named parameter: dilation.
     */
    operation2d& dilation(nvinfer1::DimsHW dilation) {
        m_dilation = dilation;
        return *this;
    }
    operation2d& dilation(int dilation) {
        m_dilation = nvinfer1::DimsHW({dilation, dilation});
        return *this;
    }

public:
    // Standard getters.
    bool hasActivation() const {
        return (ACT != ActivationType::NONE);
    }
    bool hasBatchNorm() const {
        return BN;
    }

    ActivationType activationType() const {
        return ACT;
    }
    PaddingType paddingType() const {
        return PAD;
    }
    int noutputs() const {
        return m_noutputs;
    }
    nvinfer1::DimsHW ksize() const {
        return m_ksize;
    }
    nvinfer1::DimsHW stride() const {
        return m_stride;
    }
    nvinfer1::DimsHW dilation() const {
        return m_dilation;
    }

    /** Compute real padding values depending on the padding type parameter.
     * Not exactly equivalent to TF padding because of symmetry.
     */
    nvinfer1::DimsHW padding(const nvinfer1::DimsCHW& inshape) const {
        if(PAD == PaddingType::SAME) {
            // auto ksize = this->ksize();
            // auto stride = this->stride();
            // // Try to follow TF convention, depending on input shape.
            // int pad_along_height, pad_along_width;
            // if (inshape.h() % stride.h() == 0) {
            //     pad_along_height = std::max(ksize.h() - stride.h(), 0);
            // }
            // else {
            //     pad_along_height = std::max(ksize.h() - (inshape.h() % stride.h()), 0);
            // }
            // if (inshape.w() % stride.w() == 0) {
            //     pad_along_width = std::max(ksize.w() - stride.w(), 0);
            // }
            // else {
            //     pad_along_width = std::max(ksize.w() - (inshape.w() % stride.w()), 0);
            // }
            // // Use the left padding as general value, and adapt the output size.
            // return nvinfer1::DimsHW{pad_along_height / 2, pad_along_width / 2};
            // Rough estimate...
            int ph = m_ksize.h() / 2;
            ph *= m_dilation.h();
            int pw = m_ksize.w() / 2;
            pw *= m_dilation.w();
            return nvinfer1::DimsHW{ph, pw};
        }
        else if(PAD == PaddingType::VALID) {
            // VALID: no padding required.
            return nvinfer1::DimsHW{0, 0};
        }
        else if(PAD == PaddingType::CUSTOM) {
            return m_padding;
        }
    }

    virtual nvinfer1::ITensor* operator()(nvinfer1::ITensor* input) = 0;

protected:
    /** Set up a batch normalization operation.
     */
    nvinfer1::ITensor* batch_norm(nvinfer1::ITensor* input) {
        if(BN) {
            auto inshape = static_cast<nvinfer1::DimsCHW&&>(input->getDimensions());
            auto bnshape = this->bn_weights_shape(inshape);
            LOG(INFO) << "OP Batch Norm. Input shape: " << dims_str(inshape);
            // TODO: transform moving mean and variance in export...
            nvinfer1::IScaleLayer* bnlayer = nullptr;
            tfrt::scope bnsc = this->m_scope.sub("BatchNorm");
            // Get the weights.
            auto mean = bnsc.weights("moving_mean", bnshape);
            auto variance = bnsc.weights("moving_variance", bnshape);
            auto beta = bnsc.weights("beta", bnshape);
            auto gamma = bnsc.weights("gamma", bnshape);
            nvinfer1::Weights power{mean.type, nullptr, 0};

            // Add two scale layers...
            bnlayer = this->m_scope.network()->addScale(
                *input, nvinfer1::ScaleMode::kCHANNEL, mean, variance, power);
            CHECK_NOTNULL(bnlayer);
            bnlayer->setName(bnsc.subname("normalize").c_str());
            // Second layer: shift + scale.
            input = bnlayer->getOutput(0);
            bnlayer = this->m_scope.network()->addScale(
                *input, nvinfer1::ScaleMode::kCHANNEL, beta, gamma, power);
            CHECK_NOTNULL(bnlayer);
            bnlayer->setName(bnsc.subname("scale").c_str());
            return bnlayer->getOutput(0);
        }
        return input;
    }
    /** Set up an activation operation.
     */
    nvinfer1::ITensor* activation(nvinfer1::ITensor* input, std::string suffix="") {
        LOG(INFO) << "OP activation. Type: " << ActivationName(ACT)
                << " Input shape: " << dims_str(input->getDimensions());
        std::string aname = ActivationName(ACT);
        aname += suffix;
        if(ACT != ActivationType::NONE && ACT != ActivationType::SOFTMAX) {
            nvinfer1::ActivationType kAct{nvinfer1::ActivationType(ACT)};
            // Generic activation layer...
            auto actlayer = this->m_scope.network()->addActivation(*input, kAct);
            CHECK_NOTNULL(actlayer);
            actlayer->setName(this->m_scope.subname(aname).c_str());
            return actlayer->getOutput(0);
        }
        else if(ACT == ActivationType::SOFTMAX) {
            // Specific layer for Sotfmax activation. Why???
            auto actlayer = this->m_scope.network()->addSoftMax(*input);
            CHECK_NOTNULL(actlayer);
            actlayer->setName(this->m_scope.subname(aname).c_str());
            return actlayer->getOutput(0);
        }
        return input;
    }

protected:
    /** Get batch norm weights shape. */
    nvinfer1::Dims bn_weights_shape(const nvinfer1::DimsCHW& inshape)
    {
        return nvinfer1::DimsC{m_noutputs};
    }
protected:
    // Number of outputs.
    int  m_noutputs;
    // Kernel size of the operation.
    nvinfer1::DimsHW    m_ksize;
    // Striding of the operation.
    nvinfer1::DimsHW    m_stride;
    // Custom padding values.
    nvinfer1::DimsHW    m_padding;
    /** Dilation? */
    nvinfer1::DimsHW  m_dilation;
};

/** Input layer.
 */
class input : public layer
{
public:
    /** Constructor, initialize input parameters (shape, name, ...) from
     * the scope tfrt::network object.
     */
    input(const tfrt::scope& sc) :
        layer(sc, sc.tfrt_network()->input_name(false)),
        m_shape{sc.tfrt_network()->input_shape()} {}
    /** Named parameter: input shape. */
    input& shape(nvinfer1::DimsCHW shape) {
        m_shape = shape;
        return *this;
    }
    /** Input construction */
    virtual nvinfer1::ITensor* operator()() {
        auto dt = m_scope.tfrt_network()->datatype();
        dt = nvinfer1::DataType::kFLOAT;
        // TensorRT input.
        nvinfer1::ITensor* input = m_scope.network()->addInput(
            m_scope.name().c_str(), dt, DIMRT(this->m_shape));
        LOG(INFO) << "LAYER input '" << m_scope.name() << "'. "
            << "Shape: " << dims_str(input->getDimensions());
        // Input scaling.
        input = this->scale(input);
        // return this->mark_output(input);
        return input;
    }
protected:
    /** Scaling input tensor.
     */
    nvinfer1::ITensor* scale(nvinfer1::ITensor* input) {
        auto inshape = static_cast<nvinfer1::DimsCHW&&>(input->getDimensions());
        auto wshape = this->weights_shape(inshape);
        // Get the scaling weights.
        nvinfer1::Weights shift = m_scope.weights("shift", wshape);
        nvinfer1::Weights scale = m_scope.weights("scale", wshape);
        if(shift.values || scale.values) {
            LOG(INFO) << "OP input pre-scaling (shift + scale).";
            nvinfer1::Weights power{shift.type, nullptr, 0};
            auto layer = this->m_scope.network()->addScale(
                *input, nvinfer1::ScaleMode::kUNIFORM, shift, scale, power);
            CHECK_NOTNULL(layer);
            layer->setName(m_scope.subname("scaling").c_str());
            input = layer->getOutput(0);
        }
        return input;
    }
    // Should not be used!
    nvinfer1::ITensor* operator()(nvinfer1::ITensor*) { return nullptr; }
    /** Get weights shape. */
    nvinfer1::Dims weights_shape(const nvinfer1::DimsCHW& inshape)
    {
        return nvinfer1::DimsC{1};
    }

protected:
    // Input shape.
    nvinfer1::DimsCHW  m_shape;
};

}

#endif

