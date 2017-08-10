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
#ifndef TFRT_LAYERS_H
#define TFRT_LAYERS_H

#include <vector>
#include <algorithm>

#include "utils.h"
#include "scope.h"

namespace tfrt
{

/* ============================================================================
 * Layers classes.
 * ========================================================================== */
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
/** TensorFlow padding types: SAME or VALID.
 */
enum class PaddingType : int
{
    SAME = 0,           //!< TF SAME padding.
    VALID = 1,          //!< TF VALID padding.
    CUSTOM = 2          //!< Give custom values.
};

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
        return input;
    }
protected:
    /** Scaling input tensor.
     */
    nvinfer1::ITensor* scale(nvinfer1::ITensor* input) {
        auto inshape = static_cast<nvinfer1::DimsNCHW&&>(input->getDimensions());
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
    nvinfer1::Dims weights_shape(const nvinfer1::DimsNCHW& inshape)
    {
        return nvinfer1::DimsC{1};
    }

protected:
    // Input shape.
    nvinfer1::DimsCHW  m_shape;
};

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
        auto inshape = static_cast<nvinfer1::DimsNCHW&&>(net->getDimensions());
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
    nvinfer1::Dims weights_shape(const nvinfer1::DimsNCHW& inshape)
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
/** Element wise max.
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
        layer(sc, lname), m_noutputs{0}, m_ksize{1, 1}, m_stride{1, 1}, m_padding{0, 0} {
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

    /** Compute real padding values depending on the padding type parameter.
     * Not exactly equivalent to TF padding because of symmetry.
     */
    nvinfer1::DimsHW padding() const {
        if(PAD == PaddingType::SAME) {
            // Rough estimate...
            return nvinfer1::DimsHW{m_ksize.h() / 2, m_ksize.w() / 2};
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
            auto inshape = static_cast<nvinfer1::DimsNCHW&&>(input->getDimensions());
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
    nvinfer1::ITensor* activation(nvinfer1::ITensor* input) {
        LOG(INFO) << "OP activation. Type: " << ActivationName(ACT)
                << " Input shape: " << dims_str(input->getDimensions());
        std::string aname = ActivationName(ACT);
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
    nvinfer1::Dims bn_weights_shape(const nvinfer1::DimsNCHW& inshape)
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
};

/** Classic 2D convolution layer.
 */
template <ActivationType ACT, PaddingType PAD, bool BN>
class convolution2d : public operation2d<ACT, PAD, BN>
{
public:
    /** Constructor: declare the layer.
     */
    convolution2d(const tfrt::scope& sc, const std::string& lname="Conv2d") :
        operation2d<ACT, PAD, BN>(sc, lname) {
    }
    /** Add the layer to network graph, using operator(root).
     * 2D convolution + batch norm + activation.
     */
    virtual nvinfer1::ITensor* operator()(nvinfer1::ITensor* net) {
        LOG(INFO) << "LAYER 2D contrib convolution '" << this->m_scope.name() << "'. "
            << "Input shape: " << dims_str(net->getDimensions());
        net = this->convolution(net);
        net = this->batch_norm(net);
        net = this->activation(net);
        return this->mark_output(net);
    }

protected:
    /** Set up the convolution operation.
     */
    nvinfer1::ITensor* convolution(nvinfer1::ITensor* input,
                                   int ngroups=1,
                                   std::string wname="weights",
                                   std::string bname="biases",
                                   std::string lnamesuffix="") {
        auto inshape = static_cast<nvinfer1::DimsNCHW&&>(input->getDimensions());
        auto wshape = this->weights_shape(inshape);
        auto bshape = this->biases_shape(inshape);
        LOG(INFO) << "OP 2D convolution. "
            << "Input shape: " << dims_str(inshape)
            << ". PARAMETERS: "
            << "ksize: " << dims_str(this->ksize()) << " | "
            << "noutputs: " << this->noutputs() << " | "
            << "ngroups: " << ngroups << " | "
            << "stride: " << dims_str(this->stride()) << " | "
            << "padding: " << dims_str(this->padding());
        nvinfer1::IConvolutionLayer* convlayer = nullptr;
        // Batch normalization: no bias.
        if(BN) {
            auto weights = this->m_scope.weights(wname, wshape);
            nvinfer1::Weights biases{weights.type, nullptr, 0};
            convlayer = this->m_scope.network()->addConvolution(
                *input, this->noutputs(), DIMRT(this->ksize()), weights, biases);
        }
        // Normal convolution with bias.
        else {
            auto weights = this->m_scope.weights(wname, wshape);
            auto biases = this->m_scope.weights(bname, bshape);
            convlayer = this->m_scope.network()->addConvolution(
                *input, this->noutputs(), DIMRT(this->ksize()), weights, biases);
        }
        CHECK_NOTNULL(convlayer);
        // Set name, padding, stride and nb groups.
        convlayer->setName((this->m_scope.name() + lnamesuffix).c_str());
        convlayer->setPadding(DIMRT(this->padding()));
        convlayer->setStride(DIMRT(this->stride()));
        convlayer->setNbGroups(ngroups);
        return convlayer->getOutput(0);
    }
    /** Get convolution weights shape. Note: TensorRT uses the convention GKCRS,
     * where G is the number of groups,
     * K the number of output feature maps,
     * C the number of input channels, and
     * R and S are the height and width of the filter.
     */
    nvinfer1::Dims weights_shape(const nvinfer1::DimsNCHW& inshape)
    {
        auto ksize = this->ksize();
        return nvinfer1::DimsNACHW{1, this->noutputs(), inshape.c(), ksize.h(), ksize.w()};
    }
    /** Bias shape. */
    nvinfer1::Dims biases_shape(const nvinfer1::DimsNCHW& inshape)
    {
        return nvinfer1::DimsC{this->noutputs()};
    }
};
/** Separable 2D convolution layer.
 */
template <ActivationType ACT, PaddingType PAD, bool BN>
class separable_convolution2d : public convolution2d<ACT, PAD, BN>
{
public:
    /** Constructor: declare the layer.
     */
    separable_convolution2d(const tfrt::scope& sc, const std::string& lname="SeparableConv2d") :
        convolution2d<ACT, PAD, BN>(sc, lname), m_depth_multiplier{1} {
    }
    /** Add the layer to network graph, using operator(root).
     * 2D convolution + batch norm + activation.
     */
    virtual nvinfer1::ITensor* operator()(nvinfer1::ITensor* net) {
        auto inshape = static_cast<nvinfer1::DimsNCHW&&>(net->getDimensions());
        LOG(INFO) << "LAYER 2D contrib separable convolution '" << this->m_scope.name() << "'. "
            << "Input shape: " << dims_str(inshape);
        // Number of groups: input channel size.
        int ngroups = dims_channels(inshape);
        // Depthwise convolution, with depth multiplier.
        separable_convolution2d dw_conv2d(*this);
        dw_conv2d.noutputs(ngroups * m_depth_multiplier);
        // TODO: TensorRT bug. Replace group conv. by classic convolution.
        ngroups = 1;
        net = dw_conv2d.convolution(net, ngroups, "depthwise_weights", "depthwise_biases", "_dw");
        // Pointwise convolution.
        separable_convolution2d pw_conv2d(*this);
        pw_conv2d.ksize({1, 1}).stride({1, 1});
        net = pw_conv2d.convolution(net, 1, "pointwise_weights", "biases", "_pw");
        net = pw_conv2d.batch_norm(net);
        net = pw_conv2d.activation(net);
        return this->mark_output(net);
    }
    /** Named parameter: depth multiplier.
     */
    separable_convolution2d& depthmul(int depth_multiplier) {
        m_depth_multiplier = depth_multiplier;
        return *this;
    }
    int depthmul() const {
        return m_depth_multiplier;
    }

protected:
    // Depth multiplier.
    int  m_depth_multiplier;
};
/** Transpose 2D convolution layer.
 */
template <ActivationType ACT, PaddingType PAD, bool BN>
class convolution2d_transpose : public operation2d<ACT, PAD, BN>
{
public:
    /** Constructor: declare the layer.
     */
    convolution2d_transpose(const tfrt::scope& sc, const std::string& lname="Conv2d_transpose") :
        operation2d<ACT, PAD, BN>(sc, lname), m_ceil_mode{false}, m_ceil_formula{} {
    }
    /** Add the layer to network graph, using operator(root).
     * 2D tranpose convolution + batch norm + activation.
     */
    virtual nvinfer1::ITensor* operator()(nvinfer1::ITensor* net) {
        LOG(INFO) << "LAYER 2D contrib transpose convolution '" << this->m_scope.name()
            << ". Input shape: " << dims_str(net->getDimensions());
        net = this->tr_convolution(net);
        net = this->batch_norm(net);
        net = this->activation(net);
        return this->mark_output(net);
    }
    /** Named parameter: ceil mode. */
    convolution2d_transpose& ceil(bool mode) {
        m_ceil_mode = mode;
        return *this;
    }
    bool ceil() const {
        return m_ceil_mode;
    }

public:
    /** Ceil computation method of output size. Basically suppose that
     * the layer is the up-scaling equivalent of convolution downscaled layer
     * where the formula was approximating the output shape. TODO: not very clear!
     */
    class ceil_formula : public nvinfer1::IOutputDimensionsFormula
    {
    public:
        virtual nvinfer1::DimsHW compute(nvinfer1::DimsHW inputDims, nvinfer1::DimsHW ksize,
            nvinfer1::DimsHW stride, nvinfer1::DimsHW padding, const char* layerName)
        {
            // Change a bit the formula...
            nvinfer1::DimsHW odims{
                (inputDims.h() - 1) * stride.h() + ksize.h() - 2 * padding.h() + 1,
                (inputDims.w() - 1) * stride.w() + ksize.w() - 2 * padding.w() + 1
            };
            return odims;
        }
    };

protected:
    /** Set up the convolution operation.
     */
    nvinfer1::ITensor* tr_convolution(nvinfer1::ITensor* input,
                                      std::string wname="weights",
                                      std::string bname="biases",
                                      std::string lnamesuffix="") {
        auto inshape = static_cast<nvinfer1::DimsNCHW&&>(input->getDimensions());
        auto wshape = this->weights_shape(inshape);
        auto bshape = this->biases_shape(inshape);
        LOG(INFO) << "OP 2D transpose convolution. "
            << "Input shape: " << dims_str(inshape)
            << ". PARAMETERS: "
            << "ksize: " << dims_str(this->ksize()) << " | "
            << "noutputs: " << this->noutputs() << " | "
            << "stride: " << dims_str(this->stride()) << " | "
            << "padding: " << dims_str(this->padding());
        nvinfer1::IDeconvolutionLayer* convlayer = nullptr;
        // Output formula used?
        if (m_ceil_mode) {
            this->m_scope.network()->setDeconvolutionOutputDimensionsFormula(&m_ceil_formula);
        }
        else {
            this->m_scope.network()->setDeconvolutionOutputDimensionsFormula(nullptr);
        }
        // Batch normalization: no bias.
        if(BN) {
            auto weights = this->m_scope.weights(wname, wshape);
            nvinfer1::Weights biases{weights.type, nullptr, 0};
            convlayer = this->m_scope.network()->addDeconvolution(
                *input, this->noutputs(), DIMRT(this->ksize()), weights, biases);
        }
        // Normal convolution with bias.
        else {
            auto weights = this->m_scope.weights(wname, wshape);
            auto biases = this->m_scope.weights(bname, bshape);
            convlayer = this->m_scope.network()->addDeconvolution(
                *input, this->noutputs(), DIMRT(this->ksize()), weights, biases);
        }
        // this->m_scope.network()->setDeconvolutionOutputDimensionsFormula(nullptr);
        CHECK_NOTNULL(convlayer);
        // Set name, padding, stride and nb groups.
        convlayer->setName((this->m_scope.name() + lnamesuffix).c_str());
        convlayer->setPadding(DIMRT(this->padding()));
        convlayer->setStride(DIMRT(this->stride()));
        return convlayer->getOutput(0);
    }

    nvinfer1::Dims weights_shape(const nvinfer1::DimsNCHW& inshape)
    {
        auto ksize = this->ksize();
        return nvinfer1::DimsNACHW{1, this->noutputs(), inshape.c(), ksize.h(), ksize.w()};
    }
    /** Bias shape. */
    nvinfer1::Dims biases_shape(const nvinfer1::DimsNCHW& inshape)
    {
        return nvinfer1::DimsC{this->noutputs()};
    }
private:
    // Ceil mode for computing the deconvolution formula.
    bool  m_ceil_mode;
    // Output formula object in ceil mode.
    ceil_formula  m_ceil_formula;
};

/** Activation layer.
 */
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
        nvinfer1::IPoolingLayer* poollayer = nullptr;
        poollayer = this->m_scope.network()->addPooling(
            *input, POOL, DIMRT(this->ksize()));
        CHECK_NOTNULL(poollayer);
        // Set name, padding and stride.
        poollayer->setName(this->m_scope.cname());
        poollayer->setPadding(DIMRT(this->padding()));
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

/** Default layer configurations: ReLU activation, SAME padding and
 * no batch normalization.
 */
typedef convolution2d<ActivationType::RELU, PaddingType::SAME, false>           conv2d;
typedef separable_convolution2d<ActivationType::RELU, PaddingType::SAME, false> separable_conv2d;
typedef convolution2d_transpose<ActivationType::RELU, PaddingType::SAME, false>           conv2d_transpose;

typedef max_pooling2d<PaddingType::SAME>  max_pool2d;
typedef avg_pooling2d<PaddingType::SAME>  avg_pool2d;

typedef activation<ActivationType::RELU>    relu;
typedef activation<ActivationType::SIGMOID> sigmoid;
typedef activation<ActivationType::TANH>    tanh;
typedef activation<ActivationType::SOFTMAX> softmax;

}

#endif
