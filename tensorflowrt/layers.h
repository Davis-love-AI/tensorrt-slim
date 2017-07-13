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
    nvinfer1::ITensor* mark_output(nvinfer1::ITensor* tensor) {
        tensor->setName(this->m_scope.sub("output").cname());
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
        // Get the scaling weights.
        nvinfer1::Weights shift = m_scope.weights("shift");
        nvinfer1::Weights scale = m_scope.weights("scale");
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

protected:
    // Input shape.
    nvinfer1::DimsCHW  m_shape;
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
            LOG(INFO) << "OP Batch Norm. "
                << "Input shape: " << dims_str(input->getDimensions());
            // TODO: transform moving mean and variance in export...
            nvinfer1::IScaleLayer* bnlayer = nullptr;
            tfrt::scope bnsc = this->m_scope.sub("BatchNorm");
            // Get the weights.
            auto mean = bnsc.weights("moving_mean");
            auto variance = bnsc.weights("moving_variance");
            auto beta = bnsc.weights("beta");
            auto gamma = bnsc.weights("gamma");
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
        LOG(INFO) << "OP 2D convolution. "
            << "Input shape: " << dims_str(input->getDimensions())
            << ". PARAMETERS: "
            << "ksize: " << dims_str(this->ksize()) << " | "
            << "noutputs: " << this->noutputs() << " | "
            << "ngroups: " << ngroups << " | "
            << "stride: " << dims_str(this->stride()) << " | "
            << "padding: " << dims_str(this->padding());
        nvinfer1::IConvolutionLayer* convlayer = nullptr;
        // Batch normalization: no bias.
        if(BN) {
            auto weights = this->m_scope.weights(wname);
            nvinfer1::Weights biases{weights.type, nullptr, 0};
            convlayer = this->m_scope.network()->addConvolution(
                *input, this->noutputs(), DIMRT(this->ksize()), weights, biases);
        }
        // Normal convolution with bias.
        else {
            auto weights = this->m_scope.weights(wname);
            auto biases = this->m_scope.weights(bname);
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
        LOG(INFO) << "LAYER 2D contrib separable convolution '" << this->m_scope.name() << "'. "
            << "Input shape: " << dims_str(net->getDimensions());
        // Number of groups: input channel size.
        int ngroups = dims_channels(net->getDimensions());
        // Depthwise convolution, with depth multiplier.
        separable_convolution2d dw_conv2d(*this);
        dw_conv2d.noutputs(ngroups * m_depth_multiplier);
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

typedef max_pooling2d<PaddingType::SAME>  max_pool2d;
typedef avg_pooling2d<PaddingType::SAME>  avg_pool2d;

typedef activation<ActivationType::RELU>    relu;
typedef activation<ActivationType::SIGMOID> sigmoid;
typedef activation<ActivationType::TANH>    tanh;
typedef activation<ActivationType::SOFTMAX> softmax;

}

#endif
