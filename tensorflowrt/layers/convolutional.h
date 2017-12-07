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
#ifndef TFRT_LAYERS_CONVOLUTIONAL_H
#define TFRT_LAYERS_CONVOLUTIONAL_H

#include "abstract.h"

namespace tfrt
{
/** TensorFlow output formula.  */
class tf_conv2d_formula : public nvinfer1::IOutputDimensionsFormula
{
public:
    virtual nvinfer1::DimsHW compute(
        nvinfer1::DimsHW inshape, nvinfer1::DimsHW ksize,
        nvinfer1::DimsHW stride, nvinfer1::DimsHW padding,
        #if NV_TENSORRT_MAJOR == 3
        nvinfer1::DimsHW dilation,
        #endif
        const char* layerName)
    {
        // if (padding.h() == 0 && padding.w() == 0) {
        //     // Zero padding, assume it is VALID TF padding.
        //     nvinfer1::DimsHW odims{
        //         int(std::ceil(float(inshape.h() - ksize.h() + 1) / float(stride.h()))),
        //         int(std::ceil(float(inshape.w() - ksize.w() + 1) / float(stride.w())))
        //     };
        //     return odims;
        // }
        // else {
            // SAME TF padding?
            nvinfer1::DimsHW odims{
                int(std::ceil(float(inshape.h()) / float(stride.h()))),
                int(std::ceil(float(inshape.w()) / float(stride.w())))
            };
            return odims;
        // }
    }
    // TensorRT 1 compatibility.
    #ifndef NV_TENSORRT_MAJOR
    virtual nvinfer1::Dims2 compute(
        nvinfer1::Dims2 inshape, nvinfer1::Dims2 ksize,
        nvinfer1::Dims2 stride, nvinfer1::Dims2 padding, const char* layerName)
    {
        // SAME TF padding?
        nvinfer1::Dims2 odims{
            int(std::ceil(float(inshape.h / float(stride.h)))),
            int(std::ceil(float(inshape.w / float(stride.w))))
        };
        return odims;
    }
    #endif
};
/** TensorFlow output formula. Do you floor the output dimensions or not?
 */
class tf_conv2d_transpose_formula : public nvinfer1::IOutputDimensionsFormula
{
public:
    tf_conv2d_transpose_formula(int cut_val=0) : m_cutshape{cut_val, cut_val} {}
    tf_conv2d_transpose_formula(nvinfer1::DimsHW cutshape) : m_cutshape(cutshape) {}
    virtual nvinfer1::DimsHW compute(
        nvinfer1::DimsHW inshape, nvinfer1::DimsHW ksize,
        nvinfer1::DimsHW stride, nvinfer1::DimsHW padding,
        #if NV_TENSORRT_MAJOR == 3
        nvinfer1::DimsHW dilation,
        #endif
        const char* layerName)
    {
        // Change a bit the formula...
        nvinfer1::DimsHW odims{
            (inshape.h() - 1) * stride.h() + ksize.h() - 2 * padding.h() - m_cutshape.h(),
            (inshape.w() - 1) * stride.w() + ksize.w() - 2 * padding.w() - m_cutshape.w()
        };
        return odims;
    }
    // TensorRT 1 compatibility.
    #ifndef NV_TENSORRT_MAJOR
    virtual nvinfer1::Dims2 compute(
        nvinfer1::Dims2 inshape, nvinfer1::Dims2 ksize,
        nvinfer1::Dims2 stride, nvinfer1::Dims2 padding, const char* layerName)
    {
        // Change a bit the formula...
        nvinfer1::Dims2 odims{
            (inshape.h - 1) * stride.h + ksize.h - 2 * padding.h - m_cutshape.h(),
            (inshape.w - 1) * stride.w + ksize.w - 2 * padding.w - m_cutshape.w()
        };
        return odims;
    }
    #endif
private:
    nvinfer1::DimsHW  m_cutshape;
};


/* ============================================================================
 * CONVOLUTION layers definitions.
 * ========================================================================== */
/** Classic 2D convolution layer.
 */
template <ActivationType ACT, PaddingType PAD, bool BN>
class convolution2d : public operation2d<ACT, PAD, BN>
{
public:
    /** Constructor: declare the layer.
     */
    convolution2d(const tfrt::scope& sc, const std::string& lname="Conv2d") :
        operation2d<ACT, PAD, BN>(sc, lname)
    {
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
        auto inshape = static_cast<nvinfer1::DimsCHW&&>(input->getDimensions());
        auto wshape = this->weights_shape(inshape, ngroups);
        auto bshape = this->biases_shape(inshape);
        LOG(INFO) << "OP 2D convolution. "
            << "Input shape: " << dims_str(inshape)
            << ". PARAMETERS: "
            << "ksize: " << dims_str(this->ksize()) << " | "
            << "noutputs: " << this->noutputs() << " | "
            << "ngroups: " << ngroups << " | "
            << "stride: " << dims_str(this->stride()) << " | "
            << "dilation: " << dims_str(this->dilation()) << " | "
            << "padding: " << dims_str(this->padding(inshape));
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
        convlayer->setPadding(DIMRT(this->padding(inshape)));
        convlayer->setStride(DIMRT(this->stride()));
        convlayer->setDilation(DIMRT(this->dilation()));
        convlayer->setNbGroups(ngroups);
        return convlayer->getOutput(0);
    }
    /** Get convolution weights shape. Note: TensorRT uses the convention GKCRS,
     * where G is the number of groups,
     * K the number of output feature maps,
     * C the number of input channels, and
     * R and S are the height and width of the filter.
     */
    nvinfer1::Dims weights_shape(const nvinfer1::DimsCHW& inshape, int ngroups)
    {
        auto ksize = this->ksize();
        return nvinfer1::DimsNACHW{ngroups, this->noutputs() / ngroups, inshape.c() / ngroups, ksize.h(), ksize.w()};
    }
    /** Bias shape. */
    nvinfer1::Dims biases_shape(const nvinfer1::DimsCHW& inshape)
    {
        return nvinfer1::DimsC{this->noutputs()};
    }
};

/** Grouped 2D convolution layer.
 */
template <ActivationType ACT, PaddingType PAD, bool BN>
class convolution2d_grouped : public convolution2d<ACT, PAD, BN>
{
public:
    /** Constructor: declare the layer.
     */
    convolution2d_grouped(const tfrt::scope& sc, const std::string& lname="Conv2dGrouped") :
        convolution2d<ACT, PAD, BN>(sc, lname), m_num_groups{1} {
    }
    /** Add the layer to network graph, using operator(root).
     * 2D convolution + batch norm + activation.
     */
    virtual nvinfer1::ITensor* operator()(nvinfer1::ITensor* net) {
        auto inshape = static_cast<nvinfer1::DimsCHW&&>(net->getDimensions());
        LOG(INFO) << "LAYER 2D contrib grouped convolution '" << this->m_scope.name() << "'. "
            << "Input shape: " << dims_str(inshape);

        // Number of groups in convolution.
        net = this->convolution(net, m_num_groups);
        net = this->batch_norm(net);
        net = this->activation(net);
        return this->mark_output(net);
    }
    /** Named parameter: depth multiplier.
     */
    convolution2d_grouped& ngroups(int ngroups) {
        m_num_groups = ngroups;
        return *this;
    }
    int ngroups() const {
        return m_num_groups;
    }

protected:
    // Number of groups
    int  m_num_groups;
};


/** Separable 2D convolution layer.
 */
template <ActivationType ACT, PaddingType PAD, bool BN, bool IACT=false>
class separable_convolution2d : public convolution2d<ACT, PAD, BN>
{
public:
    /** Constructor: declare the layer.
     */
    separable_convolution2d(
        const tfrt::scope& sc, const std::string& lname="SeparableConv2d") :
        convolution2d<ACT, PAD, BN>(sc, lname),
        m_depth_multiplier{1}, m_dw_group_size{1}, m_pw_ngroups{1} {
    }
    /** Add the layer to network graph, using operator(root).
     * 2D convolution + batch norm + activation.
     */
    virtual nvinfer1::ITensor* operator()(nvinfer1::ITensor* net) {
        auto inshape = static_cast<nvinfer1::DimsCHW&&>(net->getDimensions());
        LOG(INFO) << "LAYER 2D contrib separable convolution '" << this->m_scope.name() << "'. "
            << "Input shape: " << dims_str(inshape);
        // Depthwise convolution, with depth multiplier.
        int in_channels = dims_channels(inshape);
        separable_convolution2d dw_conv2d(*this);
        dw_conv2d.noutputs(in_channels * m_depth_multiplier);
        // Number of groups estimate.
        int ngroups = std::ceil(float(in_channels) / float(m_dw_group_size));
        // ngroups = 1;
        // TensorRT bug. Best group size???
        net = dw_conv2d.convolution(net, ngroups, "depthwise_weights", "depthwise_biases", "_dw");
        // net = dw_conv2d.batch_norm(net);
        // Intermediate activation?
        if (IACT) {
            net = dw_conv2d.activation(net, "_inter");
        }
        // Pointwise convolution.
        separable_convolution2d pw_conv2d(*this);
        pw_conv2d.dilation({1, 1}).ksize({1, 1}).stride({1, 1});
        net = pw_conv2d.convolution(net, m_pw_ngroups, "pointwise_weights", "biases", "_pw");
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
    /** Named parameter: depthwise group size.
     */
    separable_convolution2d& dw_group_size(int group_size) {
        m_dw_group_size = group_size;
        return *this;
    }
    int dw_group_size() const {
        return m_dw_group_size;
    }
    /** Named parameter: pointwise number of groups.
     */
    separable_convolution2d& pw_ngroups(int ngroups) {
        m_pw_ngroups = ngroups;
        return *this;
    }
    int pw_ngroups() const {
        return m_pw_ngroups;
    }
protected:
    // Depth multiplier.
    int  m_depth_multiplier;
    // Depthwise group size.
    int  m_dw_group_size;
    // Pointwise number of groups.
    int  m_pw_ngroups;
};

/** Separable 2D convolution layer.
 */
template <ActivationType ACT, PaddingType PAD, bool BN>
class depthwise_convolution2d : public convolution2d<ACT, PAD, BN>
{
public:
    /** Constructor: declare the layer.
     */
    depthwise_convolution2d(
        const tfrt::scope& sc, const std::string& lname="DepthwiseConv2d") :
        convolution2d<ACT, PAD, BN>(sc, lname),
        m_depth_multiplier{1}, m_group_size{1} {
    }
    /** Add the layer to network graph, using operator(root).
     * 2D convolution + batch norm + activation.
     */
    virtual nvinfer1::ITensor* operator()(nvinfer1::ITensor* net) {
        auto inshape = static_cast<nvinfer1::DimsCHW&&>(net->getDimensions());
        LOG(INFO) << "LAYER 2D contrib depthwise convolution '" << this->m_scope.name() << "'. "
            << "Input shape: " << dims_str(inshape);
        // Depthwise convolution, with depth multiplier.
        int in_channels = dims_channels(inshape);
        depthwise_convolution2d dw_conv2d(*this);
        dw_conv2d.noutputs(in_channels * m_depth_multiplier);
        // Number of groups estimate.
        int ngroups = std::ceil(float(in_channels) / float(m_group_size));
        // ngroups = 1;
        // TensorRT bug. Best group size???
        net = dw_conv2d.convolution(net, ngroups, "depthwise_weights", "depthwise_biases", "_dw");
        net = dw_conv2d.batch_norm(net);
        net = dw_conv2d.activation(net);
        return this->mark_output(net);
    }
    /** Named parameter: depth multiplier.
     */
    depthwise_convolution2d& depthmul(int depth_multiplier) {
        m_depth_multiplier = depth_multiplier;
        return *this;
    }
    int depthmul() const {
        return m_depth_multiplier;
    }
    /** Named parameter: group size.
     */
    depthwise_convolution2d& group_size(int group_size) {
        m_group_size = group_size;
        return *this;
    }
    int group_size() const {
        return m_group_size;
    }

protected:
    // Depth multiplier.
    int  m_depth_multiplier;
    // Group size.
    int  m_group_size;
};


/** Separable 2D convolution layer. TESTTNG
 */
template <ActivationType ACT, PaddingType PAD, bool BN>
class separable_convolution2d_test : public convolution2d<ACT, PAD, BN>
{
public:
    /** Constructor: declare the layer.
     */
    separable_convolution2d_test(
        const tfrt::scope& sc, const std::string& lname="SeparableConv2d") :
        convolution2d<ACT, PAD, BN>(sc, lname), m_depth_multiplier{1} {
    }
    /** Add the layer to network graph, using operator(root).
     * 2D convolution + batch norm + activation.
     */
    virtual nvinfer1::ITensor* operator()(nvinfer1::ITensor* net) {
        auto inshape = static_cast<nvinfer1::DimsCHW&&>(net->getDimensions());
        LOG(INFO) << "LAYER 2D contrib separable convolution '" << this->m_scope.name() << "'. "
            << "Input shape: " << dims_str(inshape);
        // Number of groups: input channel size.
        int ngroups = dims_channels(inshape);
        // Depthwise convolution, with depth multiplier.
        separable_convolution2d_test dw_conv2d(*this);
        dw_conv2d.noutputs(ngroups * m_depth_multiplier);
        // TODO: TensorRT bug. Replace group conv. by classic convolution.
        ngroups = 1;
        net = dw_conv2d.convolution(net, ngroups, "depthwise_weights", "depthwise_biases", "_dw");
        // Pointwise convolution.
        separable_convolution2d_test pw_conv2d(*this);
        pw_conv2d.ksize({1, 1}).stride({1, 1});
        net = pw_conv2d.convolution(net, 1, "pointwise_weights", "biases", "_pw");
        net = pw_conv2d.batch_norm(net);
        net = pw_conv2d.activation(net);
        return this->mark_output(net);
    }
    /** Named parameter: depth multiplier.
     */
    separable_convolution2d_test& depthmul(int depth_multiplier) {
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
        virtual nvinfer1::DimsHW compute(
            nvinfer1::DimsHW inputDims, nvinfer1::DimsHW ksize,
            nvinfer1::DimsHW stride, nvinfer1::DimsHW padding,
            #if NV_TENSORRT_MAJOR == 3
            nvinfer1::DimsHW dilation,
            #endif
            const char* layerName)
        {
            // Change a bit the formula...
            nvinfer1::DimsHW odims{
                (inputDims.h() - 1) * stride.h() + ksize.h() - 2 * padding.h() + 1,
                (inputDims.w() - 1) * stride.w() + ksize.w() - 2 * padding.w() + 1
            };
            return odims;
        }
        // TensorRT 1 compatibility.
        #ifndef NV_TENSORRT_MAJOR
        virtual nvinfer1::Dims2 compute(
            nvinfer1::Dims2 inshape, nvinfer1::Dims2 ksize,
            nvinfer1::Dims2 stride, nvinfer1::Dims2 padding, const char* layerName)
        {
            // Change a bit the formula...
            nvinfer1::Dims2 odims{
                (inshape.h - 1) * stride.h + ksize.h - 2 * padding.h + 1 ,
                (inshape.w - 1) * stride.w + ksize.w - 2 * padding.w + 1
            };
            return odims;
        }
        #endif
    };

protected:
    /** Set up the convolution operation.
     */
    nvinfer1::ITensor* tr_convolution(nvinfer1::ITensor* input,
                                      std::string wname="weights",
                                      std::string bname="biases",
                                      std::string lnamesuffix="") {
        auto inshape = static_cast<nvinfer1::DimsCHW&&>(input->getDimensions());
        auto wshape = this->weights_shape(inshape);
        auto bshape = this->biases_shape(inshape);
        LOG(INFO) << "OP 2D transpose convolution. "
            << "Input shape: " << dims_str(inshape)
            << ". PARAMETERS: "
            << "ksize: " << dims_str(this->ksize()) << " | "
            << "noutputs: " << this->noutputs() << " | "
            << "stride: " << dims_str(this->stride()) << " | "
            << "padding: " << dims_str(this->padding(inshape));
        nvinfer1::IDeconvolutionLayer* convlayer = nullptr;
        // Output formula used?
        // if (m_ceil_mode) {
        //     this->m_scope.network()->setDeconvolutionOutputDimensionsFormula(&m_ceil_formula);
        // }
        // else {
        //     this->m_scope.network()->setDeconvolutionOutputDimensionsFormula(nullptr);
        // }
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
        convlayer->setPadding(DIMRT(this->padding(inshape)));
        convlayer->setStride(DIMRT(this->stride()));
        return convlayer->getOutput(0);
    }

    nvinfer1::Dims weights_shape(const nvinfer1::DimsCHW& inshape)
    {
        auto ksize = this->ksize();
        return nvinfer1::DimsNACHW{1, this->noutputs(), inshape.c(), ksize.h(), ksize.w()};
    }
    /** Bias shape. */
    nvinfer1::Dims biases_shape(const nvinfer1::DimsCHW& inshape)
    {
        return nvinfer1::DimsC{this->noutputs()};
    }
private:
    // Ceil mode for computing the deconvolution formula.
    bool  m_ceil_mode;
    // Output formula object in ceil mode.
    ceil_formula  m_ceil_formula;
};


/* ============================================================================
 * DEFAULT layers name.
 * ========================================================================== */
/** Default layer configurations: ReLU activation, SAME padding and
 * no batch normalization.
 */
typedef convolution2d<ActivationType::RELU, PaddingType::SAME, false>           conv2d;
typedef separable_convolution2d<ActivationType::RELU, PaddingType::SAME, false> separable_conv2d;
typedef convolution2d_transpose<ActivationType::RELU, PaddingType::SAME, false>           conv2d_transpose;

}

#endif

