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
#include <fcntl.h>
#include <unistd.h>

#include <iostream>
#include <fstream>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <VX/vx.h>
#include <VX/vxu.h>
#include <NVX/nvx.h>
#include <OVX/UtilityOVX.hpp>

#include "types.h"
#include "scope.h"
#include "network.h"
#include "tensorflowrt.h"

#include "cuda/cudaHalfPrecision.h"
#include "cuda/cudaImageNet.h"
#include "cuda/cudaCHWImage.h"

const int kProtoReadBytesLimit = INT_MAX;

namespace tfrt {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
// using google::protobuf::Message;

/* ============================================================================
 * Nice utils
 * ========================================================================== */
inline nvinfer1::Dims tensor_shape(const tfrt_pb::tensor& t)
{
    nvinfer1::Dims dims;
    for(int i = 0; i < t.shape_size(); i++) {
        dims.d[i] = t.shape(i);
    }
    dims.nbDims = t.shape_size();
    return dims;
}

/* ============================================================================
 * tfrt::network methods.
 * ========================================================================== */
network::~network()
{
    // TODO: unique_ptr + custom deleter.
    if(m_nv_engine) {
        m_nv_engine->destroy();
        m_nv_engine = nullptr;
    }
    if(m_nv_infer) {
        m_nv_infer->destroy();
        m_nv_infer = nullptr;
    }
}
void network::clear()
{
    m_pb_network->Clear();
    m_zero_tensors.clear();
}
tfrt::scope network::scope(nvinfer1::INetworkDefinition* nv_network) const
{
    return tfrt::scope(nv_network, this, this->name());
}


/* ============================================================================
 * Network weights / tensors.
 * ========================================================================== */
const tfrt_pb::tensor& network::create_tensor(
    std::string name, nvinfer1::Dims shape, float val, nvinfer1::DataType dt) const
{
    LOG(INFO) << "CREATE tfrt_pb::tensor '" << name << "'. SHAPE: " << dims_str(shape);
    // Create new tensor in the weights collection.
    auto pb_tensor = m_pb_network->add_weights();
    pb_tensor->set_name(name);
    pb_tensor->set_datatype(tfrt_pb::DataType(int(dt)));
    pb_tensor->set_size(0);
    for (int i = 0 ; i < shape.nbDims; ++i) {
        pb_tensor->add_shape(shape.d[i]);
    }
    // Create Eigen tensor... Ugly if / else combinations!
    // First create float tensor, and then convert if necessary.
    if (shape.nbDims == 1) {
        auto t_float = tfrt::c<float>::tensor(shape.d[0]);
        t_float.setConstant(val);
        pb_tensor->set_size(t_float.size());
        if (dt == nvinfer1::DataType::kHALF) {
            auto t_half = tfrt::c<uint16_t>::tensor(shape.d[0]);
            cuda_float2half_array(t_float.data(), t_half.data(), t_float.size());
            pb_tensor->set_data(t_half.data(), t_half.size() * sizeof(uint16_t));
        }
        else {
            pb_tensor->set_data(t_float.data(), t_float.size() * sizeof(float));
        }
    }
    else if (shape.nbDims == 2) {
        auto t_float = tfrt::hw<float>::tensor(shape.d[0], shape.d[1]);
        t_float.setConstant(val);
        pb_tensor->set_size(t_float.size());
        if (dt == nvinfer1::DataType::kHALF) {
            auto t_half = tfrt::hw<uint16_t>::tensor(shape.d[0], shape.d[1]);
            cuda_float2half_array(t_float.data(), t_half.data(), t_float.size());
            pb_tensor->set_data(t_half.data(), t_half.size() * sizeof(uint16_t));
        }
        else {
            pb_tensor->set_data(t_float.data(), t_float.size() * sizeof(float));
        }
    }
    else if (shape.nbDims == 3) {
        auto t_float = tfrt::chw<float>::tensor(shape.d[0], shape.d[1], shape.d[2]);
        t_float.setConstant(val);
        pb_tensor->set_size(t_float.size());
        if (dt == nvinfer1::DataType::kHALF) {
            auto t_half = tfrt::chw<uint16_t>::tensor(shape.d[0], shape.d[1], shape.d[2]);
            cuda_float2half_array(t_float.data(), t_half.data(), t_float.size());
            pb_tensor->set_data(t_half.data(), t_half.size() * sizeof(uint16_t));
        }
        else {
            pb_tensor->set_data(t_float.data(), t_float.size() * sizeof(float));
        }
    }
    else if (shape.nbDims == 4) {
        auto t_float = tfrt::nchw<float>::tensor(shape.d[0], shape.d[1], shape.d[2], shape.d[3]);
        t_float.setConstant(val);
        pb_tensor->set_size(t_float.size());
        if (dt == nvinfer1::DataType::kHALF) {
            auto t_half = tfrt::nchw<uint16_t>::tensor(shape.d[0], shape.d[1], shape.d[2], shape.d[3]);
            cuda_float2half_array(t_float.data(), t_half.data(), t_float.size());
            pb_tensor->set_data(t_half.data(), t_half.size() * sizeof(uint16_t));
        }
        else {
            pb_tensor->set_data(t_float.data(), t_float.size() * sizeof(float));
        }
    }
    else if (shape.nbDims == 5) {
        auto t_float = tfrt::nachw<float>::tensor(shape.d[0], shape.d[1], shape.d[2], shape.d[3], shape.d[4]);
        t_float.setConstant(val);
        pb_tensor->set_size(t_float.size());
        if (dt == nvinfer1::DataType::kHALF) {
            auto t_half = tfrt::nachw<uint16_t>::tensor(shape.d[0], shape.d[1], shape.d[2], shape.d[3], shape.d[4]);
            cuda_float2half_array(t_float.data(), t_half.data(), t_float.size());
            pb_tensor->set_data(t_half.data(), t_half.size() * sizeof(uint16_t));
        }
        else {
            pb_tensor->set_data(t_float.data(), t_float.size() * sizeof(float));
        }
    }
    else {
        LOG(WARNING) << "FAILED to recognize the tensor shape.";
    }
    return *pb_tensor;
}
const tfrt_pb::tensor& network::create_tensor(
    std::string name, const tfrt::nchw<float>::tensor& t, nvinfer1::DataType dt) const
{
    LOG(INFO) << "CREATE tfrt_pb::tensor '" << name;
    // Create new tensor in the weights collection.
    auto pb_tensor = m_pb_network->add_weights();
    pb_tensor->set_name(name);
    pb_tensor->set_datatype(tfrt_pb::DataType(int(dt)));
    pb_tensor->set_size(0);
    for (int i = 0 ; i < t.NumDimensions ; ++i) {
        pb_tensor->add_shape(t.dimension(i));
    }
    // Set-up the weights.
    pb_tensor->set_size(t.size());
    if (dt == nvinfer1::DataType::kHALF) {
        auto t_half = tfrt::nchw<uint16_t>::tensor(
            t.dimension(0), t.dimension(1), t.dimension(2), t.dimension(3));
        cuda_float2half_array((float*)t.data(), t_half.data(), t.size());
        pb_tensor->set_data(t_half.data(), t_half.size() * sizeof(uint16_t));
    }
    else {
        pb_tensor->set_data(t.data(), t.size() * sizeof(float));
    }
    return *pb_tensor;
}
const tfrt_pb::tensor& network::create_tensor(
    std::string name, const tfrt::chw<float>::tensor& t, nvinfer1::DataType dt) const
{
    LOG(INFO) << "CREATE tfrt_pb::tensor '" << name;
    // Create new tensor in the weights collection.
    auto pb_tensor = m_pb_network->add_weights();
    pb_tensor->set_name(name);
    pb_tensor->set_datatype(tfrt_pb::DataType(int(dt)));
    pb_tensor->set_size(0);
    for (int i = 0 ; i < t.NumDimensions ; ++i) {
        pb_tensor->add_shape(t.dimension(i));
    }
    // Set-up the weights.
    pb_tensor->set_size(t.size());
    if (dt == nvinfer1::DataType::kHALF) {
        auto t_half = tfrt::chw<uint16_t>::tensor(
            t.dimension(0), t.dimension(1), t.dimension(2));
        cuda_float2half_array((float*)t.data(), t_half.data(), t.size());
        pb_tensor->set_data(t_half.data(), t_half.size() * sizeof(uint16_t));
    }
    else {
        pb_tensor->set_data(t.data(), t.size() * sizeof(float));
    }
    return *pb_tensor;
}
const tfrt_pb::tensor& network::create_tensor(
    std::string name, const tfrt::c<float>::tensor& t, nvinfer1::DataType dt) const
{
    LOG(INFO) << "CREATE tfrt_pb::tensor '" << name;
    // Create new tensor in the weights collection.
    auto pb_tensor = m_pb_network->add_weights();
    pb_tensor->set_name(name);
    pb_tensor->set_datatype(tfrt_pb::DataType(int(dt)));
    pb_tensor->set_size(0);
    for (int i = 0 ; i < t.NumDimensions ; ++i) {
        pb_tensor->add_shape(t.dimension(i));
    }
    // Set-up the weights.
    pb_tensor->set_size(t.size());
    if (dt == nvinfer1::DataType::kHALF) {
        auto t_half = tfrt::c<uint16_t>::tensor(t.dimension(0));
        cuda_float2half_array((float*)t.data(), t_half.data(), t.size());
        pb_tensor->set_data(t_half.data(), t_half.size() * sizeof(uint16_t));
    }
    else {
        pb_tensor->set_data(t.data(), t.size() * sizeof(float));
    }
    return *pb_tensor;
}

const tfrt_pb::tensor& network::tensor_by_name(std::string name, nvinfer1::Dims wshape) const
{
    // Best search algorithm ever!
    for(int i = 0 ; i < m_pb_network->weights_size() ; ++i) {
        const tfrt_pb::tensor& tensor = m_pb_network->weights(i);
        if(tensor.name() == name) {
            DLOG(INFO) << "FOUND tfrt_pb::tensor '" << name << "'. "
                << "SHAPE: " << dims_str(tensor_shape(tensor)) << " "
                << "SIZE: " << tensor.size();
            return tensor;
        }
    }
    // Create new tensor if specified.
    if (m_missing_tensors) {
        float val = 0.0f;
        return this->create_tensor(name, wshape, val, this->datatype());
    }
    LOG(WARNING) << "FAILED to find the tfrt_pb::tensor '" << name
        << "'. Using default empty tensor." ;
    return tfrt_pb::tensor::default_instance();
}
nvinfer1::Weights network::weights_by_name(std::string name, nvinfer1::Dims wshape) const
{
    const tfrt_pb::tensor& tensor = tensor_by_name(name, wshape);
    return tensor_to_weights(tensor, this->datatype());
}

nvinfer1::Weights network::tensor_to_weights(
    const tfrt_pb::tensor& tensor, nvinfer1::DataType default_dt)
{
    nvinfer1::Weights w{
        .type = nvinfer1::DataType(int(tensor.datatype())),
        .values = tensor.data().data(),
        .count = int(tensor.size())
    };
    // Check empty weights.
    if(w.count == 0) {
        w.values = nullptr;
        w.type = default_dt;
    }
    return w;
}
nvinfer1::Weights network::empty_weights() const {
    return nvinfer1::Weights{.type = this->datatype(), .values = nullptr, .count = 0};
}

/* ============================================================================
 * Getters / setters wrapping protobuf methods.
 * ========================================================================== */
const std::string& network::name() const {
    return m_pb_network->name();
}
network& network::name(const std::string& name) {
    m_pb_network->set_name(name);
    return *this;
}
nvinfer1::DataType network::datatype() const {
    // Datatype of the network. Hopefully consistent with weights...
    auto dt = m_pb_network->datatype();
    return nvinfer1::DataType(int(dt));
}
network& network::datatype(nvinfer1::DataType dt)
{
    m_pb_network->set_datatype(tfrt_pb::DataType(int(dt)));
    return *this;
}
// Max batch size and workspace.
network& network::max_batch_size(uint32_t bsize)
{
    m_max_batch_size = bsize;
    return *this;
}
uint32_t network::max_batch_size() const
{
    return m_max_batch_size;
}
network& network::max_workspace_size(uint32_t wsize)
{
    m_workspace_size = wsize;
    return *this;
}
uint32_t network::max_workspace_size() const
{
    return m_workspace_size;
}
// Input and outputs getters / setters.
network& network::input(std::string name, nvinfer1::DimsCHW shape)
{
    // Only change shape if positive.
    if(shape.c() && shape.h() && shape.w()) {
        auto input = m_pb_network->mutable_input();
        input->set_name(name);
        input->set_c(shape.c());
        input->set_h(shape.h());
        input->set_w(shape.w());
    }
    return *this;
}
tfrt::network& network::input_shape(const nvinfer1::DimsCHW& shape)
{
    // Only change shape if positive.
    if(shape.c() && shape.h() && shape.w()) {
        auto input = m_pb_network->mutable_input();
        input->set_c(shape.c());
        input->set_h(shape.h());
        input->set_w(shape.w());
    }
    return *this;
}
nvinfer1::DimsCHW network::input_shape() const
{
    auto input = m_pb_network->input();
    return nvinfer1::DimsCHW{input.c(), input.h(), input.w()};
}
std::string network::input_name(bool fullname) const
{
    std::string iname = m_pb_network->input().name();
    if(fullname) {
        iname = (this->name() + "/") + iname;
    }
    return iname;
}

network& network::outputs(
    std::vector<std::string> names, std::vector<nvinfer1::DimsCHW> shapes)
{
    CHECK_EQ(names.size(), shapes.size()) << "Invalid size of names and shapes vectors.";
    m_pb_network->clear_outputs();
    for(size_t i = 0 ; i < names.size() ; ++i) {
        auto out = m_pb_network->add_outputs();
        out->set_name(names[i]);
        out->set_c(shapes[i].c());
        out->set_h(shapes[i].h());
        out->set_w(shapes[i].w());
    }
    return *this;
}
std::vector<nvinfer1::DimsCHW> network::outputs_shape() const
{
    std::vector<nvinfer1::DimsCHW> v;
    for(int i = 0 ; i < m_pb_network->outputs_size() ; ++i) {
        const tfrt_pb::output& output = m_pb_network->outputs(i);
        v.push_back(nvinfer1::DimsCHW{output.c(), output.h(), output.w()});
    }
    return v;
}
std::vector<std::string> network::outputs_name(bool fullname, bool suffix) const
{
    // TODO: more efficient way!
    std::vector<std::string> v;
    for(int i = 0 ; i < m_pb_network->outputs_size() ; ++i) {
        const tfrt_pb::output& output = m_pb_network->outputs(i);
        // Construct output name...
        std::string oname = output.name();
        if(fullname) {
            oname = (this->name() + "/") + oname;
        }
        if(suffix) {
            oname += "/";
            oname += output.suffix();
        }
        v.push_back(oname);
    }
    return v;
}

tfrt::cuda_tensor* network::find_cuda_output(const std::string& name) const
{
    DLOG(INFO) << "Finding CUDA output tensor named: \'" << name << "\'";
    // DCHECK(name.length()) << "Empty CUDA tensor name.";
    for(auto& t : m_cuda_outputs) {
        if(t.name.find(name) != std::string::npos && name.length()) {
            // A bit ugly hack!!!
            return (tfrt::cuda_tensor*) &t;
        }
    }
    LOG(WARNING) << "Could not find CUDA output tensor named: \'" << name << "\'";
    return nullptr;
}
bool network::create_missing_tensors() const
{
    return m_missing_tensors;
}
void network::create_missing_tensors(bool v)
{
    m_missing_tensors = v;
}

/* ============================================================================
 * load - build - serialize. The big stuff!
 * ========================================================================== */
bool network::parse_protobuf(const std::string& filename, google::protobuf::MessageLite* message)
{
    LOG(INFO) << "Parsing protobuf binary file: " << filename;
    // Highly inspired by Caffe source code!
    int fd = open(filename.c_str(), O_RDONLY);
    CHECK_NE(fd, -1) << "FILE not found: " << filename;
    ZeroCopyInputStream* raw_input = new FileInputStream(fd);
    CodedInputStream* coded_input = new CodedInputStream(raw_input);
    coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);
    bool success = message->ParseFromCodedStream(coded_input);
    delete coded_input;
    delete raw_input;
    close(fd);
    CHECK(success) << "FATAL error, could not parse protobuf file: " << filename;
    return success;
}
bool network::load_weights(const std::string& filename)
{
    if (filename.length() == 0) {
        LOG(WARNING) << "No protobuf filename provided.";
        return true;
    }
    else {
        LOG(INFO) << "Loading network parameters and weights from: " << filename;
        return parse_protobuf(filename, m_pb_network.get());
    }
}
void network::clear_weights()
{
    m_pb_network->clear_weights();
}

bool network::load(std::string filename)
{
    // Serialize model.
    // std::stringstream model_stream;
    // nvinfer1::IHostMemory* model_stream{nullptr};
    std::string model_buffer;
    this->serialize_model(filename, model_buffer, true);

    // Inference runtime + engine + execution context.
    nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(m_gie_logger);
    CHECK_NOTNULL(infer);
    nvinfer1::ICudaEngine* engine = infer->deserializeCudaEngine(
        model_buffer.data(), model_buffer.size(), nullptr);
    CHECK_NOTNULL(engine);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    CHECK_NOTNULL(context);

    // Debug and profiler options.
    if(m_enable_debug) {
        LOG(INFO) << LOG_GIE << "Enabling context debug sync.";
        context->setDebugSync(true);
    }
    if(m_enable_profiler) {
        context->setProfiler(&m_gie_profiler);
    }
    LOG(INFO) << LOG_GIE << "CUDA engine context initialized with #bindings: " << engine->getNbBindings();
    this->m_nv_infer = infer;
    this->m_nv_engine = engine;
    this->m_nv_context = context;

    // Cached binding pointers.
    nvinfer1::DimsNCHW shape;
    auto outputs_name = this->outputs_name(true, true);
    m_cached_bindings.resize(1 + outputs_name.size());

    // CUDA allocate input memory.
    const int input_idx = m_nv_engine->getBindingIndex(input_name(true).c_str());
    nvinfer1::DimsCHW inshape =
        static_cast<nvinfer1::DimsCHW&&>(engine->getBindingDimensions(input_idx));
    shape = nvinfer1::DimsNCHW{int(m_max_batch_size), inshape.c(), inshape.h(), inshape.w()};

    LOG(INFO) << LOG_GIE << "Allocating CUDA input memory for '" << input_name(true)
        << "' with shape: " << dims_str(shape);
    m_cuda_input = std::move(tfrt::cuda_tensor(input_name(true), shape));
    bool r = m_cuda_input.allocate();
    CHECK(r) << LOG_GIE << "Could not allocate memory for CUDA input: "
        << dims_str(shape) << " | "<< input_name(true);
    m_cuda_input.binding_index = input_idx;
    m_cached_bindings[input_idx] = m_cuda_input.cuda;

    // CUDA allocate outputs memory.
    m_cuda_outputs.clear();
    for(size_t i = 0 ; i < outputs_name.size() ; ++i) {
        const int output_idx = engine->getBindingIndex(outputs_name[i].c_str());
        if(output_idx > -1) {
            nvinfer1::DimsCHW outshape =
                static_cast<nvinfer1::DimsCHW&&>(engine->getBindingDimensions(output_idx));
            shape = nvinfer1::DimsNCHW{int(m_max_batch_size), outshape.c(), outshape.h(), outshape.w()};
            LOG(INFO) << LOG_GIE << "Allocating CUDA output memory for '" << outputs_name[i]
                << "' with shape: " << dims_str(shape);
            // Push CUDA tensor and allocate memory.
            m_cuda_outputs.push_back(tfrt::cuda_tensor(outputs_name[i], shape));
            r = m_cuda_outputs.back().allocate();
            CHECK(r) << LOG_GIE << "Could not allocate memory for CUDA output: "
                << dims_str(shape) << " | "<< outputs_name[i];
            m_cuda_outputs.back().binding_index = output_idx;
            m_cached_bindings[output_idx] = m_cuda_outputs.back().cuda;
        }
        else {
            LOG(ERROR) << LOG_GIE << "Could not find binding index for output tensor: " << outputs_name[i];
        }
    }
    CHECK(m_cuda_outputs.size()) << LOG_GIE << "No output found in the network.";
    return true;
}


nvinfer1::ITensor* network::build(tfrt::scope sc)
{
    return nullptr;
}
bool network::serialize_model(const std::string& filename, std::string& model_buffer, bool caching)
{
    // Load model parameters and weights.
    this->load_weights(filename);

    // Try to read serialized model from cache.
    std::ostringstream  filename_cache;
    filename_cache << filename << "."  << m_max_batch_size << ".cache";
    if(caching && filename.length()) {
        LOG(INFO) << LOG_GIE << "Try reading cached model from: "<< filename_cache.str();
        // Successful read of cached file => load and return.
        std::ifstream model_cached(filename_cache.str());
        if(model_cached) {
            // Set model stream back to beginning.
            std::stringstream model_stream;
            model_stream.seekg(0, model_stream.beg);
            LOG(INFO) << LOG_GIE << "Loading network profile from cache...";
            model_stream << model_cached.rdbuf();
            model_cached.close();
            // UGLY copy!!!
            model_buffer = model_stream.str();
            return true;
        }
        LOG(WARNING) << LOG_GIE << "Could not read cached model. Back to th' old way.";
    }
    LOG(INFO) << LOG_GIE << "Building and profiling the network model.";
    nvinfer1::IHostMemory* nv_model_stream{nullptr};
    this->profile_model(&nv_model_stream);
    this->clear_weights();
    // TODO: fix this ugly copy of buffer!
    model_buffer.assign((char*)nv_model_stream->data(), nv_model_stream->size());
    nv_model_stream->destroy();

    if(caching && filename.length()) {
        LOG(INFO) << LOG_GIE << "Writing cached model to: " << filename_cache.str();
        std::ofstream model_cached;
        model_cached.open(filename_cache.str());
        model_cached << model_buffer;
        model_cached.close();
        // model_stream.seekg(0, model_stream.beg);
    }
    return true;
}
bool network::profile_model(nvinfer1::IHostMemory** nv_model_stream)
{
    // Create API root class - must span the lifetime of the engine usage.
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(m_gie_logger);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    builder->setDebugSync(m_enable_debug);
    builder->setMinFindIterations(10);	    // allow time for TX1/2 GPU to spin up.
    builder->setAverageFindIterations(5);
    builder->setMinFindIterations(1);	    // allow time for TX1/2 GPU to spin up.
    builder->setAverageFindIterations(1);

    // Build the network.
    LOG(INFO) << LOG_GIE << "Building network from scratch!";
    auto net = this->build(this->scope(network));
    CHECK_NOTNULL(net);
    LOG(INFO) << LOG_GIE << "Network successfully built."
        << " #Inputs: " << network->getNbInputs() << " #Outputs: " << network->getNbOutputs();

    // Build the engine
    LOG(INFO) << LOG_GIE << "Configuring CUDA engine.";
    LOG(INFO) << LOG_GIE << "Max batch size: " << m_max_batch_size;
    builder->setMaxBatchSize(m_max_batch_size);
    LOG(INFO) << LOG_GIE << "Max workspace size: " << m_workspace_size;
    builder->setMaxWorkspaceSize(m_workspace_size);
    // Set up the floating mode.
    bool compatibleType = (this->datatype() == nvinfer1::DataType::kFLOAT ||
                           builder->platformHasFastFp16());
    CHECK(compatibleType) << LOG_GIE << "Can not build network with FP16 data type. Platform not compatible.";
    bool useFP16 = (this->datatype() == nvinfer1::DataType::kHALF &&
                    builder->platformHasFastFp16());
    useFP16 = builder->platformHasFastFp16();
    LOG_IF(INFO, useFP16) << LOG_GIE << "Configure network with FP16 data type.";
    LOG_IF(INFO, !useFP16) << LOG_GIE << "Configure network with FP32 data type.";
    builder->setHalf2Mode(useFP16);
    // builder->setHalf2Mode(builder->platformHasFastFp16());

    nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
    CHECK_NOTNULL(engine);
    network->destroy();
    // Serialize the engine, then close everything down
    LOG(INFO) << LOG_GIE << "Serializing the engine.";
    // engine->serialize(model_stream);
    (*nv_model_stream) = engine->serialize();
    engine->destroy();
    builder->destroy();
    return true;
}

/* ============================================================================
 * Inference methods.
 * ========================================================================== */
void network::inference(const tfrt::nchw<float>::tensor& tensor)
{
    DLOG(INFO) << "Inference on the neural network:" << this->name();
    // Check tensor dimensions.
    CHECK_EQ(tensor.dimension(1), m_cuda_input.shape.c())
        << "Input tensor with wrong channel dimension.";
    CHECK_EQ(tensor.dimension(2), m_cuda_input.shape.h())
        << "Input tensor with wrong height dimension.";
    CHECK_EQ(tensor.dimension(3), m_cuda_input.shape.w())
        << "Input tensor with wrong width dimension.";
    CHECK_LE(tensor.dimension(0), m_cuda_input.shape.n())
        << "Input tensor with wrong batch dimension.";
    std::memcpy(m_cuda_input.cpu, tensor.data(), tensor.dimension(0) * m_cuda_input.shape.c() * m_cuda_input.shape.h() * m_cuda_input.shape.w() * sizeof(float));
    CUDA(cudaDeviceSynchronize());
    m_nv_context->execute(tensor.dimension(0), (void**)m_cached_bindings.data());
    CUDA(cudaDeviceSynchronize());
}
void network::inference(float* rgba, uint32_t height, uint32_t width)
{
    DLOG(INFO) << "Inference on the neural network:" << this->name();
    // Checking inputs!
    CHECK(rgba) << "Invalid image buffer.";
    CHECK(height) << "Invalid image height.";
    CHECK(width) << "Invalid image width.";
    // Downsample and convert to RGB.
    cudaError_t r = cudaPreImageNet((float4*)rgba, width, height,
        m_cuda_input.cuda, m_cuda_input.shape.w(), m_cuda_input.shape.h());
    CHECK_EQ(r, cudaSuccess) << "Failed to resize image to network input shape. "
        << "CUDA error: " << r;
    // Execute TensorRT network (batch size = 1).
    size_t num_batches = 1;
    m_nv_context->execute(num_batches, (void**)m_cached_bindings.data());
}
void network::inference(vx_image image)
{
    DLOG(INFO) << "Inference (batch 1) on the neural network:"  << this->name();
    // Check image information.
    vx_df_image format = VX_DF_IMAGE_VIRT;
    NVXIO_SAFE_CALL( vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)) );
    CHECK_EQ(format, VX_DF_IMAGE_RGBX) << "Wrong VX image format.";
    // Inference using CUDA patch.
    nvx_image_inpatch img_patch{image};
    this->inference(img_patch);
}
void network::inference(const nvx_image_inpatch& image)
{
    const auto& img_patch = image;
    LOG(INFO) << "Converting RGBA image to CHW format.";
    auto r = cuda_rgba_to_chw(img_patch.cuda, m_cuda_input.cuda,
        m_cuda_input.shape.w(), m_cuda_input.shape.h(),
        img_patch.addr.stride_x, img_patch.addr.stride_y);
    CHECK_EQ(r, cudaSuccess) << "Failed to convert VX image to CHW format. CUDA error: " << r;
    CUDA(cudaDeviceSynchronize());
    // Execute TensorRT network (batch size = 1).
    size_t num_batches = 1;
    m_nv_context->execute(num_batches, (void**)m_cached_bindings.data());
}

void network::inference(vx_image img1, vx_image img2)
{
    LOG(INFO) << "Inference (batch 2) on the neural network:"  << this->name();
    // Set CUDA patches and convert to CHW format.
    LOG(INFO) << "Creating patches from VX images.";
    nvx_image_inpatch img_patch1{img1};
    nvx_image_inpatch img_patch2{img2};
    this->inference(img_patch1, img_patch2);
}
void network::inference(const nvx_image_inpatch& img1, const nvx_image_inpatch& img2)
{
    cudaError_t r;
    const auto& img_patch1 = img1;
    const auto& img_patch2 = img2;
    const nvinfer1::DimsNCHW& inshape{m_cuda_input.shape};
    LOG(INFO) << "Converting RGBA image to CHW format.";
    r = cuda_rgba_to_chw(img_patch1.cuda, m_cuda_input.cuda_ptr(0),
        inshape.w(), inshape.h(), img_patch1.addr.stride_x, img_patch1.addr.stride_y);
    CHECK_EQ(r, cudaSuccess) << "Failed to convert VX image 0 to CHW format. CUDA error: " << r;
    r = cuda_rgba_to_chw(img_patch2.cuda, m_cuda_input.cuda_ptr(1),
        inshape.w(), inshape.h(), img_patch2.addr.stride_x, img_patch2.addr.stride_y);
    CHECK_EQ(r, cudaSuccess) << "Failed to convert VX image 1 to CHW format. CUDA error: " << r;

    CUDA(cudaDeviceSynchronize());
    // Execute TensorRT network (batch size = 1).
    size_t num_batches = 2;
    LOG(INFO) << "Executing neural network.";
    m_nv_context->execute(num_batches, (void**)m_cached_bindings.data());
}


}
