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
#ifndef TFRT_NETWORK_H
#define TFRT_NETWORK_H

#include <memory>
#include <string>
#include <sstream>

#include <unsupported/Eigen/CXX11/Tensor>
// #include <cuda_runtime_api.h>
#include <NvInfer.h>

#include "types.h"
#include "tfrt_jetson.h"
#include "network.pb.h"
#include "cuda/cudaMappedMemory.h"
#include "misc/std_make_unique.h"

namespace tfrt
{
class scope;

/** Standard map string->tensor. */
typedef std::map<std::string, nvinfer1::ITensor*> map_tensor;
/** Add an end point to a collection (if existing).
 */
inline nvinfer1::ITensor* add_end_point(tfrt::map_tensor* end_points, const std::string& name, nvinfer1::ITensor* tensor) {
    if(end_points) {
        end_points->operator[](name) = tensor;
    }
    return tensor;
}
/** Find an end point in a collection. First partial match.
 */
inline nvinfer1::ITensor* find_end_point(tfrt::map_tensor* end_points, const std::string& name) {
    if(end_points) {
        for(auto&& point : *end_points) {
            if(point.first.find(name) != std::string::npos) {
                return point.second;
            }
        }
    }
    return nullptr;
}

/* ============================================================================
 * tfrt::cuda_tensor: shared memory between CPU and GPU/CUDA.
 * ========================================================================== */
/** CUDA tensor with shared memory between CUDA and CPU.
 */
struct cuda_tensor
{
public:
    cuda_tensor();
    /** Constructor: provide tensor shape. */
    cuda_tensor(const std::string&, const nvinfer1::DimsNCHW&);
    /** Move constructor and assignement. */
    cuda_tensor(cuda_tensor&& t);
    cuda_tensor& operator=(cuda_tensor&& t);
    /** Destructor: CUDA free memory.  */
    ~cuda_tensor();
    /** Allocate shared memory between CPU and GPU/CUDA. */
    bool allocate();
    /** Free allocated memory and reset pointers. */
    void free();
    /** Is the tensor allocated. */
    bool is_allocated() const {
        return (cpu != nullptr && cuda != nullptr);
    }

public:
    /** Get an Eigen tensor representation of the CPU tensor. */
    tfrt::nchw<float>::tensor tensor() const;

private:
    cuda_tensor(const cuda_tensor&) = default;
public:
    // Tensor name, shape and size.
    std::string  name;
    nvinfer1::DimsNCHW  shape;
    size_t  size;
    // TODO: use std::unique_ptr with custom deleter. Cleaner?
    float*  cpu;
    float*  cuda;
    // Binding index.
    int binding_index;
};

/* ============================================================================
 * tfrt::network
 * ========================================================================== */
/** Generic network class, implementation the basic building, inference
 * profiling methods.
 */
class network
{
public:
    /** Create network, specifying the name and the datatype.
     */
    network(std::string name) :
        // m_pb_network(new tfrt_pb::network()),
        m_pb_network(std::make_unique<tfrt_pb::network>()),
        m_nv_infer{nullptr}, m_nv_engine{nullptr}, m_nv_context{nullptr},
        m_max_batch_size{2}, m_workspace_size{16 << 20},
        m_enable_profiler{false}, m_enable_debug{false} {
    }
    virtual ~network();
    /** Clear the network and its weights. */
    void clear();

    /** Load network configuration and weights + build + profile model. */
    bool load(std::string filename);
    /** Get the default scope for this network. */
    tfrt::scope scope(nvinfer1::INetworkDefinition* nv_network) const;

public:
    /** Get a tensor by name. Return empty tensor if not found. */
    const tfrt_pb::tensor& tensor_by_name(std::string name) const;
    /** Get NV weights by name. Return empty weights if not found. */
    nvinfer1::Weights weights_by_name(std::string name) const;

    // General network parameters.
    const std::string& name() const;
    network& name(const std::string& name);
    nvinfer1::DataType datatype() const;
    // Input and outputs getters.
    nvinfer1::DimsCHW input_shape() const;
    std::string input_name(bool fullname) const;
    std::vector<nvinfer1::DimsCHW> outputs_shape() const;
    std::vector<std::string> outputs_name(bool fullname, bool suffix) const;
    // Input and output setters.
    tfrt::network& input_shape(const nvinfer1::DimsCHW& shape);

public:
    /** Generate empty weights. */
    nvinfer1::Weights empty_weights() const {
        return nvinfer1::Weights{.type = this->datatype(), .values = nullptr, .count = 0};
    }
    /** Convert TF protobuf tensor to NV weights. */
    static nvinfer1::Weights tensor_to_weights(const tfrt_pb::tensor& tensor, nvinfer1::DataType default_dt=nvinfer1::DataType::kFLOAT);
    /** Parse a protobuf file into a message.  */
    static bool parse_protobuf(const std::string&, google::protobuf::MessageLite*);

public:
    /** Load weights and configuration from .tfrt file. */
    virtual bool load_weights(const std::string& filename);
    /** Clear out the collections of network weights, to save memory. */
    virtual void clear_weights();

    /** Build the complete network. Input + all layers.
     * VIRTUAL: to be re-implemented in children classes.
     */
    virtual nvinfer1::ITensor* build(tfrt::scope sc);
    /** Serialize a network model. If caching=True, try to first load from
     * a cached file. If no file, construct the usual way and save the cache.
     */
    bool serialize_model(const std::string& filename, std::stringstream& model_stream, bool caching=true);
    /** Build and profile a model.
     */
    bool profile_model(std::stringstream& model_stream);

protected:
    /** Find a output CUDA tensor from the all collection! Return first partial match.
     */
    tfrt::cuda_tensor* find_cuda_output(const std::string& name) const;

protected:
	/** Prefix used for tagging printed log output. */
	#define LOG_GIE "[GIE]  "

	/** Logger class for GIE info/warning/errors.
     */
	class Logger : public nvinfer1::ILogger
	{
		void log( Severity severity, const char* msg ) override
		{
			if( severity != Severity::kINFO /*|| mEnableDebug*/ )
				printf(LOG_GIE "%s\n", msg);
		}
	} m_gie_logger;
	/** Profiler interface for measuring layer timings
	 */
	class Profiler : public nvinfer1::IProfiler
	{
	public:
		Profiler() : timingAccumulator(0.0f) {}
		virtual void reportLayerTime(const char* layerName, float ms)
		{
			printf(LOG_GIE "layer %s - %f ms\n", layerName, ms);
			timingAccumulator += ms;
		}
		float timingAccumulator;

	} m_gie_profiler;
	/** When profiling is enabled, end a profiling section and report timing statistics.
	 */
	inline void PROFILER_REPORT()	{
        if(m_enable_profiler) {
            printf(LOG_GIE "layer network time - %f ms\n", m_gie_profiler.timingAccumulator); m_gie_profiler.timingAccumulator = 0.0f;
        }
    }

protected:
    // Protobuf network object.
    std::unique_ptr<tfrt_pb::network>  m_pb_network;
    // TensorRT elements...
    nvinfer1::IRuntime*  m_nv_infer;
	nvinfer1::ICudaEngine*  m_nv_engine;
	nvinfer1::IExecutionContext*  m_nv_context;

    // TensorRT max batch size and workspace size.
    uint32_t  m_max_batch_size;
    uint32_t  m_workspace_size;
    // Profiler and debugging?
    bool  m_enable_profiler;
	bool  m_enable_debug;

    // CUDA input and outputs.
    tfrt::cuda_tensor  m_cuda_input;
    std::vector<tfrt::cuda_tensor>  m_cuda_outputs;
    // Cached bindings vector.
    std::vector<float*>  m_cached_bindings;

    // Temporary collection of zero tensors.
    std::vector<tfrt_pb::tensor>  m_zero_tensors;
};

}

#endif