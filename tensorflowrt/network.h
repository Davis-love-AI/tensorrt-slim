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
#include <VX/vx.h>

#include "utils.h"
#include "types.h"

#include "tfrt_jetson.h"
#include "network.pb.h"
#include "cuda/cudaMappedMemory.h"
#include "cuda/cudaCHWImage.h"
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
    if(end_points && name.length()) {
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
// struct cuda_tensor
// {
// public:
//     cuda_tensor();
//     /** Constructor: provide tensor shape. */
//     cuda_tensor(const std::string&, const nvinfer1::DimsNCHW&);
//     /** Move constructor and assignement. */
//     cuda_tensor(cuda_tensor&& t);
//     cuda_tensor& operator=(cuda_tensor&& t);
//     /** Destructor: CUDA free memory.  */
//     ~cuda_tensor();
//     /** Allocate shared memory between CPU and GPU/CUDA. */
//     bool allocate();
//     /** Free allocated memory and reset pointers. */
//     void free();
//     /** Is the tensor allocated. */
//     bool is_allocated() const {
//         return (cpu != nullptr && cuda != nullptr);
//     }

// public:
//     /** Get an Eigen tensor representation of the CPU tensor.  */
//     tfrt::nchw<float>::tensor tensor() const;
//     /** Get the cuda pointer, at a given batch index.  */
//     float* cuda_ptr(size_t batch_idx=0) {
//         return (cuda + batch_idx*shape.c()*shape.h()*shape.w());
//     }
//     /** Get the cpu pointer, at a given batch index.  */
//     float* cpu_ptr(size_t batch_idx=0) {
//         return (cpu + batch_idx*shape.c()*shape.h()*shape.w());
//     }


// private:
//     cuda_tensor(const cuda_tensor&) = default;
// public:
//     // Tensor name, shape and size.
//     std::string  name;
//     nvinfer1::DimsNCHW  shape;
//     size_t  size;
//     // TODO: use std::unique_ptr with custom deleter. Cleaner?
//     float*  cpu;
//     float*  cuda;
//     // Binding index.
//     int binding_index;
// };
// /* ============================================================================
//  * tfrt::cuda_tensor implementation.
//  * ========================================================================== */
// inline cuda_tensor::cuda_tensor() :
//     name{}, shape{}, size{0}, cpu{nullptr}, cuda{nullptr}, binding_index{0}
// {
// }
// inline cuda_tensor::cuda_tensor(const std::string& _name, const nvinfer1::DimsNCHW& _shape) :
//     name{_name}, shape{_shape},
//     size{_shape.n() * _shape.c() * _shape.h() * _shape.w() * sizeof(float)},
//     cpu{nullptr}, cuda{nullptr}, binding_index{0}
// {
// }
// inline cuda_tensor::cuda_tensor(cuda_tensor&& t) :
//     name{}, shape{}, size{0}, cpu{nullptr}, cuda{nullptr}
// {
//     this->operator=(std::move(t));
// }
// inline cuda_tensor& cuda_tensor::operator=(cuda_tensor&& t) {
//     // Free allocated memory...
//     free();
//     // Copy.
//     name = t.name;
//     shape = t.shape;
//     size = t.size;
//     cpu = t.cpu;
//     cuda = t.cuda;
//     // Reset.
//     t.name = "";
//     t.shape = nvinfer1::DimsNCHW();
//     t.size = 0;
//     t.cpu = nullptr;
//     t.cuda = nullptr;
//     return *this;
// }
// inline cuda_tensor::~cuda_tensor()
// {
//     free();
// }
// inline bool cuda_tensor::allocate()
// {
//     free();
//     // Double check size...
//     size = shape.n() * shape.c() * shape.h() * shape.w() * sizeof(float);
//     if(!cudaAllocMapped((void**)&cpu, (void**)&cuda, size)) {
//         LOG(FATAL) << "Failed to allocate CUDA mapped memory for tensor: " << name;
//         return false;
//     }
//     return true;
// }
// inline void cuda_tensor::free()
// {
//     if(cpu) {
//         CUDA(cudaFreeHost(cpu));
//         cpu = nullptr;
//         cuda = nullptr;
//     }
// }
// inline tfrt::nchw<float>::tensor cuda_tensor::tensor() const
// {
//     // Tensor using existing memory.
//     CHECK_NOTNULL(cpu);
//     return tfrt::nchw<float>::tensor_map(cpu, shape.n(), shape.c(), shape.h(), shape.w());
// }

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
        m_max_batch_size{2}, m_workspace_size{16 << 22},
        m_enable_profiler{false}, m_enable_debug{false},
        m_missing_tensors{false}  
    {
        this->name(name);
    }
    virtual ~network();
    /** Clear the network and its weights. */
    void clear();

    /** Load network configuration and weights + build + profile model. */
    bool load(std::string filename);
    /** Get the default scope for this network. */
    tfrt::scope scope(nvinfer1::INetworkDefinition* nv_network) const;

public:
    // General network parameters.
    const std::string& name() const;
    network& name(const std::string& name);
    nvinfer1::DataType datatype() const;
    network& datatype(nvinfer1::DataType dt);
    // Input and outputs getters.
    network& input(std::string name, nvinfer1::DimsCHW shape);
    nvinfer1::DimsCHW input_shape() const;
    std::string input_name(bool fullname) const;

    network& outputs(std::vector<std::string> names, std::vector<nvinfer1::DimsCHW> shapes);
    std::vector<nvinfer1::DimsCHW> outputs_shape() const;
    std::vector<std::string> outputs_name(bool fullname, bool suffix) const;
    // Input and output setters.
    tfrt::network& input_shape(const nvinfer1::DimsCHW& shape);
    // Create missing tensors?
    bool create_missing_tensors() const;
    void create_missing_tensors(bool v);

public:
    /// Weight tensors handling.
    /** Create a tensor, with a given name, shape, default value and type.
     * The tensor is owned by the network.
     */
    const tfrt_pb::tensor& create_tensor(std::string name, nvinfer1::Dims shape,
        float val, nvinfer1::DataType dt) const;
    const tfrt_pb::tensor& create_tensor(
        std::string name, const tfrt::nchw<float>::tensor& t, nvinfer1::DataType dt) const;   
    const tfrt_pb::tensor& create_tensor(
        std::string name, const tfrt::chw<float>::tensor& t, nvinfer1::DataType dt) const;
    const tfrt_pb::tensor& create_tensor(
        std::string name, const tfrt::c<float>::tensor& t, nvinfer1::DataType dt) const;   
    
    /** Get a tensor by name. If not found, either create a new tensor to replace
     * or just return an empty tensor. */
    const tfrt_pb::tensor& tensor_by_name(std::string name, nvinfer1::Dims wshape) const;
    /** Get NV weights by name. Return empty weights if not found. */
    nvinfer1::Weights weights_by_name(std::string name, nvinfer1::Dims wshape) const;
    /** Generate empty weights. */
    nvinfer1::Weights empty_weights() const;
    /** Convert TF protobuf tensor to NV weights. */
    static nvinfer1::Weights tensor_to_weights(const tfrt_pb::tensor& tensor,
        nvinfer1::DataType default_dt=nvinfer1::DataType::kFLOAT);
    /** Parse a protobuf file into a message.  */
    static bool parse_protobuf(const std::string&, google::protobuf::MessageLite*);

public:
    /// Loading and building the model.
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
    bool serialize_model(const std::string& filename, std::string& model_buffer, bool caching=true);
    /** Build and profile a model.
     */
    bool profile_model(nvinfer1::IHostMemory** model_stream);

protected:
    // Basic inference methods: single image, nvx images, ...
    /** Inference on a NCHW tensor (host memory). 
     * Note: easy to use, but not optimized as performing a copy + cuda sync at every call.
     */
    void inference(const tfrt::nchw<float>::tensor& tensor);
    /** Inference on a single RGBA image. */
    void inference(float* rgba, uint32_t height, uint32_t width);
    /** Inference on a single VX image.
     * Input image is supposed to be in RGBA, uint8 format.
     */
    void inference(vx_image image);
    void inference(const tfrt::nvx_image_inpatch& image);     
    /** Inference on two VX images.
     * Input images are supposed to be in RGBA, uint8 format.
     */
    void inference(vx_image img1, vx_image img2);
    void inference(const nvx_image_inpatch& img1, const nvx_image_inpatch& img2);     
    
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

public:
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

    // Create missing tensors?
    bool  m_missing_tensors;
    // Temporary collection of zero tensors.
    std::vector<tfrt_pb::tensor>  m_zero_tensors;
};

}

#endif