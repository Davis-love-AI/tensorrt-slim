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
#ifndef TFRT_UTILS_H
#define TFRT_UTILS_H

#include <glog/logging.h>

#include <VX/vx.h>
#include <VX/vxu.h>
#include <NVX/nvx.h>
#include <NvInfer.h>

#include "cuda/cudaMappedMemory.h"
#include "types.h"
#include "tfrt_jetson.h"
#include "network.pb.h"

namespace tfrt
{
/* ============================================================================
 * Dim utils.
 * ========================================================================== */
/** Get the number of channels from a dims object.
 */
inline int dims_channels(nvinfer1::Dims dims)
{
    // Suppose NCHW, CHW or HW format...
    if(dims.nbDims >= 4) {
        return dims.d[1];
    }
    else if(dims.nbDims == 3) {
        return dims.d[0];
    }
    return 1;
}
/** Generate a string describing a dims object.
 */
inline std::string dims_str(nvinfer1::Dims dims)
{
    std::ostringstream oss;
    oss << "[" << dims.d[0];
    for(int i = 1 ; i < dims.nbDims ; ++i) {
        oss << ", " << dims.d[i];
    }
    oss << "]";
    return oss.str();
}
/** Generate a NV dims object from an equivalent protobuf object.
 */
inline nvinfer1::DimsCHW dims_pb(tfrt_pb::dimsCHW dims)
{
    return {dims.c(), dims.h(), dims.w()};
}
inline nvinfer1::DimsHW dims_pb(tfrt_pb::dimsHW dims)
{
    return {dims.h(), dims.w()};
}

/* ============================================================================
 * NVX compatibility utils.
 * ========================================================================== */
/** NVX CUDA image input patch. Automatic map at construction, and unmap at destruction.
 * RAII spirit: probably not the fastest implementation, but easy to use.
 */
struct nvx_image_inpatch
{
    // Map id.
    vx_map_id  map_id;
    // Map addressing.
    vx_imagepatch_addressing_t  addr;
    // CUDA image pointer.
    vx_uint8*  cuda;
    // VX image.
    vx_image  image;
    // Usage and memory type.
    vx_enum  usage;
    vx_enum  mem_type;

public:
    nvx_image_inpatch() : 
        map_id{0}, addr{}, cuda{nullptr}, image{}, 
        usage{VX_READ_ONLY}, mem_type{NVX_MEMORY_TYPE_CUDA}
    {
    }
    /** Construction of the CUDA patch from a VX input image,
     * directly initializing the mapping.
     */
    nvx_image_inpatch(vx_image _input_img, 
        vx_enum _usage=VX_READ_ONLY, vx_enum _mem_type=NVX_MEMORY_TYPE_CUDA) : 
            map_id{0}, addr{}, cuda{nullptr}, image{_input_img},
            usage{_usage}, mem_type{_mem_type}
    {
        vxMapImagePatch(image, nullptr, 0, &map_id, &addr, (void **)&cuda, 
            usage, mem_type, 0);
    }
    /** Unmap at destruction. */
    ~nvx_image_inpatch()
    {
        if (cuda) {
            vxUnmapImagePatch(image, map_id);
        }
    }

private:
    // Deactivating copy
    nvx_image_inpatch(const nvx_image_inpatch&);
    nvx_image_inpatch(nvx_image_inpatch&&);
};

/* ============================================================================
 * tfrt::cuda_tensor_t
 * ========================================================================== */
/** CUDA tensor with shared memory between CUDA and CPU.
 */
template <typename T>
struct cuda_tensor_t
{
public:
    cuda_tensor_t() :
        name{}, shape{}, size{0}, cpu{nullptr}, cuda{nullptr}, binding_index{0}  {}
    /** Constructor: provide tensor shape. */
    cuda_tensor_t(const std::string& _name, const nvinfer1::DimsNCHW& _shape) :
        name{_name}, shape{_shape},
        size{_shape.n() * _shape.c() * _shape.h() * _shape.w() * sizeof(T)},
        cpu{nullptr}, cuda{nullptr}, binding_index{0}  {}
    /** Move constructor and assignement. */
    cuda_tensor_t(cuda_tensor_t<T>&& t)  :
        name{}, shape{}, size{0}, cpu{nullptr}, cuda{nullptr}, binding_index{0}  
    {
        this->operator=(std::move(t));
    }
    cuda_tensor_t& operator=(cuda_tensor_t<T>&& t)
    {
        // Free allocated memory...
        free();
        // Copy.
        name = t.name;
        shape = t.shape;
        size = t.size;
        cpu = t.cpu;
        cuda = t.cuda;
        binding_index = t.binding_index;
        // Reset.
        t.name = "";
        t.shape = nvinfer1::DimsNCHW();
        t.size = 0;
        t.cpu = nullptr;
        t.cuda = nullptr;
        t.binding_index = 0;
        return *this;
    }
    /** Destructor: CUDA free memory.  */
    ~cuda_tensor_t()
    {
        free();
    }
    /** Reshape the tensor. Release memory if necessary.  */
    void reshape(const nvinfer1::DimsNCHW& _shape) 
    {
        this->free();
        this->shape = _shape;
    }
    /** Allocate shared memory between CPU and GPU/CUDA. */
    bool allocate()
    {
        free();
        // Double check size...
        size = shape.n() * shape.c() * shape.h() * shape.w() * sizeof(T);
        if (size) {
            if (!cudaAllocMapped((void**)&cpu, (void**)&cuda, size)) {
                LOG(FATAL) << "Failed to allocate CUDA mapped memory for tensor: " << name;
                return false;
            }
        }   
        return true;
    }
    /** Free allocated memory and reset pointers. */
    void free()
    {
        if(cpu) {
            CUDA(cudaFreeHost(cpu));
            cpu = nullptr;
            cuda = nullptr;
        }
    }
    /** Is the tensor allocated. */
    bool is_allocated() const {
        return (cpu != nullptr && cuda != nullptr);
    }

public:
    /** Get an Eigen tensor representation of the CPU tensor.  */
    typename tfrt::nchw<T>::tensor_map tensor() const
    {
         // Tensor using existing memory.
        CHECK_NOTNULL(cpu);
        return typename tfrt::nchw<T>::tensor_map(cpu, shape.n(), shape.c(), shape.h(), shape.w());
    }
    /** Get the cuda pointer, at a given batch index.  */
    T* cuda_ptr(size_t batch_idx=0) const {
        return (cuda + batch_idx*shape.c()*shape.h()*shape.w());
    }
    /** Get the cpu pointer, at a given batch index.  */
    T* cpu_ptr(size_t batch_idx=0) const {
        return (cpu + batch_idx*shape.c()*shape.h()*shape.w());
    }

private:
    cuda_tensor_t(const cuda_tensor_t<T>&) = default;
public:
    // Tensor name, shape and size.
    std::string  name;
    nvinfer1::DimsNCHW  shape;
    size_t  size;
    // TODO: use std::unique_ptr with custom deleter. Cleaner?
    T*  cpu;
    T*  cuda;
    // Binding index.
    int binding_index;
};

// Common CUDA tensors.
typedef cuda_tensor_t<float>  cuda_tensor;
typedef cuda_tensor_t<uint8_t>  cuda_tensor_u8;

}

#endif
