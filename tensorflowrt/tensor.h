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
#ifndef TFRT_TENSOR_H
#define TFRT_TENSOR_H

#include <glog/logging.h>
#include <Eigen/Dense>

#include <VX/vx.h>
#include <VX/vxu.h>
#include <NVX/nvx.h>

#include <NVX/nvxcu.h>
#include <NvInfer.h>

#include "ros/ros.h"
#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"

#include "std_msgs/Int8MultiArray.h"
#include "std_msgs/Int16MultiArray.h"
#include "std_msgs/Int32MultiArray.h"
#include "std_msgs/Int64MultiArray.h"

#include "std_msgs/UInt8MultiArray.h"
#include "std_msgs/UInt16MultiArray.h"
#include "std_msgs/UInt32MultiArray.h"
#include "std_msgs/UInt64MultiArray.h"

#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/Float64MultiArray.h"

#include "cuda/cudaMappedMemory.h"
#include "types.h"
#include "tfrt_jetson.h"
#include "network.pb.h"

namespace tfrt
{
/* ============================================================================
 * tfrt::cuda_tensor_t
 * ========================================================================== */
/** CUDA tensor with shared memory between CUDA and CPU.
 */
template <typename T>
struct cuda_tensor_t
{
public:
    /** Value type? */
    typedef T  value_type;
    
    /** Empty CUDA tensor. */
    cuda_tensor_t() :
        name{}, shape{}, size{0}, cpu{nullptr}, cuda{nullptr}, binding_index{0},
        m_own_memory{false}  {}
    /** Constructor: provide tensor shape. */
    cuda_tensor_t(const std::string& _name, const nvinfer1::DimsNCHW& _shape) :
        name{_name}, shape{_shape},
        size{_shape.n() * _shape.c() * _shape.h() * _shape.w() * sizeof(T)},
        cpu{nullptr}, cuda{nullptr}, binding_index{0}, m_own_memory{false} {}
    /** Move constructor and assignement. */
    cuda_tensor_t(cuda_tensor_t<T>&& t)  :
        name{}, shape{}, size{0}, cpu{nullptr}, cuda{nullptr}, binding_index{0}, 
        m_own_memory{false}
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
        m_own_memory = t.m_own_memory;
        // Reset.
        t.name = "";
        t.shape = nvinfer1::DimsNCHW();
        t.size = 0;
        t.cpu = nullptr;
        t.cuda = nullptr;
        t.binding_index = 0;
        t.m_own_memory = false;
        return *this;
    }
    /** Copy constructor: copy pointers, do not own mem. anymore. */
    cuda_tensor_t(const cuda_tensor_t<T>& t) : 
        name{t.name}, shape{t.shape}, size{t.size}, 
        cpu{t.cpu}, cuda{t.cuda}, binding_index{t.binding_index}, 
        m_own_memory{false}
    {}
    /** Copy constructor with batch index. 
     * Return a view of the tensor with batch size=1.
    */
    cuda_tensor_t(const cuda_tensor_t<T>& t, size_t batch_idx) : 
        name{t.name}, 
        shape{1, t.shape.c(), t.shape.h(), t.shape.w()}, 
        size{t.shape.c()*t.shape.h()*t.shape.w()*sizeof(T)}, 
        cpu{t.cpu + batch_idx*shape.c()*shape.h()*shape.w()}, 
        cuda{t.cuda + batch_idx*shape.c()*shape.h()*shape.w()}, 
        binding_index{t.binding_index}, m_own_memory{false}
    {}
    /** Assignement operator=. */
    cuda_tensor_t& operator=(const cuda_tensor_t<T>& t)
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
        m_own_memory = false;
        return *this;
    }
         
    /** Destructor: CUDA free memory.  */
    ~cuda_tensor_t()
    {
        free();
    }

public:
    /** Reshape the tensor. Release memory if necessary. */
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
        if(cpu && m_own_memory) {
            CUDA(cudaFreeHost(cpu));
        }
        cpu = nullptr;
        cuda = nullptr;
        m_own_memory = false;
    }
    /** Is the tensor allocated. */
    bool is_allocated() const {
        return (cpu != nullptr && cuda != nullptr);
    }
    /** Is the memory owned? */
    bool is_memory_owned() const {
        return m_own_memory;
    }

public:
    /** Get a batch view with a given index. */
    cuda_tensor_t batch(size_t batch_idx) const 
    {
        return cuda_tensor_t(*this, batch_idx);
    }
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
    T* cuda_ptr(size_t batch_idx, size_t channel_idx) const {
        return (cuda + batch_idx*shape.c()*shape.h()*shape.w() + channel_idx*shape.h()*shape.w());
    }
    /** Get the cpu pointer, at a given batch index.  */
    T* cpu_ptr(size_t batch_idx=0) const {
        return (cpu + batch_idx*shape.c()*shape.h()*shape.w());
    }
    T* cpu_ptr(size_t batch_idx, size_t channel_idx) const {
        return (cpu + batch_idx*shape.c()*shape.h()*shape.w() + channel_idx*shape.h()*shape.w());
    }
    /** Convert to NVX-CUDA image plane. */
    nvxcu_pitch_linear_image_t nvxcu_image(size_t b_idx, size_t c_idx)
    {
        nvxcu_pitch_linear_image_t image;
        image.base.image_type = NVXCU_PITCH_LINEAR_IMAGE;
        image.base.format = NVXCU_DF_IMAGE_U8;
        image.base.width = shape.w();
        image.base.height = shape.h();
        image.planes[0].dev_ptr = this->cuda_ptr(b_idx, c_idx);
        image.planes[0].pitch_in_bytes = shape.w();
        return image;
    }

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

private:
    // Do I own the memory? Quick way of having shared reference to a tensor.
    bool  m_own_memory;
};

// Common CUDA tensors.
typedef cuda_tensor_t<float>  cuda_tensor;
typedef cuda_tensor_t<float>  cuda_tensor_f;
typedef cuda_tensor_t<double>  cuda_tensor_d;
typedef cuda_tensor_t<float>  cuda_tensor_f32;
typedef cuda_tensor_t<double>  cuda_tensor_f64;

typedef cuda_tensor_t<uint8_t>  cuda_tensor_u8;
typedef cuda_tensor_t<uint16_t>  cuda_tensor_u16;
typedef cuda_tensor_t<uint32_t>  cuda_tensor_u32;
typedef cuda_tensor_t<uint64_t>  cuda_tensor_u64;

typedef cuda_tensor_t<int8_t>  cuda_tensor_i8;
typedef cuda_tensor_t<int16_t>  cuda_tensor_i16;
typedef cuda_tensor_t<int32_t>  cuda_tensor_i32;
typedef cuda_tensor_t<int64_t>  cuda_tensor_i64;

/* ============================================================================
 * tfrt::ros_tensor_t => ROS multi-array msg.
 * ========================================================================== */
template <typename T>
struct ros_tensor_t
{
public:
    /** Value type? */
    typedef typename T::_data_type::value_type  value_type;
    typedef T  array_type;
    
    /** Empty tensor! */
    ros_tensor_t() : name{}, shape{}
    {
        this->init_layout();
    }
    /** Construct from shape. */
    ros_tensor_t(const std::string& _name, const nvinfer1::DimsNCHW& _shape) :
        name{_name}, shape{_shape} 
    {
        this->init_layout();
        array.data.resize(shape.n()*shape.c()*shape.h()*shape.w());
    }
    /** Construct from CUDA tensor. */
    ros_tensor_t(const cuda_tensor_t<value_type>& t) :
        name{t.name}, shape{t.shape} 
    {
        this->init_layout();
        array.data.resize(shape.n()*shape.c()*shape.h()*shape.w());
        // Copy if allocated CPU memory.
        if (t.cpu) {
            this->copy_from(t.cpu);
        }
    }
    /** Construct from EIGEN matrix. */
    template <int R, int C, int M>
    ros_tensor_t(const Eigen::Matrix<value_type, R, C, M>& m) :
        name{"matrix"}, shape{int(m.rows()), int(m.cols()), 1, 1}
    {
        this->init_layout();
        array.data.resize(shape.n()*shape.c()*shape.h()*shape.w());
        this->copy_from(m.data());
    }

    /** Size of the tensor in bytes? */
    size_t size() const {
        return array.data.size() * sizeof(value_type);
    }
    /** Data pointer. */
    const value_type* data() const {
        return array.data.data();
    }
    value_type* data() {
        return array.data.data();
    }

    /** Copy tensor memory from some input (CUDA tensor for e.g.). */
    void copy_from(const void* input) {
        std::memcpy((void*)array.data.data(), input, this->size());
    }
    /** Copy tensor memory to some output (CUDA tensor for e.g.). */
    void copy_to(void* output) {
        std::memcpy(output, (void*)array.data.data(), this->size());
    }
    /** EIGEN tensor map. */
    typename tfrt::nchw<value_type>::tensor_map tensor() const
    {
        // Tensor using existing memory.
        return typename tfrt::nchw<value_type>::tensor_map(
            array.data.data(), shape.n(), shape.c(), shape.h(), shape.w());
    }
    /** Back to multiarray! */
    operator T() {
        return this->array;
    }

private:
    /** Initialize the layout. */
    void init_layout()
    {
        array.layout.dim.push_back(std_msgs::MultiArrayDimension());
        array.layout.dim[0].size = shape.n();
        array.layout.dim[0].stride = shape.n()*shape.c()*shape.w()*shape.h();
        array.layout.dim[0].label = "N";
        
        array.layout.dim.push_back(std_msgs::MultiArrayDimension());
        array.layout.dim[1].size = shape.c();
        array.layout.dim[1].stride = shape.c()*shape.w()*shape.h();
        array.layout.dim[1].label = "C";

        array.layout.dim.push_back(std_msgs::MultiArrayDimension());
        array.layout.dim[2].size = shape.w();
        array.layout.dim[2].stride = shape.w()*shape.h();
        array.layout.dim[2].label = "H";

        array.layout.dim.push_back(std_msgs::MultiArrayDimension());
        array.layout.dim[3].size = shape.h();
        array.layout.dim[3].stride = shape.h();
        array.layout.dim[3].label = "W";
    }

public:
    // Tensor name, shape and size.
    std::string  name;
    nvinfer1::DimsNCHW  shape;
    // ROS multiarray
    T  array;
};

// Common ROS tensors.
typedef ros_tensor_t<std_msgs::Int8MultiArray>  ros_tensor_i8;
typedef ros_tensor_t<std_msgs::Int16MultiArray>  ros_tensor_i16;
typedef ros_tensor_t<std_msgs::Int32MultiArray>  ros_tensor_i32;
typedef ros_tensor_t<std_msgs::Int64MultiArray>  ros_tensor_i64;

typedef ros_tensor_t<std_msgs::UInt8MultiArray>  ros_tensor_u8;
typedef ros_tensor_t<std_msgs::UInt16MultiArray>  ros_tensor_u16;
typedef ros_tensor_t<std_msgs::UInt32MultiArray>  ros_tensor_u32;
typedef ros_tensor_t<std_msgs::UInt64MultiArray>  ros_tensor_u64;

typedef ros_tensor_t<std_msgs::Float32MultiArray>  ros_tensor_f;
typedef ros_tensor_t<std_msgs::Float64MultiArray>  ros_tensor_d;
typedef ros_tensor_t<std_msgs::Float32MultiArray>  ros_tensor_f32;
typedef ros_tensor_t<std_msgs::Float64MultiArray>  ros_tensor_f64;

}

#endif
