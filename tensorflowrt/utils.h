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
#include "tensor.h"
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

}

#endif
