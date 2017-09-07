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
/* Specific features necessary for Jetson TX1/2 compatibility.
 * In particular, implements backward compatibility with TensorRT 1.0.
 */
#ifndef TFRT_JETSON_H
#define TFRT_JETSON_H

#include <cmath>
#include <string>
#include <sstream>
#include <NvInfer.h>

/* ============================================================================
 * Backport TensorRT2 dims to TensorRT1.
 * ========================================================================== */
#ifndef NV_TENSORRT_MAJOR

namespace nvinfer1
{
/** Host memory object. */
typedef std::stringstream IHostMemory;

/** Dimension types. */
enum class DimensionType : int
{
    kSPATIAL = 0,			//!< elements correspond to different spatial data
    kCHANNEL = 1,			//!< elements correspond to different channels
    kINDEX = 2,				//!< elements correspond to different batch index
    kSEQUENCE = 3			//!< elements correspond to different sequence values
};
/** Dimensions classes. */
class Dims
{
public:
    static const int MAX_DIMS = 8;			//!< the maximum number of dimensions supported for a tensor
    int nbDims;								//!< the number of dimensions
    int d[MAX_DIMS];					//!< the extent of each dimension
    DimensionType type[MAX_DIMS];			//!< the type of each dimension
};
class DimsHW : public Dims
{
public:
    DimsHW()
    {
        nbDims = 2;
        type[0] = type[1] = DimensionType::kSPATIAL;
        d[0] = d[1] = 0;
    }
    DimsHW(int height, int width)
    {
        nbDims = 2;
        type[0] = type[1] = DimensionType::kSPATIAL;
        d[0] = height;
        d[1] = width;
    }
    DimsHW(const nvinfer1::Dims2& dims)
    {
        nbDims = 2;
        type[0] = type[1] = DimensionType::kSPATIAL;
        d[0] = dims.h;
        d[1] = dims.w;
    }
    int& h() { return d[0]; }
    int h() const { return d[0]; }
    int& w() { return d[1]; }
    int w() const { return d[1]; }

    // Implicit Dims2 conversion.
    // operator Dims2() const {
    //     return Dims2{d[0], d[1]};
    // }
};

class DimsCHW : public Dims
{
public:
    DimsCHW()
    {
        nbDims = 3;
        type[0] = DimensionType::kCHANNEL;
        type[1] = type[2] = DimensionType::kSPATIAL;
        d[0] = d[1] = d[2] = 0;
    }
    DimsCHW(int channels, int height, int width)
    {
        nbDims = 3;
        type[0] = DimensionType::kCHANNEL;
        type[1] = type[2] = DimensionType::kSPATIAL;
        d[0] = channels;
        d[1] = height;
        d[2] = width;
    }
    DimsCHW(const nvinfer1::Dims3& dims)
    {
        nbDims = 3;
        type[0] = DimensionType::kCHANNEL;
        type[1] = type[2] = DimensionType::kSPATIAL;
        d[0] = dims.c;
        d[1] = dims.h;
        d[2] = dims.w;
    }

    int& c() { return d[0]; }
    int c() const { return d[0]; }
    int& h() { return d[1]; }
    int h() const { return d[1]; }
    int& w() { return d[2]; }
    int w() const { return d[2]; }

    // Implicit Dims3 conversion.
    // operator Dims3() const {
    //     return Dims3{d[0], d[1], d[2]};
    // }
};

class DimsNCHW : public Dims
{
public:
    /**
    * \brief construct an empty DimsNCHW object
    */
    DimsNCHW()
    {
        nbDims = 4;
        type[0] = DimensionType::kINDEX;
        type[1] = DimensionType::kCHANNEL;
        type[2] = type[3] = DimensionType::kSPATIAL;
        d[0] = d[1] = d[2] = d[3] = 0;
    }
    DimsNCHW(int batchSize, int channels, int height, int width)
    {
        nbDims = 4;
        type[0] = DimensionType::kINDEX;
        type[1] = DimensionType::kCHANNEL;
        type[2] = type[3] = DimensionType::kSPATIAL;
        d[0] = batchSize;
        d[1] = channels;
        d[2] = height;
        d[3] = width;
    }
    DimsNCHW(const nvinfer1::Dims4& dims)
    {
        nbDims = 4;
        type[0] = DimensionType::kINDEX;
        type[1] = DimensionType::kCHANNEL;
        type[2] = type[3] = DimensionType::kSPATIAL;
        d[0] = dims.n;
        d[1] = dims.c;
        d[2] = dims.h;
        d[3] = dims.w;
    }

    int& n() { return d[0]; }
    int n() const { return d[0]; }
    int& c() { return d[1]; }
    int c() const { return d[1]; }
    int& h() { return d[2]; }
    int h() const { return d[2]; }
    int& w() { return d[3]; }
    int w() const { return d[3]; }

    // Implicit Dims4 conversion.
    // operator Dims4() const {
    //     return Dims4{d[0], d[1], d[2], d[3]};
    // }
};

}

/* ============================================================================
 * Backport some dims utils TensorRT1.
 * ========================================================================== */
namespace tfrt
{
inline int dims_channels(nvinfer1::Dims2 dims)
{
    return 1;
}
inline int dims_channels(nvinfer1::Dims3 dims)
{
    return dims.c;
}
inline int dims_channels(nvinfer1::Dims4 dims)
{
    return dims.c;
}
inline std::string dims_str(nvinfer1::Dims2 dims)
{
    std::ostringstream oss;
    oss << "[" << dims.h << ", " << dims.w << "]";
    return oss.str();
}
inline std::string dims_str(nvinfer1::Dims3 dims)
{
    std::ostringstream oss;
    oss << "[" << dims.c << ", " << dims.h << ", " << dims.w << "]";
    return oss.str();
}
inline std::string dims_str(nvinfer1::Dims4 dims)
{
    std::ostringstream oss;
    oss << "[" << dims.n << ", " << dims.c << ", " << dims.h << ", " << dims.w << "]";
    return oss.str();
}
}

inline nvinfer1::Dims2 DIMRT(const nvinfer1::DimsHW& dims)
{
    return nvinfer1::Dims2{.h=dims.h(), .w=dims.w()};
}
inline nvinfer1::Dims3 DIMRT(const nvinfer1::DimsCHW& dims)
{
    return nvinfer1::Dims3{.c=dims.c(), .h=dims.h(), .w=dims.w()};
}
inline nvinfer1::Dims4 DIMRT(const nvinfer1::DimsNCHW& dims)
{
    return nvinfer1::Dims4{.n=dims.n(), .c=dims.c(), .h=dims.h(), .w=dims.w()};
}

#endif

/* ============================================================================
 * Compatibility API for TensorRT2.
 * ========================================================================== */
#ifdef NV_TENSORRT_MAJOR

inline nvinfer1::DimsHW DIMRT(const nvinfer1::DimsHW& dims)
{
    return dims;
}
inline nvinfer1::DimsCHW DIMRT(const nvinfer1::DimsCHW& dims)
{
    return dims;
}
inline nvinfer1::DimsNCHW DIMRT(const nvinfer1::DimsNCHW& dims)
{
    return dims;
}

#endif

namespace nvinfer1
{
/** Additional dimension object. */
class DimsNACHW : public Dims
{
public:
    /**
    * \brief construct an empty DimsNCHW object
    */
    DimsNACHW()
    {
        nbDims = 5;
        type[0] = DimensionType::kINDEX;
        type[1] = DimensionType::kCHANNEL;
        type[2] = DimensionType::kCHANNEL;
        type[3] = type[4] = DimensionType::kSPATIAL;
        d[0] = d[1] = d[2] = d[3] = d[4] = 0;
    }
    DimsNACHW(int batchSize, int anchors, int channels, int height, int width)
    {
        nbDims = 5;
        type[0] = DimensionType::kINDEX;
        type[1] = DimensionType::kCHANNEL;
        type[2] = DimensionType::kCHANNEL;
        type[3] = type[4] = DimensionType::kSPATIAL;
        d[0] = batchSize;
        d[1] = anchors;
        d[2] = channels;
        d[3] = height;
        d[4] = width;
    }

    int& n() { return d[0]; }
    int n() const { return d[0]; }
    int& a() { return d[1]; }
    int a() const { return d[1]; }
    int& c() { return d[2]; }
    int c() const { return d[2]; }
    int& h() { return d[3]; }
    int h() const { return d[3]; }
    int& w() { return d[4]; }
    int w() const { return d[4]; }
};

class DimsC : public Dims
{
public:
    DimsC()
    {
        nbDims = 1;
        type[0] = DimensionType::kCHANNEL;
        d[0] = 0;
    }
    DimsC(int channels)
    {
        nbDims = 1;
        type[0] = DimensionType::kCHANNEL;
        d[0] = channels;
    }
    int& c() { return d[0]; }
    int c() const { return d[0]; }
};

}

#endif