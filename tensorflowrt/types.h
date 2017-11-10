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
#ifndef TFRT_TYPES_H
#define TFRT_TYPES_H

// Eigen and OpenCV headers
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <opencv2/core/core.hpp>

#include <NvInfer.h>

namespace tfrt
{
// ========================================================================== //
// EIGEN tensors.
// ========================================================================== //
/** C type of tensors. */
template <typename T, typename IndexType=Eigen::DenseIndex>
struct c {
    typedef Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType> tensor;
    typedef Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType> const_tensor;

    typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>,
                             Eigen::Aligned> tensor_map;
    typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType>,
                             Eigen::Aligned> const_tensor_map;

    static nvinfer1::DimsCHW shape(const tensor& t) {
        return nvinfer1::DimsCHW{int(t.dimension(0)), 1, 1};
    }
};
/** HW type of tensors. */
template <typename T, typename IndexType=Eigen::DenseIndex>
struct hw {
    typedef Eigen::Tensor<T, 2, Eigen::RowMajor, IndexType> tensor;
    typedef Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType> const_tensor;

    typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, IndexType>,
                             Eigen::Aligned> tensor_map;
    typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType>,
                             Eigen::Aligned> const_tensor_map;

    static nvinfer1::DimsCHW shape(const tensor& t) {
        return nvinfer1::DimsCHW{1, int(t.dimension(0)), int(t.dimension(1))};
    }
};
/** CHW type of tensors. */
template <typename T, typename IndexType=Eigen::DenseIndex>
struct chw {
    typedef Eigen::Tensor<T, 3, Eigen::RowMajor, IndexType> tensor;
    typedef Eigen::Tensor<const T, 3, Eigen::RowMajor, IndexType> const_tensor;

    typedef Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor, IndexType>,
                             Eigen::Aligned> tensor_map;
    typedef Eigen::TensorMap<Eigen::Tensor<const T, 3, Eigen::RowMajor, IndexType>,
                             Eigen::Aligned> const_tensor_map;

    static nvinfer1::DimsCHW shape(const tensor& t) {
        return nvinfer1::DimsCHW{
            int(t.dimension(0)), int(t.dimension(1)), int(t.dimension(2))};
    }
};
/** NHW type of tensors. */
template <typename T, typename IndexType=Eigen::DenseIndex>
struct nhw {
    typedef Eigen::Tensor<T, 3, Eigen::RowMajor, IndexType> tensor;
    typedef Eigen::Tensor<const T, 3, Eigen::RowMajor, IndexType> const_tensor;

    typedef Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor, IndexType>,
                             Eigen::Aligned> tensor_map;
    typedef Eigen::TensorMap<Eigen::Tensor<const T, 3, Eigen::RowMajor, IndexType>,
                             Eigen::Aligned> const_tensor_map;

    static nvinfer1::DimsCHW shape(const tensor& t) {
        return nvinfer1::DimsCHW{
            int(t.dimension(0)), int(t.dimension(1)), int(t.dimension(2))};
    }
};
/** NCHW type of tensors. */
template <typename T=float, typename IndexType=Eigen::DenseIndex>
struct nchw {
    typedef Eigen::Tensor<T, 4, Eigen::RowMajor, IndexType> tensor;
    typedef Eigen::Tensor<const T, 4, Eigen::RowMajor, IndexType> const_tensor;

    typedef Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor, IndexType>,
                             Eigen::Aligned> tensor_map;
    typedef Eigen::TensorMap<Eigen::Tensor<const T, 4, Eigen::RowMajor, IndexType>,
                             Eigen::Aligned> const_tensor_map;
    
    static nvinfer1::DimsNCHW shape(const tensor& t) {
        return nvinfer1::DimsNCHW{
            int(t.dimension(0)), int(t.dimension(1)), int(t.dimension(2)), int(t.dimension(3))};
    }
};
/** NACHW type of tensors.
 * A stands for the anchors dimension.
 */
template <typename T=float, typename IndexType=Eigen::DenseIndex>
struct nachw {
    typedef Eigen::Tensor<T, 5, Eigen::RowMajor, IndexType> tensor;
    typedef Eigen::Tensor<const T, 5, Eigen::RowMajor, IndexType> const_tensor;

    typedef Eigen::TensorMap<Eigen::Tensor<T, 5, Eigen::RowMajor, IndexType>,
                             Eigen::Aligned> tensor_map;
    typedef Eigen::TensorMap<Eigen::Tensor<const T, 5, Eigen::RowMajor, IndexType>,
                             Eigen::Aligned> const_tensor_map;
};


// ========================================================================== //
// EIGEN matrices.
// ========================================================================== //
typedef Eigen::Matrix<float, 2, 2, Eigen::RowMajor>  matrix_22f_rm;
typedef Eigen::Matrix<float, 2, 3, Eigen::RowMajor>  matrix_23f_rm;
typedef Eigen::Matrix<float, 3, 2, Eigen::RowMajor>  matrix_32f_rm;

typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor>  matrix_33f_rm;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor>  matrix_44f_rm;
typedef Eigen::Matrix<float, 3, 4, Eigen::RowMajor>  matrix_34f_rm;
typedef Eigen::Matrix<float, 4, 3, Eigen::RowMajor>  matrix_43f_rm;

typedef Eigen::Matrix<float, 5, 5, Eigen::RowMajor>  matrix_55f_rm;
typedef Eigen::Matrix<float, 6, 6, Eigen::RowMajor>  matrix_66f_rm;
typedef Eigen::Matrix<float, 7, 7, Eigen::RowMajor>  matrix_77f_rm;
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor>  matrix_88f_rm;

typedef Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>  matrix_d2f_rm;
typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>  matrix_d3f_rm;
typedef Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor>  matrix_d4f_rm;

typedef Eigen::Matrix<float, 3, 3, Eigen::ColMajor>  matrix_33f;
typedef Eigen::Matrix<float, 4, 4, Eigen::ColMajor>  matrix_44f;
typedef Eigen::Matrix<float, 3, 4, Eigen::ColMajor>  matrix_34f;
typedef Eigen::Matrix<float, 4, 3, Eigen::ColMajor>  matrix_43f;

typedef Eigen::Matrix<float, 5, 5, Eigen::ColMajor>  matrix_55f;
typedef Eigen::Matrix<float, 6, 6, Eigen::ColMajor>  matrix_66f;
typedef Eigen::Matrix<float, 7, 7, Eigen::ColMajor>  matrix_77f;
typedef Eigen::Matrix<float, 8, 8, Eigen::ColMajor>  matrix_88f;


typedef Eigen::Matrix<float, 2, 1>  vector_2f;
typedef Eigen::Matrix<float, 3, 1>  vector_3f;
typedef Eigen::Matrix<float, 4, 1>  vector_4f;
typedef Eigen::Matrix<float, 5, 1>  vector_5f;
typedef Eigen::Matrix<float, 6, 1>  vector_6f;
typedef Eigen::Matrix<float, 7, 1>  vector_7f;
typedef Eigen::Matrix<float, 8, 1>  vector_8f;

typedef Eigen::Matrix<float, Eigen::Dynamic, 1>  vector_df;

typedef Eigen::Matrix<float, Eigen::Dynamic, 2>  matrix_d2f;
typedef Eigen::Matrix<float, Eigen::Dynamic, 3>  matrix_d3f;
typedef Eigen::Matrix<float, Eigen::Dynamic, 4>  matrix_d4f;

/** Old stuff, to remove.  */
typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor>  Matrix3f;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor>  Matrix4f;

typedef Eigen::Matrix<float, 3, 4, Eigen::RowMajor>  Matrix34f;
typedef Eigen::Matrix<float, 4, 3, Eigen::RowMajor>  Matrix43f;

typedef Eigen::Matrix<float, 3, 1>  Vector3f;
typedef Eigen::Matrix<float, 4, 1>  Vector4f;

// ========================================================================== //
// EIGEN tensor/matrix => OpenCV mat.
// ========================================================================== //
inline cv::Mat tensor_to_cvmat(tfrt::nchw<uint8_t>::tensor& t, size_t n=0, size_t c=0)
{
    auto shape = tfrt::nchw<uint8_t>::shape(t);
    uint8_t* ptr = t.data() + n*shape.h()*shape.w()*shape.c() + c*shape.h()*shape.w();
    return cv::Mat(shape.h(), shape.w(), CV_8UC1, ptr);
}
inline cv::Mat tensor_to_cvmat(tfrt::chw<uint8_t>::tensor& t)
{
    auto shape = tfrt::chw<uint8_t>::shape(t);
    uint8_t* ptr = t.data();
    return cv::Mat(shape.h(), shape.w(), CV_8UC1, ptr);
}

inline cv::Mat tensor_to_cvmat(tfrt::nchw<float>::tensor& t, size_t n=0, size_t c=0)
{
    auto shape = tfrt::nchw<float>::shape(t);
    float* ptr = t.data() + n*shape.h()*shape.w()*shape.c() + c*shape.h()*shape.w();
    return cv::Mat(shape.h(), shape.w(), CV_32FC1, ptr);
}
inline cv::Mat tensor_to_cvmat(tfrt::chw<float>::tensor& t)
{
    auto shape = tfrt::chw<float>::shape(t);
    float* ptr = t.data();
    return cv::Mat(shape.h(), shape.w(), CV_32FC1, ptr);
}

}

#endif
