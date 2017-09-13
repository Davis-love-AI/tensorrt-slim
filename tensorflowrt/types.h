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

#include <unsupported/Eigen/CXX11/Tensor>

namespace tfrt
{
/** C type of tensors. */
template <typename T, typename IndexType=Eigen::DenseIndex>
struct c {
    typedef Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType> tensor;
    typedef Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType> const_tensor;

    typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>,
                             Eigen::Aligned> tensor_map;
    typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType>,
                             Eigen::Aligned> const_tensor_map;
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


/** EIGEN typedefs
 * TODO: naming convention + column major. 
 */
typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor>  Matrix3f;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor>  Matrix4f;

typedef Eigen::Matrix<float, 3, 4, Eigen::RowMajor>  Matrix34f;
typedef Eigen::Matrix<float, 4, 3, Eigen::RowMajor>  Matrix43f;

typedef Eigen::Matrix<float, 3, 1>  Vector3f;
typedef Eigen::Matrix<float, 4, 1>  Vector4f;


}

#endif
