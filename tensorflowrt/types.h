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


}

#endif
