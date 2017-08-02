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

#ifndef NVX_VSTAB_NODES_HPP
#define NVX_VSTAB_NODES_HPP

#include <NVX/nvx.h>

#include <algorithm>
#include <Eigen/Dense>

// row-major storage order
typedef Eigen::Matrix<vx_float32, 3, 3, Eigen::RowMajor> Matrix3x3f_rm;
typedef Eigen::Matrix<vx_float32, 3, 4, Eigen::RowMajor> Matrix3x4f_rm;

// Register homographyFilter kernel in OpenVX context
vx_status registerHomographyFilterKernel(vx_context context);

// Create homographyFilter node
vx_node homographyFilterNode(vx_graph graph, vx_matrix input,
                             vx_matrix homography, vx_image image,
                             vx_array mask);


// Register matrixSmoother kernel in OpenVX context
vx_status registerMatrixSmootherKernel(vx_context context);

// Create matrixSmoother node
vx_node matrixSmootherNode(vx_graph graph,
                      vx_delay matrices, vx_matrix smoothed);


// Register truncateStabTransform kernel in OpenVX context
vx_status registerTruncateStabTransformKernel(vx_context context);

/* Create truncateStabTransform node.
 * cropMargin - proportion of the width(height) of the frame
 * that is allowed to be cropped for stabilizing of the frames. The value should be less than 0.5.
 * If cropMargin is negative then the truncation procedure is turned off.
 */
vx_node truncateStabTransformNode(vx_graph graph, vx_matrix stabTransform, vx_matrix truncatedTransform,
                                  vx_image image, vx_scalar cropMargin);

#endif
