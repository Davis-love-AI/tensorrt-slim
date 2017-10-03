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
#include <glog/logging.h>

#include "seg_network.h"
#include "cuda/cudaSegmentation.h"

namespace tfrt
{
// ========================================================================== //
// Segmentation Network.
// ========================================================================== //
void seg_network::seg_cuda_output(const tfrt::cuda_tensor& t)
{
    this->cuda_output(t, 0);
}
void seg_network::init_tensors_cached()
{
    const tfrt::cuda_tensor& cuda_output = m_cuda_outputs[0];
    const auto& oshape = cuda_output.shape;
    if (!m_rclasses_cached.is_allocated() ) {
        m_rclasses_cached = tfrt::cuda_tensor_u8(
            "classes", {oshape.n(), 1, oshape.h(), oshape.w()});
        m_rclasses_cached.allocate();
    }
    if (!m_rscores_cached.is_allocated() ) {
        m_rscores_cached = tfrt::cuda_tensor(
            "scores", {oshape.n(), 1, oshape.h(), oshape.w()});
        m_rscores_cached.allocate();
    }
}
void seg_network::post_processing()
{
    this->init_tensors_cached();
    // For God sake, used a fucking CUDA kernel for that!
    const auto& rtensor = m_cuda_outputs[0].tensor();
    const auto& oshape = m_cuda_outputs[0].shape;
    LOG(INFO) << "SEGNET: post-processing of output with shape: "
        << dims_str(m_cuda_outputs[0].shape);
    // CUDA(cudaDeviceSynchronize());
    for (long n = 0 ; n < rtensor.dimension(0) ; ++n) {
        for (long i = 0 ; i < rtensor.dimension(2) ; ++i) {
            for (long j = 0 ; j < rtensor.dimension(3) ; ++j) {
                uint8_t max_idx = 0;
                float max_score = 0.0f;
                // Channel loop.
                for (long k = 0 ; k < rtensor.dimension(1) ; ++k) {
                    float score = rtensor(n, k, i, j);
                    if (score > max_score) {
                        max_idx = uint8_t(k);
                        max_score = score;
                    }
                    // Save to cached tensors.
                    long idx = n * oshape.h() * oshape.w() + i * oshape.w() + j;
                    if (max_score > m_detection_threshold) {
                        m_rclasses_cached.cpu[idx] = max_idx + int(!m_empty_class);
                        m_rscores_cached.cpu[idx] = max_score;
                    }
                    else {
                        m_rclasses_cached.cpu[idx] = 0;
                        m_rscores_cached.cpu[idx] = max_score;
                    }
                }
            }
        }
    }
    // CUDA(cudaDeviceSynchronize());
    LOG(INFO) << "SEGNET: done with post-processing of output";
}

// void seg_network::inference(vx_image image)
// {
//     LOG(INFO) << "SEGNET: inference with single input.";
//     network::inference(image);
//     // Post-processing of the output: computing classes and scores.
//     this->post_processing();
// }
// void seg_network::inference(const nvx_image_inpatch& image)
// {
//     LOG(INFO) << "SEGNET: inference with single input.";
//     network::inference(image);
//     // Post-processing of the output: computing classes and scores.
//     this->post_processing();
// }

// void seg_network::inference(vx_image img1, vx_image img2)
// {
//     LOG(INFO) << "SEGNET: inference with two inputs.";
//     network::inference(img1, img2);
//     // CUDA(cudaDeviceSynchronize());
//     // this->post_processing();
// }
// void seg_network::inference(const nvx_image_inpatch& img1, const nvx_image_inpatch& img2)
// {
//     LOG(INFO) << "SEGNET: inference with two inputs.";
//     network::inference(img1, img2);
//     // CUDA(cudaDeviceSynchronize());
//     // this->post_processing();
// }

// ========================================================================== //
// Post-segmentation network.
// ========================================================================== //
seg_network_post::seg_network_post() :
    m_seg_outshape{0, 0, 0},
    m_empty_class{false},
    m_detection_threshold{0.0f}
{
}
seg_network_post::seg_network_post(nvinfer1::DimsCHW _seg_outshape, 
        bool _empty_class, float _detection_threshold) :
    m_seg_outshape{_seg_outshape},
    m_empty_class{_empty_class},
    m_detection_threshold{_detection_threshold}
{
    // Allocate temporary buffers.
    m_rclasses_cached = tfrt::cuda_tensor_u8(
        "classes", {1, 1, m_seg_outshape.h(), m_seg_outshape.w()});
    m_rclasses_cached.allocate();
    m_rscores_cached = tfrt::cuda_tensor(
        "scores", {1, 1, m_seg_outshape.h(), m_seg_outshape.w()});
    m_rscores_cached.allocate();

    // Default transformation matrix.
    m_transformation_matrix = tfrt::cuda_tensor("tr_matrix", {1, 1, 3, 3});
    m_transformation_matrix.allocate();
    this->transformation_matrix(tfrt::matrix_33f_rm::Identity());
}
void seg_network_post::apply(const tfrt::cuda_tensor& seg_output_raw, size_t batch_idx)
{
    // CUDA optimized implementation! Finally!
    cuda_seg_post_process(
        seg_output_raw.cuda_ptr(batch_idx), m_rclasses_cached.cuda, m_rscores_cached.cuda, 
        m_transformation_matrix.cuda,
        m_seg_outshape.w(), m_seg_outshape.h(), seg_output_raw.shape.c(),
        m_empty_class, m_detection_threshold);
}

void seg_network_post::transformation_matrix(const tfrt::matrix_33f_rm& m)
{
    // Just copy the memory!
    memcpy(m_transformation_matrix.cpu, m.data(), m_transformation_matrix.size);
}
tfrt::matrix_33f_rm seg_network_post::transformation_matrix() const
{
    // Just copy the memory!
    tfrt::matrix_33f_rm m;
    memcpy(m.data(), m_transformation_matrix.cpu, m_transformation_matrix.size);
    return m;
}

}