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

namespace tfrt
{
void seg_network::init_tensors_cached()
{
    const tfrt::cuda_tensor& cuda_output = m_cuda_outputs[0];
    const auto& oshape = cuda_output.shape;
    if (!m_rclasses_cached.size()) {
        m_rclasses_cached = tfrt::nhw<uint8_t>::tensor(oshape.n(), oshape.h(), oshape.w());
    }
    if (!m_rscores_cached.size()) {
        m_rscores_cached = tfrt::nhw<float>::tensor(oshape.n(), oshape.h(), oshape.w());
    }
}
void seg_network::post_processing()
{
    this->init_tensors_cached();
    // For God sake, used a fucking CUDA kernel for that!
    const auto& rtensor = m_cuda_outputs[0].tensor();
    DLOG(INFO) << "SEGNET: post-processing of output with shape: " << dims_str(m_cuda_outputs[0].shape);
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
                    m_rclasses_cached(n, i, j) = max_idx + int(!m_empty_class);
                    m_rscores_cached(n, i, j) = max_score;
                }
            }
        }
    }
    DLOG(INFO) << "SEGNET: done with post-processing of output";
}

void seg_network::inference(vx_image image)
{
    network::inference(image);
    // Post-processing of the output: computing classes and scores.
    this->post_processing();
}
void seg_network::inference(vx_image img1, vx_image img2)
{
    network::inference(img1, img2);
    CUDA(cudaDeviceSynchronize());
    this->post_processing();
}

}