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

#include "seg_network.h"

namespace tfrt
{
void seg_network::init_tensors_cached()
{
    const tfrt::cuda_tensor& cuda_output = m_cuda_outputs[0];
    const auto& oshape = cuda_output.shape;
    if (m_rclasses_cached.size() != cuda_output.size) {
        m_rclasses_cached = tfrt::nchw<uint8_t>::tensor(oshape.n(), oshape.c(), oshape.h(), oshape.w());
    }
    if (m_rscores_cached.size() != cuda_output.size) {
        m_rscores_cached = tfrt::nchw<float>::tensor(oshape.n(), oshape.c(), oshape.h(), oshape.w());
    }
}
void seg_network::inference(vx_image image)
{
    network::inference(image);
    // Post-processing of the output.

}
void seg_network::inference(vx_image img1, vx_image img2)
{
    network::inference(img1, img2);

    

}

}