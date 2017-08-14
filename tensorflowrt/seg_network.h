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
#ifndef TFRT_SEG_NETWORK_H
#define TFRT_SEG_NETWORK_H

#include <tuple>
#include <memory>
#include <NvInfer.h>

#include "network.h"

namespace tfrt
{

class seg_network : public tfrt::network
{
public:
    /** Create segmentation network, with a given name.
     */
    seg_network(std::string name, uint32_t num_classes_seg=2, bool empty_class=true) :
        tfrt::network(name), m_num_classes{num_classes_seg}, m_empty_class{empty_class},
        m_desc_classes{num_classes_seg, "Nothing"} {}

   
public:
    /** Number of classes. */
    uint32_t num_classes() const {
        return m_num_classes;
    }
    /** Description of a class. */
    std::string description(uint32_t idx) const {
        return m_desc_classes[idx];
    }

public:
    /** Inference on a single VX image. */
    void inference(vx_image image);
    /** Inference on two VX images. */
    void inference(vx_image img1, vx_image img2);

    // Getting raw results.
    const tfrt::nhw<uint8_t>::tensor& raw_classes() const {
        return m_rclasses_cached;
    }
    const tfrt::nhw<float>::tensor& raw_scores() const {
        return m_rscores_cached;
    }
    tfrt::nchw<float>::tensor raw_probabilities() const {
        return m_cuda_outputs[0].tensor();
    }

protected:
    /** Initialize the cached tensors. */
    void init_tensors_cached();
    /** Post-processing of outputs. Ugly way! */
    void post_processing();

protected:
    // Number of classes in the model.
    uint32_t  m_num_classes;
    // Empty class in the first coordinate?
    bool  m_empty_class;
    // Segmentation classes descriptions
    std::vector<std::string>  m_desc_classes;

    // Cached result tensors.
    tfrt::nhw<uint8_t>::tensor  m_rclasses_cached;
    tfrt::nhw<float>::tensor  m_rscores_cached;
};

}

#endif
