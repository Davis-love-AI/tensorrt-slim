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

#include "utils.h"
#include "network.h"

namespace tfrt
{
// ========================================================================== //
// Segmentation Network.
// ========================================================================== //
class seg_network : public tfrt::network
{
public:
    /** Create segmentation network, with a given name and a number of classes.
     * 'empty_class': if the network predict the 0-index empty class?
     */
    seg_network(std::string name, uint32_t num_classes_seg=2, bool empty_class=true) :
        tfrt::network(name), m_num_classes{num_classes_seg},
        m_empty_class{empty_class},
        m_detection_threshold{0.0f},
        m_desc_classes{num_classes_seg, "Nothing"}
    {}


public:
    /** Number of classes. */
    uint32_t num_classes() const {
        return m_num_classes;
    }
    /** Description of a class. */
    std::string description(uint32_t idx) const {
        return m_desc_classes[idx];
    }
    /** Empty class? */
    bool empty_class() {
        return m_empty_class;
    }
    void empty_class(bool v) {
        m_empty_class = v;
    }
    /** Detection threshold. */
    float detection_threshold() {
        return m_detection_threshold;
    }
    void detection_threshold(float v) {
        m_detection_threshold = v;
    }

public:
    /** Inference on a single VX image. */
    void inference(vx_image image);
    void inference(const nvx_image_inpatch& image);
    /** Inference on two VX images. */
    void inference(vx_image img1, vx_image img2);
    void inference(const nvx_image_inpatch& img1, const nvx_image_inpatch& img2);

    // Getting raw results.
    const tfrt::cuda_tensor_u8& raw_classes() const {
        return m_rclasses_cached;
    }
    const tfrt::cuda_tensor& raw_scores() const {
        return m_rscores_cached;
    }
    const tfrt::cuda_tensor& raw_probabilities() const {
        return m_cuda_outputs[0];
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
    // Detection threshold.
    float  m_detection_threshold;
    // Segmentation classes descriptions
    std::vector<std::string>  m_desc_classes;

    // Cached result tensors.
    tfrt::cuda_tensor_u8  m_rclasses_cached;
    tfrt::cuda_tensor  m_rscores_cached;
};

// ========================================================================== //
// Post-processing of segmentation.
// ========================================================================== //
/** Post-processing of segmentation raw output. 
* For now, only batch of 1 supported.
*/
class seg_network_post
{
public:
    /** Empty constructor, nothing allocated. */
    seg_network_post();
    /** Build, with all parameters. */
    seg_network_post(nvinfer1::DimsCHW _seg_outshape, 
        bool _empty_class, float _detection_threshold);

    /** Apply the post-processing algorithm. */
    void apply(const tfrt::cuda_tensor& seg_output_raw, size_t batch_idx);

public:
    /** Empty class? */
    bool empty_class() {
        return m_empty_class;
    }
    void empty_class(bool v) {
        m_empty_class = v;
    }
    /** Detection threshold. */
    float detection_threshold() {
        return m_detection_threshold;
    }
    void detection_threshold(float v) {
        m_detection_threshold = v;
    }

private:
    /** Segmentation output shape. */
    nvinfer1::DimsCHW  m_seg_outshape;
    /** Empty class in the first coordinate? */
    bool  m_empty_class;
    /** Detection threshold. */
    float  m_detection_threshold;
    
    // Cached result tensors.
    tfrt::cuda_tensor_u8  m_rclasses_cached;
    tfrt::cuda_tensor  m_rscores_cached;
};

}

#endif
