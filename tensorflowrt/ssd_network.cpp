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

#include "utils.h"
#include "ssd_network.h"
#include "cuda/cudaImageNet.h"

namespace tfrt
{

/* ============================================================================
 * tfrt::ssd_anchor2d methods.
 * ========================================================================== */
ssd_anchor2d::ssd_anchor2d(const tfrt_pb::ssd_anchor2d& anchor) :
    size{anchor.size()},
    scales{anchor.scales().begin(), anchor.scales().end()}
{}

/* ============================================================================
 * tfrt::ssd_feature methods.
 * ========================================================================== */
ssd_feature::ssd_feature(const tfrt_pb::ssd_feature& feature) :
    name{feature.name()}, fullname{feature.fullname()},
    shape{dims_pb(feature.shape())},
    anchors2d{feature.anchors2d().begin(), feature.anchors2d().end()},
    outputs{nullptr, nullptr, nullptr, nullptr}
{}

tfrt::nachw<float>::tensor ssd_feature::predictions2d() const
{
    DLOG(INFO) << "Reshaping 2D predictions into NACHW tensor.";
    // Eigen::DSizes<long int, 4>
    // Reshape NCHW tensor. Should not require memory reallocation.
    size_t nanchors2d = this->num_anchors2d_total();
    auto t = this->outputs.predictions2d->tensor();
    std::array<long, 5> shape{
        t.dimension(0), long(nanchors2d), long(t.dimension(1) / nanchors2d),
        t.dimension(2), t.dimension(3)};
    tfrt::nachw<float>::tensor a = t.reshape(shape);
    return a;
}
tfrt::nachw<float>::tensor ssd_feature::boxes2d() const
{
    DLOG(INFO) << "Reshaping 2D boxes into NACHW tensor.";
    // Reshape NCHW tensor. Should not require memory reallocation.
    size_t nanchors2d = this->num_anchors2d_total();
    auto t = this->outputs.boxes2d->tensor();
    std::array<long, 5> shape{
        t.dimension(0), long(nanchors2d), 4, t.dimension(2), t.dimension(3)};
    tfrt::nachw<float>::tensor a = t.reshape(shape);
    return a;
}

/* ============================================================================
 * tfrt::ssd_network methods.
 * ========================================================================== */
ssd_network::~ssd_network()
{
}
size_t ssd_network::nb_features() const
{
    return m_pb_ssd_network->features_size();
}
void ssd_network::clear_cache()
{
    m_cached_features.clear();
}

const std::vector<ssd_feature>& ssd_network::features() const
{
    if(m_cached_features.size() == 0) {
        // Copy from protobuf objects.
        std::vector<ssd_feature> features{
            m_pb_ssd_network->features().begin(), m_pb_ssd_network->features().end()};
        // A bit hacky here...
        for(size_t i = 0 ; i < features.size() ; ++i) {
            auto& fout = features[i].outputs;
            auto& pb_feat = m_pb_ssd_network->features(i);
            auto& pb_fout = pb_feat.outputs();
            // Find CUDA tensors.
            fout.predictions2d = this->find_cuda_output(pb_fout.predictions2d());
            fout.boxes2d = this->find_cuda_output(pb_fout.boxes2d());
            fout.predictions3d = this->find_cuda_output(pb_fout.predictions3d());
            fout.boxes3d = this->find_cuda_output(pb_fout.boxes3d());
        }
        m_cached_features = features;
    }
    return m_cached_features;
}

bool ssd_network::load_weights(const std::string& filename)
{
    // Free everything!
    m_cached_features.empty();

    LOG(INFO) << "Loading SSD network parameters and weights from: " << filename;
    bool r = parse_protobuf(filename, m_pb_ssd_network.get());
    // Hacky swaping!
    m_pb_network.reset(m_pb_ssd_network->release_network());
    return r;
}


tfrt::boxes2d::bboxes2d ssd_network::raw_detect2d(
        float* rgba, uint32_t height, uint32_t width, float threshold, size_t max_detections)
{
    CHECK(rgba) << "Invalid image buffer.";
    CHECK(height) << "Invalid image height.";
    CHECK(width) << "Invalid image width.";
    // Downsample and convert to RGB.
    cudaError_t r = cudaPreImageNet((float4*)rgba, width, height,
        m_cuda_input.cuda, m_cuda_input.shape.w(), m_cuda_input.shape.h());
    CHECK_EQ(r, cudaSuccess) << "Failed to resize image to ImageNet network input shape."
        << "CUDA error: " << r;

    // Execute TensorRT network (batch size = 1) TODO.
    DLOG(INFO) << "Raw 2D detections from SSD network.";
    size_t num_batches = 1;
    m_nv_context->execute(num_batches, (void**)m_cached_bindings.data());

    // Post-processing of outputs of every feature layer.
    DLOG(INFO) << "Post-processing of SSD raw outputs, selecting 2D boxes. "
        << "Max detections: " << max_detections << " Threshold: " << threshold;
    const auto& features = this->features();
    size_t batch = 0;
    size_t bboxes2d_idx = 0;
    tfrt::boxes2d::bboxes2d  bboxes2d{max_detections};
    for(auto& f : features) {
        DLOG(INFO) << "Extracting raw 2D boxes from feature: " << f.name;
        // Get the Eigen output tensors.
        tfrt::nachw<float>::tensor pred2d = f.predictions2d();
        tfrt::nachw<float>::tensor boxes2d = f.boxes2d();
        // Fill 2D bounding boxes with raw values. Stop at max detections.
        this->fill_bboxes_2d(pred2d, boxes2d, threshold, max_detections,
                             batch, bboxes2d_idx, bboxes2d);
    }
    // Sort by decreasing score.
    DLOG(INFO) << "Sort SSD raw 2D boxes by decreasing score.";
    bboxes2d.sort_by_score(true);
    // Simple post-processing of outputs of every feature layer.
    return bboxes2d;
}

void ssd_network::fill_bboxes_2d(
    const tfrt::nachw<float>::tensor& predictions2d,
    const tfrt::nachw<float>::tensor& boxesd2d,
    float threshold, size_t max_detections, size_t batch,
    size_t& bboxes2d_idx, tfrt::boxes2d::bboxes2d& bboxes2d) const
{
    // No silver bullet here! Have to go the hard loop-way!
    // size_t idx{bboxes2d_idx};
    float y, x, h, w;
    // Loop on  height, width and anchors dimensions.
    for(long i = 0 ; i < predictions2d.dimension(3) ; ++i) {
        for(long j = 0 ; j < predictions2d.dimension(4) ; ++j) {
            for(long k = 0 ; k < predictions2d.dimension(1) ; ++k) {
                // Check index did not reach bounds...
                if(bboxes2d_idx >= max_detections-1) {
                    return;
                }
                // Initialize with no-object class.
                size_t max_idx = 0;
                float max_pred = predictions2d(batch, k, 0, i, j);
                // Loop over the classes!
                for(long l = 1 ; l < predictions2d.dimension(2) ; ++l) {
                    if(predictions2d(batch, k, l, i, j) > max_pred) {
                        max_idx = l;
                        max_pred = predictions2d(batch, k, l, i, j);
                    }
                }
                // Assign bounding box.
                if(max_idx > 0 && max_pred > threshold) {
                    bboxes2d.classes(bboxes2d_idx) = max_idx;
                    bboxes2d.scores(bboxes2d_idx) = max_pred;
                    // Recall: raw output in y, x, h, w format.
                    y = boxesd2d(batch, k, 0, i, j);
                    x = boxesd2d(batch, k, 1, i, j);
                    h = boxesd2d(batch, k, 2, i, j);
                    w = boxesd2d(batch, k, 3, i, j);
                    // Convert to ymin, xmin, ymax, xmax.
                    bboxes2d.boxes(bboxes2d_idx, 0) = y - h / 2.;
                    bboxes2d.boxes(bboxes2d_idx, 1) = x - w / 2.;
                    bboxes2d.boxes(bboxes2d_idx, 2) = y + h / 2.;
                    bboxes2d.boxes(bboxes2d_idx, 3) = x + w / 2.;
                    bboxes2d_idx++;
                }
            }
        }
    }
}

void ssd_network::draw_bboxes_2d(float* input, float* output,
    uint32_t width, uint32_t height, const tfrt::boxes2d::bboxes2d& bboxes2d) const
{

}

tfrt::cuda_tensor& ssd_network::colors_2d() {
    if(!m_cuda_colors_2d.is_allocated()) {
        this->generate_colors_2d();
    }
    return m_cuda_colors_2d;
}
tfrt::cuda_tensor& ssd_network::colors_3d() {
    if(!m_cuda_colors_3d.is_allocated()) {
        this->generate_colors_3d();
    }
    return m_cuda_colors_3d;
}
tfrt::cuda_tensor& ssd_network::colors_seg() {
    if(!m_cuda_colors_3d.is_allocated()) {
        this->generate_colors_seg();
    }
    return m_cuda_colors_seg;
}

void ssd_network::generate_colors_2d()
{
    // Allocate color tensor.
    auto num_classes = this->num_classes_2d();
    m_cuda_colors_2d.shape = {num_classes, 4, 1, 1};
    m_cuda_colors_2d.allocate();
    // Default colors...
    for(int n = 0 ; n < num_classes ; n++) {
        m_cuda_colors_2d.cpu[n*4+0] = 0.0f;	    // r
        m_cuda_colors_2d.cpu[n*4+1] = 255.0f;	// g
        m_cuda_colors_2d.cpu[n*4+2] = 175.0f;	// b
        m_cuda_colors_2d.cpu[n*4+3] = 100.0f;	// a
    }
    // Use MS COCO colors???
}
void ssd_network::generate_colors_3d()
{
    // Allocate color tensor.
    m_cuda_colors_3d.shape = {this->num_classes_3d(), 4, 1, 1};

}
void ssd_network::generate_colors_seg()
{

}


// void test_values(float* ptensor, const tfrt::nachw<float>::tensor& tensor)
// {
//     LOG(INFO) << "Looking at values...";
//     LOG(INFO) << "#0 " << ptensor[0] << " | " << tensor(0,0,0,0,0);
//     LOG(INFO) << "#1 " << ptensor[1] << " | " << tensor(0,0,0,0,1);
//     LOG(INFO) << "#2 " << ptensor[2];
//     LOG(INFO) << "#3 " << ptensor[3];
// }


}
