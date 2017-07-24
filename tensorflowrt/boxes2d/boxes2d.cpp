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

#include <numeric>

#include "boxes2d.h"

namespace tfrt
{
namespace boxes2d
{

/* ============================================================================
 * Bounding boxes 2D
 * ========================================================================== */
bboxes2d::bboxes2d(size_t size) :
    classes{vec_int::Zero(size)}, scores{vec_float::Zero(size)}, boxes{boxes2d::Zero(size, 4)}
{}

void bboxes2d::sort_by_score(bool decreasing)
{
    // Trick: create vector of indexes to sort...
    size_t _size = this->size();
    std::vector<size_t> idxes(_size);
    std::iota(idxes.begin(), idxes.end(), 0);
    // Eigen::VectorXi::LinSpaced(4, 7, 10);
    // std::sort with custom lambda.
    const auto& ref_scores{this->scores};
    if(decreasing) {
        std::sort(idxes.begin(), idxes.end(),
            [&ref_scores](size_t i1, size_t i2) {  return ref_scores[i1] > ref_scores[i2];  });
    }
    else {
        std::sort(idxes.begin(), idxes.end(),
            [&ref_scores](size_t i1, size_t i2) {  return ref_scores[i1] < ref_scores[i2];  });
    }
    // Sort elements from the bboxes2d.
    vec_int _classes{vec_int(_size)};
    vec_float _scores{vec_float(_size)};
    boxes2d _boxes{boxes2d(_size, 4)};
    for(size_t i = 0 ; i < idxes.size() ; ++i) {
        _classes[i] = classes[idxes[i]];
        _scores[i] = scores[idxes[i]];
        _boxes.row(i) = boxes.row(idxes[i]);
    }
    classes.swap(_classes);
    scores.swap(_scores);
    boxes.swap(_boxes);
    // TODO: fix this UGLYYYY implementation using row swap.
    // scores.row(0).swap(scores.row(0));
}


}
}
