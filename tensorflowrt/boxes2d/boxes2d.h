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
#ifndef TFRT_BOXES2D_H
#define TFRT_BOXES2D_H

#include <chrono>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

// tfrt::boxes2d namespace
namespace tfrt
{
namespace boxes2d
{
/** Note: 2D boxes are supposed to be stored using
 * the CV convention ymin, xmin, ymax, xmax.
 */
// Usual types for boxes2d.
typedef Eigen::Array<int, Eigen::Dynamic, 1>  vec_int;
typedef Eigen::Array<float, Eigen::Dynamic, 1>  vec_float;
typedef Eigen::Array<float, Eigen::Dynamic, 4>  boxes2d;

/* ============================================================================
 * Bounding boxes 2D
 * ========================================================================== */
/** Structure representing 2D bounding boxes. Aim to be a complete
 * representation of properties of a collection of boxes, i.e.
 *  - classes
 *  - scores
 *  - boxes2d (coordinates...)
 * Other?
 */
struct bboxes2d
{
    typedef std::chrono::high_resolution_clock clock;

    /** Reference time point of the measurement.  */
    std::chrono::time_point<clock> time;
    /** Vector of classes.  */
    vec_int  classes;
    /** Vector of scores.  */
    vec_float scores;
    /** Array of boxes coordinates. */
    boxes2d  boxes;

public:
    /** Create a collection of certain size. Initialize all to members zeros.
     */
    bboxes2d(size_t size=0);
    /** Inplace sort of 2D bboxes by score (decreasing order by default).  */
    void sort_by_score(bool decreasing=true);

public:
    /** Get the number of bboxes.  */
    size_t size() const {
        assert(classes.size() == scores.size());
        assert(classes.size() == boxes.rows());
        return classes.size();
    }
};

}
}

#endif

// boxes2d operations.
#include "operations.h"
