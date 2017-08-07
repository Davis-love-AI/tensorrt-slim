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
#ifndef NVX_STABILIZER_HPP
#define NVX_STABILIZER_HPP

#include <memory>
#include <VX/vx.h>

namespace nvx
{
/** Abstract video stabilization class.
 */
class VideoStabilizer
{
public:
/** Common stabilization parameters.
    */
struct VideoStabilizerParams
{
    /** Number of smoothing frames. Taken from the interval [-numOfSmoothingFrames_; numOfSmoothingFrames_] in the current frame's vicinity */
    vx_size  num_smoothing_frames;
    /** Output height after cropping and warp transform. */
    vx_uint32  output_height;
    /** Output width after cropping and warp transform. */
    vx_uint32  output_width;
    /** Crop margin. TODO: not be removed... */
    vx_float32  crop_margin;
    /* Percentage to crop after stabilization, on every side of image. */
    vx_float32  crop_y;
    vx_float32  crop_x;
    vx_float32  crop_scale_y;
    vx_float32  crop_scale_x;

    /** Default constructor taking values specified in app FLAGS. */
    VideoStabilizerParams();
};

static std::unique_ptr<nvx::VideoStabilizer> createImageBasedVStab(
    vx_context context, const VideoStabilizerParams& params = VideoStabilizerParams());

virtual ~VideoStabilizer() {}
/** Initialize the stabilizer with a first frame.  */
virtual void init(vx_image firstFrame) = 0;
/** Initialize the stabilizer with a first frame dimension.  */
virtual void init(vx_uint32 width, vx_uint32 height) = 0;

/** Process a frame. */
virtual void process(vx_image newFrame) = 0;
/** Get the stabilized frame. */
virtual vx_image get_frame_stabilized() const = 0;
/** Get the original frame corresponding. */
virtual vx_image get_frame_original() const = 0;

/** Performance? */
virtual void print_performances() const = 0;
};

/** Init a delay of matrixes with identity matrices.  */
vx_status initDelayOfMatrices(vx_delay delayOfMatrices);
/** Init a delay of images, copying the first image in the collectionl.  */
vx_status initDelayOfImages(vx_context context, vx_delay delayOfImages);
}

#endif
