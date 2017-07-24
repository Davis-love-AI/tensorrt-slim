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
#ifndef TFRT_BOXES2D_OPS_H
#define TFRT_BOXES2D_OPS_H

#include "boxes2d.h"

namespace tfrt
{
namespace boxes2d
{

/** Rescaling of 2D boxes. rx, ry corresponding to x and y scaling factors.
 */
inline void rescale(boxes2d& boxes, float ry, float rx)
{
    boxes.col(0) *= ry;
    boxes.col(1) *= rx;
    boxes.col(2) *= ry;
    boxes.col(3) *= rx;
}


}
}

#endif
