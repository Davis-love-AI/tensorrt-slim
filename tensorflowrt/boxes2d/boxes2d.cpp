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

}
}
