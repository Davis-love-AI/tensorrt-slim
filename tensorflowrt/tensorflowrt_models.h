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
#ifndef TFRT_MODELS
#define TFRT_MODELS

// ========================================================================== //
// ImageNet models
// ========================================================================== //
#include "nets/inception1.h"
#include "nets/inception2.h"
#include "nets/nasnet.h"

#include "nets/resnet_v1.h"
#include "nets/resnext.h"
#include "nets/mobilenets.h"

// ========================================================================== //
// SSD models.
// ========================================================================== //


// ========================================================================== //
// Segmentation models
// ========================================================================== //
#include "models/ssd_inception2_v0.h"
#include "models/seg_inception2_v0.h"
#include "models/seg_inception2_2x2.h"
#include "models/seg_inception2_v1.h"
#include "models/seg_inception2_v1_5x5.h"
#include "models/seg_inception2_logits_v1.h"
// #include "inception2.h"

namespace tfrt
{
/** Neural Nets factory. */
std::unique_ptr<tfrt::network>&& nets_factory(const std::string& name);

}

#endif
