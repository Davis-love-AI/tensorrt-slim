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
#ifndef TFRT_UTILS_H
#define TFRT_UTILS_H

#include <glog/logging.h>
#include <NvInfer.h>

#include "tfrt_jetson.h"

namespace tfrt
{
/* ============================================================================
 * Dim utils.
 * ========================================================================== */
inline int dims_channels(nvinfer1::Dims dims)
{
    // Suppose NCHW, CHW or HW format...
    if(dims.nbDims >= 4) {
        return dims.d[1];
    }
    else if(dims.nbDims == 3) {
        return dims.d[0];
    }
    return 1;
}
inline std::string dims_str(nvinfer1::Dims dims)
{
    std::ostringstream oss;
    oss << "[" << dims.d[0];
    for(int i = 1 ; i < dims.nbDims ; ++i) {
        oss << ", " << dims.d[i];
    }
    oss << "]";
    return oss.str();
}

}

#endif
