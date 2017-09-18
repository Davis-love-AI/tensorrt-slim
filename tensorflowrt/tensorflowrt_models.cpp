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
// #include <fcntl.h>
// #include <unistd.h>
#include <memory>

#include "utils.h"
#include "tensorflowrt_models.h"

namespace tfrt
{
/** TODO: fix this messy implementation. */
std::unique_ptr<tfrt::network>&& nets_factory(const std::string& name)
{
    static std::map<std::string, std::unique_ptr<tfrt::network> > nets;
    // Fill the map at first call!
    if (nets.empty()) {
        // ImageNet classification.
        nets["inception1"] = std::make_unique<inception1::net>();
        nets["inception2"] = std::make_unique<inception2::net>();
        nets["resnet_v1_50"] = std::make_unique<resnet_v1_50::net>();
        nets["resnet_v1_101"] = std::make_unique<resnet_v1_101::net>();
        nets["resnet_v1_152"] = std::make_unique<resnet_v1_152::net>();

        // Segmentation networks.
        nets["ssd_inception2_v0"] = std::make_unique<ssd_inception2_v0::net>();
        nets["seg_inception2_v1"] = std::make_unique<seg_inception2_v1::net>();
        nets["seg_inception2_v1_5x5"] = std::make_unique<seg_inception2_v1_5x5::net>();
        nets["seg_inception2_logits_v1"] = std::make_unique<seg_inception2_logits_v1::net>();
        nets["seg_inception2_2x2"] = std::make_unique<seg_inception2_2x2::net>();
    }
    return std::move(nets.at(name));
}
}
