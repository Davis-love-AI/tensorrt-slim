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
#include <stdexcept>

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
        nets["inception_v1"] = std::make_unique<inception1::net>();
        nets["inception_v2"] = std::make_unique<inception2::net>();
        nets["inception_v2b"] = std::make_unique<inception_v2b::net>();
        nets["inception_v2c"] = std::make_unique<inception_v2c::net>();
        nets["inception_v2d"] = std::make_unique<inception_v2d::net>();
        nets["inception_v2_group"] = std::make_unique<inception2_group::net>();

        nets["inception_v4"] = std::make_unique<inception4::net>();
        nets["inception_v4b"] = std::make_unique<inception_v4b::net>();

        nets["resnet_v1_50"] = std::make_unique<resnet_v1_50::net>();
        nets["resnet_v1_101"] = std::make_unique<resnet_v1_101::net>();
        nets["resnet_v1_152"] = std::make_unique<resnet_v1_152::net>();

        nets["resnet_v3_50"] = std::make_unique<resnet_v3_50::net>();
        nets["resnet_v3_50b"] = std::make_unique<resnet_v3_50_bis::net>();
        nets["resnet_v3_101"] = std::make_unique<resnet_v3_101::net>();
        nets["resnet_v3_152"] = std::make_unique<resnet_v3_152::net>();

        nets["resnext_50"] = std::make_unique<resnext_50::net>();

        nets["mobilenets_v1"] = std::make_unique<mobilenets::net>(1);
        nets["mobilenets_v1_gs4"] = std::make_unique<mobilenets::net>(4);
        nets["mobilenets_v1_gs8"] = std::make_unique<mobilenets::net>(8);
        nets["mobilenets_v1_gs16"] = std::make_unique<mobilenets::net>(16);
        nets["mobilenets_v1_gs32"] = std::make_unique<mobilenets::net>(32);

        nets["nasnet_mobile"] = std::make_unique<nasnet_mobile::net>();
        nets["nasnet_large"] = std::make_unique<nasnet_large::net>();

        nets["nasnet_v1b_mobile"] = std::make_unique<nasnet_v1b_mobile::net>();
        nets["nasnet_v1b_large"] = std::make_unique<nasnet_v1b_large::net>();
        nets["nasnet_v1c_mobile"] = std::make_unique<nasnet_v1c_mobile::net>();
        nets["nasnet_v1c_large"] = std::make_unique<nasnet_v1c_large::net>();

        nets["nasnet_v2_small"] = std::make_unique<nasnet_v2_small::net>();
        nets["nasnet_v2_medium"] = std::make_unique<nasnet_v2_medium::net>();

        // Segmentation networks.
        nets["ssd_inception2_v0"] = std::make_unique<ssd_inception2_v0::net>();
        nets["seg_inception2_v1"] = std::make_unique<seg_inception2_v1::net>();
        nets["seg_inception2_v1_5x5"] = std::make_unique<seg_inception2_v1_5x5::net>();
        nets["seg_inception2_logits_v1"] = std::make_unique<seg_inception2_logits_v1::net>();
        nets["seg_inception2_2x2"] = std::make_unique<seg_inception2_2x2::net>();
    }
    if (nets.find(name) == nets.end()) {
        LOG(ERROR) << "NETWORK: can find network named '" << name << "'.";
        std::cout << "Networks available:" << std::endl;
        for (auto it = nets.begin(); it != nets.end(); ++it) {
            std::cout << " * " << it->first << std::endl;
        }
        throw std::out_of_range("NETWORK: can find network.");
    }
    return std::move(nets.at(name));
}
}
