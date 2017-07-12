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
#ifndef TFRT_IMAGENET_NETWORK_H
#define TFRT_IMAGENET_NETWORK_H

#include <tuple>
#include <memory>
#include <NvInfer.h>

#include "network.h"

namespace tfrt
{

class imagenet_network : public tfrt::network
{
public:
    /** Create imagenet network, with a given name.
     */
    imagenet_network(std::string name, uint32_t num_classes=1000, bool empty_class=false) :
        tfrt::network(name), m_num_classes{num_classes}, m_empty_class{empty_class} {}

    /** Load ImageNet classes information and descriptions.
     */
    bool load_info(const std::string& filename);

    /** Classify an image. Return a tuple <class, score>
     */
    std::pair<int, float> classify(float* rgba, uint32_t height, uint32_t width);

public:
    /** Number of classes. */
    uint32_t num_classes() const {
        return m_num_classes;
    }
    /** Synset code of a class. */
    std::string synset(uint32_t idx) const {
        return m_synset_classes[idx];
    }
    /** Description of a class. */
    std::string description(uint32_t idx) const {
        return m_desc_classes[idx];
    }

protected:
    // Number of classes in the model.
    uint32_t  m_num_classes;
    // Empty class in the first coordinate?
    bool  m_empty_class;

    // ImageNet classes synset(?) and descriptions
    std::vector<std::string>  m_synset_classes; // 1000 class ID's (ie n01580077, n04325704)
	std::vector<std::string>  m_desc_classes;
};

}

#endif
