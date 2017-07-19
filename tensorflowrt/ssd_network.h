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
#ifndef TFRT_SSD_NETWORK_H
#define TFRT_SSD_NETWORK_H

#include <cmath>
#include <memory>
#include <NvInfer.h>

#include "network.h"
#include "ssd_network.pb.h"

namespace tfrt
{

class ssd_network : public tfrt::network
{
    /** Create SSD network, specifying the name.
     */
    ssd_network(std::string name) :
        tfrt::network(name),
        m_pb_ssd_network(std::make_unique<tfrt_pb::ssd_network>()) {
    }
    virtual ~ssd_network();


protected:
    // Protobuf network object.
    std::unique_ptr<tfrt_pb::ssd_network>  m_pb_ssd_network;
};

}

#endif
